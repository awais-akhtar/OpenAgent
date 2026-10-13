"""Local native Segment/Edge operator for explicitly configured text families."""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import os

from .execution_service import ExecutionCancelled,_admit_checkpoint_publication
from .session_recovery import _path
from .native_library_identity import _stamp
from .publisher_media_files import read_regular
from .closed_set_scoring import ClosedSetScoringMixin

_REGISTRY = {}
_LOCK = threading.RLock()


def _pairs(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ValueError('duplicate native family metadata field: '+key)
        result[key] = value
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--family',required=True,choices=('mixtral','qwen3_5_moe_text','qwen4_exp_text','glm5_next_text','inkling_text','glm5_next','glm_moe_dsa'))
    result.add_argument('--checkpoint',required=True,type=Path)
    result.add_argument('--runtime-library',type=Path)
    result.add_argument('--kernel-library',type=Path)
    result.add_argument('--expert-library',type=Path)
    result.add_argument('--publisher-expert-mib',type=int)
    result.add_argument('--conversion-mib',type=int)
    result.add_argument('--device')
    result.add_argument('--attention-mib',type=int,default=64)
    result.add_argument('--head-mib',type=int,default=64)
    result.add_argument('--tokenizer',type=Path)
    result.add_argument('--context',required=True,type=int)
    result.add_argument('--runtime-mib',required=True,type=int)
    result.add_argument('--aggregate-runtime-mib',type=int,
                        help='aggregate declared budgets across all native range, edge and session owners')
    result.add_argument('--staging-mib',required=True,type=int)
    result.add_argument('--snapshot-mib',required=True,type=int)
    result.add_argument('--transfer-mib',required=True,type=int)
    result.add_argument('--layers-per-range',required=True,type=int)
    result.add_argument('--prefill-chunk',type=int,default=32)
    result.add_argument('--max-tokens',type=int,default=128)
    result.add_argument('--token-ids',required=True)
    result.add_argument('--stop-token-id',type=int,action='append',default=[])
    result.add_argument('--stop-prefix',type=str,action='append',default=[])
    result.add_argument('--expert-manifest',type=Path)
    result.add_argument('--expert-kernel')
    result.add_argument('--expert-cache-mib',type=int)
    result.add_argument('--expert-work-mib',type=int)
    result.add_argument('--ple-manifest',type=Path,
                        help='version-one explicit ordered Qwen4 PLE shard bindings')
    result.add_argument('--ple-cache-mib',type=int)
    result.add_argument('--stream-ple',action='store_true',
                        help='derive ordered PLE bindings from the publisher shard naming contract')
    result.add_argument('--temperature',type=float,default=0.0)
    result.add_argument('--top-k',type=int,default=0)
    result.add_argument('--top-p',type=float,default=1.0)
    result.add_argument('--min-p',type=float,default=0.0)
    result.add_argument('--repetition-penalty',type=float,default=1.0)
    result.add_argument('--frequency-penalty',type=float,default=0.0)
    result.add_argument('--presence-penalty',type=float,default=0.0)
    result.add_argument('--seed',type=int,default=0)
    result.add_argument('--continue-session',action='store_true')
    result.add_argument('--restore-cache',type=Path)
    result.add_argument('--save-cache',type=Path)
    result.add_argument('--cache-mib',type=int)
    return result


class _Cancellation:
    def __init__(self,callback):
        self.callback = callback

    def is_set(self):
        return bool(self.callback is not None and self.callback())


class _DeviceOwner:
    def __init__(self,root,args,manifest,config):
        self.family = args.family
        self.root,self.device,self.weight_bytes,self.runtime_bytes = root,args.device,args.staging_mib*1024**2,args.runtime_mib*1024**2
        if any(type(value) is not int or value<=0 for value in (args.attention_mib,args.head_mib)):
            raise ValueError('positive explicit device attention/head scratch required')
        self.attention_bytes,self.head_bytes = args.attention_mib*1024**2,args.head_mib*1024**2
        self.total_bytes = (getattr(args,'aggregate_runtime_mib',None) or args.runtime_mib)*1024**2
        self.store,self.provider = None,None
        if self.family=='glm_moe_dsa' and args.publisher_expert_mib is not None:
            import importlib
            from .publisher_checkpoint.storage import snapshot
            from .device_glm_dsa_experts import DeviceGlmDsaExperts
            torch=importlib.import_module('torch')
            original=snapshot(root,max_metadata_bytes=self.weight_bytes,max_tensors=200000,max_shards=4096)
            self.provider=DeviceGlmDsaExperts(original,config,torch=torch,device=self.device,
                max_bytes=args.publisher_expert_mib*1024**2,max_host_bytes=args.conversion_mib*1024**2)
        if manifest is not None:
            from .expert_store import DiskExpertStore
            from .torch_mixtral_experts import TorchMixtralExperts
            from .torch_qwen35_experts import TorchQwen35Experts
            from .torch_qwen4_experts import TorchQwen4Experts
            regions = {layer.id+'/'+expert:region for layer in manifest.layers for expert,region in layer.regions.items()}
            self.store = DiskExpertStore(root,regions,capacity_bytes=args.expert_cache_mib*1024**2)
            try:
                lattice = grid = None
                for layer in manifest.layers:
                    if layer.dtype=='iq3-format6-98-v1':
                        from .iq3_lattice import IQ3Grid
                        grid = IQ3Grid.read(root/'openagent-iq3-grid.bin',expected_sha256=layer.contract['grid_sha256'])
                    elif layer.dtype=='e8-lattice-caller-v1':
                        from .e8_lattice_expert import LatticeContract
                        lattice = LatticeContract.read(root/'openagent-e8-contract.bin',expected_identity=layer.contract['contract_sha256'])
                if self.family=='glm_moe_dsa':
                    if self.provider is not None:
                        raise ValueError('choose original or packed DSA expert streaming')
                    from .torch_glm_dsa_experts import TorchGlmDsaExperts
                    self.provider=TorchGlmDsaExperts(manifest,self.store,config,
                        max_work_bytes=args.expert_work_mib*1024**2,lattice_contract=lattice,iq3_grid=grid)
                else:
                    provider_type = {'mixtral':TorchMixtralExperts,'qwen3_5_moe_text':TorchQwen35Experts,
                                     'qwen4_exp_text':TorchQwen4Experts}[self.family]
                    self.provider = provider_type(manifest,self.store,max_work_bytes=args.expert_work_mib*1024**2,
                                                       lattice_contract=lattice,iq3_grid=grid)
            except BaseException:
                self.store.close()
                raise

    def create_execution_service(self,config,**options):
        from .execution_factories import open_device_mixtral_directory_service,open_device_qwen35_directory_service,open_device_qwen4_directory_service
        if self.family in ('inkling_text','glm5_next_text','glm_moe_dsa'):
            from .execution_factories import (open_device_inkling_directory_service,
                open_device_glm5_directory_service,open_device_glm_dsa_directory_service)
            if self.family!='glm5_next_text' and options['context_tokens']!=config['max_position_embeddings']:
                raise ValueError('device context must match admitted checkpoint capacity')
            factory = {'inkling_text':open_device_inkling_directory_service,
                'glm5_next_text':open_device_glm5_directory_service,
                'glm_moe_dsa':open_device_glm_dsa_directory_service}[self.family]
            capacity = {'context_tokens':options['context_tokens']} if self.family=='glm5_next_text' else {}
            if self.family=='glm_moe_dsa':capacity['expert_provider']=self.provider
            return factory(self.root,device=self.device,max_weight_bytes=self.weight_bytes,
                runtime_max_bytes=self.runtime_bytes,max_total_model_bytes=self.total_bytes,
                max_new_tokens=options['max_new_tokens'],max_sessions=options['max_sessions'],
                prefill_chunk_tokens=options['prefill_chunk_tokens'],
                max_snapshots_per_session=options.get('max_snapshots_per_session',4),
                max_logit_buffer_bytes=options['max_transfer_bytes'],
                max_work_bytes=self.attention_bytes,max_head_bytes=self.head_bytes,**capacity)
        if options['context_tokens']!=config['max_position_embeddings']:
            raise ValueError('device service context must match admitted checkpoint capacity')
        factory = {'mixtral':open_device_mixtral_directory_service,'qwen3_5_moe_text':open_device_qwen35_directory_service,
                   'qwen4_exp_text':open_device_qwen4_directory_service}[self.family]
        scratch = {'max_attention_bytes':self.attention_bytes} if self.family=='mixtral' else {'max_layer_bytes':self.attention_bytes}
        return factory(self.root,device=self.device,max_weight_bytes=self.weight_bytes,
            runtime_max_bytes=self.runtime_bytes,max_new_tokens=options['max_new_tokens'],max_sessions=options['max_sessions'],
            prefill_chunk_tokens=options['prefill_chunk_tokens'],max_logit_buffer_bytes=options['max_transfer_bytes'],
            max_snapshots_per_session=options.get('max_snapshots_per_session',4),
            max_total_model_bytes=self.total_bytes,expert_provider=self.provider,max_head_bytes=self.head_bytes,**scratch)


class _FlatOwner:
    """Borrow admitted immutable artifacts; services own native model state."""
    def __init__(self,root,args,manifest=None,config=None):
        self.root,self.family,self.library = root,args.family,args.kernel_library
        self.runtime_bytes,self.weight_bytes = args.runtime_mib*1024**2,args.staging_mib*1024**2
        self.total_bytes = (getattr(args,'aggregate_runtime_mib',None) or 2*args.runtime_mib)*1024**2
        self.expert_library=getattr(args,'expert_library',None)
        self.expert_bytes=(getattr(args,'publisher_expert_mib',None) or 0)*1024**2
        self.conversion_bytes=(getattr(args,'conversion_mib',None) or 0)*1024**2
        self.provider=self.expert_runtime=self.kernel_resource=None
        if manifest is not None:
            if self.family!='glm_moe_dsa':
                raise ValueError('flat packed experts require the exact DSA family')
            from .olmoe_cli import _kernel_resources
            from .expert_runtime import SparseExpertRuntime
            from .glm_dsa_streamed_experts import GlmDsaStreamedExperts
            try:
                kernels,self.kernel_resource=_kernel_resources(root,args.expert_kernel,args.expert_cache_mib/1024)
                if kernels is None:raise ValueError('explicit compiled DSA expert kernel required')
                self.expert_runtime=SparseExpertRuntime(root,manifest,capacity_bytes=args.expert_cache_mib*1024**2,
                    kernels=kernels,preferred_kernel=args.expert_kernel,allow_kernel_fallback=False)
                self.provider=GlmDsaStreamedExperts(self.expert_runtime,config,
                    max_work_bytes=args.expert_work_mib*1024**2)
            except BaseException:
                if self.expert_runtime is not None:self.expert_runtime.close()
                if self.kernel_resource is not None and callable(getattr(self.kernel_resource,'close',None)):
                    self.kernel_resource.close()
                raise

    def create_execution_service(self,config,**options):
        if self.family=='glm_moe_dsa':
            from .execution_factories import open_native_glm_dsa_directory_service
            return open_native_glm_dsa_directory_service(self.root,library=self.library,
                max_weight_bytes=self.weight_bytes,runtime_max_bytes=self.runtime_bytes,
                max_total_model_bytes=self.total_bytes,context_tokens=options['context_tokens'],
                max_new_tokens=options['max_new_tokens'],max_sessions=options['max_sessions'],
                prefill_chunk_tokens=options['prefill_chunk_tokens'],
                max_snapshots_per_session=options.get('max_snapshots_per_session',4),
                max_logit_buffer_bytes=options['max_transfer_bytes'],expert_provider=self.provider)
        if self.family=='glm5_next':
            from .execution_factories import open_publisher_glm5_text_service
            return open_publisher_glm5_text_service(self.root,library=self.library,
                expert_library=self.expert_library,max_expert_bytes=self.expert_bytes,
                max_conversion_bytes=self.conversion_bytes,max_weight_bytes=self.weight_bytes,
                runtime_max_bytes=self.runtime_bytes,max_total_model_bytes=self.total_bytes,
                context_tokens=options['context_tokens'],max_new_tokens=options['max_new_tokens'],
                max_sessions=options['max_sessions'],prefill_chunk_tokens=options['prefill_chunk_tokens'],
                max_snapshots_per_session=options.get('max_snapshots_per_session',4))
        from .execution_factories import open_flat_native_directory_service
        return open_flat_native_directory_service(self.root,family=self.family,library=self.library,
            max_weight_bytes=self.weight_bytes,runtime_max_bytes=self.runtime_bytes,
            max_total_model_bytes=self.total_bytes,context_tokens=options['context_tokens'],
            max_new_tokens=options['max_new_tokens'],max_sessions=options['max_sessions'],
            prefill_chunk_tokens=options['prefill_chunk_tokens'],
            max_snapshots_per_session=options.get('max_snapshots_per_session',4))


class PublisherSession(ClosedSetScoringMixin):
    """Gateway-compatible owned service; public registry owners stay pinned."""
    def __init__(self,args,*,cancelled=None):
        self.service,self.session_id,self.tokenizer = None,None,None
        self._lock = threading.RLock()
        self._closed = False
        self.family=args.family
        self._key = tuple((name,str(getattr(args,name))) for name in
            ('family','checkpoint','tokenizer','runtime_library','kernel_library','context','runtime_mib',
             'staging_mib','snapshot_mib','transfer_mib','layers_per_range','prefill_chunk'))
        self._key += tuple((name,str(getattr(args,name,None))) for name in
            ('expert_manifest','expert_kernel','expert_cache_mib','expert_work_mib','device','attention_mib','head_mib','aggregate_runtime_mib','ple_manifest','ple_cache_mib','stream_ple','expert_library','publisher_expert_mib','conversion_mib'))
        if args.device is not None and args.family not in ('mixtral','qwen3_5_moe_text','qwen4_exp_text','inkling_text','glm5_next_text','glm_moe_dsa'):
            raise ValueError('unsupported explicit device text family')
        flat_family = args.family in ('glm5_next_text','inkling_text','glm5_next') or (
            args.family=='glm_moe_dsa' and args.runtime_library is None)
        if args.family=='glm5_next':
            if args.expert_library is None or any(type(v) is not int or v<=0 for v in
                    (args.publisher_expert_mib,args.conversion_mib)):
                raise ValueError('original GLM loading requires an explicit expert library and streaming/conversion budgets')
        elif args.family=='glm_moe_dsa' and args.device is not None:
            if args.expert_library is not None:
                raise ValueError('DSA device expert streaming uses the requested device runtime')
            if any(value is not None for value in (args.publisher_expert_mib,args.conversion_mib)) and any(
                    type(value) is not int or value<=0 for value in (args.publisher_expert_mib,args.conversion_mib)):
                raise ValueError('DSA original expert streaming requires both positive device and conversion budgets')
        elif any(getattr(args,name,None) is not None for name in ('expert_library','publisher_expert_mib','conversion_mib')):
            raise ValueError('original publisher expert options require the GLM wrapper family')
        if args.device is None and (args.kernel_library is None or (not flat_family and args.runtime_library is None)):
            raise ValueError('native execution requires explicit runtime and kernel libraries')
        cancel = _Cancellation(cancelled)
        if cancel.is_set():
            raise ExecutionCancelled('native family load cancelled')
        for name in ('context','runtime_mib','staging_mib','snapshot_mib','transfer_mib','layers_per_range','max_tokens'):
            value = getattr(args,name)
            if type(value) is not int or value<=0:
                raise ValueError('positive explicit native family admission required: '+name)
        root = _path(args.checkpoint).resolve(strict=True)
        self._artifact_root=root
        aggregate_mib = getattr(args, 'aggregate_runtime_mib', None)
        if aggregate_mib is not None and (type(aggregate_mib) is not int or aggregate_mib <= 0):
            raise ValueError('positive aggregate runtime MiB required')
        if not root.is_dir():
            raise ValueError('local native family checkpoint directory required')
        raw=read_regular(root/'config.json',max_bytes=1024**2)
        config_digest=hashlib.sha256(raw).hexdigest()
        if len(raw)>1024*1024:
            raise MemoryError('native family config exceeds admission')
        config = json.loads(raw,object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if type(config) is not dict or config.get('model_type')!=args.family:
            raise ValueError('configured native family differs from checkpoint identity')
        if args.family=='glm5_next':
            from .publisher_family_contracts import component_contract
            config=component_contract(config,arithmetic='fp32').text_config
        self.vocabulary = (config.get('unpadded_vocab_size') or config['vocab_size']) if args.family=='inkling_text' else config['vocab_size']
        eos = config.get('eos_token_id')
        self.stops = [] if eos is None else [eos] if type(eos) is int else list(eos)
        tokenizer_path = _path(args.tokenizer or root/'tokenizer.json').resolve(strict=True)
        if tokenizer_path.stat().st_size>args.staging_mib*1024**2:
            raise MemoryError('tokenizer exceeds native metadata admission')
        from tokenizers import Tokenizer
        tokenizer_bytes=read_regular(tokenizer_path,max_bytes=args.staging_mib*1024**2)
        tokenizer_digest=hashlib.sha256(tokenizer_bytes).hexdigest()
        self.tokenizer = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
        del tokenizer_bytes
        files = {root/'config.json',tokenizer_path}
        parsed_digests={root/'config.json':config_digest,tokenizer_path:tokenizer_digest}
        parsed_stamps={}
        files.update(_path(path).resolve(strict=True) for path in (args.runtime_library,args.kernel_library,args.expert_library) if path is not None)
        manifest = None
        ple_bindings = None
        ple_inventory = None
        ple_manifest = getattr(args,'ple_manifest',None)
        ple_budget = getattr(args,'ple_cache_mib',None)
        if getattr(args,'stream_ple',False):
            if ple_manifest is not None:
                raise ValueError('choose derived publisher PLE bindings or an explicit manifest')
            if args.family!='qwen4_exp_text' or args.device is not None or type(ple_budget) is not int or ple_budget<=0:
                raise ValueError('native CPU Qwen4 PLE streaming requires positive cache admission')
            from .publisher_checkpoint.storage import snapshot
            from .publisher_weight_mapping import qwen4_ple_bindings
            ple_inventory=snapshot(root,max_metadata_bytes=args.staging_mib*1024**2,
                max_tensors=200000,max_shards=4096)
            ple_bindings=qwen4_ple_bindings(ple_inventory,config)
            if not ple_bindings: raise ValueError('checkpoint contains no configured PLE layers')
        if ple_manifest is not None:
            if args.family!='qwen4_exp_text' or args.device is not None:
                raise ValueError('PLE row streaming requires the native CPU Qwen4 text decoder')
            if type(ple_budget) is not int or ple_budget<=0:
                raise ValueError('explicit positive PLE cache admission required')
            path = _path(ple_manifest).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError('PLE manifest escapes admitted checkpoint root')
            document=read_regular(path,max_bytes=1024**2)
            parsed_digests[path]=hashlib.sha256(document).hexdigest()
            if len(document)>1024**2:
                raise MemoryError('PLE bindings exceed metadata admission')
            specification = json.loads(document,object_pairs_hook=_pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if type(specification) is not dict or set(specification)!={'version','bindings'} or specification['version']!=1 or type(specification['bindings']) is not list:
                raise ValueError('version-one explicit PLE binding inventory required')
            ple_bindings = {}
            for binding in specification['bindings']:
                if type(binding) is not dict or set(binding)!={'layer','tensors'}:
                    raise ValueError('PLE binding requires layer and ordered tensors')
                layer,names = binding['layer'],binding['tensors']
                if type(layer) is not int or layer<0 or layer in ple_bindings or type(names) is not list or not names or any(type(name) is not str or not name for name in names):
                    raise ValueError('invalid or duplicate PLE layer binding')
                ple_bindings[layer] = names
            files.add(path)
        elif ple_budget is not None and not getattr(args,'stream_ple',False):
            raise ValueError('PLE cache admission requires an explicit manifest')
        if flat_family and args.family!='glm_moe_dsa' and any(getattr(args,name,None) is not None for name in
                ('expert_manifest','expert_kernel','expert_cache_mib','expert_work_mib')):
            raise ValueError('flat native family operator requires its complete resident checkpoint inventory')
        if args.expert_manifest is not None:
            from .expert_runtime import SparseExpertManifest
            if (args.device is None and not args.expert_kernel) or any(type(value) is not int or value<=0 for value in
                    (args.expert_cache_mib,args.expert_work_mib)):
                raise ValueError('packed execution requires explicit kernel/cache/work admission')
            manifest_path = _path(args.expert_manifest).resolve(strict=True)
            if not manifest_path.is_relative_to(root):
                raise ValueError('expert manifest must be inside admitted model root')
            manifest_stamp=_stamp(manifest_path.stat())
            manifest = SparseExpertManifest.read(manifest_path)
            if _stamp(_path(manifest_path).stat())!=manifest_stamp:
                raise ValueError('expert manifest changed during admission')
            parsed_stamps[manifest_path]=manifest_stamp
            if manifest.architecture!=args.family:
                raise ValueError('packed manifest family differs from configured decoder')
            files.add(manifest_path)
            for layer in manifest.layers:
                for region in layer.regions.values():
                    if region.storage!='':
                        raise ValueError('native family operator admits local expert storage only')
                    path = _path(root/region.path).resolve(strict=True)
                    if not path.is_relative_to(root):
                        raise ValueError('packed expert container escapes admitted model root')
                    files.add(path)
            for filename in ('openagent-iq3-grid.bin','openagent-e8-contract.bin'):
                path = root/filename
                if path.is_file():
                    files.add(path)
        elif any(value is not None for value in (args.expert_kernel,args.expert_cache_mib,args.expert_work_mib)):
            raise ValueError('expert options require an explicit packed manifest')
        files.update(root.glob('*.safetensors'))
        index = _path(root/'model.safetensors.index.json')
        self._shard_inventory=set(root.glob('*.safetensors'))
        self._index_present=index.exists()
        if index.is_file():
            files.add(index)
            document=read_regular(index,max_bytes=16*1024**2)
            parsed_digests[index]=hashlib.sha256(document).hexdigest()
            if len(document)>16*1024**2:
                raise MemoryError('native shard index exceeds admission')
            mapping = json.loads(document,object_pairs_hook=_pairs).get('weight_map')
            if type(mapping) is not dict or any(type(value) is not str for value in mapping.values()):
                raise ValueError('exact native checkpoint shard mapping required')
            for name in set(mapping.values()):
                path = _path(root/name).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError('native checkpoint shard escapes model root')
                files.add(path)
        digest = hashlib.sha256()
        self._artifacts={}
        for path in sorted(files,key=str):
            path = _path(path).resolve(strict=True)
            before = path.stat()
            if path in parsed_stamps and _stamp(before)!=parsed_stamps[path]:
                raise ValueError('parsed native manifest changed before artifact admission')
            if not path.is_file():
                raise ValueError('regular native artifact required')
            digest.update(str(path).encode())
            content=hashlib.sha256();count=0
            with path.open('rb') as source:
                if _stamp(os.fstat(source.fileno()))!=_stamp(before):
                    raise ValueError('native artifact changed during opening')
                for data in iter(lambda: source.read(1024*1024),b''):
                    if cancel.is_set():
                        raise ExecutionCancelled('native artifact admission cancelled')
                    count+=len(data)
                    if count>before.st_size:raise ValueError('native artifact grew during admission')
                    digest.update(data);content.update(data)
                if count!=before.st_size or _stamp(os.fstat(source.fileno()))!=_stamp(before):
                    raise ValueError('native artifact changed during reading')
            after = _path(path).stat()
            if _stamp(before)!=_stamp(after):
                raise ValueError('native artifact changed during admission')
            expected=parsed_digests.get(path)
            if expected is not None and content.hexdigest()!=expected:
                raise ValueError('native metadata changed before artifact admission')
            self._artifacts[path]=_stamp(after)
        self.checkpoint_identity = digest.hexdigest()
        registry_key = (self._key,self.checkpoint_identity)
        with _LOCK:
            owner = _REGISTRY.get(registry_key)
            if owner is None and flat_family and args.device is None:
                owner = _FlatOwner(root,args,manifest,config)
                _REGISTRY[registry_key] = owner
            if owner is None and args.device is not None:
                owner = _DeviceOwner(root,args,manifest,config)
                _REGISTRY[registry_key] = owner
            if owner is None:
                from .mixtral_public_adapters import NativeMixtralPublicAdapters
                from .qwen35_public_adapters import NativeQwen35PublicAdapters
                from .qwen4_public_adapters import NativeQwen4PublicAdapters
                from .glm_dsa_public_adapters import NativeGlmDsaPublicAdapters
                types = {'mixtral':NativeMixtralPublicAdapters,'qwen3_5_moe_text':NativeQwen35PublicAdapters,
                         'qwen4_exp_text':NativeQwen4PublicAdapters,'glm_moe_dsa':NativeGlmDsaPublicAdapters}
                suffix = hashlib.sha256(repr(registry_key).encode()).hexdigest()[:24]
                runtime = resource = ple_store = None
                try:
                    provider = None
                    if ple_bindings is not None:
                        from .publisher_checkpoint.storage import snapshot
                        from .qwen4_ple_store import Qwen4PLEStore
                        inventory = ple_inventory or snapshot(root,max_metadata_bytes=args.staging_mib*1024**2,
                            max_tensors=200000,max_shards=4096)
                        ple_store = Qwen4PLEStore(inventory,config,ple_bindings,
                            max_bytes=ple_budget*1024**2)
                    if manifest is not None:
                        from .olmoe_cli import _kernel_resources
                        from .expert_runtime import SparseExpertRuntime
                        from .mixtral_streamed_experts import MixtralStreamedExperts
                        from .qwen35_streamed_experts import Qwen35StreamedExperts
                        from .qwen4_streamed_experts import Qwen4StreamedExperts
                        kernels,resource = _kernel_resources(root,args.expert_kernel,args.expert_cache_mib/1024)
                        if kernels is None:
                            raise ValueError('explicit compiled packed-expert kernel required')
                        runtime = SparseExpertRuntime(root,manifest,capacity_bytes=args.expert_cache_mib*1024**2,
                            kernels=kernels,preferred_kernel=args.expert_kernel,allow_kernel_fallback=False)
                        provider_types = {'mixtral':MixtralStreamedExperts,'qwen3_5_moe_text':Qwen35StreamedExperts,
                                          'qwen4_exp_text':Qwen4StreamedExperts}
                        mixtral = args.family=='mixtral'
                        if args.family=='glm_moe_dsa':
                            from .glm_dsa_streamed_experts import GlmDsaStreamedExperts
                            provider=GlmDsaStreamedExperts(runtime,config,max_work_bytes=args.expert_work_mib*1024**2)
                        else:
                            provider = provider_types[args.family](runtime,
                                layer_ids=tuple(str(index) for index in range(config['num_hidden_layers'])),
                                hidden_size=config['hidden_size'],
                                intermediate_size=config['intermediate_size' if mixtral else 'moe_intermediate_size'],
                                expert_count=config['num_local_experts' if mixtral else 'num_experts'],
                                selected_experts=config['num_experts_per_tok'],
                                max_work_bytes=args.expert_work_mib*1024**2)
                    ple_options = {'ple_provider':ple_store} if ple_store is not None else {}
                    expert_options = {'expert_provider':provider}
                    if args.family=='qwen3_5_moe_text':expert_options['context_tokens']=args.context
                    owner = types[args.family](args.runtime_library,kernel_library=args.kernel_library,
                        model_directory=root,max_staging_bytes=args.staging_mib*1024**2,
                        max_snapshot_bytes=args.snapshot_mib*1024**2,max_batch_rows=args.prefill_chunk,
                        max_resource_bytes=(aggregate_mib if aggregate_mib is not None else
                            args.runtime_mib*2*(2*((config['num_hidden_layers']+args.layers_per_range-1)//args.layers_per_range)+1))*1024**2,
                        segment_id='oa-native-segment-'+suffix,edge_id='oa-native-edge-'+suffix,
                        **expert_options,**ple_options)
                    # Registered callbacks borrow these resources for process
                    # lifetime; retain them even if edge registration fails.
                    owner.operator_expert_runtime,owner.operator_kernel_resource = runtime,resource
                    owner.operator_ple_store = ple_store
                    owner.register()
                    _REGISTRY[registry_key] = owner
                except BaseException as primary:
                    if owner is not None and owner.registered_segment:
                        _REGISTRY[registry_key] = owner
                    else:
                        previous=getattr(primary,'cleanup',None)
                        pending=([previous] if callable(previous) else [])
                        pending.extend(value.close for value in (ple_store,runtime,resource)
                            if value is not None and callable(getattr(value,'close',None)))
                        def cleanup_construction():
                            while pending:
                                pending[0]();pending.pop(0)
                        try:cleanup_construction()
                        except BaseException as cleanup:
                            primary.cleanup=cleanup_construction
                            primary.add_note('native bank construction cleanup retained: '+repr(cleanup))
                    raise
            if args.device is None and not flat_family and not owner.registered_edge:
                owner.register()
        try:
            self.service = owner.create_execution_service(config,checkpoint_identity=self.checkpoint_identity,
                context_tokens=args.context,native_memory_bytes=args.runtime_mib*1024**2,
                layers_per_range=args.layers_per_range,max_transfer_bytes=args.transfer_mib*1024**2,
                max_new_tokens=args.context,max_sessions=2,prefill_chunk_tokens=args.prefill_chunk)
            self.session_id = self.service.open_session()
            self._artifacts_unchanged()
            if cancel.is_set():
                raise ExecutionCancelled('native family service load cancelled')
        except BaseException as error:
            previous=getattr(error,'cleanup',None)
            pending=([previous] if callable(previous) and previous!=self.close else [])+[self.close]
            def cleanup_session():
                while pending:
                    pending[0]();pending.pop(0)
            try:cleanup_session()
            except BaseException as cleanup:
                error.cleanup=cleanup_session
                error.add_note('native family session cleanup remains pending: '+repr(cleanup))
            raise

    def _artifacts_unchanged(self):
        root=_path(self._artifact_root)
        if (set(root.glob('*.safetensors'))!=self._shard_inventory or
                _path(root/'model.safetensors.index.json').exists()!=self._index_present):
            raise ValueError('native checkpoint shard inventory changed after admission')
        for path,stamp in self._artifacts.items():
            if _stamp(_path(path).stat())!=stamp:
                raise ValueError('native artifact changed after admission: '+str(path))

    def run(self,args,*,emit=None,cancelled=None,checkpoint_sink=None,max_checkpoint_bytes=0):
        with self._lock:
            _admit_checkpoint_publication(checkpoint_sink,max_checkpoint_bytes)
            if self._closed:
                raise RuntimeError('native family session is closed')
            self._artifacts_unchanged()
            key = tuple((name,str(getattr(args,name,None))) for name,_ in self._key)
            if key!=self._key:
                raise ValueError('native session options changed after admission')
            tokens = json.loads(args.token_ids)
            if type(tokens) is not list or not tokens or len(tokens)>args.context or any(
                    type(token) is not int or not 0<=token<self.vocabulary for token in tokens):
                raise ValueError('bounded native prompt token inventory required')
            stops = tuple(self.stops)+tuple(args.stop_token_id)
            if any(type(token) is not int or not 0<=token<self.vocabulary for token in stops):
                raise ValueError('invalid native stop token ID')
            prefixes=[]
            for raw_prefix in getattr(args,'stop_prefix',()):
                try:
                    parts=raw_prefix.split(',')
                    if not raw_prefix or any(not value.strip() for value in parts):
                        raise ValueError('stop prefixes require comma-separated token IDs')
                    prefix=tuple(int(value.strip()) for value in parts)
                except (TypeError, ValueError) as error:
                    raise ValueError('stop prefixes require comma-separated token IDs') from error
                if (not prefix or len(prefix)>128 or
                        any(token<0 or token>=self.vocabulary for token in prefix)):
                    raise ValueError('invalid native stop prefix')
                prefixes.append(prefix)
            if len(prefixes)>64 or len(set(prefixes))!=len(prefixes):
                raise ValueError('bounded unique native stop-prefix inventory required')
            from .sampling import SamplingPolicy
            sampling = SamplingPolicy(temperature=args.temperature,top_k=args.top_k,
                top_p=args.top_p,min_p=args.min_p,repetition_penalty=args.repetition_penalty,
                frequency_penalty=args.frequency_penalty,presence_penalty=args.presence_penalty,
                seed=args.seed)
            if sampling.top_k>self.vocabulary:
                raise ValueError('sampling top-k exceeds admitted model vocabulary')
            if args.restore_cache is not None or args.save_cache is not None:
                if type(args.cache_mib) is not int or args.cache_mib<=0:
                    raise ValueError('explicit complete-cache transport budget required')
            if args.restore_cache is not None:
                self.service.load_session_cache(self.session_id,args.restore_cache,max_bytes=args.cache_mib*1024**2)
            elif not args.continue_session:
                self.service.reset(self.session_id)
            def event(value):
                if emit is not None and value.get('event')=='generation.token':
                    emit(dict(event='tokens',token_ids=[value['token_id']]))
            result = self.service.generate(self.session_id,tokens,max_new_tokens=args.max_tokens,
                stop_ids=stops,stop_prefixes=prefixes,cancel=_Cancellation(cancelled),on_event=event,sampling=sampling,
                checkpoint_sink=checkpoint_sink,max_checkpoint_bytes=max_checkpoint_bytes,
                application_identity=self.checkpoint_identity)
            generated = list(result.token_ids)
            cache_receipt = None
            if args.save_cache is not None:
                cache_receipt = self.service.save_session_cache(self.session_id,args.save_cache,max_bytes=args.cache_mib*1024**2)
            return dict(generated_ids=generated,text=self.tokenizer.decode(generated,skip_special_tokens=True),
                family=args.family,checkpoint_identity=self.checkpoint_identity,
                finish_reason=result.finish_reason,context_tokens=result.context_tokens,
                sampling=dict(vars(sampling)),cache=cache_receipt,
                numerical_parity_verified=False,platform_certified=False)

    def resume_generation(self,source,*,max_bytes,max_new_tokens=None,emit=None,cancelled=None,
                          checkpoint_sink=None,max_checkpoint_bytes=0):
        with self._lock:
            _admit_checkpoint_publication(checkpoint_sink,max_checkpoint_bytes)
            if self._closed:raise RuntimeError('native publisher session is closed')
            self._artifacts_unchanged()
            if max_new_tokens is not None and (type(max_new_tokens) is not int or
                    not 1<=max_new_tokens<=self.service.limits.max_new_tokens):
                raise ValueError('admitted positive generation continuation required')
            cancel=_Cancellation(cancelled)
            boundary=self.service.restore_generation_checkpoint(self.session_id,source,max_bytes=max_bytes,
                cancel=cancel,application_identity=self.checkpoint_identity)
            emitted=[]
            def event(value):
                if emit is not None and value.get('event')=='generation.token':
                    emitted.append(value['token_id'])
                    emit(dict(event='tokens',token_ids=[value['token_id']],
                        text=self.tokenizer.decode(emitted,skip_special_tokens=True)))
            result=self.service.resume_generation(self.session_id,boundary,max_new_tokens=max_new_tokens,
                cancel=cancel,on_event=event,checkpoint_sink=checkpoint_sink,max_checkpoint_bytes=max_checkpoint_bytes,
                application_identity=self.checkpoint_identity)
            generated=list(result.token_ids)
            return dict(generated_ids=generated,text=self.tokenizer.decode(generated,skip_special_tokens=True),
                family=self.family,checkpoint_identity=self.checkpoint_identity,finish_reason=result.finish_reason,
                context_tokens=result.context_tokens,numerical_parity_verified=False,platform_certified=False)

    def continuation_store(self,directory,*,max_entries,max_total_bytes,max_cache_bytes,expiry_seconds=None,indexed=True):
        from .continuation_store import DurableContinuationStore
        from .indexed_continuation_store import IndexedContinuationStore
        with self._lock:
            if self._closed:raise RuntimeError('native publisher session is closed')
            self._artifacts_unchanged()
            if type(indexed) is not bool:
                raise ValueError('explicit boolean continuation index policy required')
            factory = IndexedContinuationStore if indexed else DurableContinuationStore
            return factory(directory,model_identity=self.checkpoint_identity,
                max_entries=max_entries,max_total_bytes=max_total_bytes,max_cache_bytes=max_cache_bytes,
                expiry_seconds=expiry_seconds)

    def save_continuation(self,store,key):
        with self._lock:
            if self._closed or store.identity!=self.checkpoint_identity:
                raise ValueError('open matching native continuation owner required')
            self._artifacts_unchanged()
            return store.save(self.service,self.session_id,key)

    def recover_continuation(self,store,key):
        with self._lock:
            if self._closed or store.identity!=self.checkpoint_identity:
                raise ValueError('open matching native continuation owner required')
            self._artifacts_unchanged()
            return store.recover_into(self.service,self.session_id,key)

    def close(self):
        with self._lock:
            if self.service is not None:
                self.service.close()
                self.service = None
            self.session_id,self.tokenizer = None,None
            self._closed = True


def main(argv=None):
    args = parser().parse_args(argv)
    session = PublisherSession(args)
    try:
        result = session.run(args)
        print(json.dumps(result,sort_keys=True,allow_nan=False))
    finally:
        session.close()
    return 0


__all__ = ['parser','PublisherSession','main']

if __name__=='__main__':
    raise SystemExit(main())
