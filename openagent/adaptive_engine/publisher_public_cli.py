"""Explicit original K3/V4.1 text serving through native Segment and Edge ABI."""
import argparse
import copy
import hashlib
import json
import threading
from pathlib import Path

from .native_family_cli import PublisherSession as _TextPublisherSession
from .publisher_media_files import read_regular
from .session_recovery import _path
from .native_library_identity import _stamp, NativeLibraryIdentity
from .owned_resources import OwnedResources

public_publisher_operator=True


def parser(*, family):
    if family=='kimi_k3':
        from .publisher_k3_cli import parser as base_parser
    elif family=='deepseek_v41':
        from .publisher_v41_cli import parser as base_parser
    else:
        raise ValueError('exact native public publisher family required')
    result=base_parser()
    result.set_defaults(family=family,public_text_abi=True,top_k=0,top_p=1.0,min_p=0.0,
        repetition_penalty=1.0,frequency_penalty=0.0,presence_penalty=0.0)
    for name in ('executor','resource','work','snapshot'):
        result.add_argument('--public-'+name+'-mib',type=int,required=True)
    result.add_argument('--public-runtime-library',type=Path,required=True)
    result.add_argument('--public-snapshots',type=int,default=4)
    result.add_argument('--restore-cache',type=Path)
    result.add_argument('--save-cache',type=Path)
    result.add_argument('--cache-mib',type=int)
    result.add_argument('--continue-session',action='store_true')
    return result


class PublisherSession(_TextPublisherSession):
    """Own admitted checkpoint sources until all public callback sessions close."""
    def __init__(self,args,*,cancelled=None):
        self._lock=threading.RLock()
        self._closed=False
        self.service=self.session_id=self.tokenizer=None
        self._resources=OwnedResources()
        self.family=args.family
        if self.family not in ('kimi_k3','deepseek_v41'):
            raise ValueError('original native public publisher identity required')
        # This adapter publishes the complete *text* backbone over the
        # Segment/Edge wire.  It must reject every media/auxiliary switch at
        # the public boundary instead of silently accepting an option that its
        # factory cannot carry through the text-only ABI.  In particular K3's
        # image-token and vision-tower options are separate from ``--image``.
        media_or_auxiliary = (
            'device', 'arithmetic_device', 'fla_version', 'include_vision',
            'image', 'image_token_id', 'image_mib', 'vision_tower_epsilon',
            'dspark',
        )
        if any(getattr(args, name, None) for name in media_or_auxiliary):
            raise ValueError('public text composition requires native text arithmetic; media and DSpark retain their original operators')
        if any(type(getattr(args,'public_'+name+'_mib',None)) is not int or
               getattr(args,'public_'+name+'_mib')<=0 for name in ('executor','resource','work','snapshot')):
            raise ValueError('explicit positive complete public publisher budgets required')
        if cancelled is not None and not callable(cancelled):
            raise TypeError('callable public publisher cancellation required')
        def check():
            if cancelled is not None and cancelled():
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('public publisher construction cancelled')
        check()
        root=_path(args.checkpoint).resolve(strict=True)
        args.checkpoint=root
        args.tokenizer=_path(args.tokenizer or root/'tokenizer.json').resolve(strict=True)
        self._artifact_root=root
        self._shard_inventory=set(root.glob('*.safetensors'))
        self._index_present=_path(root/'model.safetensors.index.json').exists()
        paths={root/'config.json',args.tokenizer,*self._shard_inventory}
        if self._index_present:
            index=root/'model.safetensors.index.json'
            document=json.loads(read_regular(index,max_bytes=args.metadata_mib*1024**2))
            mapping=document.get('weight_map')
            if type(mapping) is not dict or any(type(name) is not str for name in mapping.values()):
                raise ValueError('complete original publisher shard mapping required')
            paths.add(index)
            for name in set(mapping.values()):
                path=_path(root/name).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError('public publisher shard escapes original model root')
                paths.add(path)
        paths.update(value for name,value in vars(args).items() if name.endswith('_library') and value is not None)
        self._artifacts={}
        digest=hashlib.sha256()
        for path in sorted(paths,key=str):
            path=_path(path).resolve(strict=True)
            stamp=_stamp(path.stat())
            if not path.is_file():
                raise ValueError('regular pinned public publisher source file required')
            self._artifacts[path]=stamp
            digest.update(repr((str(path),stamp)).encode())
        original=json.loads(read_regular(root/'config.json',max_bytes=args.metadata_mib*1024**2))
        if original.get('model_type')!=self.family:
            raise ValueError('original checkpoint identity differs from public operator')
        token_bytes=read_regular(args.tokenizer,max_bytes=getattr(args,'input_mib',args.metadata_mib)*1024**2)
        digest.update(hashlib.sha256(token_bytes).digest())
        for path in sorted((path for path in paths if path.suffix.lower() in ('.dll','.so','.dylib')),key=str):
            digest.update(NativeLibraryIdentity.read(path).sha256.encode())
        self.checkpoint_identity='openagent-public-publisher-artifacts-v1:'+digest.hexdigest()
        self._key=tuple((name,str(value)) for name,value in sorted(vars(args).items())
            if name not in ('prompt','token_ids','max_tokens','temperature','seed','stop_token_id',
                            'stream','restore_cache','save_cache','continue_session','cache_mib'))
        self.stops=original.get('text_config',original).get('eos_token_id',())
        self.stops=(self.stops,) if type(self.stops) is int else tuple(self.stops or ())
        try:
            from tokenizers import Tokenizer
            from .execution_factories import (K3Libraries,create_public_k3_execution_service,
                open_publisher_v41_service,compose_public_publisher_service)
            self.tokenizer=Tokenizer.from_str(token_bytes.decode('utf-8'))
            self._tokenizer=self.tokenizer
            check()
            mib=1024**2
            public=dict(runtime_library=args.public_runtime_library,model_directory=root,
                checkpoint_identity=self.checkpoint_identity,max_resource_bytes=args.public_resource_mib*mib,
                max_work_bytes=args.public_work_mib*mib,max_snapshot_bytes=args.public_snapshot_mib*mib,
                max_batch_rows=args.prefill_chunk,max_snapshots_per_session=args.public_snapshots)
            if self.family=='kimi_k3':
                from .publisher_checkpoint.service import PublisherCheckpoint
                from .k3_execution.exact_admission import load_exact_evidence
                checkpoint=PublisherCheckpoint.open(root,family=self.family,
                    max_metadata_bytes=args.metadata_mib*mib,max_tensors=args.max_tensors,max_shards=args.max_shards)
                from .publisher_checkpoint.k3_adapter import k3_route_identity, k3_source_format
                source_format=k3_source_format(checkpoint)
                packed=(source_format=='MXFP4' if args.expert_storage=='auto'
                        else args.expert_storage=='packed-mxfp4')
                expert_format='e2m1-low-first-e8m0-g32' if packed else 'fp32'
                if packed and source_format!='MXFP4':
                    raise ValueError('packed K3 expert storage requires a compressed MXFP4 checkpoint')
                k3_route_identity(checkpoint,backend='public-range',
                    arithmetic='fp32-approximate',expert_format=expert_format)
                payload=checkpoint.load_k3_execution_payload(capacity=args.context,
                    max_weight_bytes=args.resident_mib*mib,max_chunk_bytes=args.chunk_mib*mib,
                    expert_format=expert_format,
                    use_expert_provider=packed,max_expert_bytes=args.expert_mib*mib if packed else None,
                    max_mapped_bytes=args.mapped_mib*mib if packed else None,
                    projection_epsilon=args.projection_epsilon,l2_epsilon=args.l2_epsilon,
                    source_backend='public-range',source_arithmetic='fp32-approximate')
                if payload.expert_provider is not None:
                    self._resources.callback(payload.expert_provider.close)
                evidence=None
                if args.exact_evidence is not None:
                    evidence=load_exact_evidence(args.exact_evidence,expected_sha256=args.exact_evidence_sha256,
                        max_bytes=args.exact_evidence_mib*mib)
                elif args.exact_evidence_sha256 is not None:
                    raise ValueError('exact evidence pin requires its source file')
                if args.public_executor_mib!=args.runtime_mib:
                    raise ValueError('K3 public executor reservation must match its complete runtime budget')
                self.service=create_public_k3_execution_service(payload,K3Libraries(
                    recurrence=args.kimi_library,operators=args.operators_library,
                    attention=args.attention_library,mxfp4=args.mxfp4_library),
                    runtime_max_bytes=args.runtime_mib*mib,max_new_tokens=args.context,
                    prefill_chunk_tokens=args.prefill_chunk,require_publisher_exact=args.require_publisher_exact,
                    exact_evidence=evidence,**public)
            else:
                backing=open_publisher_v41_service(root,execution_library=args.execution_library,
                    mxfp4_library=args.mxfp4_library,fp8_library=args.fp8_library,context_tokens=args.context,
                    max_weight_bytes=args.resident_mib*mib,runtime_max_bytes=args.runtime_mib*mib,
                    arithmetic_max_bytes=args.arithmetic_mib*mib,max_expert_bytes=args.expert_mib*mib,
                    max_projection_bytes=args.projection_mib*mib,max_metadata_bytes=args.metadata_mib*mib,
                    bf16_projection_max_bytes=args.bf16_projection_mib*mib,tokenizer=self.tokenizer,
                    max_new_tokens=args.context,prefill_chunk_tokens=args.prefill_chunk,
                    max_total_bytes=None if args.total_mib is None else args.total_mib*mib)
                self.service=compose_public_publisher_service(backing,family='deepseek-v41',
                    config=backing.publisher_payload.profile,executor_bytes=args.public_executor_mib*mib,**public)
            self._resources.callback(self.service.close)
            self.vocabulary=self.service.limits.vocabulary
            if any(type(token) is not int or not 0<=token<self.vocabulary for token in self.stops):
                raise ValueError('original public publisher EOS inventory differs')
            self.session_id=self.service.open_session()
            self._artifacts_unchanged()
            check()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as failure:
                from .failure_reporting import retain_cleanup_failure
                retain_cleanup_failure(primary,self.close,failure,
                    'public publisher construction cleanup retained: ')
            raise

    def run(self,args,*,emit=None,cancelled=None,checkpoint_sink=None,max_checkpoint_bytes=0):
        request=copy.copy(args)
        if request.prompt is not None:
            request.token_ids=json.dumps(self.tokenizer.encode(request.prompt,add_special_tokens=False).ids)
            request.prompt=None
        result=super().run(request,emit=emit,cancelled=cancelled,
            checkpoint_sink=checkpoint_sink,max_checkpoint_bytes=max_checkpoint_bytes)
        result['text']=self.tokenizer.decode(result['generated_ids'],skip_special_tokens=self.family!='kimi_k3')
        result['public_native_abi']=True
        return result

    def generate(self,*,_request_args,media_items=None,emit=None,cancelled=None,
                 checkpoint_sink=None,checkpoint_max_bytes=None):
        if media_items is not None:
            raise ValueError('public text operator requires its text input; original media operator owns expansion')
        return self.run(_request_args,emit=emit,cancelled=cancelled,checkpoint_sink=checkpoint_sink,
            max_checkpoint_bytes=0 if checkpoint_sink is None else checkpoint_max_bytes)

    def resume_generation(self,source,*,max_bytes=None,max_new_tokens=None,
                          checkpoint_max_bytes=None,max_tokens=None,emit=None,cancelled=None,
                          checkpoint_sink=None,max_checkpoint_bytes=0):
        budget=max_bytes if max_bytes is not None else checkpoint_max_bytes
        maximum=max_new_tokens if max_new_tokens is not None else max_tokens
        emitted=[]
        def event(row):
            if emit is not None:
                if row.get('event')=='tokens':
                    emitted.extend(row['token_ids'])
                    row=dict(row,text=self.tokenizer.decode(emitted,skip_special_tokens=self.family!='kimi_k3'))
                emit(row)
        result=super().resume_generation(source,max_bytes=budget,max_new_tokens=maximum,emit=event,
            cancelled=cancelled,checkpoint_sink=checkpoint_sink,
            max_checkpoint_bytes=(checkpoint_max_bytes if checkpoint_sink is not None and checkpoint_max_bytes is not None
                                  else max_checkpoint_bytes))
        result['text']=self.tokenizer.decode(result['generated_ids'],skip_special_tokens=self.family!='kimi_k3')
        result['public_native_abi']=True
        return result

    def close(self):
        with self._lock:
            if self._closed and not self._resources.pending:
                return
            # OwnedResources retains the failing callback.  Do not mark the
            # session closed or discard its references until dependency-ordered
            # release has actually completed; callers can retry cleanup.
            self._resources.close()
            self._closed=True
            self.service=self.session_id=self.tokenizer=self._tokenizer=None


def main(argv=None):
    selector=argparse.ArgumentParser(add_help=False)
    selector.add_argument('--family',required=True,choices=('kimi_k3','deepseek_v41'))
    selected,remaining=selector.parse_known_args(argv)
    args=parser(family=selected.family).parse_args(remaining)
    with PublisherSession(args) as session:
        print(json.dumps(session.run(args),sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
