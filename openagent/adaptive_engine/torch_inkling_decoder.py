"""Owned Inkling accelerator execution and physical continuation transport."""
from contextlib import contextmanager,nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import threading
from types import SimpleNamespace
from uuid import uuid4
import numpy as np
from .independent_inkling import IndependentInklingDecoder,logits_selection,GLOBAL_NAMES
from .device_inkling_forward import inkling_device_forward
from .device_logits import device_logits
from .device_execution_policy import admit_no_fallback,admit_fp32

_SOURCE_ADMISSION = object()

@dataclass(frozen=True)
class InklingDeviceSnapshot:
    owner:str
    identity:str
    key:str
    position:int

class TorchInklingDecoder:
    implementation='torch-inkling-device-fp32'
    def __init__(self,config,tensors,*,device,max_bytes,max_snapshots=4,max_work_bytes=64*1024**2,max_head_bytes=64*1024**2,
                 expert_provider=None,projection_provider=None,_borrow_globals=None,_shared_global_identity=None,
                 _source_admission=None):
        if _source_admission is not _SOURCE_ADMISSION:
            raise ValueError('Inkling device construction requires metadata-admitted source tensors')
        admit_no_fallback(device)
        inventory=IndependentInklingDecoder(config,{},_inventory_only=True,max_bytes=max_bytes,_omit_experts=expert_provider is not None)
        if any(type(v) is not int or v<=0 for v in (max_bytes,max_work_bytes,max_head_bytes)) or type(max_snapshots) is not int or not 1<=max_snapshots<=64:
            raise ValueError('explicit Inkling accelerator memory and snapshot limits required')
        if type(device) is not str or (device not in ('cuda','mps') and not (device.startswith('cuda:') and device[5:].isdigit())):
            raise ValueError('explicit CUDA/HIP or Metal Inkling device required')
        self.expert_provider,self.projection_provider=expert_provider,projection_provider
        provider_bytes=0
        if expert_provider is not None:
            from .device_inkling_publisher_experts import DeviceInklingPublisherExperts
            if type(expert_provider) is not DeviceInklingPublisherExperts or expert_provider.contract!=inventory.config:
                raise ValueError('exact Inkling device expert provider required')
            provider_bytes+=expert_provider.max_bytes
        supplied=dict(tensors);streamed=set();borrowed={}
        if _borrow_globals is not None:
            if set(_borrow_globals)!=set(GLOBAL_NAMES) or type(_shared_global_identity) is not str or not _shared_global_identity:
                raise ValueError('complete owned backbone global tensors and identity required')
            borrowed=dict(_borrow_globals)
            if set(borrowed)&set(supplied):raise ValueError('borrowed Inkling globals overlap supplied tensors')
            for name,value in borrowed.items():
                if tuple(value.shape)!=inventory.tensor_shapes[name] or str(value.dtype)!='torch.float32':
                    raise ValueError('shared Inkling device global geometry differs')
            supplied.update(borrowed)
        if projection_provider is not None:
            from .device_publisher_projections import DevicePublisherProjections
            if type(projection_provider) is not DevicePublisherProjections:
                raise TypeError('Inkling original projection bank required')
            if device=='cuda' and projection_provider.device.type=='cuda': device=str(projection_provider.device)
            if str(projection_provider.device)!=device: raise ValueError('Inkling projection device differs')
            for name,value in projection_provider.matrices.items():
                if name not in inventory.tensor_shapes or value.shape!=inventory.tensor_shapes[name] or name in supplied or name.endswith('rel_logits_proj.proj'):
                    raise ValueError('Inkling streamed projection differs or overlaps: '+name)
                supplied[name]=value;streamed.add(name)
            provider_bytes+=projection_provider.max_bytes
        if set(supplied)!=set(inventory.tensor_shapes):
            raise ValueError('complete exact Inkling accelerator tensor inventory required')
        self.config=inventory.config;self.geometry=inventory.geometry
        self.weight_bytes=sum(math.prod(shape)*4 for name,shape in inventory.tensor_shapes.items() if name not in streamed and name not in borrowed)+provider_bytes
        if 3*self.weight_bytes+max_work_bytes+max_head_bytes>max_bytes:
            raise MemoryError('Inkling accelerator source/device construction exceeds admission')
        digest=hashlib.sha256(json.dumps(dict(config),sort_keys=True,allow_nan=False).encode());admitted={}
        if borrowed:digest.update(_shared_global_identity.encode())
        for provider in (expert_provider,projection_provider):
            if provider is not None: digest.update(provider.identity.encode())
        for name,shape in inventory.tensor_shapes.items():
            if name in streamed or name in borrowed: continue
            value=np.asarray(supplied[name])
            if value.dtype!=np.float32 or tuple(value.shape)!=shape or not np.isfinite(value).all():
                raise ValueError('invalid Inkling accelerator tensor: '+name)
            admitted[name]=np.array(value,copy=True,order='C')
            digest.update(name.encode());digest.update(memoryview(admitted[name]).cast('B'))
        torch=importlib.import_module('torch')
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('requested Inkling CUDA/HIP device unavailable')
            index=torch.cuda.current_device() if device=='cuda' else int(device.split(':')[1])
            if index>=torch.cuda.device_count():
                raise ValueError('Inkling device index unavailable')
            device='cuda:'+str(index)
        elif not torch.backends.mps.is_available():
            raise RuntimeError('requested Inkling Metal device unavailable')
        self.torch,self.device=torch,device
        self.weights={name:torch.tensor(value,device=device,dtype=torch.float32) for name,value in admitted.items()}
        for name,value in borrowed.items():
            if str(value.device)!=device:raise ValueError('shared Inkling global tensor device differs')
            self.weights[name]=value
        if projection_provider is not None: self.weights.update(projection_provider.matrices)
        digest.update(repr((device,torch.__version__,self.implementation)).encode())
        self.identity='openagent-torch-inkling-v1:'+digest.hexdigest()
        self.max_bytes,self.max_snapshots,self.work_bytes,self.head_bytes=max_bytes,max_snapshots,max_work_bytes,max_head_bytes
        self._lock=threading.RLock();self._local=threading.local();self._closed=self._busy=False
        self._owner=uuid4().hex;self._snapshots={};self._snapshot_bytes=0;self._cache=();self.position=0

    @classmethod
    def from_directory(cls,directory,*,device,max_weight_bytes,max_bytes,read_limits=None,cancel=None,**options):
        # The provider contract currently binds configuration/geometry but not
        # the exact source checkpoint weight bytes.  Omitting a bank and
        # replacing it with a separately opened provider could therefore mix
        # checkpoint revisions while still satisfying the tensor shapes.
        if options.get('expert_provider') is not None or options.get('projection_provider') is not None:
            raise ValueError(
                'Inkling external device expert/projection providers are disabled '
                'until source checkpoint content identity is shared')
        from .native_checkpoint import read_native_text_checkpoint,_finish_decoder
        def select(config,inventory):
            shapes=IndependentInklingDecoder(config,{},_inventory_only=True,max_bytes=max_bytes).tensor_shapes
            if set(shapes)!=set(inventory) or any(
                    tuple(inventory[name]['shape'])!=shape or
                    inventory[name].get('dtype') not in ('F32','F16','BF16')
                    for name,shape in shapes.items()):
                raise ValueError('exact full Inkling accelerator checkpoint required')
            omitted=set();expert=options.get('expert_provider');projections=options.get('projection_provider')
            if expert is not None:
                omitted.update(name for name in shapes if name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj')))
            if projections is not None:
                for name,value in projections.matrices.items():
                    if name not in shapes or name in omitted or tuple(value.shape)!=shapes[name]:
                        raise ValueError('streamed Inkling checkpoint projection inventory differs')
                    omitted.add(name)
            selected={name:shape for name,shape in shapes.items() if name not in omitted}
            reservations=sum(provider.max_bytes for provider in (expert,projections) if provider is not None)
            if 4*sum(math.prod(shape)*4 for shape in selected.values())+reservations>max_bytes:
                raise MemoryError('Inkling checkpoint conversion exceeds aggregate admission')
            return tuple(sorted(selected))
        config,weights=read_native_text_checkpoint(directory,model_type='inkling_text',max_weight_bytes=max_weight_bytes,
            read_limits=read_limits,cancel=cancel,tensor_selection=select,source_backend='device')
        return _finish_decoder(cls(config,weights,device=device,max_bytes=max_bytes,
            _source_admission=_SOURCE_ADMISSION,**options),cancel)

    def _check(self):
        if self._closed or self._busy:
            raise RuntimeError('Inkling device session is closed or executing')
    def _cancel(self):
        event=getattr(self._local,'cancel',None)
        if event is not None and event.is_set():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('Inkling device transaction cancelled')
    @contextmanager
    def cancellation_scope(self,event):
        if event is not None and not callable(getattr(event,'is_set',None)):
            raise TypeError('event-like Inkling cancellation required')
        old=getattr(self._local,'cancel',None);self._local.cancel=event
        try:
            yield self
        finally:
            self._local.cancel=old
    def _state_size(self,g):
        c=self.config
        return 2*c['max_position_embeddings']*g.kv_width+2*c['conv_kernel_size']*(g.kv_width+c['hidden_size'])
    def _state_bytes(self):
        return sum(value.numel()*4 for value in self._cache)
    def _admit(self,extra):
        if self.weight_bytes+self._snapshot_bytes+self._state_bytes()+extra>self.max_bytes:
            raise MemoryError('Inkling device continuation transaction exceeds admission')
    def forward(self,token_ids,*,attention_mask=None,position_ids=None,logits_to_keep=0,use_cache=True,inputs_embeds=None,
                _raw_hidden=False,_position_origin=0):
        admit_fp32(self.torch,self.device)
        with self._lock:
            self._check();self._cancel();ids=np.asarray(token_ids)
            if ids.ndim==1: ids=ids[None,:]
            if (ids.ndim!=2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or ids.shape[0]>128 or ids.shape[1]>4096
                    or np.any(ids<0) or np.any(ids>=self.config['vocab_size'])):
                raise ValueError('bounded Inkling integer token batch required')
            if attention_mask is not None or position_ids is not None or type(use_cache) is not bool:
                raise ValueError('Inkling uses contiguous unmasked physical positions')
            if type(_raw_hidden) is not bool or type(_position_origin) is not int or not 0<=_position_origin<=32:
                raise ValueError('explicit bounded Inkling MTP hidden/position contract required')
            start=self.position if use_cache else 0;batch,tokens=ids.shape
            if start+tokens>self.config['max_position_embeddings'] or (use_cache and self._cache and self._cache[0].shape[0]!=batch):
                raise ValueError('Inkling continuation capacity/batch differs')
            state=batch*sum(self._state_size(g) for g in self.geometry)*4
            rows=self.config['unpadded_vocab_size'] or self.config['vocab_size']
            output=batch*tokens*(self.config['hidden_size']+(0 if _raw_hidden else self.config['vocab_size'])+
                len(self.geometry)*(self.config['n_routed_experts']+self.config['n_shared_experts']))*4
            media_bytes=0 if inputs_embeds is None else batch*tokens*self.config['hidden_size']*8
            self._admit(2*state+2*output+self.work_bytes+(0 if _raw_hidden else self.head_bytes)+ids.size*16+media_bytes)
            torch=self.torch;media=None
            if inputs_embeds is not None:
                source=np.asarray(inputs_embeds)
                if source.shape!=(*ids.shape,self.config['hidden_size']) or source.dtype!=np.float32 or not np.isfinite(source).all():
                    raise ValueError('matching finite Inkling FP32 media embeddings required')
                media=torch.tensor(source,device=self.device)
            selection=logits_selection(logits_to_keep,tokens);self._busy=True
            try:
                with torch.inference_mode(), (self.projection_provider.cancellation_scope(getattr(self._local,'cancel',None))
                        if self.projection_provider is not None else nullcontext()):
                    hidden,routing,pending=inkling_device_forward(torch,torch.tensor(ids.astype(np.int64),device=self.device),
                        self.weights,self.config,self.geometry,previous=self._cache if use_cache else (),start=start,
                        max_work_bytes=self.work_bytes,inputs_embeds=media,cancel=getattr(self._local,'cancel',None),expert_provider=self.expert_provider,
                        raw_hidden=_raw_hidden,position_origin=_position_origin)
                    logits=None if _raw_hidden else device_logits(torch,hidden[:,selection]/self.config['logits_mup_width_multiplier'],
                        self.weights['lm_head.weight'],max_device_bytes=self.head_bytes,
                        max_host_bytes=batch*tokens*self.config['vocab_size']*4,cancel=getattr(self._local,'cancel',None))
                    result=SimpleNamespace(logits=None if logits is None else logits[...,:rows],last_hidden_state=hidden.detach().cpu().numpy().copy(),
                        routing_weights=routing.detach().cpu().numpy().copy())
                    self._cancel()
                    if use_cache: self._cache,self.position=pending,start+tokens
                    return result
            finally:
                self._busy=False
    def snapshot(self):
        with self._lock:
            self._check();self._cancel();size=self._state_bytes()
            if len(self._snapshots)>=self.max_snapshots: raise MemoryError('Inkling snapshot inventory full')
            self._admit(2*size)
            banks=tuple(value.detach().cpu().numpy().copy() for value in self._cache)
            for value in banks: value.flags.writeable=False
            receipt=InklingDeviceSnapshot(self._owner,self.identity,uuid4().hex,self.position)
            self._cancel();self._snapshots[receipt.key]=(receipt,banks,size);self._snapshot_bytes+=size
            return receipt
    def _snapshot(self,receipt):
        if type(receipt) is not InklingDeviceSnapshot or receipt.owner!=self._owner or receipt.identity!=self.identity:
            raise ValueError('owned Inkling accelerator snapshot required')
        value=self._snapshots.get(receipt.key)
        if value is None or value[0]!=receipt: raise ValueError('released Inkling snapshot')
        return value
    def restore(self,receipt):
        with self._lock:
            self._check();_,banks,size=self._snapshot(receipt);self._admit(2*size);self._cancel()
            pending=();committed=False
            try:
                pending=tuple(self.torch.tensor(value,device=self.device) for value in banks)
                self._cancel();self._cache,self.position=pending,receipt.position;committed=True
            finally:
                if not committed: pending=()
    def release_snapshot(self,receipt):
        with self._lock:
            self._check();self._snapshot_bytes-=self._snapshot(receipt)[2];del self._snapshots[receipt.key]
    def reset(self):
        with self._lock:
            self._check();self._cache,self.position=(),0
    def write_cache(self,output,*,max_bytes):
        from .torch_inkling_cache import write_cache
        return write_cache(self,output,max_bytes=max_bytes)
    def restore_cache(self,source,*,max_bytes):
        from .torch_inkling_cache import restore_cache
        return restore_cache(self,source,max_bytes=max_bytes)
    def save_cache(self,destination,*,max_bytes):
        from .cache_file_recovery import save_executor_cache
        return save_executor_cache(self,destination,max_bytes=max_bytes)
    def load_cache(self,source,*,max_bytes):
        from .cache_file_recovery import load_executor_cache
        return load_executor_cache(self,source,max_bytes=max_bytes)
    def status(self):
        return dict(model_type='inkling_text',implementation=self.implementation,device=self.device,position=self.position,closed=self._closed)
    def close(self):
        with self._lock:
            if self._closed: return
            self._check();self.weights.clear();self._cache=();self._snapshots.clear();self._snapshot_bytes=0;self.position=0;self._closed=True

__all__=['TorchInklingDecoder','InklingDeviceSnapshot']
