"""Explicit CUDA/HIP or Metal Mixtral decoder with owned device continuation."""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from .native_mixtral_decoder import configuration,GLOBAL_NAMES,LAYER_NAMES
from .device_execution_policy import admit_no_fallback,admit_fp32


@dataclass(frozen=True)
class TorchMixtralSnapshot:
    identity: str
    owner: str
    handle: str
    position: int


class TorchMixtralDecoder:
    """Full resident FP32 decoder arithmetic executes on an explicit device.

    No CPU fallback is selected. Host arrays provide input/output boundaries and
    owned recovery snapshots. Device kernels may reassociate products; this
    implementation makes no publisher equivalence or platform certification claim.
    """
    implementation = 'torch-mixtral-device-fp32'

    @classmethod
    def from_directory(cls,directory,*,device,max_weight_bytes,max_bytes,max_snapshots=4,
                       read_limits=None,cancel=None,expert_provider=None,
                       max_attention_bytes=64*1024**2,max_head_bytes=64*1024**2):
        from .native_checkpoint import read_native_text_checkpoint,_finish_decoder
        from .mixtral_inventory import admit_mixtral_inventory
        layouts = []
        def select(config,inventory):
            c,_,layout,_ = admit_mixtral_inventory(config,inventory)
            layouts.append(layout)
            names = tuple(sorted(inventory))
            if expert_provider is not None:
                names = tuple(name for name in names if not name.endswith((LAYER_NAMES[7],LAYER_NAMES[8]))
                    and '.block_sparse_moe.experts.' not in name)
            size = sum(math.prod(inventory[name]['shape'])*4 for name in names)
            if c['tie_word_embeddings'] and 'lm_head.weight' not in inventory:
                size += 4*c['hidden_size']*c['vocab_size']
            if 4*size+getattr(expert_provider,'max_bytes',0)>max_bytes:
                raise MemoryError('Mixtral device checkpoint conversion exceeds admission')
            return names
        config,weights = read_native_text_checkpoint(directory,model_type='mixtral',
            max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,tensor_selection=select)
        if layouts==['split']:
            c,_ = configuration(config,None)
            converted = {name:weights[name] for name in GLOBAL_NAMES if name in weights}
            e,m,h = c['num_local_experts'],c['intermediate_size'],c['hidden_size']
            for index in range(c['num_hidden_layers']):
                p = f'model.layers.{index}.'
                for name in LAYER_NAMES[:6]:
                    converted[p+name] = weights[p+name]
                converted[p+LAYER_NAMES[6]] = weights[p+'block_sparse_moe.gate.weight']
                if expert_provider is not None:
                    continue
                gate_up = np.empty((e,2*m,h),np.float32)
                down = np.empty((e,h,m),np.float32)
                for expert in range(e):
                    source = p+f'block_sparse_moe.experts.{expert}.'
                    gate_up[expert,:m] = weights[source+'w1.weight']
                    gate_up[expert,m:] = weights[source+'w3.weight']
                    down[expert] = weights[source+'w2.weight']
                converted[p+LAYER_NAMES[7]],converted[p+LAYER_NAMES[8]] = gate_up,down
            weights = converted
        return _finish_decoder(cls(config,weights,device=device,max_bytes=max_bytes,
                                    max_snapshots=max_snapshots,expert_provider=expert_provider,
                                    max_attention_bytes=max_attention_bytes,max_head_bytes=max_head_bytes),cancel)

    def __init__(self,config,tensors,*,device,max_bytes,max_snapshots=4,expert_provider=None,
                 max_attention_bytes=64*1024**2,max_head_bytes=64*1024**2):
        admit_no_fallback(device)
        self.config,self.capacity = configuration(config,None)
        if type(max_bytes) is not int or max_bytes<=0 or type(max_snapshots) is not int or not 1<=max_snapshots<=64:
            raise ValueError('explicit decoder/snapshot admission required')
        if type(device) is not str or (device!='mps' and device!='cuda' and
                not (device.startswith('cuda:') and device[5:].isdigit())):
            raise ValueError('explicit CUDA/HIP or Metal device required')
        self.max_bytes,self.max_snapshots = max_bytes,max_snapshots
        if type(max_attention_bytes) is not int or max_attention_bytes<=0:
            raise ValueError('explicit device attention scratch budget required')
        self.attention_bytes = min(max_attention_bytes,max_bytes)
        if type(max_head_bytes) is not int or max_head_bytes<=0:
            raise ValueError('explicit vocabulary device scratch budget required')
        self.head_bytes = min(max_head_bytes,max_bytes)
        self._lock,self._closed,self._busy = threading.RLock(),False,False
        self._cancellation = threading.local()
        self._owner,self._snapshots,self._snapshot_bytes = uuid4().hex,{},0
        self.expert_provider = expert_provider
        self._expert_bytes,self._expert_identity = 0,None
        c = self.config
        h,m,e,d = (c[key] for key in ('hidden_size','intermediate_size','num_local_experts','head_dim'))
        q,k = c['num_attention_heads']*d,c['num_key_value_heads']*d
        shapes = dict(zip(GLOBAL_NAMES,((c['vocab_size'],h),(h,),(c['vocab_size'],h))))
        layer_shapes = ((h,),(h,),(q,h),(k,h),(k,h),(h,q),(e,h),(e,2*m,h),(e,h,m))
        for index in range(c['num_hidden_layers']):
            shapes.update((f'model.layers.{index}.'+name,shape) for name,shape in zip(LAYER_NAMES,layer_shapes))
        if expert_provider is not None:
            from .torch_mixtral_experts import TorchMixtralExperts
            from .native_mixtral_decoder import mixtral_metadata_fingerprint
            if (type(expert_provider) is not TorchMixtralExperts or len(expert_provider.layers)!=c['num_hidden_layers'] or
                    (expert_provider.hidden_size,expert_provider.intermediate_size,expert_provider.expert_count,expert_provider.selected_experts)!=
                    (h,m,e,c['num_experts_per_tok']) or
                    expert_provider.manifest.fingerprint!=mixtral_metadata_fingerprint(config)):
                raise ValueError('exact Mixtral device packed provider contract required')
            self._expert_bytes,self._expert_identity = expert_provider.max_bytes,expert_provider.identity
            shapes = {name:shape for name,shape in shapes.items() if not name.endswith((LAYER_NAMES[7],LAYER_NAMES[8]))}
        supplied = dict(tensors)
        if c['tie_word_embeddings']:
            if 'lm_head.weight' in supplied and not np.array_equal(supplied.get('model.embed_tokens.weight'),supplied['lm_head.weight']):
                raise ValueError('inconsistent Mixtral tied weights')
            supplied['lm_head.weight'] = supplied.get('model.embed_tokens.weight')
        if set(supplied)!=set(shapes):
            raise ValueError('exact complete fused Mixtral tensor inventory required')
        self.weight_bytes = 4*sum(math.prod(shape) for shape in shapes.values())
        if 3*self.weight_bytes+self._expert_bytes>max_bytes:
            raise MemoryError('Mixtral source/device weight admission exceeded')
        digest = hashlib.sha256(json.dumps(dict(c),sort_keys=True,separators=(',',':')).encode())
        admitted = {}
        for name,shape in shapes.items():
            value = np.asarray(supplied[name])
            if value.dtype!=np.float32 or value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('invalid fused Mixtral tensor: '+name)
            value = np.array(value,order='C',copy=True)
            digest.update(name.encode());digest.update(memoryview(value).cast('B'))
            admitted[name] = value
        torch = importlib.import_module('torch')
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('requested CUDA/HIP device unavailable')
            index = int(device.split(':')[1]) if ':' in device else torch.cuda.current_device()
            if index>=torch.cuda.device_count():
                raise ValueError('requested device index unavailable')
            device = 'cuda:'+str(index)
        elif not torch.backends.mps.is_available():
            raise RuntimeError('requested Metal device unavailable')
        self.torch,self.device = torch,device
        self.weights = {name:torch.tensor(value,device=device,dtype=torch.float32) for name,value in admitted.items()}
        digest.update(repr((device,torch.__version__,self.implementation,'window-kv-block-attention-head-v3',
                            self.attention_bytes,self.head_bytes)).encode())
        if self._expert_identity is not None:
            digest.update(self._expert_identity.encode())
        self.identity = 'openagent-torch-mixtral-v1:'+digest.hexdigest()
        self._cache,self._mask,self.position = (),None,0

    def _check(self):
        if self._closed or self._busy:
            raise RuntimeError('Mixtral device decoder closed or busy')

    @contextmanager
    def cancellation_scope(self,cancel):
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise ValueError('device cancellation must provide is_set')
        previous = getattr(self._cancellation,'event',None)
        self._cancellation.event = cancel
        try:
            yield
        finally:
            self._cancellation.event = previous

    def _check_cancel(self):
        cancel = getattr(self._cancellation,'event',None)
        if cancel is not None and cancel.is_set():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('Mixtral device execution cancelled; continuation unchanged')

    def _norm(self,value,weight):
        return value*self.torch.rsqrt(value.square().mean(dim=-1,keepdim=True)+self.config['rms_norm_eps'])*weight

    def _rotate(self,value,positions):
        torch,d = self.torch,self.config['head_dim']
        frequency = self.config['rope_theta']**(-torch.arange(0,d,2,device=self.device,dtype=torch.float32)/d)
        frequency = frequency/self.config['rope_factor']
        angles = positions.to(torch.float32)[...,None]*frequency
        cosine,sine = angles.cos()[:,None],angles.sin()[:,None]
        left,right = value[...,:d//2],value[...,d//2:]
        return torch.cat((left*cosine-right*sine,right*cosine+left*sine),dim=-1)

    def forward(self,token_ids,*,attention_mask=None,position_ids=None):
        admit_fp32(self.torch,self.device)
        with self._lock:
            self._check()
            torch,c = self.torch,self.config
            ids = np.asarray(token_ids)
            if ids.ndim==1:
                ids = ids[None]
            if ids.ndim!=2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=c['vocab_size']):
                raise ValueError('bounded integer Mixtral token rows required')
            b,t = ids.shape
            end = self.position+t
            if b>128 or t>4096 or end>self.capacity:
                raise ValueError('Mixtral device batch/context admission exceeded')
            h,d,nq,nk = (c[key] for key in ('hidden_size','head_dim','num_attention_heads','num_key_value_heads'))
            old_bytes = sum(value.numel()*4 for bank in self._cache for value in bank)
            retained = self.position if c['sliding_window'] is None else min(self.position,c['sliding_window'])
            cache_start = self.position-retained
            key_count = retained+t
            kv = 8*c['num_hidden_layers']*b*key_count*nk*d
            work = 4*b*t*(20*h+6*nq*d+8*c['intermediate_size']+4*c['num_local_experts']+c['vocab_size'])
            work += max(self.attention_bytes,self.head_bytes)
            if self.weight_bytes+self._snapshot_bytes+old_bytes+2*kv+work+self._expert_bytes>self.max_bytes:
                raise MemoryError('Mixtral device continuation/work admission exceeded')
            mask = np.ones((b,end),np.uint8) if attention_mask is None else np.asarray(attention_mask)
            if mask.shape!=(b,end) or not np.isin(mask,(0,1)).all():
                raise ValueError('binary full-prefix Mixtral mask required')
            if self._mask is not None and not np.array_equal(self._mask,mask[:,:self.position]):
                raise ValueError('committed batch/mask changed')
            positions = np.broadcast_to(np.arange(self.position,end),ids.shape) if position_ids is None else np.asarray(position_ids)
            if positions.shape!=ids.shape or positions.dtype.kind not in 'iu' or np.any(positions<0) or np.any(positions>=1048576):
                raise ValueError('bounded explicit Mixtral rotary positions required')
            self._busy = True
            try:
                self._check_cancel()
                with torch.inference_mode():
                    if self.expert_provider is not None and self.expert_provider.identity!=self._expert_identity:
                        raise ValueError('Mixtral device provider changed after admission')
                    indices = torch.tensor(np.array(ids,dtype=np.int64),device=self.device)
                    rotary = torch.tensor(np.array(positions,dtype=np.int64),device=self.device)
                    occupied = torch.tensor(np.array(mask,dtype=np.bool_),device=self.device)
                    x = self.weights['model.embed_tokens.weight'][indices]
                    staged = []
                    for index in range(c['num_hidden_layers']):
                        self._check_cancel()
                        p = f'model.layers.{index}.'
                        z = self._norm(x,self.weights[p+LAYER_NAMES[0]])
                        q = (z@self.weights[p+LAYER_NAMES[2]].T).reshape(b,t,nq,d).transpose(1,2)
                        k = (z@self.weights[p+LAYER_NAMES[3]].T).reshape(b,t,nk,d).transpose(1,2)
                        v = (z@self.weights[p+LAYER_NAMES[4]].T).reshape(b,t,nk,d).transpose(1,2)
                        q,k = self._rotate(q,rotary),self._rotate(k,rotary)
                        if self._cache:
                            k = torch.cat((self._cache[index][0],k),dim=2)
                            v = torch.cat((self._cache[index][1],v),dim=2)
                        if c['sliding_window'] is None:
                            staged.append((k,v))
                        else:
                            # Own the suffix so a view cannot retain the full
                            # prefill allocation through shared tensor storage.
                            staged.append((k[:,:,-c['sliding_window']:].clone(),
                                           v[:,:,-c['sliding_window']:].clone()))
                        from .device_attention import device_attention
                        attended = device_attention(torch,q,k,v,key_mask=occupied[:,cache_start:end],
                            query_start=self.position,key_start=cache_start,window=c['sliding_window'],
                            max_scratch_bytes=self.attention_bytes,cancel=getattr(self._cancellation,'event',None))
                        context = attended.transpose(1,2).reshape(b,t,nq*d)
                        x = x+context@self.weights[p+LAYER_NAMES[5]].T
                        z = self._norm(x,self.weights[p+LAYER_NAMES[1]])
                        router = (z@self.weights[p+LAYER_NAMES[6]].T).softmax(dim=-1)
                        selected = torch.argsort(router,dim=-1,descending=True,stable=True)[...,:c['num_experts_per_tok']]
                        scales = router.gather(-1,selected)
                        scales = scales/scales.sum(dim=-1,keepdim=True)
                        flat = z.reshape(-1,h)
                        out = torch.zeros_like(flat)
                        for expert in range(c['num_local_experts']):
                            self._check_cancel()
                            locations = torch.nonzero(selected.reshape(-1,c['num_experts_per_tok'])==expert,as_tuple=False)
                            if not locations.numel():
                                continue
                            rows,ranks = locations[:,0],locations[:,1]
                            if self.expert_provider is not None:
                                result = self.expert_provider.execute(index,expert,flat[rows],torch=torch,device=self.device,
                                    cancel=getattr(self._cancellation,'event',None))
                            else:
                                projected = flat[rows]@self.weights[p+LAYER_NAMES[7]][expert].T
                                gate,up = projected.chunk(2,dim=-1)
                                product = torch.nn.functional.silu(gate)*up
                                result = product@self.weights[p+LAYER_NAMES[8]][expert].T
                            out.index_add_(0,rows,result*scales.reshape(-1,c['num_experts_per_tok'])[rows,ranks,None])
                        x = x+out.reshape(b,t,h)
                    self._check_cancel()
                    from .device_logits import device_logits
                    result = device_logits(torch,self._norm(x,self.weights['model.norm.weight']),
                        self.weights['lm_head.weight'],max_device_bytes=self.head_bytes,
                        max_host_bytes=b*t*c['vocab_size']*4,cancel=getattr(self._cancellation,'event',None))
                    if not np.isfinite(result).all() or any(not bool(torch.isfinite(value).all().item()) for bank in staged for value in bank):
                        raise ArithmeticError('nonfinite Mixtral device continuation/output')
                    self._check_cancel()
                    self._cache,self._mask,self.position = tuple(staged),np.array(mask,dtype=np.uint8,copy=True),end
                    return SimpleNamespace(logits=result)
            finally:
                self._busy = False

    def reset(self):
        with self._lock:
            self._check()
            self._cache,self._mask,self.position = (),None,0

    def snapshot(self):
        with self._lock:
            self._check()
            size = sum(value.numel()*4 for bank in self._cache for value in bank)+(0 if self._mask is None else self._mask.nbytes)
            if len(self._snapshots)>=self.max_snapshots or self.weight_bytes+self._snapshot_bytes+3*size+self._expert_bytes>self.max_bytes:
                raise MemoryError('Mixtral device snapshot admission exceeded')
            banks = tuple(tuple(value.cpu().numpy().tobytes() for value in bank) for bank in self._cache)
            mask = None if self._mask is None else (self._mask.shape,self._mask.tobytes())
            handle = TorchMixtralSnapshot(self.identity,self._owner,uuid4().hex,self.position)
            self._snapshots[handle.handle] = (handle,banks,mask,size)
            self._snapshot_bytes += size
            return handle

    def restore(self,handle):
        with self._lock:
            self._check()
            entry = self._snapshots.get(getattr(handle,'handle',None))
            if entry is None or type(handle) is not TorchMixtralSnapshot or entry[0]!=handle:
                raise ValueError('owned Mixtral device snapshot required')
            _,banks,mask,size = entry
            if self.weight_bytes+self._snapshot_bytes+3*size+self._expert_bytes>self.max_bytes:
                raise MemoryError('Mixtral device restore staging exceeds admission')
            c = self.config
            extent = handle.position if c['sliding_window'] is None else min(handle.position,c['sliding_window'])
            shape = None if mask is None else (mask[0][0],c['num_key_value_heads'],extent,c['head_dim'])
            staged = tuple(tuple(self.torch.tensor(np.frombuffer(data,dtype=np.float32).reshape(shape),
                device=self.device) for data in bank) for bank in banks)
            restored_mask = None if mask is None else np.frombuffer(mask[1],dtype=np.uint8).reshape(mask[0]).copy()
            self._cache,self._mask,self.position = staged,restored_mask,handle.position

    def release_snapshot(self,handle):
        with self._lock:
            self._check()
            entry = self._snapshots.get(getattr(handle,'handle',None))
            if entry is None or entry[0]!=handle:
                raise ValueError('owned Mixtral device snapshot required')
            self._snapshot_bytes -= entry[3]
            del self._snapshots[handle.handle]

    def write_cache(self,output,*,max_bytes):
        from .torch_mixtral_cache import write_device_cache
        return write_device_cache(self,output,max_bytes=max_bytes)

    def restore_cache(self,source,*,max_bytes):
        from .torch_mixtral_cache import restore_device_cache
        return restore_device_cache(self,source,max_bytes=max_bytes)

    def save_cache(self,destination,*,max_bytes):
        from .cache_file_recovery import save_executor_cache
        return save_executor_cache(self,destination,max_bytes=max_bytes)

    def load_cache(self,source,*,max_bytes):
        from .cache_file_recovery import load_executor_cache
        return load_executor_cache(self,source,max_bytes=max_bytes)

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('cannot close active Mixtral device decoder')
            self._closed = True
            self._cache,self._mask,self.weights = (),None,{}
            self.expert_provider = None
            self._snapshots.clear();self._snapshot_bytes = 0


__all__ = ['TorchMixtralDecoder','TorchMixtralSnapshot']
