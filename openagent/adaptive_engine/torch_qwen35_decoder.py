"""Explicit device Qwen3.5 text execution with atomic hybrid continuation."""
from contextlib import contextmanager,nullcontext
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import threading
from uuid import uuid4

import numpy as np

from .independent_qwen35 import IndependentQwen35TextDecoder
from .device_qwen_delta import qwen35_device_delta
from .device_qwen35_layers import qwen35_device_attention,qwen35_device_moe
from .device_logits import device_logits
from .device_execution_policy import admit_no_fallback,admit_fp32


@dataclass(frozen=True)
class TorchQwen35Snapshot:
    identity: str
    owner: str
    handle: str
    position: int


class TorchQwen35Decoder:
    implementation = 'torch-qwen35-device-fp32'

    def __init__(self,config,tensors,*,device,max_bytes,max_snapshots=4,
                 max_layer_bytes=64*1024**2,max_head_bytes=64*1024**2,expert_provider=None,projection_provider=None,context_tokens=None):
        admit_no_fallback(device)
        inventory = IndependentQwen35TextDecoder(config,{},_inventory_only=True)
        self.config,self.rotary_dim,self.theta = inventory.config,inventory.rotary_dim,inventory.theta
        self.mrope_section=inventory.mrope_section
        self.capacity=self.config.max_position_embeddings if context_tokens is None else context_tokens
        if type(self.capacity) is not int or not 1<=self.capacity<=self.config.max_position_embeddings:
            raise ValueError('bounded Qwen3.5 physical context required')
        if type(max_bytes) is not int or max_bytes<=0 or type(max_snapshots) is not int or not 1<=max_snapshots<=64:
            raise ValueError('explicit decoder and snapshot budgets required')
        if any(type(value) is not int or value<=0 for value in (max_layer_bytes,max_head_bytes)):
            raise ValueError('explicit device workspace budgets required')
        if type(device) is not str or (device not in ('cuda','mps') and not (device.startswith('cuda:') and device[5:].isdigit())):
            raise ValueError('explicit CUDA/HIP or Metal device required')
        shapes = dict(inventory.tensor_shapes)
        self.expert_provider,self._expert_bytes,self._expert_identity = expert_provider,0,None
        if expert_provider is not None:
            from .torch_qwen35_experts import TorchQwen35Experts
            from .qwen35_publisher_experts import DeviceQwen35PublisherExperts
            from .qwen35_streamed_experts import qwen35_metadata_fingerprint
            c = self.config
            fingerprint=(expert_provider.fingerprint if type(expert_provider) is DeviceQwen35PublisherExperts else
                expert_provider.manifest.fingerprint if type(expert_provider) is TorchQwen35Experts else None)
            if (type(expert_provider) not in (TorchQwen35Experts,DeviceQwen35PublisherExperts) or
                    len(expert_provider.layers)!=c.num_hidden_layers or
                    (expert_provider.hidden_size,expert_provider.intermediate_size,expert_provider.expert_count,expert_provider.selected_experts)!=
                    (c.hidden_size,c.moe_intermediate_size,c.num_experts,c.num_experts_per_tok) or
                    fingerprint!=qwen35_metadata_fingerprint(config)):
                raise ValueError('exact Qwen3.5 packed device provider required')
            self._expert_bytes,self._expert_identity = expert_provider.max_bytes,expert_provider.identity
            shapes = {name:shape for name,shape in shapes.items() if not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
        supplied = dict(tensors)
        self.projection_provider=projection_provider;self._projection_bytes=0
        streamed=set()
        if projection_provider is not None:
            from .device_publisher_projections import DevicePublisherProjections
            if type(projection_provider) is not DevicePublisherProjections:
                raise TypeError('explicit original Qwen3.5 device projection provider required')
            if device=='cuda' and projection_provider.device.type=='cuda':device=str(projection_provider.device)
            if str(projection_provider.device)!=device:raise ValueError('projection bank and decoder devices differ')
            for name,value in projection_provider.matrices.items():
                if name not in shapes or shapes[name]!=value.shape or name in supplied:
                    raise ValueError('duplicate or foreign Qwen3.5 dense projection binding: '+name)
                supplied[name]=value
            streamed=set(projection_provider.matrices);self._projection_bytes=projection_provider.max_bytes
        if config.get('tie_word_embeddings',False):
            embedding=supplied.get('model.embed_tokens.weight')
            if 'lm_head.weight' in supplied:
                head=supplied['lm_head.weight']
                same=(embedding is head if 'model.embed_tokens.weight' in streamed or 'lm_head.weight' in streamed
                      else np.array_equal(embedding,head))
                if not same:raise ValueError('inconsistent tied Qwen3.5 weights')
            supplied['lm_head.weight'] = embedding
            if 'model.embed_tokens.weight' in streamed:streamed.add('lm_head.weight')
        if set(supplied)!=set(shapes):
            raise ValueError('complete exact Qwen3.5 tensor inventory required')
        self.weight_bytes = 4*sum(math.prod(shape) for name,shape in shapes.items() if name not in streamed)
        if 3*self.weight_bytes+self._expert_bytes+self._projection_bytes>max_bytes:
            raise MemoryError('Qwen3.5 source/device weights exceed admission')
        digest = hashlib.sha256(json.dumps(dict(config),sort_keys=True,separators=(',',':')).encode())
        admitted = {}
        for name,shape in shapes.items():
            if name in streamed:continue
            value = np.asarray(supplied[name])
            if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('invalid Qwen3.5 FP32 tensor: '+name)
            value = np.array(value,copy=True,order='C')
            digest.update(name.encode());digest.update(memoryview(value).cast('B'))
            admitted[name] = value
        torch = importlib.import_module('torch')
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('requested CUDA/HIP device unavailable')
            index = int(device.split(':')[1]) if ':' in device else torch.cuda.current_device()
            if index>=torch.cuda.device_count():
                raise ValueError('requested CUDA/HIP index unavailable')
            device = 'cuda:'+str(index)
        elif not torch.backends.mps.is_available():
            raise RuntimeError('requested Metal device unavailable')
        self.torch,self.device = torch,device
        self.weights = {name:torch.tensor(value,dtype=torch.float32,device=device) for name,value in admitted.items()}
        if projection_provider is not None:
            self.weights.update(projection_provider.matrices)
            if config.get('tie_word_embeddings',False) and 'model.embed_tokens.weight' in streamed:
                self.weights['lm_head.weight']=self.weights['model.embed_tokens.weight']
            digest.update(projection_provider.identity.encode())
        self.max_bytes,self.max_snapshots = max_bytes,max_snapshots
        self.layer_bytes,self.head_bytes = max_layer_bytes,max_head_bytes
        digest.update(repr((device,torch.__version__,self.implementation,max_layer_bytes,max_head_bytes)).encode())
        if self._expert_identity is not None:
            digest.update(self._expert_identity.encode())
        self.identity = 'openagent-torch-qwen35-v1:'+digest.hexdigest()
        if self.capacity!=self.config.max_position_embeddings:
            self.identity+=':physical-context:'+str(self.capacity)
        self._lock,self._closed,self._busy = threading.RLock(),False,False
        self._cancellation = threading.local()
        self._owner,self._snapshots,self._snapshot_bytes = uuid4().hex,{},0
        self._cache,self._mask,self.position = (),None,0

    @classmethod
    def from_directory(cls,directory,*,device,max_weight_bytes,max_bytes,read_limits=None,cancel=None,expert_provider=None,projection_provider=None,**options):
        from .native_checkpoint import read_native_text_checkpoint,_finish_decoder
        def select(config,inventory):
            capacity=options.get('context_tokens',config['max_position_embeddings'])
            if capacity is None:capacity=config['max_position_embeddings']
            if type(capacity) is not int or not 1<=capacity<=config['max_position_embeddings']:
                raise ValueError('bounded device Qwen3.5 physical context required before payload reads')
            expected = IndependentQwen35TextDecoder(config,{},_inventory_only=True).tensor_shapes
            if config.get('tie_word_embeddings',False) and 'lm_head.weight' not in inventory:
                expected = {name:shape for name,shape in expected.items() if name!='lm_head.weight'}
            if set(expected)!=set(inventory) or any(tuple(inventory[name]['shape'])!=shape for name,shape in expected.items()):
                raise ValueError('exact Qwen3.5 checkpoint inventory required')
            names = tuple(sorted(expected))
            if expert_provider is not None:
                names = tuple(name for name in names if not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj')))
            if projection_provider is not None:names=tuple(name for name in names if name not in projection_provider.matrices)
            if 4*sum(math.prod(expected[name])*4 for name in names)+getattr(expert_provider,'max_bytes',0)+getattr(projection_provider,'max_bytes',0)>max_bytes:
                raise MemoryError('Qwen3.5 checkpoint conversion exceeds admission')
            return names
        config,weights = read_native_text_checkpoint(directory,model_type='qwen3_5_moe_text',
            max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,tensor_selection=select)
        return _finish_decoder(cls(config,weights,device=device,max_bytes=max_bytes,expert_provider=expert_provider,
            projection_provider=projection_provider,**options),cancel)

    def _check(self):
        if self._closed or self._busy:
            raise RuntimeError('Qwen3.5 device decoder closed or busy')

    def _cancel(self):
        event = getattr(self._cancellation,'event',None)
        if event is not None and event.is_set():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('Qwen3.5 device execution cancelled')

    @contextmanager
    def cancellation_scope(self,cancel):
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise ValueError('cancellation must provide is_set')
        previous = getattr(self._cancellation,'event',None)
        self._cancellation.event = cancel
        try:
            yield
        finally:
            self._cancellation.event = previous

    def _norm(self,hidden,name):
        return hidden*self.torch.rsqrt(hidden.square().mean(dim=-1,keepdim=True)+self.config.rms_norm_eps)*(1+self.weights[name])

    def _state_bytes(self,cache=None):
        return sum(value.numel()*4 for pair in (self._cache if cache is None else cache) for value in pair)

    routing_observation_supported=True

    def forward(self,token_ids,*,attention_mask=None,position_ids=None,inputs_embeds=None,capture_routing=False):
        admit_fp32(self.torch,self.device)
        with self._lock:
            self._check();self._cancel()
            if type(capture_routing) is not bool:raise ValueError('explicit Qwen3.5 routing observation policy required')
            if self.expert_provider is not None and self.expert_provider.identity!=self._expert_identity:
                raise ValueError('Qwen3.5 device expert provider changed')
            ids = np.asarray(token_ids)
            if ids.ndim==1:
                ids = ids[None,:]
            c = self.config
            if ids.ndim!=2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=c.vocab_size):
                raise ValueError('invalid Qwen3.5 token sequence')
            batch,tokens = ids.shape
            end = self.position+tokens
            if batch>128 or tokens>4096 or end>self.capacity:
                raise ValueError('Qwen3.5 sequence admission exceeded')
            mask = np.ones((batch,end),np.uint8) if attention_mask is None else np.asarray(attention_mask)
            if mask.shape!=(batch,end) or not np.isin(mask,(0,1)).all():
                raise ValueError('complete binary attention mask required')
            mask = np.array(mask,dtype=np.uint8,copy=True)
            if self._mask is not None and not np.array_equal(mask[:,:self.position],self._mask):
                raise ValueError('committed mask or batch changed')
            positions = np.broadcast_to(np.arange(self.position,end),ids.shape) if position_ids is None else np.asarray(position_ids)
            spatial=positions.shape==(3,batch,tokens)
            if ((positions.shape!=ids.shape and not (spatial and self.mrope_section is not None)) or
                    positions.dtype.kind not in 'iu' or np.any(positions<0) or np.any(positions>=c.max_position_embeddings)):
                raise ValueError('invalid rotary positions')
            embedded=None if inputs_embeds is None else np.asarray(inputs_embeds)
            if embedded is not None and (embedded.dtype!=np.float32 or embedded.shape!=(batch,tokens,c.hidden_size)
                    or not np.isfinite(embedded).all()):
                raise ValueError('finite FP32 Qwen3.5 embedding rows required')
            pending_bytes = 0
            for kind in c.layer_types:
                if kind=='full_attention':
                    pending_bytes += batch*c.num_key_value_heads*end*c.head_dim*8
                else:
                    width = 2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
                    pending_bytes += batch*(width*c.linear_conv_kernel_dim+c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim)*4
            host_logits = batch*tokens*c.vocab_size*4
            required = self.weight_bytes+self._expert_bytes+self._projection_bytes+self._snapshot_bytes+self._state_bytes()+2*pending_bytes+mask.nbytes+host_logits+batch*tokens*c.hidden_size*32+max(self.layer_bytes,self.head_bytes)+positions.size*16
            if capture_routing:required+=batch*tokens*c.num_hidden_layers*c.num_experts*32
            if required>self.max_bytes:
                raise MemoryError('Qwen3.5 pending continuation/work exceeds admission')
            self._busy = True
            try:
                observations={};seen=set()
                if capture_routing:
                    shape=(batch,tokens,c.num_hidden_layers,c.num_experts)
                    observations={'router_logits':np.empty(shape,np.float32),'routing_weights':np.zeros(shape,np.float32)}
                def observe(layer,raw,selected,scales):
                    self._cancel();rows=batch*tokens
                    if layer in seen or not 0<=layer<c.num_hidden_layers or tuple(raw.shape)!=(rows,c.num_experts) or tuple(selected.shape)!=(rows,c.num_experts_per_tok) or scales.shape!=selected.shape:
                        raise ValueError('Qwen3.5 executed routing observation geometry differs')
                    values=raw.detach().cpu().numpy();indices=selected.detach().cpu().numpy();gammas=scales.detach().cpu().numpy()
                    if indices.dtype!=np.int64 or np.any(indices<0) or np.any(indices>=c.num_experts) or not np.isfinite(values).all() or not np.isfinite(gammas).all():
                        raise ValueError('invalid Qwen3.5 executed routing observation')
                    observations['router_logits'][:,:,layer,:]=values.reshape(batch,tokens,c.num_experts)
                    dense=np.zeros((rows,c.num_experts),np.float32);np.put_along_axis(dense,indices,gammas,axis=-1)
                    observations['routing_weights'][:,:,layer,:]=dense.reshape(batch,tokens,c.num_experts)
                    seen.add(layer);self._cancel()
                torch = self.torch
                event = getattr(self._cancellation,'event',None)
                with torch.inference_mode(), (self.projection_provider.cancellation_scope(event) if self.projection_provider is not None else nullcontext()):
                    hidden = (self.weights['model.embed_tokens.weight'][torch.tensor(ids.astype(np.int64),device=self.device)]
                        if embedded is None else torch.tensor(np.array(embedded,copy=True,order='C'),device=self.device))
                    dm = torch.tensor(mask.astype(bool),device=self.device)
                    dp = torch.tensor(np.array(positions,dtype=np.int64,copy=True),device=self.device)
                    pending = []
                    for index,kind in enumerate(c.layer_types):
                        self._cancel()
                        prefix = f'model.layers.{index}.'
                        normalized = self._norm(hidden,prefix+'input_layernorm.weight')
                        previous = self._cache[index] if self._cache else None
                        if kind=='full_attention':
                            mixed,state = qwen35_device_attention(torch,normalized,self.weights,c,prefix=prefix,
                                previous=previous,positions=dp,mask=dm,start=self.position,rotary_dim=self.rotary_dim,
                                theta=self.theta,max_scratch_bytes=self.layer_bytes,cancel=event)
                        else:
                            mixed,state = qwen35_device_delta(torch,normalized,self.weights,c,prefix=prefix,
                                previous=previous,mask=dm[:,self.position:],max_scratch_bytes=self.layer_bytes,cancel=event)
                        hidden = hidden+mixed
                        hidden = hidden+qwen35_device_moe(torch,self._norm(hidden,prefix+'post_attention_layernorm.weight'),
                            self.weights,c,prefix=prefix,max_scratch_bytes=self.layer_bytes,cancel=event,
                            expert_provider=self.expert_provider,layer_index=index,
                            routing_observer=(lambda raw,selected,scales,layer=index:observe(layer,raw,selected,scales)) if capture_routing else None)
                        pending.append(state)
                    logits = device_logits(torch,self._norm(hidden,'model.norm.weight'),self.weights['lm_head.weight'],
                        max_device_bytes=self.head_bytes,max_host_bytes=host_logits,cancel=event)
                    if any(not bool(torch.isfinite(value).all()) for pair in pending for value in pair):
                        raise FloatingPointError('nonfinite Qwen3.5 device continuation')
                    if capture_routing and len(seen)!=c.num_hidden_layers:raise ValueError('complete Qwen3.5 routing observation inventory required')
                    self._cancel()
                    self._cache,self._mask,self.position = tuple(pending),mask,end
                    return {'logits':logits,**observations} if capture_routing else logits
            finally:
                self._busy = False

    def snapshot(self):
        with self._lock:
            self._check()
            size = self._state_bytes()+(0 if self._mask is None else self._mask.nbytes)
            if len(self._snapshots)>=self.max_snapshots or self.weight_bytes+self._expert_bytes+self._projection_bytes+self._state_bytes()+self._snapshot_bytes+2*size>self.max_bytes:
                raise MemoryError('Qwen3.5 snapshot admission exceeded')
            banks = tuple(tuple(value.detach().cpu().numpy().copy() for value in pair) for pair in self._cache)
            handle = uuid4().hex
            receipt = TorchQwen35Snapshot(self.identity,self._owner,handle,self.position)
            self._snapshots[handle] = (receipt,banks,None if self._mask is None else self._mask.copy(),size)
            self._snapshot_bytes += size
            return receipt

    def _snapshot(self,receipt):
        if type(receipt) is not TorchQwen35Snapshot or receipt.identity!=self.identity or receipt.owner!=self._owner:
            raise ValueError('owned Qwen3.5 snapshot required')
        value = self._snapshots.get(receipt.handle)
        if value is None or value[0]!=receipt:
            raise ValueError('unknown Qwen3.5 snapshot')
        return value

    def restore(self,receipt):
        with self._lock:
            self._check()
            _,banks,mask,size = self._snapshot(receipt)
            if self.weight_bytes+self._expert_bytes+self._projection_bytes+self._state_bytes()+self._snapshot_bytes+2*size>self.max_bytes:
                raise MemoryError('Qwen3.5 restore admission exceeded')
            pending = tuple(tuple(self.torch.tensor(value,device=self.device,dtype=self.torch.float32) for value in pair) for pair in banks)
            self._cache,self._mask,self.position = pending,None if mask is None else mask.copy(),receipt.position

    def release_snapshot(self,receipt):
        with self._lock:
            self._check()
            self._snapshot_bytes -= self._snapshot(receipt)[3]
            del self._snapshots[receipt.handle]

    def reset(self):
        with self._lock:
            self._check()
            self._cache,self._mask,self.position = (),None,0

    def write_cache(self,output,*,max_bytes):
        from .torch_qwen35_cache import write_device_cache
        return write_device_cache(self,output,max_bytes=max_bytes)

    def restore_cache(self,source,*,max_bytes):
        from .torch_qwen35_cache import restore_device_cache
        return restore_device_cache(self,source,max_bytes=max_bytes)

    def save_cache(self,destination,*,max_bytes):
        from .cache_file_recovery import save_executor_cache
        return save_executor_cache(self,destination,max_bytes=max_bytes)

    def load_cache(self,source,*,max_bytes):
        from .cache_file_recovery import load_executor_cache
        return load_executor_cache(self,source,max_bytes=max_bytes)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._check()
            self.weights.clear();self._snapshots.clear()
            self.expert_provider = None
            self.projection_provider=None
            self._cache,self._mask,self.position = (),None,0
            self._snapshot_bytes,self._closed = 0,True

    def status(self):
        with self._lock:
            return dict(model_type='qwen3_5_moe_text',implementation=self.implementation,
                        device=self.device,position=self.position,closed=self._closed,
                        physical_context_tokens=self.capacity,publisher_context_tokens=self.config.max_position_embeddings)
