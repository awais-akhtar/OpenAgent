"""Owned GLM5-next accelerator decoder with KDA/shared-DSA recovery."""
import hashlib
import importlib
import json
import math
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4
import numpy as np
from .independent_glm5_next import Glm5NextTextContract,tensor_layout
from .torch_inkling_decoder import TorchInklingDecoder
from .device_glm5_forward import glm5_device_forward
from .device_logits import device_logits
from .device_execution_policy import admit_no_fallback,admit_fp32

_SOURCE_ADMISSION = object()

def _canonical(name):
    for prefix in ('model.language_model.','language_model.','model.'):
        if name.startswith(prefix): return name[len(prefix):]
    return name

class TorchGlm5NextDecoder(TorchInklingDecoder):
    implementation='torch-glm5-next-device-fp32'
    cache_magic=b'OATGLM01'
    def __init__(self,config,tensors,*,device,max_bytes,max_tokens=None,max_snapshots=4,
                 max_work_bytes=64*1024**2,max_head_bytes=64*1024**2,expert_provider=None,projection_provider=None,
                 _source_admission=None):
        if _source_admission is not _SOURCE_ADMISSION:
            raise ValueError('GLM5 device construction requires metadata-admitted source tensors')
        admit_no_fallback(device)
        self.contract=Glm5NextTextContract.parse(config);c=self.contract
        capacity=c.max_position_embeddings if max_tokens is None else max_tokens
        if type(capacity) is not int or not 1<=capacity<=c.max_position_embeddings:
            raise ValueError('bounded GLM accelerator context capacity required')
        if any(type(v) is not int or v<=0 for v in (max_bytes,max_work_bytes,max_head_bytes)) or type(max_snapshots) is not int or not 1<=max_snapshots<=64:
            raise ValueError('explicit GLM accelerator memory and snapshot limits required')
        if type(device) is not str or (device not in ('cuda','mps') and not (device.startswith('cuda:') and device[5:].isdigit())):
            raise ValueError('explicit GLM CUDA/HIP or Metal device required')
        shapes,self._slots=tensor_layout(c);supplied={}
        self.expert_provider=expert_provider
        provider_bytes=0
        if expert_provider is not None:
            from .device_glm5_publisher_experts import DeviceGlm5PublisherExperts
            if type(expert_provider) is not DeviceGlm5PublisherExperts or expert_provider.contract!=c:
                raise ValueError('exact GLM device expert provider required')
            omitted={name for name in shapes if name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
            shapes={name:shape for name,shape in shapes.items() if name not in omitted}
            self._slots=tuple(tuple(None if name in omitted else name for name in layer) for layer in self._slots)
            provider_bytes=expert_provider.max_bytes
        for name,value in tensors.items():
            canonical=_canonical(name)
            if canonical in supplied: raise ValueError('duplicate GLM canonical tensor: '+canonical)
            supplied[canonical]=value
        shapes['lm_head.weight']=(c.vocab_size,c.hidden_size)
        self.projection_provider=projection_provider
        streamed=set()
        if projection_provider is not None:
            from .device_publisher_projections import DevicePublisherProjections
            if type(projection_provider) is not DevicePublisherProjections:
                raise TypeError('GLM original device projection bank required')
            if device=='cuda' and projection_provider.device.type=='cuda': device=str(projection_provider.device)
            if str(projection_provider.device)!=device: raise ValueError('GLM projection device differs')
            for name,value in projection_provider.matrices.items():
                if name not in shapes or value.shape!=shapes[name] or name in supplied:
                    raise ValueError('GLM streamed projection overlaps or differs from tensor inventory: '+name)
                if name=='lm_head.weight' and c.tie_word_embeddings and value is not projection_provider.matrices.get('embed_tokens.weight'):
                    raise ValueError('GLM tied head must share its original embedding source')
                supplied[name]=value;streamed.add(name)
            provider_bytes+=projection_provider.max_bytes
        if c.tie_word_embeddings:
            embedding=supplied.get('embed_tokens.weight')
            if 'lm_head.weight' in supplied and ((('embed_tokens.weight' in streamed) and supplied['lm_head.weight'] is not embedding)
                    or ('embed_tokens.weight' not in streamed and not np.array_equal(supplied['lm_head.weight'],embedding))):
                raise ValueError('GLM tied head differs from embedding')
            supplied['lm_head.weight']=embedding
            if 'embed_tokens.weight' in streamed:streamed.add('lm_head.weight')
        if set(supplied)!=set(shapes): raise ValueError('complete exact GLM accelerator tensor inventory required')
        self.weight_bytes=sum(math.prod(shape)*4 for name,shape in shapes.items() if name not in streamed)+provider_bytes
        if 3*self.weight_bytes+max_work_bytes+max_head_bytes>max_bytes:
            raise MemoryError('GLM source/device construction exceeds aggregate admission')
        digest=hashlib.sha256(json.dumps(dict(config),sort_keys=True,allow_nan=False).encode());admitted={}
        if expert_provider is not None: digest.update(expert_provider.identity.encode())
        if projection_provider is not None: digest.update(projection_provider.identity.encode())
        for name,shape in shapes.items():
            if name in streamed: continue
            value=np.asarray(supplied[name])
            if value.dtype!=np.float32 or value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('invalid GLM device tensor: '+name)
            admitted[name]=np.array(value,copy=True,order='C');digest.update(name.encode());digest.update(memoryview(admitted[name]).cast('B'))
        torch=importlib.import_module('torch')
        if device.startswith('cuda'):
            if not torch.cuda.is_available(): raise RuntimeError('requested GLM CUDA/HIP device unavailable')
            index=torch.cuda.current_device() if device=='cuda' else int(device.split(':')[1])
            if index>=torch.cuda.device_count(): raise ValueError('GLM device index unavailable')
            device='cuda:'+str(index)
        elif not torch.backends.mps.is_available(): raise RuntimeError('requested GLM Metal device unavailable')
        self.torch,self.device=torch,device
        self.weights={name:torch.tensor(value,device=device,dtype=torch.float32) for name,value in admitted.items()}
        if projection_provider is not None: self.weights.update(projection_provider.matrices)
        if c.tie_word_embeddings and 'embed_tokens.weight' in streamed:
            self.weights['lm_head.weight']=self.weights['embed_tokens.weight']
        digest.update(repr((device,torch.__version__,capacity,self.implementation)).encode())
        self.identity='openagent-torch-glm5-next-v1:'+digest.hexdigest()
        self.config=dict(model_type='glm5_next_text',vocab_size=c.vocab_size,hidden_size=c.hidden_size,max_position_embeddings=capacity)
        self.geometry=tuple(c.layer_types);self.max_tokens=capacity
        self.max_bytes,self.max_snapshots,self.work_bytes,self.head_bytes=max_bytes,max_snapshots,max_work_bytes,max_head_bytes
        self._lock=threading.RLock();self._local=threading.local();self._closed=self._busy=False
        self._owner=uuid4().hex;self._snapshots={};self._snapshot_bytes=0;self._cache=();self.position=0
    def _state_size(self,kind):
        c=self.contract
        if kind=='linear_attention':
            return 3*c.linear_num_heads*c.linear_head_dim*c.linear_conv_kernel_dim+c.linear_num_heads*c.linear_head_dim**2
        return self.max_tokens*(c.num_attention_heads*(c.qk_nope_head_dim+c.v_head_dim)+2*c.index_head_dim)
    @classmethod
    def from_directory(cls,directory,*,device,max_weight_bytes,max_bytes,read_limits=None,cancel=None,**options):
        # These providers are independently constructed from a publisher
        # snapshot.  Matching their configuration/geometry does not prove
        # their expert/projection payloads came from this directory's exact
        # checkpoint revision.  Until a shared immutable snapshot/content
        # identity is threaded through both owners, do not permit an external
        # provider to replace tensors from the admitted source inventory.
        if options.get('expert_provider') is not None or options.get('projection_provider') is not None:
            raise ValueError(
                'GLM5 external device expert/projection providers are disabled '
                'until source checkpoint content identity is shared')
        from .native_checkpoint import read_native_text_checkpoint,_finish_decoder
        def select(config,inventory):
            c=Glm5NextTextContract.parse(config);shapes,_=tensor_layout(c)
            canonical={}
            for name,record in inventory.items():
                key=_canonical(name)
                if key in canonical: raise ValueError('duplicate GLM checkpoint namespace')
                canonical[key]=(name,record)
            if not c.tie_word_embeddings or 'lm_head.weight' in canonical:
                shapes['lm_head.weight']=(c.vocab_size,c.hidden_size)
            if set(shapes)!=set(canonical) or any(
                    tuple(canonical[name][1]['shape'])!=shape or
                    canonical[name][1].get('dtype') not in ('F32','F16','BF16')
                    for name,shape in shapes.items()):
                raise ValueError('exact full GLM accelerator checkpoint required')
            from .source_scope import require_source_inventory_admission
            require_source_inventory_admission(
                family='glm5-next',model_type='glm5_next_text',backend='device',
                arithmetic='fp32',persisted_format='float32',inventory=inventory)
            omitted=set();expert=options.get('expert_provider');projections=options.get('projection_provider')
            if expert is not None:
                omitted.update(name for name in shapes if name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj')))
            if projections is not None:
                for name,value in projections.matrices.items():
                    if name=='lm_head.weight' and c.tie_word_embeddings and name not in shapes:
                        if value is not projections.matrices.get('embed_tokens.weight'):raise ValueError('GLM tied source differs')
                        continue
                    if name not in shapes or name in omitted or tuple(value.shape)!=shapes[name] or (name=='lm_head.weight' and c.tie_word_embeddings and value is not projections.matrices.get('embed_tokens.weight')):
                        raise ValueError('streamed GLM checkpoint projection inventory differs')
                    omitted.add(name)
            selected={name:shape for name,shape in shapes.items() if name not in omitted}
            reservations=sum(provider.max_bytes for provider in (expert,projections) if provider is not None)
            if 4*sum(math.prod(shape)*4 for shape in selected.values())+reservations>max_bytes:
                raise MemoryError('GLM checkpoint conversion exceeds aggregate admission')
            return tuple(canonical[name][0] for name in sorted(selected))
        config,weights=read_native_text_checkpoint(directory,model_type='glm5_next_text',max_weight_bytes=max_weight_bytes,
            read_limits=read_limits,cancel=cancel,tensor_selection=select,source_backend='device')
        return _finish_decoder(cls(config,weights,device=device,max_bytes=max_bytes,
            _source_admission=_SOURCE_ADMISSION,**options),cancel)
    def forward(self,token_ids,*,attention_mask=None,position_ids=None,use_cache=True,inputs_embeds=None,capture_routing=False):
        admit_fp32(self.torch,self.device)
        with self._lock:
            self._check();self._cancel();ids=np.asarray(token_ids);c=self.contract
            if ids.ndim==1: ids=ids[None,:]
            if ids.ndim!=2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or ids.shape[0]>128 or ids.shape[1]>4096 or np.any(ids<0) or np.any(ids>=c.vocab_size):
                raise ValueError('bounded GLM integer token batch required')
            if attention_mask is not None or position_ids is not None or type(use_cache) is not bool:
                raise ValueError('GLM device uses contiguous unmasked NoPE positions')
            if type(capture_routing) is not bool:raise ValueError('boolean GLM routing observation policy required')
            start=self.position if use_cache else 0;batch,tokens=ids.shape
            if start+tokens>self.max_tokens or (use_cache and self._cache and self._cache[0].shape[0]!=batch):
                raise ValueError('GLM continuation context/batch differs')
            state=batch*sum(self._state_size(kind) for kind in self.geometry)*4
            output=batch*tokens*(c.hidden_size+c.vocab_size)*4
            if capture_routing:output+=batch*tokens*c.num_hidden_layers*c.n_routed_experts*8
            self._admit(2*state+2*output+self.work_bytes+self.head_bytes+ids.size*16)
            media=None
            if inputs_embeds is not None:
                source=np.asarray(inputs_embeds)
                if source.shape!=(*ids.shape,c.hidden_size) or source.dtype!=np.float32 or not np.isfinite(source).all():
                    raise ValueError('matching finite GLM FP32 media embeddings required')
                media=self.torch.tensor(source,device=self.device)
            self._busy=True
            try:
                with self.torch.inference_mode(), (self.projection_provider.cancellation_scope(getattr(self._local,'cancel',None))
                        if self.projection_provider is not None else nullcontext()):
                    observed=glm5_device_forward(self.torch,self.torch.tensor(ids.astype(np.int64),device=self.device),self.weights,
                        c,self._slots,previous=self._cache if use_cache else (),start=start,capacity=self.max_tokens,
                        max_work_bytes=self.work_bytes,inputs_embeds=media,cancel=getattr(self._local,'cancel',None),expert_provider=self.expert_provider,
                        capture_routing=capture_routing)
                    hidden,pending=observed[:2]
                    logits=device_logits(self.torch,hidden,self.weights['lm_head.weight'],max_device_bytes=self.head_bytes,
                        max_host_bytes=batch*tokens*c.vocab_size*4,cancel=getattr(self._local,'cancel',None))
                    result=SimpleNamespace(logits=logits,last_hidden_state=hidden.detach().cpu().numpy().copy())
                    if capture_routing:
                        result.router_logits=observed[2].detach().cpu().numpy().copy()
                        result.routing_weights=observed[3].detach().cpu().numpy().copy()
                    self._cancel()
                    if use_cache: self._cache,self.position=pending,start+tokens
                    return result
            finally:
                self._busy=False
    def status(self):
        return dict(model_type='glm5_next_text',implementation=self.implementation,device=self.device,position=self.position,closed=self._closed)

__all__=['TorchGlm5NextDecoder']
