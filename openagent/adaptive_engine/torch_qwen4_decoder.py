"""Owned explicit Qwen4 accelerator decoder and atomic hybrid continuation."""
import hashlib
from contextlib import nullcontext
import importlib
import json
import math
import threading
from dataclasses import dataclass
from uuid import uuid4
import numpy as np

from .independent_qwen4 import IndependentQwen4TextDecoder
from .torch_qwen35_decoder import TorchQwen35Decoder
from .device_qwen4_forward import qwen4_device_forward
from .device_execution_policy import admit_no_fallback,admit_fp32


@dataclass(frozen=True)
class TorchQwen4Snapshot:
    identity: str
    owner: str
    handle: str
    position: int


class TorchQwen4Decoder(TorchQwen35Decoder):
    implementation = 'torch-qwen4-device-fp32'

    def __init__(self,config,tensors,*,device,max_bytes,max_snapshots=4,
                 max_layer_bytes=64*1024**2,max_head_bytes=64*1024**2,expert_provider=None,projection_provider=None):
        admit_no_fallback(device)
        inventory = IndependentQwen4TextDecoder(config,{},_inventory_only=True)
        self.config = inventory.config
        self.rotary_dim,self.theta,self.rope_scale = inventory.rotary_dim,getattr(inventory,'theta',10000.0),inventory.rope_scale
        self.mrope_section=inventory.mrope_section
        if any(type(value) is not int or value<=0 for value in (max_bytes,max_layer_bytes,max_head_bytes)) or type(max_snapshots) is not int or not 1<=max_snapshots<=64:
            raise ValueError('explicit bounded Qwen4 device budgets required')
        if type(device) is not str or (device not in ('cuda','mps') and not (device.startswith('cuda:') and device[5:].isdigit())):
            raise ValueError('explicit CUDA/HIP or Metal device required')
        shapes = dict(inventory.tensor_shapes)
        self.expert_provider,self._expert_bytes,self._expert_identity = expert_provider,0,None
        if expert_provider is not None:
            from .torch_qwen4_experts import TorchQwen4Experts
            from .device_qwen4_publisher_experts import DeviceQwen4PublisherExperts
            from .qwen4_streamed_experts import qwen4_metadata_fingerprint
            c = self.config
            if (type(expert_provider) not in (TorchQwen4Experts,DeviceQwen4PublisherExperts) or len(expert_provider.layers)!=c.num_hidden_layers or
                    (expert_provider.hidden_size,expert_provider.intermediate_size,expert_provider.expert_count,expert_provider.selected_experts)!=
                    (c.hidden_size,c.moe_intermediate_size,c.num_experts,c.num_experts_per_tok) or
                    (expert_provider.fingerprint if type(expert_provider) is DeviceQwen4PublisherExperts else expert_provider.manifest.fingerprint)!=qwen4_metadata_fingerprint(config)):
                raise ValueError('exact Qwen4 device expert provider required')
            self._expert_bytes,self._expert_identity = expert_provider.max_bytes,expert_provider.identity
            shapes = {name:shape for name,shape in shapes.items() if not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
        supplied = dict(tensors)
        self.projection_provider=projection_provider
        self._owns_projection_provider=False
        self._projection_bytes=0
        if projection_provider is not None:
            from .device_publisher_projections import DevicePublisherProjections
            if type(projection_provider) is not DevicePublisherProjections:
                raise TypeError('owned original device projection provider required')
            if device=='cuda' and projection_provider.device.type=='cuda':
                device=str(projection_provider.device)
            if str(projection_provider.device)!=device:
                raise ValueError('projection provider and decoder device must match explicitly')
            for name,value in projection_provider.matrices.items():
                if name not in shapes or tuple(shapes[name])!=value.shape:
                    raise ValueError('projection binding is outside the dense decoder inventory: '+name)
                if name in supplied:
                    raise ValueError('duplicate resident and streamed projection: '+name)
                supplied[name]=value
            self._projection_bytes=projection_provider.max_bytes
        streamed=set() if projection_provider is None else set(projection_provider.matrices)
        if config.get('tie_word_embeddings',False):
            if 'lm_head.weight' in streamed and supplied.get('lm_head.weight') is not supplied.get('model.embed_tokens.weight'):
                raise ValueError('tied heads must use the same embedding source, without a second projection binding')
            if 'lm_head.weight' in supplied and (('model.embed_tokens.weight' in streamed and
                    supplied['lm_head.weight'] is not supplied['model.embed_tokens.weight']) or
                    ('model.embed_tokens.weight' not in streamed and not np.array_equal(supplied.get('model.embed_tokens.weight'),supplied['lm_head.weight']))):
                raise ValueError('inconsistent Qwen4 tied weights')
            supplied['lm_head.weight'] = supplied.get('model.embed_tokens.weight')
            if 'model.embed_tokens.weight' in streamed:streamed.add('lm_head.weight')
        if set(supplied)!=set(shapes)|set(inventory.integer_metadata):
            raise ValueError('complete exact Qwen4 device tensor inventory required')
        self.weight_bytes = sum(math.prod(shape)*4 for name,shape in shapes.items() if name not in streamed)+sum(value.nbytes for value in inventory.integer_metadata.values())
        self._expert_bytes+=self._projection_bytes
        if 3*self.weight_bytes+self._expert_bytes>max_bytes:
            raise MemoryError('Qwen4 source/device weights exceed admission')
        admitted = {}
        digest = hashlib.sha256(json.dumps(dict(config),sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        for name,shape in shapes.items():
            if name in streamed:
                continue
            value = np.asarray(supplied[name])
            if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('invalid device Qwen4 FP32 tensor: '+name)
            admitted[name] = np.array(value,copy=True,order='C')
        for name,expected in inventory.integer_metadata.items():
            value = np.asarray(supplied[name])
            if value.dtype.kind not in 'iu' or not np.array_equal(value,expected):
                raise ValueError('invalid Qwen4 hash metadata: '+name)
            admitted[name] = expected.copy()
        for name,value in sorted(admitted.items()):
            digest.update(name.encode());digest.update(memoryview(value).cast('B'))
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
        self.weights = {name:torch.tensor(value,device=device,dtype=torch.int64 if name in inventory.integer_metadata else torch.float32) for name,value in admitted.items()}
        if projection_provider is not None:
            self.weights.update(projection_provider.matrices)
            if config.get('tie_word_embeddings',False) and 'model.embed_tokens.weight' in streamed:
                self.weights['lm_head.weight']=self.weights['model.embed_tokens.weight']
            digest.update(projection_provider.identity.encode())
        digest.update(repr((device,torch.__version__,self.implementation,max_layer_bytes,max_head_bytes)).encode())
        if self._expert_identity is not None:
            digest.update(self._expert_identity.encode())
        self.identity = 'openagent-torch-qwen4-v1:'+digest.hexdigest()
        self.max_bytes,self.max_snapshots,self.layer_bytes,self.head_bytes = max_bytes,max_snapshots,max_layer_bytes,max_head_bytes
        self._lock,self._closed,self._busy = threading.RLock(),False,False
        self._cancellation = threading.local()
        self._owner,self._snapshots,self._snapshot_bytes = uuid4().hex,{},0
        self._cache,self._ple,self._mask,self._positions,self.position = (),(),None,None,0

    @classmethod
    def from_directory(cls,directory,*,device,max_weight_bytes,max_bytes,read_limits=None,cancel=None,expert_provider=None,projection_provider=None,
                       max_projection_bytes=None,max_projection_host_bytes=None,max_projection_cache_bytes=0,**options):
        from .native_checkpoint import read_native_text_checkpoint,_finish_decoder
        owned=False
        if max_projection_bytes is not None:
            if projection_provider is not None:
                raise ValueError('choose an external or an owned projection bank')
            from .publisher_checkpoint.storage import snapshot,thaw
            from .device_publisher_projections import DevicePublisherProjections
            metadata=snapshot(directory,max_metadata_bytes=16*1024**2,max_tensors=200000,max_shards=8192)
            inventory=IndependentQwen4TextDecoder(thaw(metadata.config),{},_inventory_only=True)
            names=tuple(name for name,shape in inventory.tensor_shapes.items()
                        if len(shape)==2
                        and not (name=='lm_head.weight' and metadata.config.get('tie_word_embeddings',False)))
            torch=importlib.import_module('torch')
            if device=='cuda':
                device='cuda:'+str(torch.cuda.current_device())
            projection_provider=DevicePublisherProjections.from_snapshot(metadata,names,torch=torch,device=device,
                max_bytes=max_projection_bytes,max_host_bytes=max_projection_host_bytes,
                max_cache_bytes=max_projection_cache_bytes)
            owned=True
            if metadata.config.get('tie_word_embeddings',False):
                try:
                    from .publisher_tensor_reader import TensorSlice
                    record=metadata.tensors.get('lm_head.weight')
                    projection_provider.alias('lm_head.weight','model.embed_tokens.weight',
                        expected_binding=None if record is None else TensorSlice('lm_head.weight',tuple(record.shape)),cancel=cancel)
                except BaseException as primary:
                    try:projection_provider.close()
                    except BaseException as cleanup:
                        primary.cleanup=projection_provider.close;primary.add_note('tied projection owner cleanup retained: '+repr(cleanup))
                    raise
        def select(config,inventory):
            admitted = IndependentQwen4TextDecoder(config,{},_inventory_only=True)
            shapes = dict(admitted.tensor_shapes)
            shapes.update((name,value.shape) for name,value in admitted.integer_metadata.items())
            if config.get('tie_word_embeddings',False) and 'lm_head.weight' not in inventory:
                del shapes['lm_head.weight']
            if set(shapes)!=set(inventory) or any(tuple(inventory[name]['shape'])!=tuple(shape) for name,shape in shapes.items()):
                raise ValueError('exact full Qwen4 checkpoint inventory required')
            names = tuple(sorted(shapes))
            if expert_provider is not None:
                names = tuple(name for name in names if not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj')))
            if projection_provider is not None:
                names=tuple(name for name in names if name not in projection_provider.matrices)
            size = sum(math.prod(shapes[name])*(8 if name in admitted.integer_metadata else 4) for name in names)
            if 4*size+getattr(expert_provider,'max_bytes',0)+getattr(projection_provider,'max_bytes',0)>max_bytes:
                raise MemoryError('Qwen4 device checkpoint conversion exceeds admission')
            return names
        try:
            config,weights = read_native_text_checkpoint(directory,model_type='qwen4_exp_text',max_weight_bytes=max_weight_bytes,
                read_limits=read_limits,cancel=cancel,tensor_selection=select)
            if projection_provider is not None:
                from .publisher_checkpoint.storage import freeze
                projection_provider.reader.snapshot.assert_unchanged()
                if projection_provider.reader.snapshot.config!=freeze(config):
                    raise ValueError('projection checkpoint configuration differs from decoder source')
            decoder=cls(config,weights,device=device,max_bytes=max_bytes,expert_provider=expert_provider,projection_provider=projection_provider,**options)
            decoder._owns_projection_provider=owned
            return _finish_decoder(decoder,cancel)
        except BaseException:
            if owned:
                projection_provider.close()
            raise

    def _state_bytes(self,cache=None):
        banks = self._cache+self._ple if cache is None else cache
        return sum(value.numel()*value.element_size() for pair in banks if pair is not None for value in pair)+(0 if self._positions is None else self._positions.numel()*8)

    routing_observation_supported = True

    def forward(self,token_ids,*,attention_mask=None,position_ids=None,inputs_embeds=None,capture_routing=False):
        admit_fp32(self.torch,self.device)
        with self._lock:
            self._check();self._cancel()
            if self.expert_provider is not None and self.expert_provider.identity!=self._expert_identity:
                raise ValueError('Qwen4 device provider changed after admission')
            c = self.config
            if type(capture_routing) is not bool:
                raise ValueError('explicit Qwen4 routing observation policy required')
            ids = np.asarray(token_ids)
            if ids.ndim==1:
                ids = ids[None,:]
            if ids.ndim!=2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=c.vocab_size):
                raise ValueError('invalid Qwen4 token sequence')
            batch,tokens = ids.shape
            end = self.position+tokens
            if batch>128 or tokens>4096 or end>c.max_position_embeddings:
                raise ValueError('Qwen4 sequence admission exceeded')
            mask = np.ones((batch,end),np.uint8) if attention_mask is None else np.asarray(attention_mask)
            if mask.shape!=(batch,end) or not np.isin(mask,(0,1)).all():
                raise ValueError('complete binary Qwen4 mask required')
            mask = np.array(mask,dtype=np.uint8,copy=True)
            if self._mask is not None and not np.array_equal(mask[:,:self.position],self._mask):
                raise ValueError('committed Qwen4 mask/batch changed')
            pos = np.broadcast_to(np.arange(self.position,end),ids.shape) if position_ids is None else np.asarray(position_ids)
            if self.mrope_section is not None and pos.shape==ids.shape:
                pos=np.broadcast_to(pos,(3,*ids.shape))
            position_shape=(3,*ids.shape) if self.mrope_section is not None else ids.shape
            if pos.shape!=position_shape or pos.dtype.kind not in 'iu' or np.any(pos<0) or np.any(pos>=c.max_position_embeddings):
                raise ValueError('bounded Qwen4 text positions required')
            linear_width = 2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
            state = (24 if self.mrope_section is not None else 8)*batch*end
            for kind in c.layer_types:
                if kind=='linear_attention':
                    state += 4*batch*(linear_width*c.linear_conv_kernel_dim+c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim)
                else:
                    state += 4*batch*end*(2*c.num_key_value_heads*c.head_dim+c.indexer_head_dim)
            state += len(c.ple_layer_ids)*batch*(8*(c.ngram_size-1)+4*c.hc_count*c.hidden_size*(c.ple_conv_kernel_size-1)*c.ngram_size)
            host = batch*tokens*c.vocab_size*4
            routing_bytes = batch*tokens*c.num_hidden_layers*c.num_experts*32 if capture_routing else 0
            if self.weight_bytes+self._expert_bytes+self._snapshot_bytes+self._state_bytes()+2*state+mask.nbytes+host+routing_bytes+batch*tokens*c.hc_count*c.hidden_size*24+max(self.layer_bytes,self.head_bytes)>self.max_bytes:
                raise MemoryError('Qwen4 pending device continuation/work exceeds admission')
            media=None
            if inputs_embeds is not None:
                source=np.asarray(inputs_embeds)
                if source.shape!=(*ids.shape,c.hidden_size) or source.dtype!=np.float32 or not np.isfinite(source).all():
                    raise ValueError('matching finite Qwen4 FP32 media embeddings required')
                media=self.torch.tensor(source,device=self.device)
            self._busy = True
            try:
                observations = None
                observer = None
                if capture_routing:
                    shape = (batch,tokens,c.num_hidden_layers,c.num_experts)
                    observations = {'router_logits':np.empty(shape,np.float32),
                                    'routing_weights':np.zeros(shape,np.float32)}
                    seen = set()
                    def observer(layer, raw, selected, scales):
                        self._cancel()
                        if layer in seen or not 0 <= layer < c.num_hidden_layers:
                            raise ValueError('duplicate or invalid Qwen4 routing observation layer')
                        rows = batch*tokens
                        if tuple(raw.shape)!=(rows,c.num_experts) or tuple(selected.shape)!=(rows,c.num_experts_per_tok) or scales.shape!=selected.shape:
                            raise ValueError('Qwen4 device routing observation geometry differs')
                        logits_host = raw.detach().cpu().numpy()
                        ids_host = selected.detach().cpu().numpy()
                        scales_host = scales.detach().cpu().numpy()
                        if (not np.isfinite(logits_host).all() or not np.isfinite(scales_host).all() or
                                ids_host.dtype!=np.int64 or np.any(ids_host<0) or np.any(ids_host>=c.num_experts)):
                            raise ValueError('invalid Qwen4 device routing observation')
                        observations['router_logits'][:,:,layer,:] = logits_host.reshape(batch,tokens,c.num_experts)
                        dense = np.zeros((rows,c.num_experts),np.float32)
                        np.put_along_axis(dense,ids_host,scales_host,axis=-1)
                        observations['routing_weights'][:,:,layer,:] = dense.reshape(batch,tokens,c.num_experts)
                        seen.add(layer)
                        self._cancel()
                torch = self.torch
                with torch.inference_mode(), (self.projection_provider.cancellation_scope(getattr(self._cancellation,'event',None))
                        if self.projection_provider is not None else nullcontext()):
                    dp = torch.tensor(np.array(pos,dtype=np.int64,copy=True),device=self.device)
                    full = dp if self._positions is None else torch.cat((self._positions,dp),dim=-1)
                    logits,cache,ple = qwen4_device_forward(torch,torch.tensor(ids.astype(np.int64),device=self.device),self.weights,c,
                        previous=self._cache or (None,)*c.num_hidden_layers,previous_ple=self._ple or (None,)*c.num_hidden_layers,
                        mask=torch.tensor(mask.astype(bool),device=self.device),positions=dp,full_positions=full,start=self.position,
                        rotary_dim=self.rotary_dim,theta=self.theta,rope_scale=self.rope_scale,max_layer_bytes=self.layer_bytes,
                        max_head_bytes=self.head_bytes,max_host_logits_bytes=host,expert_provider=self.expert_provider,
                        cancel=getattr(self._cancellation,'event',None),inputs_embeds=media,mrope_section=self.mrope_section,
                        routing_observer=observer)
                    if capture_routing and len(seen)!=c.num_hidden_layers:
                        raise ValueError('complete Qwen4 device routing inventory required')
                    self._cancel()
                    self._cache,self._ple,self._mask,self._positions,self.position = cache,ple,mask,full,end
                    return {'logits':logits,**observations} if capture_routing else logits
            finally:
                self._busy = False

    def reset(self):
        with self._lock:
            self._check()
            self._cache,self._ple,self._mask,self._positions,self.position = (),(),None,None,0

    def snapshot(self):
        with self._lock:
            self._check();self._cancel()
            size = self._state_bytes()+(0 if self._mask is None else self._mask.nbytes)
            if len(self._snapshots)>=self.max_snapshots or self.weight_bytes+self._expert_bytes+self._snapshot_bytes+self._state_bytes()+2*size>self.max_bytes:
                raise MemoryError('Qwen4 complete snapshot exceeds admission')
            def freeze(value):
                array = value.detach().cpu().numpy().copy()
                array.flags.writeable = False
                return array
            cache = tuple(None if bank is None else tuple(freeze(value) for value in bank) for bank in self._cache)
            ple = tuple(None if bank is None else tuple(freeze(value) for value in bank) for bank in self._ple)
            positions = None if self._positions is None else freeze(self._positions)
            mask = None if self._mask is None else self._mask.copy()
            if mask is not None:
                mask.flags.writeable = False
            self._cancel()
            receipt = TorchQwen4Snapshot(self.identity,self._owner,uuid4().hex,self.position)
            self._snapshots[receipt.handle] = (receipt,cache,ple,positions,mask,size)
            self._snapshot_bytes += size
            return receipt

    def _snapshot(self,receipt):
        if type(receipt) is not TorchQwen4Snapshot or receipt.owner!=self._owner or receipt.identity!=self.identity:
            raise ValueError('owned Qwen4 device snapshot required')
        entry = self._snapshots.get(receipt.handle)
        if entry is None or entry[0]!=receipt:
            raise ValueError('unknown or released Qwen4 snapshot')
        return entry

    def restore(self,receipt):
        with self._lock:
            self._check();self._cancel()
            _,cache,ple,positions,mask,size = self._snapshot(receipt)
            if self.weight_bytes+self._expert_bytes+self._snapshot_bytes+self._state_bytes()+2*size>self.max_bytes:
                raise MemoryError('Qwen4 complete snapshot restore exceeds admission')
            def device(value):
                return self.torch.tensor(value,device=self.device,dtype=self.torch.int64 if value.dtype.kind in 'iu' else self.torch.float32)
            pending = tuple(None if bank is None else tuple(device(value) for value in bank) for bank in cache)
            lexical = tuple(None if bank is None else tuple(device(value) for value in bank) for bank in ple)
            dp = None if positions is None else device(positions)
            dm = None if mask is None else mask.copy()
            self._cancel()
            self._cache,self._ple,self._positions,self._mask,self.position = pending,lexical,dp,dm,receipt.position

    def release_snapshot(self,receipt):
        with self._lock:
            self._check()
            self._snapshot_bytes -= self._snapshot(receipt)[5]
            del self._snapshots[receipt.handle]

    def write_cache(self,output,*,max_bytes):
        from .torch_qwen4_cache import write_device_cache
        return write_device_cache(self,output,max_bytes=max_bytes)

    def restore_cache(self,source,*,max_bytes):
        from .torch_qwen4_cache import restore_device_cache
        return restore_device_cache(self,source,max_bytes=max_bytes)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._check()
            if self._owns_projection_provider:
                self.projection_provider.close()
            self.reset();self.weights.clear();self._snapshots.clear()
            self.expert_provider = None
            self.projection_provider=None
            self._snapshot_bytes,self._closed = 0,True

    def status(self):
        with self._lock:
            return dict(model_type='qwen4_exp_text',implementation=self.implementation,
                        device=self.device,position=self.position,closed=self._closed)
