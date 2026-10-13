"""Owned Qwen4 conditional-generation composition and media continuation.

Accepts expanded processor tokens and patches. Native text arithmetic consumes
the original ids for lexical features and the scattered vision embeddings.
The wrapper owns rope deltas with cache snapshots and durable transport.
"""
from dataclasses import dataclass
from contextlib import ExitStack
import hashlib
import hmac
import json
import struct
import threading
from uuid import uuid4
import numpy as np
from .publisher_family_contracts import component_contract,partition_weights
from .native_qwen4 import NativeQwen4TextDecoder
from .qwen4_vision import Qwen4VisionEncoder,_cancel
from .segment_snapshot import _read,_write,_pairs

@dataclass(frozen=True)
class Qwen4MediaSnapshot:
    owner: str
    key: str
    identity: str
    position: int

class Qwen4MultimodalDecoder:
    implementation='native-c-qwen4-conditional-fp32'
    native_range_execution=True
    cache_magic=b'OAQ4MM01'
    def __init__(self,config,tensors,*,library,text_library,max_bytes,max_snapshots=4,arithmetic='fp32',
                 ple_provider=None,own_ple_provider=False,expert_provider=None,own_expert_provider=False,text_device=None,
                 projection_provider=None,own_projection_provider=False,native_composition=False,
                 native_store_runtime_library=None,native_expert_max_bytes=0,native_ple_max_bytes=0,
                 segment_runtime_library=None,segment_model_directory=None,segment_work_bytes=None,
                 segment_layer_ranges=None):
        self.components=component_contract(config,arithmetic=arithmetic)
        if self.components.variant.wrapper!='qwen4_exp':
            raise ValueError('Qwen4 publisher wrapper required')
        if type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('explicit aggregate media/text budget required')
        self.config=self.components.original_config
        self.max_bytes=max_bytes;self._lock=threading.RLock();self._busy=False;self._closed=False
        self._owner=uuid4().hex;self._snapshots={};self._orphaned_snapshots=[];self.max_snapshots=max_snapshots
        self.rope_delta=0;self.text=self.vision=None
        self._native_store_runtime=self._native_experts=self._native_ple=None
        if type(native_composition) is not bool:raise ValueError('boolean complete native composition policy required')
        if any(type(value) is not int or value<0 for value in (native_expert_max_bytes,native_ple_max_bytes)):
            raise ValueError('nonnegative original native store budgets required')
        if native_composition and text_device is not None:raise ValueError('native composition requires native text arithmetic')
        if segment_runtime_library is not None:
            if text_device is not None or not native_composition or segment_model_directory is None:
                raise ValueError('public Qwen4 requires native composition and an admitted checkpoint directory')
            if type(segment_work_bytes) is not int or segment_work_bytes<=0:
                raise ValueError('explicit native public work admission required')
        elif any(value is not None for value in (segment_model_directory,segment_work_bytes,segment_layer_ranges)):
            raise ValueError('public Qwen4 options require a runtime library')
        if native_store_runtime_library is not None or native_expert_max_bytes or native_ple_max_bytes:
            if not native_composition or native_store_runtime_library is None:
                raise ValueError('native original stores require explicit runtime and native composition')
            if bool(expert_provider is not None)!=bool(native_expert_max_bytes) or bool(ple_provider is not None)!=bool(native_ple_max_bytes):
                raise ValueError('each original streamed bank requires its matching native owner budget')
        elif native_composition and (expert_provider is not None or ple_provider is not None):
            raise ValueError('streamed composition requires native original store ownership')
        self._ple_provider=ple_provider;self._own_ple_provider=own_ple_provider
        self._expert_provider=expert_provider;self._own_expert_provider=own_expert_provider
        self._projection_provider=projection_provider;self._own_projection_provider=own_projection_provider
        text,vision,audio,auxiliary=partition_weights(tensors,self.components)
        if audio:
            raise ValueError('Qwen4 wrapper does not own an audio encoder')
        # Auxiliary tensor ownership remains explicit even when generation does
        # not request draft prediction; no checkpoint namespace is discarded.
        auxiliary_bytes=sum(np.asarray(a).nbytes for a in auxiliary.values())
        if 2*auxiliary_bytes>=max_bytes:
            raise MemoryError('auxiliary construction staging exhausts admission')
        for name,value in auxiliary.items():
            value=np.asarray(value)
            if value.dtype.kind!='f' or not np.isfinite(value).all():
                raise ValueError('finite owned Qwen4 auxiliary weights required: '+name)
        self.auxiliary={name:np.frombuffer(np.asarray(value).tobytes(),dtype=np.asarray(value).dtype).reshape(np.asarray(value).shape)
                        for name,value in auxiliary.items()}
        available=max_bytes-auxiliary_bytes
        if available<=0:
            raise MemoryError('auxiliary checkpoint storage exhausts admission')
        try:
            if (native_expert_max_bytes+native_ple_max_bytes+getattr(expert_provider,'max_bytes',0)
                    +getattr(ple_provider,'max_bytes',0)>=available-available//2):
                raise MemoryError('complete Qwen4 native store ownership exceeds text admission')
            if native_store_runtime_library is not None:
                from .segment_edge_runtime import _Functions
                self._native_store_runtime=_Functions(native_store_runtime_library)
                if expert_provider is not None:
                    from .native_original_experts import NativeOriginalExperts
                    self._native_experts=NativeOriginalExperts(self._native_store_runtime,expert_provider,
                        max_bytes=native_expert_max_bytes)
                if ple_provider is not None:
                    from .native_original_ple import NativeOriginalPLE
                    self._native_ple=NativeOriginalPLE(self._native_store_runtime,ple_provider,max_bytes=native_ple_max_bytes)
            self.vision=Qwen4VisionEncoder(self.components.vision_config,vision,library=library,max_bytes=available//2)
            if text_device is None:
                if projection_provider is not None:
                    raise ValueError('device projection bank requires an explicit text device')
                self.text=NativeQwen4TextDecoder(self.components.text_config,text,library=text_library,
                                               max_bytes=available-available//2,max_snapshots=max_snapshots,
                                               ple_provider=ple_provider,expert_provider=expert_provider,
                                               native_composition=native_composition,native_expert_bank=self._native_experts,
                                               native_ple_bank=self._native_ple)
                if segment_runtime_library is not None:
                    from .native_qwen4_public_text import NativeQwen4PublicTextDecoder
                    self.text=NativeQwen4PublicTextDecoder(segment_runtime_library,self.text,
                        directory=segment_model_directory,max_bytes=available-available//2,
                        max_work_bytes=segment_work_bytes,max_snapshots=max_snapshots,
                        layer_ranges=segment_layer_ranges)
                    self.implementation='native-segment-edge-qwen4-conditional-fp32'
            else:
                if ple_provider is not None:
                    raise ValueError('CPU lexical provider cannot be used by a device decoder')
                from .torch_qwen4_decoder import TorchQwen4Decoder
                self.text=TorchQwen4Decoder(self.components.text_config,text,device=text_device,
                    max_bytes=available-available//2,max_snapshots=max_snapshots,expert_provider=expert_provider,
                    projection_provider=projection_provider)
                self.implementation='torch-qwen4-conditional-fp32';self.native_range_execution=False
            if self.text.mrope_section is None:
                raise ValueError('conditional Qwen4 requires an explicit MRoPE contract')
            if self.components.vision_config['out_hidden_size']!=self.components.text_config['hidden_size']:
                raise ValueError('Qwen4 vision projection and language embedding widths differ')
            digest=hashlib.sha256((self.components.identity+self.text.identity+self.vision.identity).encode())
            for name,array in sorted(self.auxiliary.items()):
                digest.update(name.encode());digest.update(memoryview(array).cast('B'))
            self.identity='openagent-qwen4-media-v1:'+digest.hexdigest()
            self.text.identity+=':wrapper:'+self.identity
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:
                primary.cleanup=self.close;primary.add_note('Qwen4 native component cleanup retained: '+repr(cleanup))
            raise

    @property
    def position(self):
        return self.text.position

    def _check(self):
        if self._closed or self._busy:
            raise RuntimeError('Qwen4 media owner is closed or executing')

    def _positions(self,ids,types,image_grids,video_grids):
        merge=self.components.vision_config['spatial_merge_size']
        queues={1:[],2:[]}
        for modality,grids in ((1,image_grids),(2,video_grids)):
            if grids is None: continue
            for temporal,height,width in np.asarray(grids):
                if modality==2:
                    queues[2].extend([(1,int(height),int(width))]*int(temporal))
                else: queues[1].append((int(temporal),int(height),int(width)))
        used={1:0,2:0};positions=np.empty((3,1,ids.shape[1]),np.int64)
        cursor=self.position+self.rope_delta;start=0
        while start<ids.shape[1]:
            kind=int(types[0,start]);end=start+1
            while end<ids.shape[1] and types[0,end]==kind: end+=1
            if kind==0:
                positions[:,0,start:end]=np.arange(cursor,cursor+end-start)[None]
                cursor+=end-start
            else:
                if used[kind]>=len(queues[kind]):
                    raise ValueError('media type span has no corresponding publisher grid')
                temporal,height,width=queues[kind][used[kind]];used[kind]+=1
                if height%merge or width%merge or end-start!=temporal*(height//merge)*(width//merge):
                    raise ValueError('media placeholder span differs from its merged grid')
                axes=np.meshgrid(np.arange(temporal),np.arange(height//merge),np.arange(width//merge),indexing='ij')
                positions[:,0,start:end]=np.stack(axes).reshape(3,-1)+cursor
                cursor+=max(height,width)//merge
            start=end
        if any(used[k]!=len(queues[k]) for k in used):
            raise ValueError('publisher grids have unused media items')
        delta=int(positions.max())+1-(self.position+ids.shape[1])
        return positions,delta

    @property
    def routing_observation_supported(self):
        return bool(getattr(self.text,'routing_observation_supported',False))

    def forward(self,token_ids,*,pixel_values=None,image_grid_thw=None,pixel_values_videos=None,
                video_grid_thw=None,mm_token_type_ids=None,cancelled=None,capture_routing=False):
        with self._lock:
            self._check();self._busy=True
            try:
                _cancel(cancelled);ids=np.asarray(token_ids)
                if type(capture_routing) is not bool:
                    raise ValueError('explicit media routing observation policy required')
                class Cancellation:
                    def is_set(self):
                        return cancelled is not None and cancelled()
                event=Cancellation()
                routing={'capture_routing':True} if capture_routing and self.routing_observation_supported else {}
                if ids.ndim==1: ids=ids[None]
                c=self.components.text_config
                if ids.ndim!=2 or ids.shape[0]!=1 or not ids.shape[1] or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=c['vocab_size']):
                    raise ValueError('Qwen4 media expects unpadded batch-one expanded processor token ids')
                image_id=self.config['image_token_id'];video_id=self.config['video_token_id']
                types=np.where(ids==image_id,1,np.where(ids==video_id,2,0))
                if mm_token_type_ids is not None and not np.array_equal(types,np.asarray(mm_token_type_ids)):
                    raise ValueError('modality types disagree with the actual placeholder tokens')
                if ((pixel_values is None)!=(image_grid_thw is None) or
                    (pixel_values_videos is None)!=(video_grid_thw is None)):
                    raise ValueError('processor pixel values and grids must be supplied together')
                embedding_bytes=ids.size*c['hidden_size']*4
                if embedding_bytes*3>min(self.text.max_bytes,self.vision.max_bytes):
                    raise MemoryError('media insertion staging exceeds complete component admission')
                if hasattr(self.text,'torch'):
                    embedding=self.text.weights['model.embed_tokens.weight']
                    with ExitStack() as scopes:
                        scopes.enter_context(self.text.torch.inference_mode())
                        scopes.enter_context(self.text.cancellation_scope(event))
                        provider=getattr(self.text,'projection_provider',None)
                        if provider is not None:
                            scopes.enter_context(provider.cancellation_scope(event))
                        device_ids=self.text.torch.tensor(ids.astype(np.int64),device=self.text.device)
                        embeds=embedding[device_ids].detach().cpu().numpy().copy()
                else:
                    if hasattr(self.text,'embed_tokens'):
                        embeds=self.text.embed_tokens(ids,cancelled=cancelled)
                    else:
                        embeds=np.array(self.text.weights['model.embed_tokens.weight'][ids],dtype=np.float32,order='C')
                for modality,pixels,grids in ((1,pixel_values,image_grid_thw),(2,pixel_values_videos,video_grid_thw)):
                    count=int(np.count_nonzero(types==modality))
                    if pixels is None:
                        if count: raise ValueError('media placeholder has no owned features')
                        continue
                    values=self.vision.forward(pixels,grids,cancelled=cancelled)
                    projected=np.concatenate(values,axis=0)
                    if projected.shape!=(count,c['hidden_size']):
                        raise ValueError('vision features and media placeholders differ')
                    embeds[types==modality]=projected
                positions,delta=self._positions(ids,types,image_grid_thw,video_grid_thw)
                if hasattr(self.text,'torch'):
                    with self.text.cancellation_scope(event):
                        result=self.text.forward(ids,inputs_embeds=embeds,position_ids=positions,**routing)
                else:
                    result=self.text.forward(ids,inputs_embeds=embeds,position_ids=positions,cancelled=cancelled,**routing)
                self.rope_delta=delta
                return result
            finally:
                self._busy=False

    def snapshot(self):
        with self._lock:
            self._check()
            if len(self._snapshots)>=self.max_snapshots:
                raise MemoryError('media snapshot inventory exhausted')
            inner=self.text.snapshot()
            try:
                receipt=Qwen4MediaSnapshot(self._owner,uuid4().hex,self.identity,self.position)
                self._snapshots[receipt.key]=(receipt,inner,self.rope_delta)
                return receipt
            except BaseException as primary:
                try:
                    self.text.release_snapshot(inner)
                except BaseException as cleanup:
                    self._orphaned_snapshots.append(inner)
                    primary.add_note('unpublished media snapshot cleanup retained for close: '+repr(cleanup))
                raise

    def _snapshot(self,receipt):
        if type(receipt) is not Qwen4MediaSnapshot or receipt.owner!=self._owner or receipt.identity!=self.identity:
            raise ValueError('foreign media snapshot')
        state=self._snapshots.get(receipt.key)
        if state is None or state[0] is not receipt:
            raise ValueError('released or forged media snapshot')
        return state

    def restore(self,receipt):
        with self._lock:
            self._check();_,inner,delta=self._snapshot(receipt)
            self.text.restore(inner);self.rope_delta=delta

    def release_snapshot(self,receipt):
        with self._lock:
            self._check();_,inner,_=self._snapshot(receipt)
            self.text.release_snapshot(inner);del self._snapshots[receipt.key]

    def reset(self):
        with self._lock:
            self._check();self.text.reset();self.rope_delta=0

    def write_cache(self,output,*,max_bytes):
        with self._lock:
            self._check()
            if type(max_bytes) is not int or not 44<max_bytes<=self.max_bytes:
                raise ValueError('explicit bounded media continuation budget required')
            self._busy=True
            try:
                header=json.dumps(dict(version=1,identity=self.identity,rope_delta=self.rope_delta),sort_keys=True,separators=(',',':')).encode()
                overhead=12+len(header)+32
                if max_bytes<=overhead or max_bytes>self.max_bytes:
                    raise MemoryError('explicit bounded media continuation extent required')
                # Prefix has its own digest; the nested text frame independently
                # authenticates every physical cache and positional bank.
                prefix=self.cache_magic+struct.pack('<I',len(header))+header
                _write(output,prefix);_write(output,hashlib.sha256(prefix).digest())
                return overhead+self.text.write_cache(output,max_bytes=max_bytes-overhead)
            finally:
                self._busy=False

    def restore_cache(self,source,*,max_bytes):
        with self._lock:
            self._check()
            if type(max_bytes) is not int or not 44<max_bytes<=self.max_bytes:
                raise ValueError('explicit bounded media continuation budget required')
            self._busy=True
            try:
                prefix=_read(source,12)
                if prefix[:8]!=self.cache_magic: raise ValueError('invalid media continuation magic')
                length=struct.unpack('<I',prefix[8:])[0]
                if not 0<length<=min(65536,max_bytes-44): raise MemoryError('media continuation header exceeds budget')
                raw=_read(source,length);digest=_read(source,32)
                if not hmac.compare_digest(bytes(digest),hashlib.sha256(prefix+raw).digest()):
                    raise ValueError('media continuation header checksum differs')
                header=json.loads(raw,object_pairs_hook=_pairs)
                if (type(header) is not dict or set(header)!={'version','identity','rope_delta'} or
                    type(header['version']) is not int or header['version']!=1 or header['identity']!=self.identity or
                    type(header['rope_delta']) is not int or abs(header['rope_delta'])>self.components.text_config['max_position_embeddings']):
                    raise ValueError('media continuation identity or rotary delta differs')
                position=self.text.restore_cache(source,max_bytes=max_bytes-44-length)
                self.rope_delta=header['rope_delta'];return position
            finally:
                self._busy=False

    def close(self):
        with self._lock:
            if self._closed: return
            if self._busy: raise RuntimeError('media execution owns resources')
            errors=[]
            for snapshot in tuple(self._orphaned_snapshots):
                try:
                    self.text.release_snapshot(snapshot)
                    self._orphaned_snapshots.remove(snapshot)
                except BaseException as error: errors.append(error)
            for owner in (self.text,self.vision):
                if owner is not None:
                    try:
                        owner.close()
                        if owner is self.text:self._orphaned_snapshots.clear()
                    except BaseException as error: errors.append(error)
            if errors: raise RuntimeError('media component cleanup remains pending') from errors[0]
            for name in ('_native_ple','_native_experts','_native_store_runtime'):
                owner=getattr(self,name)
                if owner is not None:owner.close();setattr(self,name,None)
            if self._own_ple_provider and self._ple_provider is not None:
                self._ple_provider.close();self._ple_provider=None
            if self._own_expert_provider and self._expert_provider is not None:
                self._expert_provider.close();self._expert_provider=None
            if self._own_projection_provider and self._projection_provider is not None:
                self._projection_provider.close();self._projection_provider=None
            self.auxiliary={};self._snapshots.clear();self._closed=True

__all__=['Qwen4MultimodalDecoder','Qwen4MediaSnapshot']
