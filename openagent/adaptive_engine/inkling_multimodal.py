"""Owned Inkling image/audio insertion, native text and physical continuation."""
import hashlib
import threading
import numpy as np
from .publisher_family_contracts import component_contract,partition_weights
from .native_inkling import NativeInklingDecoder
from .flat_native_session import FlatNativeSession
from .inkling_media import InklingMediaEncoder
from .qwen4_vision import _cancel

class InklingMultimodalDecoder:
    implementation='native-c-inkling-conditional-fp32'
    native_range_execution=True
    def __init__(self,config,tensors,*,library,text_library,max_bytes,context_tokens,
                 max_snapshots=4,arithmetic='fp32',mtp_max_bytes=0,mtp_work_bytes=0,
                 expert_provider=None,own_expert_provider=False,text_device=None,
                 projection_provider=None,own_projection_provider=False,
                 segment_runtime_library=None,segment_work_bytes=0,model_directory=None,segment_layer_span=0):
        self.components=component_contract(config,arithmetic=arithmetic)
        if self.components.variant.wrapper!='inkling_mm_model': raise ValueError('exact Inkling multimodal wrapper required')
        self.config=self.components.original_config
        if type(segment_layer_span) is not int or segment_layer_span<0 or (segment_layer_span and segment_runtime_library is None):
            raise ValueError('nonnegative native layer span requires the Segment runtime')
        if type(max_bytes) is not int or max_bytes<=0 or type(context_tokens) is not int or not 1<=context_tokens<=min(65536,self.components.text_config['max_position_embeddings']):
            raise ValueError('explicit bounded Inkling aggregate memory and resident context capacity required')
        self._lock=threading.RLock();self._busy=False;self._closed=False
        self.max_bytes=max_bytes;self.max_snapshots=max_snapshots;self.text=self.media=self.session=None
        self.mtp=None;self.continuation=None
        self._expert_provider=expert_provider;self._own_expert_provider=own_expert_provider
        self._projection_provider=projection_provider;self._own_projection_provider=own_projection_provider
        if segment_runtime_library is not None and (model_directory is None or text_device is not None or
                type(segment_work_bytes) is not int or segment_work_bytes<=0):
            raise ValueError('public Inkling execution requires a checkpoint root, CPU text execution and explicit work admission')
        if type(mtp_max_bytes) is not int or mtp_max_bytes<0 or type(mtp_work_bytes) is not int or mtp_work_bytes<0:
            raise ValueError('explicit nonnegative MTP admission required')
        if bool(mtp_max_bytes)!=bool(mtp_work_bytes) or (mtp_max_bytes and mtp_work_bytes>=mtp_max_bytes):
            raise ValueError('Inkling MTP component and smaller work budgets must be supplied together')
        text,vision,audio,auxiliary=partition_weights(tensors,self.components)
        from .inkling_mtp import admit_mtp_inventory
        headers={}
        for name,value in auxiliary.items():
            value=np.asarray(value)
            if value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError('finite converted Inkling auxiliary weights required: '+name)
            headers[name]=(value.shape,'F32')
        admit_mtp_inventory(self.components.text_config,self.components.auxiliary_config,headers)
        auxiliary_bytes=sum(np.asarray(value).nbytes for value in auxiliary.values())
        if 2*auxiliary_bytes>=max_bytes: raise MemoryError('Inkling auxiliary staging exceeds admission')
        self.auxiliary={name:np.frombuffer(np.asarray(value).tobytes(),dtype=np.asarray(value).dtype).reshape(np.asarray(value).shape)
                        for name,value in auxiliary.items()}
        available=max_bytes-auxiliary_bytes-mtp_max_bytes
        if available<=0: raise MemoryError('Inkling MTP and auxiliary reservations exhaust aggregate admission')
        text_config=dict(self.components.text_config);text_config['max_position_embeddings']=context_tokens
        try:
            self.media=InklingMediaEncoder(self.components.vision_config,self.components.audio_config,
                dict(vision,**audio),library=library,text_hidden=text_config['hidden_size'],max_bytes=available//2)
            if text_device is None:
                if projection_provider is not None: raise ValueError('device projection bank requires device text execution')
                self.text=NativeInklingDecoder(text_config,text,library=text_library,max_bytes=available-available//2,
                    expert_provider=expert_provider)
                if segment_runtime_library is None:
                    self.session=FlatNativeSession(self.text,family='inkling_text',max_bytes=available-available//2,max_snapshots=max_snapshots)
            else:
                from .torch_inkling_decoder import TorchInklingDecoder
                self.text=TorchInklingDecoder(text_config,text,device=text_device,
                    max_bytes=available-available//2,max_snapshots=max_snapshots,
                    expert_provider=expert_provider,projection_provider=projection_provider)
                self.session=self.text
                self.implementation='torch-inkling-conditional-fp32';self.native_range_execution=False
            if mtp_max_bytes:
                from .inkling_mtp import InklingMTP
                c=self.config['text_config'];depth=c.get('num_mtp_layers')
                if type(depth) is not int or depth<=0 or not 0<mtp_work_bytes<mtp_max_bytes:
                    raise ValueError('publisher MTP depth and explicit work/stage budgets required')
                self.mtp=InklingMTP(self.text,self.auxiliary,c,text_library=text_library,media_library=library,
                    stage_max_bytes=(mtp_max_bytes-mtp_work_bytes)//depth,max_total_bytes=mtp_max_bytes,
                    max_work_bytes=mtp_work_bytes,max_snapshots=max_snapshots)
            if segment_runtime_library is not None:
                from .native_family_public_text import NativeFamilyPublicTextDecoder
                # MTP has captured its shared immutable globals before native
                # registration transfers and closes the original text owner.
                self.text=NativeFamilyPublicTextDecoder(segment_runtime_library,self.text,directory=model_directory,
                    max_bytes=available-available//2,max_work_bytes=segment_work_bytes,max_snapshots=max_snapshots,
                    layer_ranges=None if not segment_layer_span else tuple((begin,min(begin+segment_layer_span,
                        self.config['text_config']['num_hidden_layers'])) for begin in
                        range(0,self.config['text_config']['num_hidden_layers'],segment_layer_span)))
                self.session=self.text;self.implementation='native-segment-edge-inkling-conditional-fp32'
            digest=hashlib.sha256((self.components.identity+self.media.identity+self.session.identity).encode())
            for name,array in sorted(self.auxiliary.items()):
                digest.update(name.encode());digest.update(memoryview(array).cast('B'))
            self.identity='openagent-inkling-conditional-v1:'+digest.hexdigest()
            self.session.identity=self.identity
            if self.mtp is not None:
                from .grouped_continuation import GroupedContinuation
                self.identity='openagent-inkling-conditional-mtp-v1:'+hashlib.sha256((self.identity+self.mtp.identity).encode()).hexdigest()
                self.continuation=GroupedContinuation((self.session,self.mtp),identity=self.identity,max_snapshots=max_snapshots)
            else: self.continuation=self.session
        except BaseException as primary:
            try: self.close()
            except BaseException as failure:
                # Preserve the construction error while exporting a retryable
                # handle for any owner whose interrupted close did not finish.
                primary.cleanup=self.close
                primary.add_note('Inkling conditional construction cleanup remains pending: '+repr(failure))
            raise

    @property
    def position(self): return self.text.position

    def _check(self):
        if self._closed or self._busy: raise RuntimeError('Inkling conditional owner is closed or executing')

    def forward(self,token_ids,*,pixel_values=None,audio_input_ids=None,audio_input_ids_mask=None,cancelled=None):
        with self._lock:
            self._check();self._busy=True
            try:
                _cancel(cancelled);ids=np.asarray(token_ids)
                if ids.ndim==1: ids=ids[None]
                c=self.text.config
                if ids.ndim!=2 or ids.shape[0]!=1 or not ids.shape[1] or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=c['vocab_size']):
                    raise ValueError('expanded unpadded batch-one Inkling processor ids required')
                width=c['hidden_size'];staging=ids.size*width*16
                if staging>min(self.media.max_bytes,self.text.max_bytes):
                    raise MemoryError('Inkling media insertion staging exceeds component admission')
                if hasattr(self.text,'embed_tokens'):
                    embeds=self.text.embed_tokens(ids,cancelled=cancelled)
                elif hasattr(self.text,'torch'):
                    with self.text.torch.inference_mode():
                        lookup=self.text.weights['model.embed_tokens.weight'][self.text.torch.tensor(ids.astype(np.int64),device=self.text.device)]
                        normalized=lookup*self.text.torch.rsqrt(lookup.square().mean(dim=-1,keepdim=True)+c['rms_norm_eps'])*self.text.weights['model.embed_norm.weight']
                        embeds=normalized.detach().cpu().numpy().copy()
                else:
                    embeds=self.media.ops.norm(np.ascontiguousarray(self.text.weights['model.embed_tokens.weight'][ids]),
                        self.text.weights['model.embed_norm.weight'],rms=True,epsilon=c['rms_norm_eps'])
                image_id=self.config.get('image_token_id',200054);audio_id=self.config.get('audio_token_id',200053)
                if image_id==audio_id: raise ValueError('media placeholder token identities overlap')
                for token_id,value,kind in ((image_id,pixel_values,'image'),(audio_id,audio_input_ids,'audio')):
                    selected=ids==token_id;count=int(selected.sum())
                    if value is None:
                        if count: raise ValueError('Inkling placeholder has no '+kind+' features')
                        continue
                    features=(self.media.vision(value,cancelled=cancelled) if kind=='image' else
                        self.media.audio(value,mask=audio_input_ids_mask,cancelled=cancelled))
                    if features.shape!=(count,width): raise ValueError('Inkling '+kind+' features and placeholders differ')
                    embeds[selected]=features
                _cancel(cancelled)
                if hasattr(self.text,'torch'):
                    with self.text.cancellation_scope(cancelled):
                        return self.text.forward(ids,inputs_embeds=embeds)
                return self.text.forward(ids,inputs_embeds=embeds,cancelled=cancelled)
            finally: self._busy=False

    def snapshot(self):
        with self._lock: self._check();return self.continuation.snapshot()
    def restore(self,receipt):
        with self._lock: self._check();return self.continuation.restore(receipt)
    def release_snapshot(self,receipt):
        with self._lock: self._check();return self.continuation.release_snapshot(receipt)
    def reset(self):
        with self._lock: self._check();return self.continuation.reset()
    def write_cache(self,output,*,max_bytes):
        with self._lock: self._check();return self.continuation.write_cache(output,max_bytes=max_bytes)
    def restore_cache(self,source,*,max_bytes):
        with self._lock: self._check();return self.continuation.restore_cache(source,max_bytes=max_bytes)

    def draft(self,shifted_ids,previous_hidden,*,cancelled=None):
        with self._lock:
            self._check()
            if self.mtp is None: raise RuntimeError('Inkling MTP stages require explicit aggregate admission')
            ids=np.asarray(shifted_ids)
            if ids.ndim!=2 or self.mtp.position+ids.shape[-1]!=self.position:
                raise ValueError('MTP must consume exactly the unprocessed committed backbone hidden prefix')
            self._busy=True
            try: return self.mtp.draft(shifted_ids,previous_hidden,cancelled=cancelled)
            finally: self._busy=False
    def close(self):
        with self._lock:
            if self._closed: return
            if self._busy: raise RuntimeError('Inkling inference owns native resources')
            errors=[]
            if self.mtp is not None:
                try:
                    self.mtp.close();self.mtp=None
                except BaseException as error: errors.append(error)
            # These owners are independent: a failed MTP/text close must not
            # strand the media tower (or vice versa). Keep failed references so
            # callers can retry cleanup after the underlying operation settles.
            session=self.session;text_owner_failed=self.mtp is not None
            if self.mtp is None:
                text_owner=session if session is not None else self.text
                if text_owner is not None:
                    try:
                        text_owner.close()
                        if session is not None: self.session=None
                        else: self.text=None
                    except BaseException as error:
                        errors.append(error);text_owner_failed=True
            if self.media is not None:
                try:
                    self.media.close();self.media=None
                except BaseException as error: errors.append(error)
            # Providers are borrowed by text/MTP execution. Do not retire them
            # until every dependent owner has released its callback references.
            if not text_owner_failed:
                if self._own_expert_provider and self._expert_provider is not None:
                    try:
                        self._expert_provider.close();self._expert_provider=None
                    except BaseException as error: errors.append(error)
                if self._own_projection_provider and self._projection_provider is not None:
                    try:
                        self._projection_provider.close();self._projection_provider=None
                    except BaseException as error: errors.append(error)
            if errors: raise RuntimeError('Inkling cleanup remains pending') from errors[0]
            self.auxiliary={};self._closed=True

__all__=['InklingMultimodalDecoder']
