"""Owned original GLM publisher text, vision insertion and physical continuation."""
import hashlib
import json
from pathlib import Path
import threading
import numpy as np
from .publisher_family_contracts import component_contract
from .publisher_glm5_loader import load_publisher_glm5_text_decoder,admit_publisher_glm5_inventory
from .publisher_tensor_reader import PublisherTensorReader,TensorSlice
from .publisher_weight_mapping import WeightBinding,materialize
from .publisher_checkpoint.storage import snapshot
from .glm5_vision import Glm5VisionEncoder
from .flat_native_session import FlatNativeSession
from .qwen4_vision import _cancel
from .publisher_media_files import read_regular

class Glm5MultimodalDecoder:
    implementation='native-c-glm5-next-conditional-fp32'
    native_range_execution=True
    def __init__(self,directory,config,*,library,text_library,max_bytes,max_weight_bytes,
                 max_conversion_bytes,max_expert_bytes,context_tokens,max_snapshots=4,text_device=None,max_projection_bytes=0,
                 segment_runtime_library=None,segment_work_bytes=0,segment_layer_span=0):
        self.components=component_contract(config,arithmetic='fp32');self.config=self.components.original_config
        if type(segment_layer_span) is not int or segment_layer_span<0 or (segment_layer_span and segment_runtime_library is None):
            raise ValueError('nonnegative native GLM layer span requires the Segment runtime')
        if self.components.variant.wrapper!='glm5_next': raise ValueError('original GLM publisher wrapper required')
        if type(max_bytes) is not int or max_bytes<=0: raise ValueError('positive aggregate GLM conditional admission required')
        self._lock=threading.RLock();self._busy=False;self._closed=False
        self.max_bytes=max_bytes;self.owner=self.text=self.vision=self.session=None;self.auxiliary={}
        if segment_runtime_library is not None and (text_device is not None or
                type(segment_work_bytes) is not int or segment_work_bytes<=0):
            raise ValueError('native public GLM requires CPU text execution and explicit native work admission')
        inventory=snapshot(directory,max_metadata_bytes=min(max_weight_bytes,256*1024**2),max_tensors=1000000,max_shards=4096)
        from .native_checkpoint import _pairs
        raw=read_regular(Path(directory)/'config.json',max_bytes=1024**2)
        if (len(raw)>1024**2 or hashlib.sha256(raw).hexdigest()!=inventory.config_sha256 or
                json.loads(raw,object_pairs_hook=_pairs)!=config):
            raise ValueError('GLM conditional checkpoint configuration changed after admission')
        quant=config.get('quantization_config');blocks=None if quant is None else quant.get('weight_block_size')
        reader=PublisherTensorReader(inventory,max_work_bytes=max_conversion_bytes,block_shape=blocks)
        reader.admit_fp8_inventory()
        try:
            backend=('public-range' if segment_runtime_library is not None else
                     'device' if text_device is not None else 'native')
            admit_publisher_glm5_inventory(inventory,self.components,backend=backend)
            bindings=[]
            for name,record in inventory.tensors.items():
                if not name.startswith('model.visual.'): continue
                if name.endswith(('.weight_scale_inv','_scale_inv')): continue
                bindings.append(WeightBinding(name,(TensorSlice(name,record.shape),),record.shape))
            media=materialize(reader,bindings,max_bytes=min(max_weight_bytes,max_bytes//2))
            # GLM5 MTP is explicitly rejected by inventory admission until a
            # corresponding draft consumer exists; this owner therefore only
            # retains the visual component here.
            auxiliary={}
            auxiliary_bytes=0
            self.auxiliary=auxiliary;available=max_bytes-auxiliary_bytes
            self.vision=Glm5VisionEncoder(self.components.vision_config,media,library=library,max_bytes=available//2)
            del media
            if self.vision.config['out_hidden_size']!=self.components.text_config['hidden_size']:
                raise ValueError('GLM vision output and text embedding widths differ')
            self.owner=load_publisher_glm5_text_decoder(directory,library=text_library,expert_library=library,
                max_weight_bytes=max_weight_bytes,max_bytes=available-available//2,
                max_expert_bytes=max_expert_bytes,max_conversion_bytes=max_conversion_bytes,context_tokens=context_tokens,
                text_device=text_device,max_snapshots=max_snapshots,max_projection_bytes=max_projection_bytes)
            self.text=self.owner.decoder
            if segment_runtime_library is not None:
                from .native_family_public_text import NativeFamilyPublicTextDecoder
                public=NativeFamilyPublicTextDecoder(segment_runtime_library,self.text,directory=directory,
                    max_bytes=available-available//2,max_work_bytes=segment_work_bytes,
                    max_snapshots=max_snapshots,
                    layer_ranges=None if not segment_layer_span else tuple((begin,min(begin+segment_layer_span,
                        self.components.text_config['num_hidden_layers'])) for begin in
                        range(0,self.components.text_config['num_hidden_layers'],segment_layer_span)))
                self.owner.decoder=public;self.text=public;self.session=public
                self.implementation='native-segment-edge-glm5-conditional-fp32'
            elif text_device is None:
                self.session=FlatNativeSession(self.text,family='glm5_next_text',max_bytes=available-available//2,max_snapshots=max_snapshots)
            else:
                self.session=self.text;self.implementation='torch-glm5-next-conditional-fp32';self.native_range_execution=False
            digest=hashlib.sha256((self.components.identity+self.session.identity+self.vision.identity).encode())
            for name,value in sorted(auxiliary.items()): digest.update(name.encode());digest.update(memoryview(value).cast('B'))
            self.identity='openagent-glm5-conditional-v1:'+digest.hexdigest()
            if segment_runtime_library is None:self.session.identity=self.identity
            inventory.assert_unchanged()
        except BaseException as error:
            try: self.close()
            except BaseException as failure:
                from .failure_reporting import retain_cleanup_failure
                retain_cleanup_failure(error,self.close,failure,
                    'GLM conditional cleanup remains pending: ')
            raise
        finally: reader.close()

    @property
    def position(self): return self.text.position

    def _check(self):
        if self._closed or self._busy: raise RuntimeError('GLM conditional owner is closed or executing')

    def forward(self,token_ids,*,pixel_values=None,image_grid_thw=None,pixel_values_videos=None,
                video_grid_thw=None,cancelled=None):
        with self._lock:
            self._check();ids=np.asarray(token_ids)
            if ids.ndim==1: ids=ids[None]
            c=getattr(self.text,'contract',self.text.config)
            if ids.ndim!=2 or ids.shape[0]!=1 or not ids.shape[1] or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=c.vocab_size):
                raise ValueError('expanded unpadded batch-one GLM processor ids required')
            capacity=getattr(self.text,'max_tokens',None)
            position=getattr(self.text,'position',None)
            if (type(capacity) is not int or type(position) is not int or position<0 or
                    ids.shape[1]>capacity-position):
                # Reject an impossible continuation before embedding lookup or
                # running the potentially expensive vision tower. The text
                # decoder repeats this check at its own mutation boundary.
                raise ValueError('GLM multimodal prompt exceeds remaining text context')
            image_id=self.config.get('image_token_id',154854)
            start_id=self.config.get('video_start_token_id',154832)
            end_id=self.config.get('video_end_token_id',154833)
            if (any(type(value) is not int or not 0<=value<c.vocab_size
                    for value in (image_id,start_id,end_id)) or
                    len({image_id,start_id,end_id})!=3):
                raise ValueError('GLM image placeholder and video span delimiters require distinct in-vocabulary token IDs')
            if ids.size*c.hidden_size*16>min(self.text.max_bytes,self.vision.max_bytes):
                raise MemoryError('GLM embedding and media insertion exceeds staging admission')
            self._busy=True
            try:
                _cancel(cancelled)
                if hasattr(self.text,'embed_tokens'):
                    embeds=self.text.embed_tokens(ids,cancelled=cancelled)
                elif hasattr(self.text,'torch'):
                    with self.text.torch.inference_mode():
                        embeds=self.text.weights['embed_tokens.weight'][self.text.torch.tensor(ids.astype(np.int64),device=self.text.device)].detach().cpu().numpy().copy()
                else:
                    embeds=np.array(self.text.weights['embed_tokens.weight'][ids],order='C',copy=True)
                spans=np.zeros(ids.shape,bool);inside=False
                for index,token in enumerate(ids[0]):
                    if index%1024==0: _cancel(cancelled)
                    if token==start_id:
                        if inside: raise ValueError('nested GLM video spans are not admitted')
                        inside=True
                    elif token==end_id:
                        if not inside: raise ValueError('GLM video end has no opening span')
                        inside=False
                    spans[0,index]=inside
                if inside: raise ValueError('GLM media prefills require complete video spans')
                placeholders=ids==image_id
                for video,pixels,grids in ((False,pixel_values,image_grid_thw),(True,pixel_values_videos,video_grid_thw)):
                    selected=placeholders&(spans if video else ~spans);count=int(selected.sum())
                    if pixels is None:
                        if grids is not None or count: raise ValueError('GLM media placeholders require matching processor patches')
                        continue
                    if grids is None: raise ValueError('GLM processor media requires explicit grids')
                    if not count:
                        # Do not materialize an otherwise orphaned image/video
                        # tower result only to reject its feature count below.
                        raise ValueError('GLM processor media has no matching expanded-token placeholders')
                    if video:
                        grid=np.asarray(grids)
                        if grid.ndim!=2 or grid.shape[1]!=3 or grid.dtype.kind not in 'iu' or np.any(grid<=0) or np.any(grid>65536):
                            raise ValueError('bounded GLM selected video grids required')
                        frames=sum(int(row[0]) for row in grid)
                        if frames>65536: raise MemoryError('GLM video frame grid expansion exceeds admission')
                        expanded=np.empty((frames,3),np.int64);offset=0
                        for grid_index,(temporal,height,width) in enumerate(grid):
                            if grid_index%1024==0: _cancel(cancelled)
                            end=offset+int(temporal);expanded[offset:end]=(1,int(height),int(width));offset=end
                        grids=expanded
                    features=np.concatenate(self.vision.forward(pixels,grids,cancelled=cancelled),axis=0)
                    if features.shape!=(count,c.hidden_size): raise ValueError('GLM projected media and placeholder counts differ')
                    embeds[selected]=features
                _cancel(cancelled)
                if hasattr(self.text,'torch'):
                    with self.text.cancellation_scope(cancelled):
                        return self.text.forward(ids,inputs_embeds=embeds)
                return self.text.forward(ids,inputs_embeds=embeds,cancelled=cancelled)
            finally: self._busy=False

    def snapshot(self):
        with self._lock: self._check();return self.session.snapshot()
    def restore(self,receipt):
        with self._lock: self._check();return self.session.restore(receipt)
    def release_snapshot(self,receipt):
        with self._lock: self._check();return self.session.release_snapshot(receipt)
    def reset(self):
        with self._lock: self._check();return self.session.reset()
    def write_cache(self,output,*,max_bytes):
        with self._lock: self._check();return self.session.write_cache(output,max_bytes=max_bytes)
    def restore_cache(self,source,*,max_bytes):
        with self._lock: self._check();return self.session.restore_cache(source,max_bytes=max_bytes)
    def close(self):
        with self._lock:
            if self._busy: raise RuntimeError('GLM media inference owns resources')
            if self._closed: return
            errors=[]
            # The continuation session owns snapshots around the decoder also
            # referenced by the text owner. Retire it first, then let the text
            # owner close its decoder/provider resources; still attempt every
            # independent owner when any close fails.
            if self.session is not None:
                try: self.session.close();self.session=None
                except BaseException as error: errors.append(error)
            if self.owner is not None:
                try: self.owner.close();self.owner=None
                except BaseException as error: errors.append(error)
            if self.vision is not None:
                try: self.vision.close();self.vision=None
                except BaseException as error: errors.append(error)
            if errors:
                failure=RuntimeError('GLM conditional cleanup remains pending')
                from .failure_reporting import add_failure_note, failure_summary
                for error in errors:
                    add_failure_note(failure,failure_summary(error))
                raise failure from errors[0]
            self.auxiliary.clear();self._closed=True

__all__=['Glm5MultimodalDecoder']
