"""Generation service contract for owned native conditional decoders."""
from contextlib import contextmanager
import threading
import numpy as np

class MultimodalSessionExecutor:
    def __init__(self,decoder,*,max_media_bytes):
        from .qwen4_multimodal import Qwen4MultimodalDecoder
        from .qwen35_multimodal import Qwen35MultimodalDecoder
        from .inkling_multimodal import InklingMultimodalDecoder
        from .glm5_multimodal import Glm5MultimodalDecoder
        if type(decoder) not in (Qwen35MultimodalDecoder,Qwen4MultimodalDecoder,InklingMultimodalDecoder,Glm5MultimodalDecoder):
            raise ValueError('explicit native conditional decoder required')
        if type(max_media_bytes) is not int or max_media_bytes<=0:
            raise ValueError('explicit processor media staging budget required')
        self.decoder=decoder;self.max_media_bytes=max_media_bytes;self._local=threading.local()
        self._closed=False
        self._fields=(frozenset(('pixel_values','image_grid_thw','pixel_values_videos',
            'video_grid_thw','mm_token_type_ids')) if type(decoder) in (Qwen35MultimodalDecoder,Qwen4MultimodalDecoder) else
            frozenset(('pixel_values','audio_input_ids','audio_input_ids_mask')))
        if type(decoder) is Glm5MultimodalDecoder:
            self._fields=frozenset(('pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw'))

    def __getattr__(self,name): return getattr(self.decoder,name)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('multimodal session executor is closed')

    def close(self):
        if self._closed:
            return
        closer=getattr(self.decoder,'close',None)
        if callable(closer):
            closer()
        self._closed=True

    @contextmanager
    def cancellation_scope(self,cancel):
        self._ensure_open()
        old=getattr(self._local,'cancel',None);self._local.cancel=cancel
        try: yield self
        finally: self._local.cancel=old

    @staticmethod
    def _rows(value):
        logits=value['logits'] if isinstance(value,dict) else getattr(value,'logits',value)
        if not isinstance(logits,np.ndarray) or logits.ndim!=3 or logits.shape[0]!=1:
            raise ValueError('conditional session requires batch-one native logits')
        return logits[0]

    def forward(self,token_ids):
        self._ensure_open()
        cancel=getattr(self._local,'cancel',None)
        return self._rows(self.decoder.forward(token_ids,cancelled=None if cancel is None else cancel.is_set))

    def forward_media(self,token_ids,media,cancel):
        self._ensure_open()
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise ValueError('media forward cancellation must provide is_set')
        if type(media) is not dict or not media or set(media)-self._fields:
            raise ValueError('media fields differ from this conditional family contract')
        def boundary():
            if cancel is not None and cancel.is_set():
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('conditional media staging cancelled')
        # Accept arrays only, preventing an implicit unbounded sequence conversion.
        used=0
        for name,value in media.items():
            boundary()
            if not isinstance(value,np.ndarray) or value.dtype.kind not in 'biuf' or not value.size:
                raise ValueError('bounded numeric processor array required: '+name)
            used+=value.nbytes
            if 2*used>self.max_media_bytes:
                raise MemoryError('processor inputs and immutable staging exceed media admission')
        owned={}
        for name,value in media.items():
            boundary()
            owned[name]=np.frombuffer(value.tobytes(),value.dtype).reshape(value.shape)
            # `tobytes` is a single non-interruptible copy. Observe cancellation
            # immediately afterwards and never dispatch a partially staged set.
            boundary()
        callback=None if cancel is None else cancel.is_set
        return self._rows(self.decoder.forward(token_ids,cancelled=callback,**owned))

__all__=['MultimodalSessionExecutor']
