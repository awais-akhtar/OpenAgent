"""Generation service contract for owned native conditional decoders."""
from contextlib import contextmanager
import inspect
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
        self._lifecycle=threading.RLock();self._active_thread=None
        self._closed=False;self._close_complete=False
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
        with self._lifecycle:
            if self._active_thread==threading.get_ident():
                raise RuntimeError('cannot close a conditional decoder from its active callback')
            if self._close_complete:
                return
            # Refuse new operations before invoking teardown. A failed native
            # close remains retryable, but must never leave the decoder open to
            # service requests while resource destruction is pending.
            self._closed=True
            closer=getattr(self.decoder,'close',None)
            if callable(closer):
                closer()
            self._close_complete=True

    def _owned(self,name,*args,**kwargs):
        with self._operation():
            operation=getattr(self.decoder,name,None)
            if not callable(operation):
                raise RuntimeError('conditional decoder does not implement '+name)
            return operation(*args,**kwargs)

    def snapshot(self):
        return self._owned('snapshot')

    def restore(self,receipt):
        return self._owned('restore',receipt)

    def release_snapshot(self,receipt):
        return self._owned('release_snapshot',receipt)

    def reset(self):
        return self._owned('reset')

    def _owned_cache(self,name,stream,*,max_bytes,cancelled):
        if cancelled is not None and not callable(cancelled):
            raise ValueError('conditional cache cancellation observer must be callable')
        with self._operation():
            operation=getattr(self.decoder,name,None)
            if not callable(operation):
                raise RuntimeError('conditional decoder does not implement '+name)
            kwargs={'max_bytes':max_bytes}
            if cancelled is not None:
                if 'cancelled' not in inspect.signature(operation).parameters:
                    raise ValueError('conditional decoder cache operation lacks cooperative cancellation: '+name)
                kwargs['cancelled']=cancelled
            return operation(stream,**kwargs)

    def write_cache(self,output,*,max_bytes,cancelled=None):
        return self._owned_cache('write_cache',output,max_bytes=max_bytes,cancelled=cancelled)

    def restore_cache(self,source,*,max_bytes,cancelled=None):
        return self._owned_cache('restore_cache',source,max_bytes=max_bytes,cancelled=cancelled)

    @contextmanager
    def _operation(self):
        # Hold ownership across staging, native execution and result decoding:
        # close from another thread waits, while callback reentry is rejected.
        with self._lifecycle:
            self._ensure_open()
            owner=threading.get_ident()
            if self._active_thread==owner:
                raise RuntimeError('conditional decoder operation reentered from its active callback')
            self._active_thread=owner
            try:
                yield
            finally:
                self._active_thread=None

    @contextmanager
    def cancellation_scope(self,cancel):
        self._ensure_open()
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise ValueError('conditional cancellation scope must provide is_set')
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
        with self._operation():
            cancel=getattr(self._local,'cancel',None)
            if cancel is not None and cancel.is_set():
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('conditional forward cancelled before execution')
            result=self.decoder.forward(token_ids,cancelled=None if cancel is None else cancel.is_set)
            if cancel is not None and cancel.is_set():
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('conditional forward cancelled before result publication')
            rows=self._rows(result)
            if cancel is not None and cancel.is_set():
                from .execution_service import ExecutionCancelled
                raise ExecutionCancelled('conditional forward cancelled during result validation')
            return rows

    def forward_media(self,token_ids,media,cancel):
        with self._operation():
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
            try:
                for name,value in media.items():
                    boundary()
                    detached=np.empty(value.shape,dtype=value.dtype,order='C')
                    destination=detached.reshape(-1)
                    offset=0
                    # Avoid one potentially very large ``tobytes`` allocation and
                    # make cancellation observable while detaching borrowed
                    # processor arrays.  Buffered external-loop iteration handles
                    # both contiguous and strided source views in C order.
                    chunks=np.nditer(value,flags=['external_loop','buffered'],
                        op_flags=['readonly'],order='C',buffersize=65536)
                    for chunk in chunks:
                        boundary()
                        if value.dtype.kind=='f' and not np.isfinite(chunk).all():
                            raise ValueError('nonfinite processor media values: '+name)
                        end=offset+chunk.size
                        destination[offset:end]=chunk
                        offset=end
                    boundary()
                    if offset!=value.size:
                        raise ValueError('processor media copy extent changed during staging: '+name)
                    owned[name]=detached
            except BaseException:
                # Drop already-staged buffers immediately rather than retaining a
                # partial media transaction through the exception traceback.
                owned.clear()
                destination=None
                chunks=None
                detached=None
                raise
            callback=None if cancel is None else cancel.is_set
            try:
                result=self.decoder.forward(token_ids,cancelled=callback,**owned)
                boundary()
            finally:
                # The decoder consumes these arrays synchronously. Release the
                # temporary ownership even when a family-specific tower/text path
                # raises and its traceback remains alive at the caller. The
                # loop locals also retain the final staging array and iterator,
                # so clear them along with the owning dictionary.
                owned.clear()
                destination=None
                chunks=None
                chunk=None
                detached=None
            rows=self._rows(result)
            boundary()
            return rows

__all__=['MultimodalSessionExecutor']
