"""Bounded local publisher preprocessing for native conditional execution.

Only preprocessing uses the admitted Transformers package. Decoder execution
remains in OpenAgent's native composition. Remote code and downloads are off.
"""
import copy
from io import BytesIO
from pathlib import Path
import hashlib
import json
import threading
import os
import sys
import numpy as np
from types import SimpleNamespace
from .failure_reporting import add_failure_note, failure_summary
from .session_recovery import _path
from .native_library_identity import _stamp
from .publisher_media_inputs import normalize_video_fps

class PublisherTokenizerAdapter:
    """Expose the gateway's Encoding interface over the owned publisher tokenizer."""
    def __init__(self,tokenizer): self.tokenizer=tokenizer
    def encode(self,text,*,add_special_tokens=True):
        ids=self.tokenizer.encode(text,add_special_tokens=add_special_tokens)
        return SimpleNamespace(ids=list(ids))
    def decode(self,ids,*,skip_special_tokens=True):
        return self.tokenizer.decode(list(ids),skip_special_tokens=skip_special_tokens)

class PublisherMediaProcessor:
    def __init__(self,directory,*,family,max_bytes,max_pixels,max_audio_samples,
                 source_version='5.16.1'):
        if family not in ('qwen3_5_moe','qwen4_exp','inkling_mm_model','glm5_next'):
            raise ValueError('explicit conditional publisher family required')
        if any(type(v) is not int or v<=0 for v in (max_bytes,max_pixels,max_audio_samples)):
            raise ValueError('positive processor byte, pixel and audio admissions required')
        import transformers
        if transformers.__version__!=source_version:
            raise ValueError('publisher processor package version differs from pinned source')
        root=_path(directory).resolve(strict=True)
        if not root.is_dir(): raise ValueError('local publisher processor directory required')
        self.root,self.family=root,family;self.max_bytes=max_bytes
        self.max_pixels,self.max_audio_samples=max_pixels,max_audio_samples
        self._lock=threading.RLock();self._closed=False
        digest=hashlib.sha256((family+source_version).encode());self._files={};total=0
        self._artifact_names=('config.json','processor_config.json','preprocessor_config.json',
            'video_preprocessor_config.json','tokenizer_config.json','special_tokens_map.json',
            'chat_template.jinja','tokenizer.json','vocab.json','merges.txt','tokenizer.model')
        for name in self._artifact_names:
            path=_path(root/name)
            if not path.exists(): continue
            if path.is_symlink() or not path.is_file(): raise ValueError('regular local processor artifact required')
            stamp=path.stat();total+=stamp.st_size
            if 2*total>max_bytes: raise MemoryError('processor artifact loading exceeds admission')
            with path.open('rb') as stream:
                if _stamp(os.fstat(stream.fileno()))!=_stamp(stamp):
                    raise ValueError('processor artifact changed during opening')
                count=0
                for chunk in iter(lambda:stream.read(min(1024**2,max_bytes)),b''):
                    count+=len(chunk)
                    if count>stamp.st_size:raise ValueError('processor artifact grew during admission')
                    digest.update(chunk)
                if count!=stamp.st_size or _stamp(os.fstat(stream.fileno()))!=_stamp(stamp):
                    raise ValueError('processor artifact changed during reading')
            after=path.stat()
            before=_stamp(stamp)
            if before!=_stamp(after): raise ValueError('processor artifact changed during admission')
            self._files[path]=before;digest.update(name.encode())
        self.identity='openagent-local-media-processor-v1:'+digest.hexdigest()
        processor=transformers.AutoProcessor.from_pretrained(str(root),local_files_only=True,
            trust_remote_code=False)
        self.processor=processor
        try:
            self.tokenizer=PublisherTokenizerAdapter(processor.tokenizer)
            self._unchanged()
        except BaseException as primary:
            # Processor construction owns tokenizer/image/audio subcomponents.
            # If post-load artifact validation fails, do not leave that owner
            # alive on a partially constructed PublisherMediaProcessor.
            close=getattr(processor,'close',None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    def retry_processor_cleanup():
                        close()
                    primary.cleanup=retry_processor_cleanup
                    if hasattr(primary,'add_note'):
                        add_failure_note(primary, 'publisher processor construction cleanup failed: '+failure_summary(cleanup))
            self.processor=self.tokenizer=None
            raise

    def _unchanged(self):
        for name in self._artifact_names:
            path=_path(self.root/name)
            if path.exists()!=(path in self._files):
                raise ValueError('publisher processor artifact inventory changed after admission')
        for path,stamp in self._files.items():
            value=_path(path).stat()
            if stamp!=_stamp(value):
                raise ValueError('publisher processor artifact changed after admission')

    def _image(self,value):
        from PIL import Image
        if isinstance(value,(np.ndarray,Image.Image)):
            if isinstance(value,np.ndarray):
                if value.dtype!=np.uint8 or value.ndim!=3 or value.shape[2]!=3:
                    raise ValueError('decoded publisher images must be uint8 RGB HWC arrays')
                height,width=value.shape[:2]
            else:
                width,height=value.size
                if getattr(value,'n_frames',1)!=1:
                    raise ValueError('decoded still image required')
            if width<=0 or height<=0 or width*height>self.max_pixels or width*height*32>self.max_bytes:
                raise MemoryError('decoded image pixels exceed processor admission')
            if isinstance(value,np.ndarray):
                # The PIL wrapper may borrow the caller's ndarray storage.
                # Close that temporary owner deterministically after making
                # the detached RGB image, including when conversion raises.
                with Image.fromarray(value) as image:
                    return image.convert('RGB')
            return value.convert('RGB')
        if type(value) is not bytes or not value or len(value)>self.max_bytes//8:
            raise MemoryError('encoded image exceeds processor staging admission')
        with Image.open(BytesIO(value)) as image:
            width,height=image.size
            if width<=0 or height<=0 or width*height>self.max_pixels or width*height*32>self.max_bytes:
                raise MemoryError('decoded image pixels exceed processor admission')
            if getattr(image,'n_frames',1)!=1:
                raise ValueError('animated images require explicit selected video frames')
            image.load();return image.convert('RGB')

    def prepare(self,text,*,images=(),audio=(),sampling_rate=None,videos=(),video_metadata=None,cancelled=None):
        with self._lock:
            if self._closed: raise RuntimeError('publisher media processor is closed')
            if type(text) is not str or not text or len(text.encode())>min(2*1024**2,self.max_bytes//8):
                raise ValueError('bounded rendered processor text required')
            if any(type(v) not in (list,tuple) or len(v)>64 for v in (images,audio,videos)):
                raise ValueError('bounded explicit media item sequences required')
            # Freeze request ordering before per-item decoding; individual
            # decoders below create their own owned pixel/sample buffers.
            images = tuple(images)
            audio = tuple(audio)
            videos = tuple(videos)
            # The upstream processor API receives each modality in a separate
            # argument.  It does not provide a source-order mapping from those
            # arrays back to interleaved placeholders in ``text``.  Accepting
            # a mixed request here could therefore bind correctly-shaped
            # vision/audio tensors to the wrong placeholder order.  Keep
            # within-modality ordering (which is explicit in each tuple), but
            # reject cross-modality requests until a processor-specific
            # ordering contract is available.
            if sum(bool(items) for items in (images, audio, videos)) > 1:
                raise ValueError('publisher processor requires one media modality per request until cross-modality placeholder ordering is defined')
            if self.family=='qwen3_5_moe' and audio:
                raise ValueError('Qwen3.5 publisher processor rejects audio input: no audio encoder or audio-to-text placement path is implemented')
            def boundary():
                if cancelled is not None:
                    if not callable(cancelled): raise TypeError('callable processor cancellation required')
                    if cancelled(): raise InterruptedError('publisher media preprocessing cancelled')
            boundary();self._unchanged();owned=[];owned_waveforms=[];kwargs={};result={};work=0
            try:
                kwargs.update(text=text,return_tensors='np',padding=False)
                if images:
                    decoded=[]
                    for value in images:
                        boundary();image=self._image(value);owned.append(image)
                        work+=image.width*image.height*32
                        if work>self.max_bytes: raise MemoryError('aggregate decoded image staging exceeds admission')
                        decoded.append(image)
                    kwargs['images']=decoded
                if audio:
                    if self.family!='inkling_mm_model' or type(sampling_rate) is not int or sampling_rate<=0:
                        raise ValueError('Inkling waveform input needs its explicit sampling rate')
                    waveforms=[]
                    for value in audio:
                        boundary()
                        if not isinstance(value,np.ndarray) or value.ndim!=1 or value.dtype.kind!='f' or not 0<value.size<=self.max_audio_samples:
                            raise ValueError('bounded mono floating waveform required')
                        work+=value.size*64
                        if work>self.max_bytes: raise MemoryError('aggregate audio preprocessing staging exceeds admission')
                        # Large waveform conversions used to be one opaque
                        # copy/finite scan.  Chunk both operations so stream
                        # cancellation is observed during staging, before the
                        # processor can retain or transform the complete input.
                        owned_waveform=np.empty(value.size,dtype=np.float32)
                        for start in range(0,value.size,262144):
                            boundary()
                            end=min(value.size,start+262144)
                            chunk=np.asarray(value[start:end],dtype=np.float32)
                            if not np.isfinite(chunk).all(): raise ValueError('nonfinite waveform sample')
                            owned_waveform[start:end]=chunk
                        owned_waveforms.append(owned_waveform)
                        waveforms.append(owned_waveform)
                    kwargs.update(audio=waveforms,sampling_rate=sampling_rate)
                if videos:
                    if (self.family not in ('qwen3_5_moe','qwen4_exp','glm5_next') or
                            type(video_metadata) not in (list, tuple) or
                            len(video_metadata) != len(videos)):
                        raise ValueError('selected frames require explicit publisher video metadata')
                    if any(
                            not isinstance(item, dict) for item in video_metadata):
                        raise ValueError('video metadata must be an explicit mapping per selected sequence')
                    if any(set(item) != {'frames_indices', 'total_num_frames', 'fps'}
                           for item in video_metadata):
                        raise ValueError('video metadata contains unadmitted fields')
                    # Detach the processor call from caller-owned metadata so
                    # frame ordering, timestamps, and source-count admission
                    # cannot change while preprocessing is in flight.
                    owned_video_metadata = tuple(copy.deepcopy(item) for item in video_metadata)
                    owned_videos = tuple(tuple(frames) if type(frames) in (list, tuple) else frames
                                         for frames in videos)
                    decoded_videos=[]
                    for frames, metadata in zip(owned_videos, owned_video_metadata):
                        if type(frames) not in (list,tuple) or not 1<=len(frames)<=4096:
                            raise ValueError('bounded explicit selected video frames required')
                        indices = metadata.get('frames_indices')
                        total = metadata.get('total_num_frames')
                        try:
                            fps = normalize_video_fps(metadata.get('fps'))
                        except ValueError:
                            raise ValueError('ordered selected video indices, total frame count and FPS required') from None
                        if (type(indices) not in (list, tuple) or len(indices) != len(frames) or
                                type(total) is not int or isinstance(total, bool) or total <= 0 or
                                any(type(index) is not int or isinstance(index, bool) or
                                    not 0 <= index < total for index in indices) or
                                any(left >= right for left, right in zip(indices, indices[1:]))):
                            raise ValueError('ordered selected video indices, total frame count and FPS required')
                        metadata['fps'] = fps
                        decoded=[]
                        for value in frames:
                            boundary();image=self._image(value);owned.append(image)
                            work+=image.width*image.height*32
                            if work>self.max_bytes: raise MemoryError('aggregate video staging exceeds admission')
                            decoded.append(image)
                        decoded_videos.append(decoded)
                    kwargs.update(videos=decoded_videos,video_metadata=owned_video_metadata,do_sample_frames=False)
                boundary();result=dict(self.processor(**kwargs));boundary();self._unchanged()
                allowed=({'input_ids','attention_mask','pixel_values','image_grid_thw','pixel_values_videos',
                    'video_grid_thw','mm_token_type_ids'} if self.family in ('qwen3_5_moe','qwen4_exp','glm5_next') else
                    {'input_ids','attention_mask','pixel_values','audio_input_ids','audio_input_ids_mask'})
                if set(result)-allowed: raise ValueError('publisher processor returned unowned model inputs')
                extent=0
                for name,value in result.items():
                    if not isinstance(value,np.ndarray) or value.dtype.kind not in 'biuf':
                        raise ValueError('processor must return numeric NumPy arrays: '+name)
                    extent+=value.nbytes
                    if work+2*extent>self.max_bytes: raise MemoryError('processor results exceed aggregate staging admission')
                ids=result.pop('input_ids')
                if self.family=='glm5_next': result.pop('mm_token_type_ids',None)
                if ids.ndim!=2 or ids.shape[0]!=1 or not ids.shape[1] or ids.dtype.kind not in 'iu':
                    raise ValueError('expanded unpadded batch-one processor token ids required')
                mask=result.pop('attention_mask',None)
                if mask is not None and (mask.shape!=ids.shape or not np.all(mask==1)):
                    raise ValueError('native conditional processor does not admit padded prompts')
                return tuple(int(token) for token in ids[0]),result
            finally:
                primary = sys.exc_info()[1]
                cleanup_errors = []
                # Drop references to staging inputs on both success and failure.
                # The processor result is returned separately and remains owned
                # by the caller; clearing kwargs cannot invalidate that result.
                kwargs.clear()
                owned_waveforms.clear()
                if 'owned_waveform' in locals():
                    owned_waveform = None
                if 'chunk' in locals():
                    chunk = None
                if 'value' in locals():
                    value = None
                if 'waveforms' in locals():
                    waveforms.clear()
                if 'decoded' in locals():
                    decoded.clear()
                if 'decoded_videos' in locals():
                    for decoded_frames in decoded_videos:
                        decoded_frames.clear()
                    decoded_videos.clear()
                # The tuples are local snapshots of caller collections, not
                # owned inputs. Rebind them to release this frame's references
                # without changing any sequence supplied by the caller.
                images=audio=videos=()
                if 'owned_videos' in locals():
                    owned_videos=()
                if 'owned_video_metadata' in locals():
                    owned_video_metadata=()
                video_metadata=()
                if primary is not None:
                    if type(result) is dict:
                        result.clear()
                    if 'ids' in locals():
                        ids=None
                for image in owned:
                    try:
                        image.close()
                    except BaseException as error:
                        cleanup_errors.append(error)
                owned.clear()
                if cleanup_errors:
                    if primary is not None:
                        for error in cleanup_errors:
                            if hasattr(primary, 'add_note'):
                                add_failure_note(primary, 'publisher image cleanup failed: '+failure_summary(error))
                    else:
                        first = cleanup_errors[0]
                        if type(result) is dict:
                            result.clear()
                        if 'ids' in locals():
                            ids=None
                        for error in cleanup_errors[1:]:
                            if hasattr(first, 'add_note'):
                                add_failure_note(first, 'additional publisher image cleanup failure: '+failure_summary(error))
                        raise first

    def close(self):
        with self._lock:
            if self._closed and self.processor is None:
                return
            processor=self.processor
            close=getattr(processor,'close',None)
            if callable(close):
                # Keep references alive when the underlying owner fails so
                # callers can retry rather than losing a live resource.
                close()
            self.processor=self.tokenizer=None;self._closed=True

__all__=['PublisherMediaProcessor']
