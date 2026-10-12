"""Bounded local publisher preprocessing for native conditional execution.

Only preprocessing uses the admitted Transformers package. Decoder execution
remains in OpenAgent's native composition. Remote code and downloads are off.
"""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import threading
import os
import numpy as np
from types import SimpleNamespace
from .session_recovery import _path
from .native_library_identity import _stamp

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
        self.processor=transformers.AutoProcessor.from_pretrained(str(root),local_files_only=True,
            trust_remote_code=False)
        self.tokenizer=PublisherTokenizerAdapter(self.processor.tokenizer)
        self._unchanged()

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
                with Image.fromarray(value) as image:return image.convert('RGB')
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
            def boundary():
                if cancelled is not None:
                    if not callable(cancelled): raise TypeError('callable processor cancellation required')
                    if cancelled(): raise InterruptedError('publisher media preprocessing cancelled')
            boundary();self._unchanged();owned=[];work=0
            try:
                kwargs={'text':text,'return_tensors':'np','padding':False}
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
                        if not isinstance(value,np.ndarray) or value.ndim!=1 or value.dtype.kind!='f' or not 0<value.size<=self.max_audio_samples:
                            raise ValueError('bounded mono floating waveform required')
                        work+=value.size*64
                        if work>self.max_bytes: raise MemoryError('aggregate audio preprocessing staging exceeds admission')
                        if not np.isfinite(value).all(): raise ValueError('nonfinite waveform sample')
                        waveforms.append(np.array(value,dtype=np.float32,copy=True))
                    kwargs.update(audio=waveforms,sampling_rate=sampling_rate)
                if videos:
                    if self.family not in ('qwen3_5_moe','qwen4_exp','glm5_next') or video_metadata is None or len(video_metadata)!=len(videos):
                        raise ValueError('selected frames require explicit publisher video metadata')
                    decoded_videos=[]
                    for frames in videos:
                        if type(frames) not in (list,tuple) or not 1<=len(frames)<=4096:
                            raise ValueError('bounded explicit selected video frames required')
                        decoded=[]
                        for value in frames:
                            boundary();image=self._image(value);owned.append(image)
                            work+=image.width*image.height*32
                            if work>self.max_bytes: raise MemoryError('aggregate video staging exceeds admission')
                            decoded.append(image)
                        decoded_videos.append(decoded)
                    kwargs.update(videos=decoded_videos,video_metadata=video_metadata,do_sample_frames=False)
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
                for image in owned: image.close()

    def close(self):
        with self._lock:
            self.processor=self.tokenizer=None;self._closed=True

__all__=['PublisherMediaProcessor']
