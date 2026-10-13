"""Explicit local PCM audio and selected-frame video inputs for publisher processors."""
import json
import os
import stat
from pathlib import Path
import wave
import numpy as np
from .session_recovery import _path
from .native_library_identity import _stamp

def _pairs(items):
    result={}
    for key,value in items:
        if key in result: raise ValueError('duplicate media manifest field')
        result[key]=value
    return result

def read_regular(path,*,max_bytes):
    if type(max_bytes) is not int or max_bytes<=0: raise ValueError('positive media file budget required')
    resolved=_path(path).resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink(): raise ValueError('regular local media file required')
    before=resolved.stat()
    if before.st_size>max_bytes: raise MemoryError('media file exceeds admission')
    descriptor=os.open(resolved,os.O_RDONLY|getattr(os,'O_BINARY',0)|
        getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
    try:
        opened=os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stamp(opened)!=_stamp(before):
            raise ValueError('media file changed during opening')
        stream=os.fdopen(descriptor,'rb')
    except BaseException:
        os.close(descriptor);raise
    with stream:
        raw=stream.read(before.st_size+1)
        if _stamp(os.fstat(stream.fileno()))!=_stamp(before):
            raise ValueError('media file changed during reading')
    if len(raw)!=before.st_size or _stamp(before)!=_stamp(_path(resolved).stat()):
        raise ValueError('media file changed during reading')
    return raw

def read_pcm_audio(path,*,max_bytes,max_samples):
    """Decode mono integer PCM WAV without implicit mixing or resampling."""
    import io
    raw=read_regular(path,max_bytes=max_bytes)
    with wave.open(io.BytesIO(raw),'rb') as stream:
        channels=stream.getnchannels();width=stream.getsampwidth();count=stream.getnframes();rate=stream.getframerate()
        if channels!=1 or width not in (1,2,3,4) or stream.getcomptype()!='NONE' or rate<=0:
            raise ValueError('mono uncompressed 8/16/24/32-bit integer PCM WAV required')
        if type(max_samples) is not int or not 0<count<=max_samples or len(raw)+count*32>max_bytes:
            raise MemoryError('PCM decoding exceeds explicit media admission')
        data=stream.readframes(count+1)
        if len(data)!=count*width: raise ValueError('PCM frame extent differs from declared WAV length')
    if width==1: samples=(np.frombuffer(data,np.uint8).astype(np.float32)-128)/np.float32(128)
    elif width==2: samples=np.frombuffer(data,'<i2').astype(np.float32)/np.float32(32768)
    elif width==4: samples=np.frombuffer(data,'<i4').astype(np.float32)/np.float32(2147483648)
    else:
        octets=np.frombuffer(data,np.uint8).reshape(-1,3).astype(np.int32)
        integers=octets[:,0]|(octets[:,1]<<8)|(octets[:,2]<<16)
        integers=(integers^0x800000)-0x800000
        samples=integers.astype(np.float32)/np.float32(8388608)
    return np.ascontiguousarray(samples),rate

def read_selected_video(path,*,max_bytes,max_frame_bytes,max_frames=4096,cancelled=None):
    """Manifest frames are already selected; preserve supplied publisher timestamps."""
    for value in (max_bytes,max_frame_bytes,max_frames):
        if type(value) is not int or value<=0:
            raise ValueError('positive selected-video admission required')
    source=_path(path).resolve(strict=True);root=source.parent
    raw=read_regular(source,max_bytes=min(max_bytes,1024**2))
    document=json.loads(raw,object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite video metadata: '+value)))
    if type(document) is not dict or set(document)!={'schema','frames','metadata'} or document['schema']!='openagent.publisher-selected-video.v1':
        raise ValueError('explicit selected-video manifest schema required')
    frames=document['frames'];metadata=document['metadata']
    if type(frames) is not list or not 1<=len(frames)<=max_frames or type(metadata) is not dict:
        raise ValueError('bounded frame inventory and publisher video metadata required')
    indices=metadata.get('frames_indices');total=metadata.get('total_num_frames');fps=metadata.get('fps')
    if (type(indices) is not list or len(indices)!=len(frames) or type(total) is not int or total<=0 or
            type(fps) not in (int,float) or not np.isfinite(fps) or fps<=0 or
            any(type(value) is not int or not 0<=value<total for value in indices) or
            any(a>=b for a,b in zip(indices,indices[1:]))):
        raise ValueError('ordered explicit selected frame indices, total frame count and FPS required')
    extent=len(raw);decoded=[];seen_frames=set()
    for name in frames:
        if cancelled is not None and cancelled(): raise InterruptedError('selected video file staging cancelled')
        if type(name) is not str or not name or Path(name).is_absolute(): raise ValueError('relative selected-frame path required')
        candidate=_path(root/name)
        if candidate.is_symlink(): raise ValueError('selected frame symlinks are not admitted')
        target=candidate.resolve(strict=True)
        if not target.is_relative_to(root): raise ValueError('selected frame escapes manifest directory')
        key=str(target).casefold()
        if key in seen_frames:
            raise ValueError('selected video frame inventory contains duplicate paths')
        seen_frames.add(key)
        if extent >= max_bytes:
            raise MemoryError('selected video frame inventory exceeds media admission')
        value=read_regular(target,max_bytes=min(max_frame_bytes,max_bytes-extent))
        extent+=len(value);decoded.append(value)
    return decoded,metadata

__all__=['read_regular','read_pcm_audio','read_selected_video']
