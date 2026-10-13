"""Bounded stream adapters for existing complete family checkpoint codecs.

Scratch disk stores one frame; family codecs retain their own memory admission
and validation. No array schema, token replay or pickle is introduced here.
"""
import os
from pathlib import Path
import tempfile
import sys

from .segment_snapshot import _write
from .session_recovery import _path
from .generation_checkpoint_files import _remove_staging

_CHUNK = 1024*1024


def transport_cache(model, stream, codec, *, max_bytes, restoring=False,
                    scratch_directory=None):
    if type(max_bytes) is not int or not 64<=max_bytes<=2**63-1:
        raise ValueError('explicit cache transport byte budget required')
    if type(restoring) is not bool or not callable(codec):
        raise ValueError('explicit cache direction and concrete codec required')
    if not callable(getattr(stream, 'read', None)) and restoring:
        raise TypeError('cache restore source must expose a counted read method')
    if not restoring and not callable(getattr(stream, 'write', None)):
        raise TypeError('cache publication sink must expose a counted write method')
    if scratch_directory is not None:scratch_directory=_path(scratch_directory).resolve(strict=True)
    with model._lock:
        if model._busy or model._closed:
            raise RuntimeError('executor unavailable for cache transport')
        check_cancel=getattr(model,'_check_cancelled',None)
        if not callable(check_cancel):check_cancel=getattr(model,'_cancel_rows',None)
        if not callable(check_cancel):raise TypeError('complete family cancellation boundary required')
        descriptor,name=tempfile.mkstemp(prefix='openagent-cache-',dir=scratch_directory)
        os.close(descriptor)
        path=Path(name)
        try:
            if restoring:
                model._busy=True
                try:
                    size=0
                    with path.open('wb') as output:
                        while True:
                            check_cancel()
                            count=min(_CHUNK,max_bytes-size+1)
                            data=stream.read(count)
                            if type(data) is not bytes or len(data)>count:
                                raise ValueError('binary cache source required')
                            if not data:
                                break
                            size+=len(data)
                            if size>max_bytes:
                                raise MemoryError('cache transport source exceeds admission')
                            _write(output,data)
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    model._busy=False
                codec(model,path,max_payload_bytes=max_bytes)
                position=getattr(model,'position',None)
                return getattr(model,'length',None) if position is None else position
            codec(model,path,max_payload_bytes=max_bytes)
            size=path.stat().st_size
            if size>max_bytes:
                raise MemoryError('cache transport frame exceeds admission')
            model._busy=True
            try:
                with path.open('rb') as source:
                    remaining=size
                    while remaining:
                        check_cancel()
                        data=source.read(min(_CHUNK,remaining))
                        if not data:
                            raise ValueError('scratch cache frame truncated')
                        _write(stream,data)
                        remaining-=len(data)
                return size
            finally:
                model._busy=False
        finally:
            _remove_staging(path,sys.exc_info()[1])
