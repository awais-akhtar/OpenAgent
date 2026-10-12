"""Atomic padded generation across independently owned native service slots."""
from contextlib import ExitStack,contextmanager
from dataclasses import dataclass
import tempfile
import threading
import hashlib
import hmac
import json
import struct

import numpy as np


@dataclass(frozen=True)
class ServiceBatchResult:
    session_ids: tuple
    results: tuple
    occupied_indices: tuple


class NativeServiceBatch:
    """Own a service and coordinate complete-cache rollback across a batch.

    Padding is removed before each independent sequence executes. Every slot
    retains the concrete family's arithmetic and continuation semantics. This
    coordinator does not claim simultaneous device execution.
    """
    def __init__(self,service,*,max_batch_bytes,max_rollback_bytes):
        from .execution_service import NativeExecutionService
        if type(service) is not NativeExecutionService:
            raise TypeError('owned native execution service required')
        if any(type(value) is not int or value<=0 for value in (max_batch_bytes,max_rollback_bytes)):
            raise ValueError('explicit batch and complete-cache rollback admission required')
        self.service,self.input_limit,self.rollback_limit = service,max_batch_bytes,max_rollback_bytes
        self._lock = threading.RLock()
        self._busy,self._closed,self._quarantined = False,False,False

    @contextmanager
    def _operation(self,*,closing=False):
        with self._lock:
            if self._busy or (not closing and (self._closed or self._quarantined)):
                raise RuntimeError('native batch closed, quarantined or busy')
            self._busy = True
            try:
                yield
            finally:
                self._busy = False

    def open_session(self):
        with self._operation():
            return self.service.open_session()

    def close_session(self,session_id):
        with self._operation():
            self.service.close_session(session_id)

    def generate(self,session_ids,token_ids,*,attention_mask=None,max_new_tokens=None,
                 stop_ids=(),sampling=None,cancel=None,on_event=None):
        with self._operation():
            if (type(session_ids) not in (list,tuple) or not session_ids or
                    len(session_ids)>self.service.limits.max_sessions or
                    any(type(value) is not str for value in session_ids) or len(set(session_ids))!=len(session_ids)):
                raise ValueError('unique admitted batch session inventory required')
            ids = tuple(session_ids)
            values = np.asarray(token_ids)
            if (values.ndim!=2 or values.shape[0]!=len(ids) or not values.shape[1] or
                    values.dtype.kind not in 'iu' or values.size*32>self.input_limit):
                raise ValueError('bounded rectangular integer batch tokens required')
            mask = np.ones(values.shape,bool) if attention_mask is None else np.asarray(attention_mask)
            if mask.shape!=values.shape or not np.isin(mask,(0,1)).all():
                raise ValueError('binary batch occupancy mask required')
            occupied = tuple(tuple(int(index) for index in np.flatnonzero(row)) for row in mask)
            if any(not row for row in occupied):
                raise ValueError('each batch sequence requires occupied prompt tokens')
            prompts = tuple(tuple(int(values[row,index]) for index in indices) for row,indices in enumerate(occupied))
            if any(token<0 or token>=self.service.limits.vocabulary for prompt in prompts for token in prompt):
                raise ValueError('occupied batch token outside model vocabulary')
            if cancel is not None and not callable(getattr(cancel,'is_set',None)):
                raise ValueError('batch cancellation must provide is_set')
            if on_event is not None and not callable(on_event):
                raise ValueError('batch event callback required')
            count = self.service.limits.max_new_tokens if max_new_tokens is None else max_new_tokens
            if type(count) is not int or not 1<=count<=self.service.limits.max_new_tokens:
                raise ValueError('batch generation count exceeds admission')
            if len(ids)*count*128+values.size*32>self.input_limit:
                raise MemoryError('batch results/event retention exceed admission')
            with ExitStack() as owned:
                sessions = {key:self.service._session(key) for key in ids}
                for key in sorted(ids):
                    owned.enter_context(sessions[key].lock)
                    if sessions[key].active or sessions[key].closed:
                        raise RuntimeError('batch selected a busy/closed service session')
                rollback = []
                remaining = self.rollback_limit
                for key in ids:
                    file = owned.enter_context(tempfile.TemporaryFile(mode='w+b'))
                    size = self.service.write_session_cache(key,file,max_bytes=remaining)
                    remaining -= size
                    file.seek(0)
                    rollback.append((key,file,size))
                results = []
                try:
                    for index,(key,prompt) in enumerate(zip(ids,prompts)):
                        result = self.service.generate(key,prompt,max_new_tokens=count,stop_ids=stop_ids,
                                                       sampling=sampling,cancel=cancel)
                        results.append(result)
                    if on_event is not None:
                        on_event(dict(event='generation.batch',session_ids=ids,
                            token_ids=tuple(result.token_ids for result in results),committed=True))
                except BaseException as error:
                    failures = []
                    for key,file,size in rollback:
                        try:
                            file.seek(0)
                            self.service.restore_session_cache(key,file,max_bytes=size)
                        except BaseException as failure:
                            failures.append((key,failure))
                    if failures:
                        self._quarantined = True
                        for key,failure in failures:
                            error.add_note('batch rollback failed for '+key+': '+str(failure))
                    raise
                return ServiceBatchResult(ids,tuple(results),occupied)

    def _locked_sessions(self,session_ids,stack):
        if (type(session_ids) not in (list,tuple) or not session_ids or
                len(session_ids)>self.service.limits.max_sessions or
                any(type(key) is not str for key in session_ids) or len(set(session_ids))!=len(session_ids)):
            raise ValueError('unique admitted batch recovery slots required')
        ids = tuple(session_ids)
        for key in sorted(ids):
            session = self.service._session(key)
            stack.enter_context(session.lock)
            if session.active or session.closed:
                raise RuntimeError('batch recovery selected a busy/closed slot')
        return ids

    def write_cache(self,session_ids,output,*,max_bytes):
        from .segment_snapshot import _write
        with self._operation(),ExitStack() as stack:
            if type(max_bytes) is not int or max_bytes<44:
                raise ValueError('explicit complete batch cache budget required')
            ids = self._locked_sessions(session_ids,stack)
            remaining = min(max_bytes,self.rollback_limit)-44-65536
            frames = []
            for key in ids:
                source = stack.enter_context(tempfile.TemporaryFile(mode='w+b'))
                size = self.service.write_session_cache(key,source,max_bytes=remaining)
                remaining -= size
                source.seek(0)
                frames.append((source,size))
            header = json.dumps(dict(version=1,sizes=[size for _,size in frames]),
                                sort_keys=True,separators=(',',':')).encode()
            extent = 44+len(header)+sum(size for _,size in frames)
            if len(header)>65536 or extent>max_bytes:
                raise MemoryError('complete batch cache frame exceeds admission')
            digest = hashlib.sha256()
            for data in (b'OABATCH1',struct.pack('<I',len(header)),header):
                digest.update(data);_write(output,data)
            for source,size in frames:
                left = size
                while left:
                    data = source.read(min(left,1024*1024))
                    if not data:
                        raise OSError('staged batch cache truncated')
                    left -= len(data)
                    digest.update(data);_write(output,data)
            _write(output,digest.digest())
            return extent

    def restore_cache(self,session_ids,source,*,max_bytes):
        from .segment_snapshot import _read,_pairs
        with self._operation(),ExitStack() as stack:
            if type(max_bytes) is not int or max_bytes<44:
                raise ValueError('explicit complete batch cache budget required')
            ids = self._locked_sessions(session_ids,stack)
            prefix = _read(source,12)
            if prefix[:8]!=b'OABATCH1':
                raise ValueError('invalid complete batch cache magic/version')
            length = struct.unpack('<I',prefix[8:])[0]
            if not 0<length<=min(65536,max_bytes-44):
                raise ValueError('invalid batch cache header extent')
            encoded = _read(source,length)
            header = json.loads(encoded,object_pairs_hook=_pairs)
            if type(header) is not dict or set(header)!={'version','sizes'} or type(header['version']) is not int or header['version']!=1:
                raise ValueError('invalid batch cache header')
            sizes = header['sizes']
            if type(sizes) is not list or len(sizes)!=len(ids) or any(type(size) is not int or size<88 for size in sizes):
                raise ValueError('batch cache session inventory mismatch')
            payload = sum(sizes)
            if 44+length+payload>max_bytes or payload>self.rollback_limit:
                raise MemoryError('batch cache input staging exceeds admission')
            digest = hashlib.sha256(prefix+encoded)
            frames = []
            for size in sizes:
                file = stack.enter_context(tempfile.TemporaryFile(mode='w+b'))
                left = size
                while left:
                    data = _read(source,min(left,1024*1024))
                    digest.update(data);file.write(data);left -= len(data)
                file.seek(0);frames.append(file)
            if not hmac.compare_digest(digest.digest(),_read(source,32)) or source.read(1)!=b'':
                raise ValueError('batch cache checksum/EOF mismatch')
            rollback = []
            remaining = self.rollback_limit-payload
            for key in ids:
                file = stack.enter_context(tempfile.TemporaryFile(mode='w+b'))
                size = self.service.write_session_cache(key,file,max_bytes=remaining)
                remaining -= size
                file.seek(0);rollback.append((key,file,size))
            try:
                for key,file,size in zip(ids,frames,sizes):
                    self.service.restore_session_cache(key,file,max_bytes=size)
            except BaseException as error:
                failures = []
                for key,file,size in rollback:
                    try:
                        file.seek(0)
                        self.service.restore_session_cache(key,file,max_bytes=size)
                    except BaseException as failure:
                        failures.append((key,failure))
                if failures:
                    self._quarantined = True
                    for key,failure in failures:
                        error.add_note('batch cache rollback failed for '+key+': '+str(failure))
                raise
            return tuple(self.service.history(key) for key in ids)

    def save_cache(self,session_ids,destination,*,max_bytes):
        from .batch_file_recovery import save_batch_cache
        return save_batch_cache(self,session_ids,destination,max_bytes=max_bytes)

    def load_cache(self,session_ids,source,*,max_bytes):
        from .batch_file_recovery import load_batch_cache
        return load_batch_cache(self,session_ids,source,max_bytes=max_bytes)

    def recover_cached_batch(self,source,*,max_bytes):
        from .batch_file_recovery import recover_batch_cache
        return recover_batch_cache(self,source,max_bytes=max_bytes)

    def evict_cached_batch(self,session_ids,destination,*,max_bytes):
        from .batch_file_recovery import evict_batch_cache
        return evict_batch_cache(self,session_ids,destination,max_bytes=max_bytes)

    def close(self):
        with self._operation(closing=True):
            self.service.close()
            self._closed = True


__all__ = ['NativeServiceBatch','ServiceBatchResult']
