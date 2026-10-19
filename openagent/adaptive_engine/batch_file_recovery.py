"""Atomic complete native batch files and fresh-slot recovery."""
import json
import os
import stat
import struct

from .cache_file_recovery import save_service_cache,load_service_cache,verify_service_cache_file
from .session_recovery import _path
from .segment_snapshot import _pairs
from .owned_file_descriptor import close_after_failure,fdopen_owned


class _Adapter:
    def __init__(self,batch,session_ids):
        self.batch,self.ids = batch,session_ids

    def write_session_cache(self,unused,output,*,max_bytes):
        return self.batch.write_cache(self.ids,output,max_bytes=max_bytes)

    def restore_session_cache(self,unused,source,*,max_bytes):
        return self.batch.restore_cache(self.ids,source,max_bytes=max_bytes)


def save_batch_cache(batch,session_ids,destination,*,max_bytes):
    return save_service_cache(_Adapter(batch,session_ids),None,destination,max_bytes=max_bytes)


def load_batch_cache(batch,session_ids,source,*,max_bytes):
    return load_service_cache(_Adapter(batch,session_ids),None,source,max_bytes=max_bytes)


def recover_batch_cache(batch,source,*,max_bytes):
    """Admit the catalog count, create private empty slots, then restore all."""
    if type(max_bytes) is not int or max_bytes<88:
        raise ValueError('explicit batch cache file admission required')
    path = _path(source)
    descriptor = os.open(path,os.O_RDONLY|getattr(os,'O_BINARY',0)|getattr(os,'O_NOFOLLOW',0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 88<=info.st_size<=max_bytes:
            raise ValueError('bounded regular native batch cache required')
    except BaseException as primary:
        close_after_failure(descriptor,primary,owner='batch recovery input')
        raise
    stream=fdopen_owned(descriptor,'rb',owner='batch recovery input')
    with stream:
        prefix = stream.read(12)
        if len(prefix)!=12 or prefix[:8]!=b'OABATCH1':
            raise ValueError('invalid batch recovery file magic/version')
        length = struct.unpack('<I',prefix[8:])[0]
        if not 0<length<=min(65536,max_bytes-44):
            raise ValueError('invalid batch recovery header extent')
        header = json.loads(stream.read(length), object_pairs_hook=_pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if (type(header) is not dict or set(header)!={'version','sizes'} or
                type(header['version']) is not int or header['version']!=1):
            raise ValueError('invalid batch recovery header')
        sizes = header['sizes']
        if (type(sizes) is not list or not 0<len(sizes)<=batch.service.limits.max_sessions or
                any(type(size) is not int or size<88 for size in sizes) or
                44+length+sum(sizes)!=info.st_size):
            raise ValueError('batch recovery slot inventory/extent differs')
    ids = []
    try:
        for _ in sizes:
            ids.append(batch.open_session())
        load_batch_cache(batch,tuple(ids),path,max_bytes=max_bytes)
        return tuple(ids)
    except BaseException as error:
        for key in reversed(ids):
            try:
                # Restoration may quarantine the coordinator. The service still
                # owns failed slots and admits explicit cleanup independently.
                batch.service.close_session(key)
            except BaseException as cleanup:
                error.add_note('fresh batch recovery cleanup pending for '+key+': '+str(cleanup))
        raise


def evict_batch_cache(batch,session_ids,destination,*,max_bytes):
    """Retain a verified complete file before releasing selected mutable slots."""
    from contextlib import ExitStack
    with batch._lock,ExitStack() as stack:
        if batch._busy or batch._closed or batch._quarantined:
            raise RuntimeError('cannot evict a busy/closed/quarantined batch')
        ids = batch._locked_sessions(session_ids,stack)
        receipt = save_batch_cache(batch,ids,destination,max_bytes=max_bytes)
        verify_service_cache_file(destination,receipt=receipt,max_bytes=max_bytes)
        failures = []
        for key in ids:
            try:
                batch.service.close_session(key)
            except BaseException as error:
                failures.append((key,error))
        if failures:
            error = RuntimeError('batch eviction cleanup incomplete; verified complete cache retained at '+receipt['path'])
            for key,failure in failures:
                error.add_note('slot cleanup pending for '+key+': '+str(failure))
            raise error from failures[0][1]
        return dict(receipt,session_ids=ids,cache_bytes_preserved=True)


__all__ = ['save_batch_cache','load_batch_cache','recover_batch_cache','evict_batch_cache']
