"""Bounded durable service continuations keyed by explicit application identity."""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from uuid import uuid4

from .cache_file_recovery import verify_service_cache_file
from .session_recovery import _path


def _pairs(pairs):
    result = {}
    for name,value in pairs:
        if name in result:
            raise ValueError('duplicate continuation catalog field')
        result[name] = value
    return result


class DurableContinuationStore:
    """Own one dedicated directory; never replay tokens to recover model state.

    Records publish after complete cache files are flushed and read back. Each
    cache uses a unique immutable filename so interrupted record publication
    cannot replace the cache still referenced by an older record. The directory
    must be exclusively managed by this store within one process.
    """
    def __init__(self,directory,*,model_identity,max_entries,max_total_bytes,max_cache_bytes,
                 expiry_seconds=None):
        self.root = _path(directory).resolve(strict=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError('dedicated regular continuation directory required')
        if type(model_identity) is not str or not model_identity or len(model_identity)>4096:
            raise ValueError('bounded model execution identity required')
        for value in (max_entries,max_total_bytes,max_cache_bytes):
            if type(value) is not int or value<=0:
                raise ValueError('explicit continuation count/disk/cache budgets required')
        if max_entries>100000 or max_cache_bytes>max_total_bytes:
            raise ValueError('continuation admission exceeds catalog/disk budget')
        if expiry_seconds is not None and (type(expiry_seconds) not in (int,float) or
                not 0<expiry_seconds<315360000):
            raise ValueError('bounded positive continuation expiry required')
        self.identity,self.max_entries,self.total_limit,self.cache_limit = model_identity,max_entries,max_total_bytes,max_cache_bytes
        self.expiry,self._lock = expiry_seconds,threading.RLock()
        self.prefix = 'oa-cont-'+hashlib.sha256(model_identity.encode()).hexdigest()[:24]+'-'
        from .directory_ownership import DirectoryOwnership
        self._ownership = DirectoryOwnership(self.root)

    def _check_open(self):
        if self._ownership.closed:
            raise RuntimeError('durable continuation store is closed')
        self._ownership.assert_held()

    def close(self):
        with self._lock:
            self._ownership.close()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self,*_):
        self.close()

    def _key(self,key):
        if type(key) is not str or not key or len(key.encode())>4096:
            raise ValueError('bounded explicit application continuation key required')
        return hashlib.sha256(key.encode()).hexdigest()

    def _record_path(self,key):
        return self.root/(self.prefix+self._key(key)+'.json')

    def _owned_path(self,path,*,cache=False):
        path=Path(path)
        suffix='.bin' if cache else '.json'
        name=path.name
        body=name[len(self.prefix):-len(suffix)]
        parts=body.split('-') if cache else [body]
        lengths=(64,32) if cache else (64,)
        if (path.parent!=self.root or not name.startswith(self.prefix) or
                not name.endswith(suffix) or len(parts)!=len(lengths) or
                any(len(part)!=length or any(c not in '0123456789abcdef' for c in part)
                    for part,length in zip(parts,lengths))):
            raise ValueError('invalid owned continuation filename')
        return _path(path)

    def _sync_directory(self):
        if os.name!='nt':
            descriptor=os.open(self.root,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)

    def _read(self,path):
        path=self._owned_path(path)
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError('regular continuation record required')
        before = path.stat()
        stamp = lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        if not 0<before.st_size<=65536:
            raise MemoryError('continuation catalog record extent exceeds admission')
        descriptor=os.open(path,os.O_RDONLY|getattr(os,'O_BINARY',0)|
            getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_NONBLOCK',0))
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError('regular opened continuation record required')
            source=os.fdopen(descriptor,'rb')
        except BaseException:
            os.close(descriptor);raise
        with source:
            if stamp(os.fstat(source.fileno()))!=stamp(before):
                raise ValueError('continuation record changed while opening')
            raw = source.read(65537)
            if stamp(os.fstat(source.fileno()))!=stamp(before):
                raise ValueError('continuation record changed while reading')
        if stamp(_path(path).stat())!=stamp(before) or len(raw)!=before.st_size:
            raise ValueError('continuation record replaced while reading')
        if len(raw)>65536:
            raise MemoryError('continuation catalog record exceeds admission')
        value = json.loads(raw,object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite catalog field: '+value)))
        if type(value) is not dict or set(value)!={'version','identity','cache','bytes','sha256','tokens','history_sha256','created'}:
            raise ValueError('invalid continuation record')
        name = value['cache']
        if type(name) is not str:
            raise ValueError('invalid continuation cache filename')
        self._owned_path(self.root/name,cache=True)
        if name[len(self.prefix):len(self.prefix)+64]!=path.name[len(self.prefix):-5]:
            raise ValueError('continuation cache belongs to another application key')
        if (type(value['version']) is not int or value['version']!=1 or value['identity']!=self.identity or type(name) is not str or
                not name.startswith(self.prefix) or Path(name).name!=name or not name.endswith('.bin') or
                type(value['bytes']) is not int or not 0<value['bytes']<=self.cache_limit or
                type(value['tokens']) is not int or value['tokens']<0 or
                type(value['created']) not in (int,float) or not 0<value['created']<1e12):
            raise ValueError('continuation identity/geometry mismatch')
        for field in ('sha256','history_sha256'):
            if type(value[field]) is not str or len(value[field])!=64 or any(c not in '0123456789abcdef' for c in value[field]):
                raise ValueError('invalid continuation digest')
        return value

    def _publish(self,path,value):
        path=self._owned_path(path)
        encoded = json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        descriptor,temporary = tempfile.mkstemp(prefix=self.prefix+'catalog-',dir=self.root)
        pending_descriptor=descriptor
        def cleanup_staging():
            nonlocal pending_descriptor
            if pending_descriptor is not None:
                os.close(pending_descriptor);pending_descriptor=None
            Path(temporary).unlink(missing_ok=True)
        try:
            source=os.fdopen(descriptor,'wb')
            pending_descriptor=None
            with source:
                source.write(encoded);source.flush();os.fsync(source.fileno())
            os.replace(temporary,path)
            self._sync_directory()
        except BaseException as primary:
            try:cleanup_staging()
            except BaseException as cleanup:
                primary.cleanup=cleanup_staging
                primary.add_note('continuation publication staging cleanup retained: '+repr(cleanup))
            raise
        else:
            try:cleanup_staging()
            except BaseException as cleanup:
                cleanup.cleanup=cleanup_staging
                raise

    def _remove(self,path,record):
        path=self._owned_path(path)
        cache=self._owned_path(self.root/record['cache'],cache=True)
        path.unlink(missing_ok=True)
        self._sync_directory()
        cache.unlink(missing_ok=True)
        self._sync_directory()

    def prune(self,*,reserve_bytes=0,reserve_entries=0,_protected=()):
        with self._lock:
            self._check_open()
            if (type(reserve_bytes) is not int or not 0<=reserve_bytes<=self.total_limit or
                    type(reserve_entries) is not int or not 0<=reserve_entries<=self.max_entries):
                raise ValueError('invalid continuation reservation')
            records = []
            now = time.time()
            for count,path in enumerate(self.root.glob(self.prefix+'*.json'),1):
                if count>100000:
                    raise MemoryError('continuation catalog exceeds scan admission')
                value = self._read(path)
                if path not in _protected and self.expiry is not None and now-value['created']>=self.expiry:
                    self._remove(path,value)
                else:
                    records.append((value['created'],path,value))
            records.sort(key=lambda item:(item[0],str(item[1])))
            total = sum(item[2]['bytes'] for item in records)
            while records and (len(records)+reserve_entries>self.max_entries or total+reserve_bytes>self.total_limit):
                index = next((index for index,item in enumerate(records) if item[1] not in _protected),None)
                if index is None:
                    raise MemoryError('durable replacement requires old and staged cache extents together')
                _,path,value = records.pop(index)
                self._remove(path,value)
                total -= value['bytes']
            referenced = {value['cache'] for _,_,value in records}
            for count,path in enumerate(self.root.glob(self.prefix+'*.bin'),1):
                if count>200000:
                    raise MemoryError('orphan inventory exceeds scan admission')
                path=self._owned_path(path,cache=True)
                if path.name not in referenced:
                    path.unlink()
            self._sync_directory()
            return dict(entries=len(records),bytes=total)

    def save(self,service,session_id,key,*,evict=False):
        with self._lock:
            self._check_open()
            record_path = self._record_path(key)
            # Lock the service session across its history receipt and frame.
            session = service._session(session_id)
            with session.lock:
                if session.active or session.closed:
                    raise RuntimeError('cannot persist an active/closed continuation')
                history = service.history(session_id)
                exists = record_path.exists()
                self.prune(reserve_bytes=self.cache_limit,reserve_entries=0 if exists else 1,
                           _protected=(record_path,) if exists else ())
                cache = self.root/(self.prefix+self._key(key)+'-'+uuid4().hex+'.bin')
                receipt = service.save_session_cache(session_id,cache,max_bytes=self.cache_limit)
                verify_service_cache_file(cache,receipt=receipt,max_bytes=self.cache_limit)
                record = dict(version=1,identity=self.identity,cache=cache.name,bytes=receipt['bytes'],
                    sha256=receipt['sha256'],tokens=len(history),created=time.time(),
                    history_sha256=hashlib.sha256(json.dumps(list(history),separators=(',',':')).encode()).hexdigest())
                self._publish(record_path,record)
                # Reclaim the prior immutable cache only after the new catalog
                # record has crossed its durable publication boundary.
                self.prune(_protected=(record_path,))
                if evict:
                    service.close_session(session_id)
                return dict(record,key=self._key(key),evicted=bool(evict))

    def recover(self,service,key):
        with self._lock:
            self._check_open()
            path = self._record_path(key)
            record = self._read(path)
            if self.expiry is not None and time.time()-record['created']>=self.expiry:
                self._remove(path,record)
                raise FileNotFoundError('continuation expired')
            cache = self.root/record['cache']
            verify_service_cache_file(cache,receipt=record,max_bytes=self.cache_limit)
            session_id = service.recover_cached_session(cache,max_bytes=self.cache_limit)
            try:
                history = service.history(session_id)
                digest = hashlib.sha256(json.dumps(list(history),separators=(',',':')).encode()).hexdigest()
                if len(history)!=record['tokens'] or digest!=record['history_sha256']:
                    raise ValueError('recovered continuation history differs from catalog')
            except BaseException as error:
                try:
                    service.close_session(session_id)
                except BaseException as cleanup:
                    error.add_note('recovered continuation cleanup pending: '+str(cleanup))
                raise
            return session_id

    def recover_into(self,service,session_id,key):
        """Recover an existing owner without admitting a second resident decoder."""
        with self._lock:
            self._check_open();path=self._record_path(key);record=self._read(path)
            if self.expiry is not None and time.time()-record['created']>=self.expiry:
                self._remove(path,record);raise FileNotFoundError('continuation expired')
            cache=self.root/record['cache']
            verify_service_cache_file(cache,receipt=record,max_bytes=self.cache_limit)
            service.load_session_cache(session_id,cache,max_bytes=self.cache_limit,
                history_receipt=(record['tokens'],record['history_sha256']))
            return session_id

    def delete(self,key):
        with self._lock:
            self._check_open()
            path = self._record_path(key)
            self._remove(path,self._read(path))


__all__ = ['DurableContinuationStore']
