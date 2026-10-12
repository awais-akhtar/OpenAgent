"""Indexed bounded continuation eviction with rebuildable publication metadata."""
import json
import sqlite3
import time

from .continuation_store import DurableContinuationStore
from .session_recovery import _path


class IndexedContinuationStore(DurableContinuationStore):
    """Own a catalog index alongside authoritative immutable cache frames.

    The OS directory lease serializes publishers across processes. SQLite is
    only an index: complete JSON receipts remain the publication boundary and
    startup rebuild handles a process interrupted between those two writes.
    Routine admission reads indexed sizes rather than all JSON/cache files.
    """
    def __init__(self,directory,*,model_identity,max_entries,max_total_bytes,
                 max_cache_bytes,expiry_seconds=None):
        self._database=None;self._index_dirty=False;self._database_identity=None
        super().__init__(directory,model_identity=model_identity,max_entries=max_entries,
            max_total_bytes=max_total_bytes,max_cache_bytes=max_cache_bytes,
            expiry_seconds=expiry_seconds)
        try:
            path=_path(self.root/(self.prefix+'index.sqlite3'))
            _path(str(path)+'-wal');_path(str(path)+'-shm')
            if path.is_symlink(): raise ValueError('regular owned catalog database required')
            self._database=sqlite3.connect(str(path),check_same_thread=False,isolation_level=None)
            self._database.execute('PRAGMA journal_mode=WAL')
            self._database.execute('PRAGMA synchronous=FULL')
            self._database.execute('PRAGMA busy_timeout=5000')
            self._database.execute('CREATE TABLE IF NOT EXISTS records '
                '(name TEXT PRIMARY KEY, cache TEXT UNIQUE NOT NULL, bytes INTEGER NOT NULL, '
                'created REAL NOT NULL, receipt TEXT NOT NULL)')
            self._database.execute('CREATE INDEX IF NOT EXISTS records_age ON records(created,name)')
            metadata=path.stat()
            self._database_identity=(metadata.st_dev,metadata.st_ino)
            self._database_version=self._database.execute('PRAGMA data_version').fetchone()[0]
            self.rebuild()
            self.collect_orphans()
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:
                primary.cleanup=self.close
                primary.add_note('continuation index construction cleanup retained: '+repr(cleanup))
            raise

    def _check_open(self):
        super()._check_open()
        if self._database_identity is not None:
            path=_path(self.root/(self.prefix+'index.sqlite3'))
            _path(str(path)+'-wal');_path(str(path)+'-shm')
            metadata=path.stat()
            if ((metadata.st_dev,metadata.st_ino)!=self._database_identity or
                    self._database is None or
                    self._database.execute('PRAGMA data_version').fetchone()[0]!=self._database_version):
                raise RuntimeError('owned continuation index changed outside its publisher')

    def _index(self,path,value):
        self._database.execute('INSERT INTO records(name,cache,bytes,created,receipt) '
            'VALUES(?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET cache=excluded.cache, '
            'bytes=excluded.bytes,created=excluded.created,receipt=excluded.receipt',
            (path.name,value['cache'],value['bytes'],value['created'],
            json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)))

    def rebuild(self):
        """Reconstruct metadata from complete published records, without replay."""
        with self._lock:
            self._check_open();self._index_dirty=True
            self._database.execute('BEGIN IMMEDIATE')
            try:
                self._database.execute('DELETE FROM records')
                count=0
                for path in self.root.glob(self.prefix+'*.json'):
                    count+=1
                    if count>100000: raise MemoryError('continuation rebuild exceeds catalog admission')
                    value=self._read(path)
                    cache=self.root/value['cache']
                    if cache.is_symlink() or not cache.is_file() or cache.stat().st_size!=value['bytes']:
                        raise ValueError('published continuation cache is missing or changed')
                    self._index(path,value)
                self._database.execute('COMMIT')
            except BaseException as primary:
                try:self._database.execute('ROLLBACK')
                except BaseException as cleanup:
                    primary.add_note('continuation index rollback failed; rebuild remains required: '+repr(cleanup))
                raise
            self._index_dirty=False
            return dict(entries=count,rebuild_required=False)

    def _ensure_index(self):
        if self._index_dirty: self.rebuild()

    def _publish(self,path,value):
        # Keep the authoritative receipt even if the index update fails.
        super()._publish(path,value)
        try: self._index(path,value)
        except BaseException:
            self._index_dirty=True;raise

    def _remove(self,path,record):
        self._index_dirty=True
        # Removing a receipt first makes any surviving immutable frame an
        # orphan, so an interrupted eviction cannot resurrect an old session.
        super()._remove(path,record)
        self._database.execute('DELETE FROM records WHERE name=?',(path.name,))
        self._index_dirty=False

    def prune(self,*,reserve_bytes=0,reserve_entries=0,_protected=()):
        with self._lock:
            self._check_open();self._ensure_index()
            if type(reserve_bytes) is not int or not 0<=reserve_bytes<=self.total_limit or type(reserve_entries) is not int or not 0<=reserve_entries<=self.max_entries:
                raise ValueError('invalid indexed continuation reservation')
            protected=frozenset(path.name for path in _protected)
            placeholders=','.join('?' for _ in protected)
            if self.expiry is not None:
                cutoff=time.time()-self.expiry
                query='SELECT name,receipt FROM records WHERE created<=?'
                if protected: query+=' AND name NOT IN ('+placeholders+')'
                query+=' ORDER BY created,name LIMIT 1'
                while True:
                    row=self._database.execute(query,(cutoff,*protected)).fetchone()
                    if row is None: break
                    name,encoded=row;value=self._read(self.root/name)
                    if value!=json.loads(encoded):
                        self._index_dirty=True;raise ValueError('continuation index differs from published record')
                    self._remove(self.root/name,value)
            count,total=self._database.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM records').fetchone()
            while count+reserve_entries>self.max_entries or total+reserve_bytes>self.total_limit:
                # Protected replacement receipts are a bounded caller inventory.
                query='SELECT name,receipt FROM records'
                if protected: query+=' WHERE name NOT IN ('+placeholders+')'
                query+=' ORDER BY created,name LIMIT 1'
                row=self._database.execute(query,tuple(protected)).fetchone()
                if row is None: raise MemoryError('durable replacement requires old and staged extents together')
                name,encoded=row;value=self._read(self.root/name)
                if value!=json.loads(encoded):
                    self._index_dirty=True;raise ValueError('continuation index differs from published record')
                self._remove(self.root/name,value);count-=1;total-=value['bytes']
            return dict(entries=count,bytes=total)

    def collect_orphans(self):
        """Explicit startup/maintenance reclamation under exclusive ownership."""
        with self._lock:
            self._check_open();self._ensure_index();removed=0;reclaimed=0;count=0
            for path in self.root.glob(self.prefix+'*.bin'):
                count+=1
                if count>200000: raise MemoryError('orphan inventory exceeds scan admission')
                path=self._owned_path(path,cache=True)
                if self._database.execute('SELECT 1 FROM records WHERE cache=?',(path.name,)).fetchone() is None:
                    if path.is_symlink() or not path.is_file(): raise ValueError('regular orphan cache frame required')
                    extent=path.stat().st_size;path.unlink();removed+=1;reclaimed+=extent
            self._sync_directory()
            return dict(removed=removed,reclaimed_bytes=reclaimed)

    def save(self,service,session_id,key,*,evict=False):
        with self._lock:
            self._check_open();self._ensure_index()
            path=self._record_path(key)
            previous=self._read(path) if path.exists() else None
            try:
                receipt=super().save(service,session_id,key,evict=evict)
            except BaseException as error:
                try: self.collect_orphans()
                except BaseException as cleanup:
                    error.add_note('orphan continuation reclamation remains pending: '+str(cleanup))
                raise
            # The previous frame has crossed the publication boundary and can
            # now be reclaimed without a directory-wide orphan scan.
            if previous is not None and previous['cache']!=receipt['cache']:
                self._owned_path(self.root/previous['cache'],cache=True).unlink(missing_ok=True)
                self._sync_directory()
            return receipt

    def status(self):
        with self._lock:
            self._check_open();self._ensure_index()
            count,total=self._database.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM records').fetchone()
            return dict(identity=self.identity,entries=count,bytes=total,max_entries=self.max_entries,
                max_bytes=self.total_limit,index_rebuild_required=self._index_dirty)

    def close(self):
        with self._lock:
            if self._database is not None:
                self._database.close();self._database=None
            super().close()


__all__=['IndexedContinuationStore']
