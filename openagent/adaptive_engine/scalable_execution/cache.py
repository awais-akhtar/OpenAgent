"""Payload-bearing immutable KV pages with atomic SQLite WAL transactions."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import threading
import uuid

from .resources import Budget, LeasePool, check_cancel, positive
from ..session_recovery import _path
from ..native_library_identity import _stamp
from ..directory_ownership import DirectoryOwnership


def _tokens(values, limit):
    result = []
    for value in values:
        if len(result) >= limit:
            raise ValueError("token sequence admission exceeded")
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError("tokens must be nonnegative signed-64-bit integers")
        result.append(value)
    return tuple(result), b"".join(struct.pack("<q", value) for value in result)


@dataclass(frozen=True)
class CachePageMetadata:
    ordinal: int
    position: int
    token_ids: tuple[int, ...]
    payload_bytes: int


class DurablePagedCache:
    """One process owns a cache directory; threads serialize write transactions.

    SQLite FULL synchronous WAL is the atomic recovery journal. Session/page
    refs and data commit together. OS/SQLite buffers are separate from explicit
    page-lease budgets; cache_size bounds SQLite's advised page cache.
    """

    def __init__(self, directory, geometry, *, capacity_bytes, page_tokens,
                 read_budget, max_sessions=1024, max_prefixes=1024,
                 journal_records=4096, sqlite_cache_bytes=8 * 1024 * 1024):
        self.geometry = geometry
        self.capacity_bytes = positive(capacity_bytes, "capacity_bytes")
        self.page_tokens = positive(page_tokens, "page_tokens")
        self.max_sessions = positive(max_sessions, "max_sessions")
        self.max_prefixes = positive(max_prefixes, "max_prefixes")
        self.journal_records = positive(journal_records, "journal_records")
        positive(sqlite_cache_bytes, "sqlite_cache_bytes")
        if not isinstance(read_budget, Budget):
            raise TypeError("durable cache requires an explicit read Budget")
        self.read_budget = read_budget
        self._pool = LeasePool(read_budget)
        if 2 * self.page_tokens * geometry.bytes_per_token > read_budget.capacity_bytes:
            raise MemoryError("one cache page plus read/bind copy exceeds page budget")
        self.directory = _path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active = False
        self._closed = False
        self._closing = False
        self._quarantined = False
        self._db = None
        self._ownership = None
        self._database_identity = None
        self._data_version = None
        try:
            self._ownership=DirectoryOwnership(self.directory,filename='owner.lock')
            for filename in ('cache.sqlite','cache.sqlite-wal','cache.sqlite-shm'):
                _path(self.directory/filename)
            self._db = sqlite3.connect(self.directory / "cache.sqlite", isolation_level=None,
                                       check_same_thread=False, uri=True)
            info=_path(self.directory/'cache.sqlite').stat()
            self._database_identity=(info.st_dev,info.st_ino)
            self._db.execute("PRAGMA foreign_keys=ON")
            if self._db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise RuntimeError("cache filesystem does not support SQLite WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute(f"PRAGMA cache_size=-{max(1, sqlite_cache_bytes // 1024)}")
            self._db.execute("PRAGMA mmap_size=0")
            self._db.execute("PRAGMA wal_autocheckpoint=256")
            self._db.execute("PRAGMA journal_size_limit=1048576")
            max_blob = self._db.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
            if self.page_tokens * (geometry.bytes_per_token + 8) + 1024 > max_blob:
                raise ValueError("declared cache page exceeds SQLite record geometry")
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS contract (id INTEGER PRIMARY KEY, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS pages (
                    id TEXT PRIMARY KEY, tokens BLOB NOT NULL, count INTEGER NOT NULL,
                    payload BLOB NOT NULL, digest TEXT NOT NULL, chain TEXT NOT NULL,
                    refs INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY, payload_bytes INTEGER NOT NULL);
                INSERT OR IGNORE INTO usage VALUES(1,0);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, position INTEGER NOT NULL,
                    chain TEXT NOT NULL, revision INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS session_pages (
                    session TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                    ordinal INTEGER, page TEXT REFERENCES pages(id),
                    PRIMARY KEY(session, ordinal));
                CREATE TABLE IF NOT EXISTS prefixes (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, position INTEGER NOT NULL,
                    chain TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS prefix_pages (
                    prefix TEXT REFERENCES prefixes(id) ON DELETE CASCADE,
                    ordinal INTEGER, page TEXT REFERENCES pages(id),
                    PRIMARY KEY(prefix, ordinal));
                CREATE TABLE IF NOT EXISTS journal (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL,
                    session TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS unreferenced_pages ON pages(refs);
                CREATE INDEX IF NOT EXISTS session_page_refs ON session_pages(page);
                CREATE INDEX IF NOT EXISTS prefix_page_refs ON prefix_pages(page);
                CREATE TRIGGER IF NOT EXISTS page_added AFTER INSERT ON pages BEGIN
                    UPDATE usage SET payload_bytes=payload_bytes+length(NEW.payload) WHERE id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS page_removed AFTER DELETE ON pages BEGIN
                    UPDATE usage SET payload_bytes=payload_bytes-length(OLD.payload) WHERE id=1;
                END;
                CREATE TRIGGER IF NOT EXISTS session_ref_added AFTER INSERT ON session_pages BEGIN
                    UPDATE pages SET refs=refs+1 WHERE id=NEW.page;
                END;
                CREATE TRIGGER IF NOT EXISTS session_ref_removed AFTER DELETE ON session_pages BEGIN
                    UPDATE pages SET refs=refs-1 WHERE id=OLD.page;
                END;
                CREATE TRIGGER IF NOT EXISTS prefix_ref_added AFTER INSERT ON prefix_pages BEGIN
                    UPDATE pages SET refs=refs+1 WHERE id=NEW.page;
                END;
                CREATE TRIGGER IF NOT EXISTS prefix_ref_removed AFTER DELETE ON prefix_pages BEGIN
                    UPDATE pages SET refs=refs-1 WHERE id=OLD.page;
                END;
            """)
            contract = json.dumps({"version": 2, "identity": geometry.identity,
                "bytes_per_token": geometry.bytes_per_token, "page_tokens": page_tokens,
                "cache_layouts": [{key: value for key, value in asdict(layout).items()
                                  if key != "variable_size" or value}
                                 for layout in geometry.cache_layouts],
                "context_tokens": geometry.max_context_tokens}, sort_keys=True)
            row = self._db.execute("SELECT body FROM contract WHERE id=1").fetchone()
            if row is None:
                self._db.execute("INSERT INTO contract VALUES(1, ?)", (contract,))
            elif row[0] != contract:
                raise ValueError("durable cache backend/layout identity mismatch")
            if self._used() > self.capacity_bytes:
                raise MemoryError("recovered cache exceeds configured payload budget")
            if self._db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] > max_sessions:
                raise MemoryError("recovered session count exceeds admission limit")
            if self._db.execute("SELECT COUNT(*) FROM prefixes").fetchone()[0] > max_prefixes:
                raise MemoryError("recovered prefix count exceeds admission limit")
            self._validate_metadata()
            self._data_version=self._db.execute('PRAGMA data_version').fetchone()[0]
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:
                primary.cleanup=self.close
                primary.add_note('paged cache construction cleanup retained: '+repr(cleanup))
            raise

    def _unlock(self):
        if self._ownership is not None:
            self._ownership.close();self._ownership=None

    def _check(self):
        if self._closed or self._closing:
            raise RuntimeError("cache closed")
        if self._quarantined:
            raise RuntimeError('paged cache transaction outcome requires reopening the owner')
        self._ownership.assert_held()
        for filename in ('cache.sqlite','cache.sqlite-wal','cache.sqlite-shm'):
            _path(self.directory/filename)
        info=(self.directory/'cache.sqlite').stat()
        if ((info.st_dev,info.st_ino)!=self._database_identity or
                self._db.execute('PRAGMA data_version').fetchone()[0]!=self._data_version):
            self._quarantined=True
            raise ValueError('owned paged cache database changed outside this owner')

    def _used(self):
        row=self._db.execute("SELECT payload_bytes FROM usage WHERE id=1").fetchone()
        if row is None or type(row[0]) is not int or row[0]<0:
            raise ValueError('invalid durable cache payload accounting')
        return row[0]

    def _validate_metadata(self):
        """Reject corrupt shapes/references before exposing recovered sessions."""
        db = self._db
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("cache contains dangling page references")
        if self._used() != db.execute("SELECT COALESCE(SUM(length(payload)),0) FROM pages").fetchone()[0]:
            raise ValueError("cache payload accounting mismatch")
        if db.execute("""SELECT 1 FROM pages p WHERE typeof(id) != 'text' OR length(id)=0
            OR typeof(refs) != 'integer' OR refs < 1 OR refs !=
            (SELECT COUNT(*) FROM session_pages WHERE page=p.id) +
            (SELECT COUNT(*) FROM prefix_pages WHERE page=p.id) LIMIT 1""").fetchone():
            raise ValueError("cache page reference accounting mismatch")
        extent = "length(payload) < 1 OR length(payload) > count*?" if self.geometry.variable_cache_size else "length(payload) != count*?"
        invalid = db.execute(f"""SELECT 1 FROM pages WHERE typeof(count) != 'integer'
            OR typeof(tokens) != 'blob' OR typeof(payload) != 'blob' OR count < 1 OR count > ?
            OR length(tokens) != count*8 OR ({extent})
            OR typeof(digest) != 'text' OR typeof(chain) != 'text'
            OR length(digest) != 64 OR length(chain) != 64
            OR digest GLOB '*[^0-9a-f]*' OR chain GLOB '*[^0-9a-f]*' LIMIT 1""",
            (self.page_tokens, self.geometry.bytes_per_token)).fetchone()
        if invalid:
            raise ValueError("cache page metadata/layout mismatch")
        seed = hashlib.sha256(self.geometry.identity.encode()).hexdigest()
        for table, refs, column in (("sessions", "session_pages", "session"),
                                    ("prefixes", "prefix_pages", "prefix")):
            if db.execute(f'''SELECT 1 FROM {refs} WHERE typeof({column}) != 'text'
                    OR typeof(page) != 'text' OR typeof(ordinal) != 'integer'
                    OR ordinal < 0 LIMIT 1''').fetchone():
                raise ValueError('invalid durable cache page reference')
            revision_clause=" OR typeof(revision) != 'integer' OR revision < 0" if table=='sessions' else ''
            invalid = db.execute(f"""SELECT 1 FROM {table} s WHERE
                typeof(id) != 'text' OR length(id)=0 OR typeof(owner) != 'text' OR length(owner)=0
                OR typeof(position) != 'integer' OR typeof(chain) != 'text'
                OR length(chain) != 64 OR chain GLOB '*[^0-9a-f]*' {revision_clause}
                OR position < 0 OR position > ? OR position != (
                SELECT COALESCE(SUM(p.count),0) FROM {refs} r JOIN pages p ON p.id=r.page WHERE r.{column}=s.id)
                OR (SELECT COALESCE(MAX(ordinal),-1)+1 FROM {refs} WHERE {column}=s.id)
                    != (SELECT COUNT(*) FROM {refs} WHERE {column}=s.id)
                OR (SELECT COALESCE(MIN(ordinal),0) FROM {refs} WHERE {column}=s.id) != 0
                OR chain != COALESCE((SELECT p.chain FROM {refs} r JOIN pages p ON p.id=r.page
                    WHERE r.{column}=s.id ORDER BY ordinal DESC LIMIT 1),?) LIMIT 1""",
                (self.geometry.max_context_tokens, seed)).fetchone()
            if invalid:
                raise ValueError("cache session/prefix metadata mismatch")

    def _session(self, session_id, owner):
        row = self._db.execute("SELECT owner,position,chain,revision FROM sessions WHERE id=?",
                               (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        if row[0] != owner:
            raise PermissionError("cache session belongs to another owner")
        return row

    def _validate_payloads(self):
        """Hash restored pages with a bounded SQLite blob reader before commit."""
        capacity=min(65536,self.read_budget.capacity_bytes//2)
        if not capacity:raise MemoryError('one restored cache byte exceeds read admission')
        cursor=self._db.execute('SELECT rowid,length(payload),digest FROM pages')
        try:
            for rowid,extent,expected in cursor:
                digest=hashlib.sha256()
                with self.read_budget.reserve(2*min(capacity,extent)):
                    with self._db.blobopen('pages','payload',rowid,readonly=True) as blob:
                        remaining=extent
                        while remaining:
                            chunk=blob.read(min(capacity,remaining))
                            if not chunk:raise ValueError('restored cache blob truncated')
                            digest.update(chunk);remaining-=len(chunk)
                if digest.hexdigest()!=expected:
                    raise ValueError('restored cache page checksum mismatch')
        finally:cursor.close()
        seed=hashlib.sha256(self.geometry.identity.encode()).hexdigest()
        token_capacity=(capacity//8)*8
        if not token_capacity:
            raise MemoryError('one restored token metadata value exceeds read admission')
        for table,column in (('session_pages','session'),('prefix_pages','prefix')):
            cursor=self._db.execute(f'''SELECT r.{column},r.ordinal,p.rowid,length(p.tokens),p.digest,p.chain
                FROM {table} r JOIN pages p ON p.id=r.page ORDER BY r.{column},r.ordinal''')
            previous_owner=None;previous_chain=seed
            try:
                for owner,ordinal,rowid,extent,digest,chain in cursor:
                    if owner!=previous_owner:
                        previous_owner=owner;previous_chain=seed
                    observed=hashlib.sha256(bytes.fromhex(previous_chain))
                    with self.read_budget.reserve(2*min(token_capacity,extent)):
                        with self._db.blobopen('pages','tokens',rowid,readonly=True) as blob:
                            remaining=extent
                            while remaining:
                                chunk=blob.read(min(token_capacity,remaining))
                                if not chunk or len(chunk)%8:
                                    raise ValueError('restored cache token metadata truncated')
                                if any(value[0]<0 for value in struct.iter_unpack('<q',chunk)):
                                    raise ValueError('restored cache contains negative token metadata')
                                observed.update(chunk);remaining-=len(chunk)
                    observed.update(bytes.fromhex(digest))
                    observed=observed.hexdigest()
                    if observed!=chain:
                        raise ValueError('restored cache causal chain checksum mismatch')
                    previous_chain=chain
            finally:cursor.close()

    @contextmanager
    def _gate(self, cancelled):
        while True:
            check_cancel(cancelled)
            if self._lock.acquire(timeout=0.05):
                break
        try:
            check_cancel(cancelled)
            yield
        finally:
            self._lock.release()

    @contextmanager
    def _write(self, cancelled=None):
        with self._gate(cancelled):
            self._check()
            if self._active:
                raise RuntimeError("nested cache mutation during dispatch")
            self._active = True
            try:
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    self._check()
                    yield
                    self._db.execute("COMMIT")
                except BaseException as primary:
                    try:
                        if self._db.in_transaction:self._db.execute("ROLLBACK")
                        else:self._quarantined=True
                    except BaseException as rollback:
                        self._quarantined=True
                        primary.add_note('paged cache rollback failed; owner quarantined: '+repr(rollback))
                    raise
            finally:
                self._active = False

    def _journal(self, event, session, revision, body):
        self._db.execute("INSERT INTO journal(event,session,revision,body) VALUES(?,?,?,?)",
                         (event, session, revision, json.dumps(body, sort_keys=True)))
        self._db.execute("DELETE FROM journal WHERE seq <= (SELECT COALESCE(MAX(seq),0)-? FROM journal)",
                         (self.journal_records,))

    def _gc(self):
        self._db.execute("DELETE FROM pages WHERE refs=0")

    def create_session(self, session_id, owner, *, prefix_id=None):
        if not isinstance(session_id, str) or not session_id or not isinstance(owner, str) or not owner:
            raise ValueError("nonempty session ID and owner required")
        with self._write():
            if self._db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] >= self.max_sessions:
                raise MemoryError("session admission rejected")
            position, chain = 0, hashlib.sha256(self.geometry.identity.encode()).hexdigest()
            if prefix_id is not None:
                row = self._db.execute("SELECT owner,position,chain FROM prefixes WHERE id=?",
                                       (prefix_id,)).fetchone()
                if row is None:
                    raise KeyError(prefix_id)
                if row[0] != owner:
                    raise PermissionError("prefix belongs to another owner")
                position, chain = row[1:]
            self._db.execute("INSERT INTO sessions VALUES(?,?,?,?,0)",
                             (session_id, owner, position, chain))
            if prefix_id is not None:
                self._db.execute("INSERT INTO session_pages SELECT ?,ordinal,page FROM prefix_pages WHERE prefix=?",
                                 (session_id, prefix_id))
            self._journal("create", session_id, 0, {"prefix": prefix_id, "position": position})

    def publish_prefix(self, session_id, owner):
        """Freeze a complete committed prefix, including its causal history."""
        with self._write():
            row = self._session(session_id, owner)
            if self._db.execute("SELECT COUNT(*) FROM prefixes").fetchone()[0] >= self.max_prefixes:
                raise MemoryError("prefix admission rejected")
            prefix = uuid.uuid4().hex
            self._db.execute("INSERT INTO prefixes VALUES(?,?,?,?)", (prefix, owner, row[1], row[2]))
            self._db.execute("INSERT INTO prefix_pages SELECT ?,ordinal,page FROM session_pages WHERE session=?",
                             (prefix, session_id))
            self._journal("prefix", session_id, row[3], {"prefix": prefix})
            return prefix

    def release_prefix(self, prefix_id, owner):
        with self._write():
            row = self._db.execute("SELECT owner FROM prefixes WHERE id=?", (prefix_id,)).fetchone()
            if row is None:
                raise KeyError(prefix_id)
            if row[0] != owner:
                raise PermissionError("prefix belongs to another owner")
            self._db.execute("DELETE FROM prefixes WHERE id=?", (prefix_id,))
            self._gc()
            self._journal("release-prefix", prefix_id, 0, {})

    def evict_prefixes(self, owner, *, max_count, protected=()):
        """Evict only caller-owned, unprotected immutable prefixes.

        Eviction is explicit and deterministic: protected checkpoints and all
        live sessions remain untouched, while SQLite reference triggers retain
        pages still shared by other prefixes or sessions.  The caller controls
        fairness by supplying the same bounded candidate order on every pass;
        this method never guesses which live continuation is safe to discard.
        """
        if (not isinstance(owner, str) or not owner or type(max_count) is not int
                or max_count < 0):
            raise ValueError("bounded cache eviction owner and count required")
        protected = tuple(protected)
        if (any(type(value) is not str or not value for value in protected) or
                len(set(protected)) != len(protected)):
            raise ValueError("unique protected prefix identities required")
        with self._write():
            rows = self._db.execute(
                "SELECT id FROM prefixes WHERE owner=? ORDER BY id", (owner,)).fetchall()
            candidates = [row[0] for row in rows if row[0] not in protected]
            selected = candidates[:max_count]
            if selected:
                marks = ",".join("?" for _ in selected)
                self._db.execute(
                    f"DELETE FROM prefixes WHERE owner=? AND id IN ({marks})",
                    (owner, *selected))
                self._gc()
                self._journal("evict-prefixes", owner, 0,
                              {"prefixes": selected, "protected": list(protected)})
            return tuple(selected)

    def release_session(self, session_id, owner):
        with self._write():
            row = self._session(session_id, owner)
            self._db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._gc()
            self._journal("release", session_id, row[3], {})

    def reset(self, session_id, owner):
        with self._write():
            row = self._session(session_id, owner)
            self._db.execute("DELETE FROM session_pages WHERE session=?", (session_id,))
            chain = hashlib.sha256(self.geometry.identity.encode()).hexdigest()
            self._db.execute("UPDATE sessions SET position=0,chain=?,revision=revision+1 WHERE id=?",
                             (chain, session_id))
            self._gc()
            self._journal("reset", session_id, row[3] + 1, {})

    def restore_session(self, session_id, owner, prefix_id):
        """Atomic generation rollback to a pinned immutable session checkpoint."""
        with self._write():
            current = self._session(session_id, owner)
            prefix = self._db.execute("SELECT owner,position,chain FROM prefixes WHERE id=?",
                                      (prefix_id,)).fetchone()
            if prefix is None:
                raise KeyError(prefix_id)
            if prefix[0] != owner:
                raise PermissionError("checkpoint belongs to another owner")
            self._db.execute("DELETE FROM session_pages WHERE session=?", (session_id,))
            self._db.execute("INSERT INTO session_pages SELECT ?,ordinal,page FROM prefix_pages WHERE prefix=?",
                             (session_id, prefix_id))
            self._db.execute("UPDATE sessions SET position=?,chain=?,revision=revision+1 WHERE id=?",
                             (prefix[1], prefix[2], session_id))
            self._gc()
            self._journal("restore-session", session_id, current[3] + 1,
                          {"prefix": prefix_id, "position": prefix[1]})

    def sessions(self, owner):
        """Recovered committed sessions only; no uncommitted tail is exposed."""
        with self._lock:
            self._check()
            if self._active:
                raise RuntimeError("session listing during an active transaction")
            return [dict(id=row[0], position=row[1], revision=row[2]) for row in
                    self._db.execute("SELECT id,position,revision FROM sessions WHERE owner=? ORDER BY id", (owner,))]

    @contextmanager
    def transaction(self, session_id, owner, token_ids, *, cancelled=None):
        expected, _ = _tokens(token_ids, self.geometry.max_batch_tokens)
        if not expected:
            raise ValueError("empty cache transaction")
        with self._write(cancelled=cancelled):
            check_cancel(cancelled)
            row = self._session(session_id, owner)
            if len(expected) > self.geometry.max_batch_tokens or row[1] + len(expected) > self.geometry.max_context_tokens:
                raise ValueError("cache context or batch capacity exceeded")
            tx = CacheTransaction(self, session_id, row, expected, cancelled)
            try:
                yield tx
                tx._finish()
                check_cancel(cancelled)
            finally:
                tx._open = False

    def snapshot(self, destination):
        """Atomically publish a complete SQLite backup, without a full RAM copy.

        Destination must be outside the live cache directory. On Windows the
        file is fsynced but directory-entry power-loss durability is OS-owned.
        """
        target = _path(destination).resolve()
        if target.is_relative_to(self.directory):
            raise ValueError("snapshot destination must be outside live cache directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
        with self._lock:
            self._check()
            if self._active:
                raise RuntimeError("cannot snapshot an active transaction")
            output = None
            def cleanup_snapshot():
                nonlocal output
                if output is not None:
                    output.close();output=None
                temporary.unlink(missing_ok=True)
            try:
                output = sqlite3.connect(temporary)
                self._db.backup(output, pages=64)
                output.close();output=None
                with temporary.open("rb") as source:
                    os.fsync(source.fileno())
                os.replace(temporary, target)
                if os.name != "nt":
                    fd = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            except BaseException as primary:
                try:cleanup_snapshot()
                except BaseException as cleanup:
                    primary.cleanup=cleanup_snapshot
                    primary.add_note('cache snapshot staging cleanup retained: '+repr(cleanup))
                raise
            else:
                try:cleanup_snapshot()
                except BaseException as cleanup:
                    cleanup.cleanup=cleanup_snapshot
                    raise
        return target

    def restore_snapshot(self, source):
        """Restore into an empty live cache only; never overwrite live owners."""
        path = _path(source).resolve(strict=True)
        if path.is_relative_to(self.directory):
            raise ValueError("snapshot source must be outside live cache directory")
        if not path.is_file():raise ValueError('regular standalone cache snapshot required')
        source_stamp=_stamp(path.stat())
        with self._lock:
            self._check()
            if self._active or self._db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] or self._db.execute("SELECT COUNT(*) FROM prefixes").fetchone()[0]:
                raise RuntimeError("restore requires empty inactive cache")
            # Attach as read-only and copy inside one WAL transaction. Do not
            # replace an open SQLite database or lose rollback on invalid input.
            uri = path.as_uri() + "?mode=ro&immutable=1"
            self._db.execute("ATTACH DATABASE ? AS recovery", (uri,))
            primary=None
            try:
                expected = self._db.execute("SELECT body FROM contract WHERE id=1").fetchone()
                if self._db.execute("SELECT body FROM recovery.contract WHERE id=1").fetchone() != expected:
                    raise ValueError("snapshot identity/layout mismatch")
                used = self._db.execute("SELECT COALESCE(SUM(length(payload)),0) FROM recovery.pages").fetchone()[0]
                sessions = self._db.execute("SELECT COUNT(*) FROM recovery.sessions").fetchone()[0]
                prefixes = self._db.execute("SELECT COUNT(*) FROM recovery.prefixes").fetchone()[0]
                if used > self.capacity_bytes or sessions > self.max_sessions or prefixes > self.max_prefixes:
                    raise MemoryError("snapshot exceeds admission budgets")
                with self._write():
                    self._db.execute("DELETE FROM journal")
                    self._db.execute("DELETE FROM pages")
                    self._db.execute("""INSERT INTO pages(id,tokens,count,payload,digest,chain)
                        SELECT id,tokens,count,payload,digest,chain FROM recovery.pages""")
                    for table in ("sessions", "session_pages", "prefixes", "prefix_pages"):
                        self._db.execute(f"INSERT INTO {table} SELECT * FROM recovery.{table}")
                    self._validate_metadata()
                    self._validate_payloads()
                    if _stamp(_path(path).stat())!=source_stamp:
                        raise ValueError('cache snapshot source changed during restoration')
                    self._journal("restore", "", 0, {"sessions": sessions})
            except BaseException as failure:
                primary=failure
                raise
            finally:
                try:self._db.execute("DETACH DATABASE recovery")
                except BaseException as cleanup:
                    self._quarantined=True
                    if primary is None:raise
                    primary.add_note('cache recovery database detach failed; owner quarantined: '+repr(cleanup))

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._active:
                raise RuntimeError("cannot close cache during a transaction")
            self._closing=True
            self._pool.close()
            if self._db is not None:
                self._db.close();self._db=None
            self._unlock()
            self._closed = True


class CacheTransaction:
    """Append-only COW cache view. Existing pages are read-only and shared."""

    def __init__(self, cache, session, row, expected, cancelled):
        self._cache, self._session = cache, session
        self.position, self.chain, self.revision = row[1:]
        self._expected, self._cancelled = expected, cancelled
        self._consumed = 0
        self._open = True
        self._leases = 0
        self._thread = threading.get_ident()

    def _check(self):
        if not self._open:
            raise RuntimeError("cache transaction expired")
        if threading.get_ident() != self._thread:
            raise RuntimeError("cache transaction belongs to another execution thread")
        check_cancel(self._cancelled)

    def page_count(self):
        self._check()
        return self._cache._db.execute("SELECT COUNT(*) FROM session_pages WHERE session=?", (self._session,)).fetchone()[0]

    @property
    def page_tokens(self):
        self._check()
        return self._cache.page_tokens

    def page_metadata(self):
        """Stream descriptors without retaining cache payloads in RAM."""
        self._check()
        position = 0
        cursor = self._cache._db.execute("""SELECT s.ordinal,p.tokens,p.count,length(p.payload)
            FROM session_pages s JOIN pages p ON p.id=s.page
            WHERE s.session=? ORDER BY s.ordinal""", (self._session,))
        try:
            for ordinal, encoded, count, size in cursor:
                self._check()
                if len(encoded) != count * 8 or not 1 <= count <= self._cache.page_tokens:
                    raise ValueError("cache token metadata mismatch")
                tokens = tuple(value[0] for value in struct.iter_unpack("<q", encoded))
                if any(value < 0 for value in tokens):
                    raise ValueError("cache contains negative token metadata")
                yield CachePageMetadata(ordinal, position, tokens, size)
                position += count
        finally:
            cursor.close()

    @contextmanager
    def read_page(self, ordinal):
        self._check()
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("page ordinal must be nonnegative")
        db = self._cache._db
        row = db.execute("""SELECT p.id,length(p.payload),p.digest,p.count,p.tokens,p.chain FROM pages p
            JOIN session_pages s ON s.page=p.id WHERE s.session=? AND s.ordinal=?""",
            (self._session, ordinal)).fetchone()
        if row is None:
            raise IndexError(ordinal)
        if not 1 <= row[3] <= self._cache.page_tokens or len(row[4]) != row[3] * 8:
            raise ValueError("cache page token extent mismatch")
        self._cache.geometry.admit_cache_payload(row[3], row[1])
        def load():
            return bytearray(db.execute("SELECT payload FROM pages WHERE id=?", (row[0],)).fetchone()[0])
        with self._cache._pool.acquire(2 * row[1], load) as payload:
            if hashlib.sha256(payload).hexdigest() != row[2]:
                raise ValueError("durable cache page checksum mismatch")
            previous = db.execute("""SELECT p.chain FROM pages p JOIN session_pages s ON s.page=p.id
                WHERE s.session=? AND s.ordinal=?""", (self._session, ordinal - 1)).fetchone()
            chain = previous[0] if previous else hashlib.sha256(self._cache.geometry.identity.encode()).hexdigest()
            if hashlib.sha256(bytes.fromhex(chain) + row[4] + bytes.fromhex(row[2])).hexdigest() != row[5]:
                raise ValueError("durable cache causal chain checksum mismatch")
            self._leases += 1
            try:
                with memoryview(payload) as base, base.toreadonly() as view:
                    yield view
            finally:
                self._leases -= 1
                del payload

    def append_page(self, token_ids, payload):
        self._check()
        tokens, encoded = _tokens(token_ids, self._cache.page_tokens)
        cache = self._cache
        if not tokens or len(tokens) > cache.page_tokens:
            raise ValueError("append must contain 1..page_tokens tokens")
        if tokens != self._expected[self._consumed:self._consumed + len(tokens)]:
            raise ValueError("cache append differs from admitted token sequence")
        with memoryview(payload) as view:
            if (not view.c_contiguous or view.format not in ('B', 'b', 'c') or
                    view.itemsize != 1):
                raise ValueError("cache page must be contiguous")
            cache.geometry.admit_cache_payload(len(tokens), view.nbytes)
            if view.nbytes > cache.capacity_bytes - cache._used():
                raise MemoryError("durable cache payload admission rejected")
            digest = hashlib.sha256(view).hexdigest()
            chain = hashlib.sha256(bytes.fromhex(self.chain) + encoded + bytes.fromhex(digest)).hexdigest()
            page = uuid.uuid4().hex
            ordinal = self.page_count()
            with cache.read_budget.reserve(2 * view.nbytes):
                cache._db.execute("INSERT INTO pages(id,tokens,count,payload,digest,chain) VALUES(?,?,?,?,?,?)",
                    (page, encoded, len(tokens), view, digest, chain))
            cache._db.execute("INSERT INTO session_pages VALUES(?,?,?)", (self._session, ordinal, page))
        self._consumed += len(tokens)
        self.chain = chain

    def _finish(self):
        self._check()
        if self._leases:
            raise RuntimeError("cache page lease escaped callback")
        if self._consumed != len(self._expected):
            raise ValueError("backend did not append all admitted cache tokens")
        self._cache._db.execute("UPDATE sessions SET position=?,chain=?,revision=revision+1 WHERE id=?",
            (self.position + self._consumed, self.chain, self._session))
        self._cache._journal("append", self._session, self.revision + 1,
            {"position": self.position + self._consumed, "chain": self.chain})
