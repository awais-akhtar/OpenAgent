"""Single-session paged recovery with bounded RAM and transactional publication."""
import hashlib
import hmac
import struct
import tempfile
import uuid

from .cache import CacheTransaction
from .resources import check_cancel

_HEADER = struct.Struct('<8s32sQQ32s')
_PAGE = struct.Struct('<QQ')
_MAGIC = b'OASPG001'


def _write(stream, value, cancelled):
    view = memoryview(value)
    while view:
        check_cancel(cancelled)
        count = stream.write(view)
        if type(count) is not int or not 0 < count <= len(view):
            raise OSError('paged checkpoint writer made invalid progress')
        view = view[count:]


def _read(stream, count, cancelled):
    result = bytearray()
    while len(result) < count:
        check_cancel(cancelled)
        part = stream.read(count-len(result))
        if type(part) is not bytes or not part or len(part) > count-len(result):
            raise ValueError('truncated or oversized paged checkpoint read')
        result.extend(part)
    return bytes(result)


def transport(cache, session, owner, stream, *, max_bytes, restoring=False,
              cancelled=None, vocabulary=None):
    if type(max_bytes) is not int or max_bytes < _HEADER.size+32:
        raise ValueError('explicit paged checkpoint frame budget required')
    if vocabulary is not None and (type(vocabulary) is not int or vocabulary <= 0):
        raise ValueError('positive paged checkpoint vocabulary required')
    # The live contract includes the backend, complete layouts and page size.
    with cache._write(cancelled=cancelled):
        current = cache._session(session, owner)
        contract = cache._db.execute('SELECT body FROM contract WHERE id=1').fetchone()[0]
        identity = hashlib.sha256(contract.encode()).digest()
        chunk = min(65536, cache.read_budget.capacity_bytes//16)
        if chunk < 64:
            raise MemoryError('paged checkpoint transport buffer admission rejected')
        if not restoring:
            tx = CacheTransaction(cache, session, current, (), cancelled)
            count = tx.page_count()
            payload = cache._db.execute('''SELECT COALESCE(SUM(length(p.payload)),0),
                COALESCE(SUM(length(p.tokens)),0) FROM pages p JOIN session_pages s
                ON s.page=p.id WHERE s.session=?''', (session,)).fetchone()
            size = _HEADER.size+count*_PAGE.size+sum(payload)+32
            if size > max_bytes:
                raise MemoryError('paged checkpoint exceeds frame admission')
            header = _HEADER.pack(_MAGIC, identity, current[1], count, bytes.fromhex(current[2]))
            digest = hashlib.sha256(header)
            consumed = 0
            written_pages = 0
            try:
                _write(stream, header, cancelled)
                for metadata in tx.page_metadata():
                    if metadata.ordinal != written_pages or metadata.position != consumed:
                        raise ValueError('noncontiguous paged checkpoint history')
                    with cache.read_budget.reserve(len(metadata.token_ids)*96+4096):
                        if vocabulary is not None and any(token >= vocabulary for token in metadata.token_ids):
                            raise ValueError('paged checkpoint token outside vocabulary')
                        tokens = b''.join(struct.pack('<q', token) for token in metadata.token_ids)
                        descriptor = _PAGE.pack(len(metadata.token_ids), metadata.payload_bytes)
                        for value in (descriptor, tokens):
                            digest.update(value)
                            _write(stream, value, cancelled)
                        with tx.read_page(metadata.ordinal) as data:
                            for offset in range(0, len(data), chunk):
                                part = data[offset:offset+chunk]
                                try:
                                    digest.update(part)
                                    _write(stream, part, cancelled)
                                finally:
                                    part.release()
                    consumed += len(metadata.token_ids)
                    written_pages += 1
                last = cache._db.execute('''SELECT p.chain FROM pages p JOIN session_pages s
                    ON s.page=p.id WHERE s.session=? ORDER BY s.ordinal DESC LIMIT 1''',
                    (session,)).fetchone()
                final = last[0] if last is not None else hashlib.sha256(cache.geometry.identity.encode()).hexdigest()
                if consumed != current[1] or written_pages != count or final != current[2]:
                    raise ValueError('paged checkpoint final history boundary mismatch')
                _write(stream, digest.digest(), cancelled)
                check_cancel(cancelled)
                return size
            finally:
                tx._open = False
        # Stage and authenticate the complete envelope before changing page refs.
        with cache.read_budget.reserve(chunk*8), tempfile.TemporaryFile() as staged:
            size = 0
            while True:
                check_cancel(cancelled)
                requested = min(chunk, max_bytes-size+1)
                part = stream.read(requested)
                if type(part) is not bytes or len(part) > requested:
                    raise ValueError('binary bounded paged checkpoint source required')
                if not part:
                    break
                size += len(part)
                if size > max_bytes:
                    raise MemoryError('paged checkpoint exceeds frame admission')
                _write(staged, part, cancelled)
            if size < _HEADER.size+32:
                raise ValueError('truncated paged checkpoint envelope')
            staged.seek(0)
            digest = hashlib.sha256()
            remaining = size-32
            while remaining:
                part = _read(staged, min(chunk, remaining), cancelled)
                digest.update(part)
                remaining -= len(part)
            if not hmac.compare_digest(_read(staged, 32, cancelled), digest.digest()):
                raise ValueError('paged checkpoint envelope checksum mismatch')
            staged.seek(0)
            magic, observed, position, pages, final_chain = _HEADER.unpack(
                _read(staged, _HEADER.size, cancelled))
            if (magic != _MAGIC or observed != identity or
                    position > cache.geometry.max_context_tokens or pages > position or
                    (position == 0) != (pages == 0)):
                raise ValueError('paged checkpoint identity or context mismatch')
            cache._db.execute('DELETE FROM session_pages WHERE session=?', (session,))
            cache._gc()
            chain = hashlib.sha256(cache.geometry.identity.encode()).digest()
            consumed = 0
            for ordinal in range(pages):
                count, extent = _PAGE.unpack(_read(staged, _PAGE.size, cancelled))
                if not 0 < count <= cache.page_tokens or consumed+count > position:
                    raise ValueError('paged checkpoint token extent mismatch')
                cache.geometry.admit_cache_payload(count, extent)
                if staged.tell()+count*8+extent > size-32:
                    raise ValueError('paged checkpoint page exceeds authenticated extent')
                with cache.read_budget.reserve(count*32):
                    encoded = _read(staged, count*8, cancelled)
                    if any(token[0] < 0 or (vocabulary is not None and token[0] >= vocabulary)
                           for token in struct.iter_unpack('<q', encoded)):
                        raise ValueError('paged checkpoint token outside admitted vocabulary')
                    if extent > cache.capacity_bytes-cache._used():
                        raise MemoryError('restored paged checkpoint exceeds durable capacity')
                    page = uuid.uuid4().hex
                    cursor = cache._db.execute('''INSERT INTO pages
                        (id,tokens,count,payload,digest,chain) VALUES(?,?,?,zeroblob(?),'','')''',
                        (page, encoded, count, extent))
                    page_digest = hashlib.sha256()
                    with cache._db.blobopen('pages', 'payload', cursor.lastrowid) as blob:
                        remaining = extent
                        while remaining:
                            part = _read(staged, min(chunk, remaining), cancelled)
                            page_digest.update(part)
                            blob.write(part)
                            remaining -= len(part)
                    hashed = page_digest.digest()
                    chain = hashlib.sha256(chain+encoded+hashed).digest()
                    cache._db.execute('UPDATE pages SET digest=?,chain=? WHERE id=?',
                                      (hashed.hex(), chain.hex(), page))
                    cache._db.execute('INSERT INTO session_pages VALUES(?,?,?)',
                                      (session, ordinal, page))
                    consumed += count
            if consumed != position or staged.tell() != size-32 or chain != final_chain:
                raise ValueError('paged checkpoint causal chain or final extent mismatch')
            check_cancel(cancelled)
            cache._db.execute('UPDATE sessions SET position=?,chain=?,revision=revision+1 WHERE id=?',
                              (position, chain.hex(), session))
            cache._journal('restore-stream', session, current[3]+1, {'position': position})
            return position
