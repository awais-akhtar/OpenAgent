"""Owned full continuation transport for GLM5-next and Inkling native sessions.

Stores the actual native buffers including FIR histories and recurrent state.
Restore stages every bank before publishing; it never reconstructs by replay.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import struct
import threading
from uuid import uuid4

import numpy as np

from .segment_snapshot import _read, _write, _pairs

_MAGIC = b'OAFNS001'


@dataclass(frozen=True)
class FlatNativeSnapshot:
    owner: str
    identity: str
    key: str
    position: int


class FlatNativeSession:
    def __init__(self, decoder, *, family, max_bytes, max_snapshots=4):
        from .native_glm5_next import NativeGlm5NextTextDecoder
        from .native_inkling import NativeInklingDecoder
        types = {'glm5_next_text':NativeGlm5NextTextDecoder,'inkling_text':NativeInklingDecoder}
        if family not in types or type(decoder) is not types[family]:
            raise ValueError('explicit supported flat native decoder required')
        if type(max_bytes) is not int or max_bytes<=0 or type(max_snapshots) is not int or not 1<=max_snapshots<=64:
            raise ValueError('positive continuation budget and bounded snapshot inventory required')
        self.decoder,self.family,self.max_bytes,self.max_snapshots = decoder,family,max_bytes,max_snapshots
        self._lock,self._local = threading.RLock(),threading.local()
        self._owner,self._snapshots = uuid4().hex,{}
        self._snapshot_bytes = 0
        self._closed,self._busy = False,False
        if family=='glm5_next_text':
            self.shapes = tuple(tuple(tuple(shape) for shape in pair) for pair in decoder._cache_shapes)
            self.capacity,self.vocabulary = decoder.max_tokens,decoder.config.vocab_size
        else:
            self.shapes = tuple(((1,decoder._state_size(g)),) for g in decoder.geometry)
            self.capacity = decoder.config['max_position_embeddings']
            self.vocabulary = decoder.config['unpadded_vocab_size'] or decoder.config['vocab_size']
        self._full_state_bytes = 4*sum(int(np.prod(shape,dtype=object)) for bank in self.shapes for shape in bank)
        self._weights = sum(a.nbytes for a in decoder.weights.values())
        digest = hashlib.sha256(repr((family,self.shapes,self.capacity,self.vocabulary,decoder.config)).encode())
        provider=getattr(decoder,'_expert_provider',None)
        if provider is not None: digest.update(provider.identity.encode())
        for name in sorted(decoder.weights):
            array = decoder.weights[name]
            digest.update(repr((name,array.shape,array.dtype.str)).encode())
            digest.update(memoryview(array).cast('B'))
        decoder._library_identity.assert_unchanged()
        digest.update(bytes.fromhex(decoder._library_identity.sha256))
        self.identity = 'oa-flat-native-v1:'+digest.hexdigest()
        self._admit(self._full_state_bytes)

    @property
    def position(self):
        return self.decoder.position

    def _check(self, *, check_source=True):
        if self._closed or self._busy:
            raise RuntimeError('flat native session closed or active')
        if check_source:
            self.decoder._library_identity.assert_unchanged()

    def _admit(self, staging):
        base=max(self._weights+self._full_state_bytes,
            getattr(self.decoder,'_resident_bytes',0)+getattr(self.decoder,'_metadata_bytes',0))
        if base+self._snapshot_bytes+staging>self.max_bytes:
            raise MemoryError('flat native state ownership exceeds admission')

    def _cancel(self):
        cancel = getattr(self._local,'cancel',None)
        if cancel is not None and cancel.is_set():
            from .execution_service import ExecutionCancelled
            raise ExecutionCancelled('flat native execution cancelled; continuation unchanged')
        return False

    @contextmanager
    def cancellation_scope(self, cancel):
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise TypeError('cancellation event required')
        previous = getattr(self._local,'cancel',None)
        self._local.cancel = cancel
        try:
            yield
        finally:
            self._local.cancel = previous

    def forward(self, tokens):
        with self._lock:
            self._check()
            ids = np.asarray(tokens)
            if ids.ndim!=1 or ids.dtype.kind not in 'iu' or not 0<ids.size<=4096:
                raise ValueError('bounded integer token sequence required')
            self._admit(self._full_state_bytes+ids.size*(self.vocabulary*8+16))
            self._busy = True
            try:
                self._cancel()
                result = self.decoder.forward(ids[None,:],cancelled=self._cancel)
                logits = result['logits'] if isinstance(result,dict) else result.logits
                if logits.shape!=(1,ids.size,self.vocabulary) or not np.isfinite(logits).all():
                    raise ArithmeticError('invalid full native logit output')
                return logits[0]
            finally:
                self._busy = False

    def _capture(self):
        state = self.decoder._cache
        if state is None:
            if self.position!=0:
                raise ValueError('nonempty continuation lacks actual native state')
            return None
        banks = state if self.family=='glm5_next_text' else tuple((a,) for a in state)
        if len(banks)!=len(self.shapes):
            raise ValueError('native continuation bank inventory mismatch')
        copied = []
        for bank,shapes in zip(banks,self.shapes):
            if len(bank)!=len(shapes):
                raise ValueError('native continuation array inventory mismatch')
            row = []
            for array,shape in zip(bank,shapes):
                if array.dtype!=np.float32 or array.shape!=shape or not np.isfinite(array).all():
                    raise ValueError('native continuation geometry or finite boundary mismatch')
                row.append(array.astype('<f4',copy=False).tobytes(order='C'))
            copied.append(tuple(row))
        return tuple(copied)

    def snapshot(self):
        with self._lock:
            self._check()
            if len(self._snapshots)>=self.max_snapshots:
                raise MemoryError('flat native snapshot inventory exhausted')
            self._admit(self._full_state_bytes)
            banks = self._capture()
            size = 0 if banks is None else self._full_state_bytes
            handle = FlatNativeSnapshot(self._owner,self.identity,uuid4().hex,self.position)
            self._snapshots[handle.key] = (handle,banks,size)
            self._snapshot_bytes += size
            return handle

    def _owned(self, handle):
        if (type(handle) is not FlatNativeSnapshot or handle.owner!=self._owner or
                handle.identity!=self.identity or handle.key not in self._snapshots or
                self._snapshots[handle.key][0] is not handle):
            raise ValueError('foreign, stale or released native snapshot')
        return self._snapshots[handle.key]

    def _stage(self, banks, position):
        if type(position) is not int or not 0<=position<=self.capacity:
            raise ValueError('native continuation position outside capacity')
        if banks is None:
            if position!=0:
                raise ValueError('missing native state for nonempty continuation')
            return None
        if type(banks) is not tuple or len(banks)!=len(self.shapes):
            raise ValueError('incomplete native continuation banks')
        self._admit(self._full_state_bytes)
        staged = []
        for bank,shapes in zip(banks,self.shapes):
            if type(bank) is not tuple or len(bank)!=len(shapes):
                raise ValueError('incomplete native continuation arrays')
            row = []
            for raw,shape in zip(bank,shapes):
                size = 4*int(np.prod(shape,dtype=object))
                if type(raw) is not bytes or len(raw)!=size:
                    raise ValueError('native continuation byte extent mismatch')
                array = np.frombuffer(raw,dtype='<f4').reshape(shape)
                if not np.isfinite(array).all():
                    raise ValueError('nonfinite native continuation')
                row.append(np.array(array,dtype=np.float32,order='C',copy=True))
            staged.append(tuple(row))
        return tuple(staged) if self.family=='glm5_next_text' else tuple(row[0] for row in staged)

    def restore(self, handle):
        with self._lock:
            self._check()
            _,banks,_ = self._owned(handle)
            self._busy = True
            try:
                staged = self._stage(banks,handle.position)
                self._cancel()
                self.decoder._cache,self.decoder.position = staged,handle.position
            finally:
                self._busy = False

    def release_snapshot(self, handle):
        with self._lock:
            self._check(check_source=False)
            _,_,size = self._owned(handle)
            del self._snapshots[handle.key]
            self._snapshot_bytes -= size

    def write_cache(self, output, *, max_bytes):
        with self._lock:
            self._check()
            self._admit(self._full_state_bytes)
            empty = self.decoder._cache is None
            header = json.dumps(dict(identity=self.identity,position=self.position,empty=empty,
                shapes=self.shapes),sort_keys=True,separators=(',',':')).encode()
            size = 44+len(header)+(0 if empty else self._full_state_bytes)
            if type(max_bytes) is not int or size>max_bytes or len(header)>65536:
                raise MemoryError('flat native transport exceeds admission')
            self._cancel()
            banks = self._capture()
            self._busy = True
            try:
                digest = hashlib.sha256()
                for data in (_MAGIC,struct.pack('<I',len(header)),header):
                    digest.update(data)
                    _write(output,data)
                for bank in banks or ():
                    for raw in bank:
                        self._cancel()
                        digest.update(raw)
                        _write(output,raw)
                _write(output,digest.digest())
                return size
            finally:
                self._busy = False

    def restore_cache(self, source, *, max_bytes):
        with self._lock:
            self._check()
            if type(max_bytes) is not int or max_bytes<44:
                raise ValueError('positive bounded native transport required')
            self._busy = True
            try:
                self._cancel()
                prefix = _read(source,12)
                length = struct.unpack('<I',prefix[8:])[0]
                if prefix[:8]!=_MAGIC or not 0<length<=min(65536,max_bytes-44):
                    raise ValueError('invalid flat native frame header')
                raw = _read(source,length)
                header = json.loads(raw, object_pairs_hook=_pairs,
                                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if (type(header) is not dict or set(header)!={'identity','position','empty','shapes'} or
                        header['identity']!=self.identity or type(header['empty']) is not bool or
                        header['shapes']!=json.loads(json.dumps(self.shapes))):
                    raise ValueError('native frame identity or complete shape inventory mismatch')
                total = 44+length+(0 if header['empty'] else self._full_state_bytes)
                if total>max_bytes:
                    raise MemoryError('native frame payload exceeds admission')
                self._admit(2*self._full_state_bytes)
                digest = hashlib.sha256(prefix+raw)
                banks = None if header['empty'] else []
                if banks is not None:
                    for shapes in self.shapes:
                        bank = []
                        for shape in shapes:
                            self._cancel()
                            data = _read(source,4*int(np.prod(shape,dtype=object)))
                            digest.update(data)
                            bank.append(data)
                        banks.append(tuple(bank))
                    banks = tuple(banks)
                if (not hmac.compare_digest(digest.digest(), _read(source, 32)) or
                        _read(source, 1) != b''):
                    raise ValueError('native frame checksum or EOF mismatch')
                staged = self._stage(banks,header['position'])
                self._cancel()
                self.decoder._cache,self.decoder.position = staged,header['position']
            finally:
                self._busy = False

    def save_cache(self, destination, *, max_bytes):
        from .cache_file_recovery import save_executor_cache
        return save_executor_cache(self,destination,max_bytes=max_bytes)

    def load_cache(self, source, *, max_bytes):
        from .cache_file_recovery import load_executor_cache
        return load_executor_cache(self,source,max_bytes=max_bytes)

    def reset(self):
        with self._lock:
            self._check()
            self.decoder.reset()

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('cannot close active native session')
            if not self._closed:
                self.decoder.close()
                self._snapshots.clear()
                self._snapshot_bytes = 0
                self._closed = True


__all__ = ['FlatNativeSession','FlatNativeSnapshot']
