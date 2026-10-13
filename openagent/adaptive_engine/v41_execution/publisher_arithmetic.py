"""Concrete CPU publisher transport and native arithmetic; no automatic builds.

Uses leased tiled FP8 matrices and admitted streamed/grouped projection banks,
and selected packed expert leases. Quantization is in the composition library; the
existing private FP4 codec and block-128 FP8 backend are different contracts.
"""
import ctypes as C
import hashlib
import _ctypes
import os
import weakref
import threading
from contextlib import ExitStack, contextmanager

import numpy as np

from ..native_mxfp4 import NativeMXFP4
from .execution import V41Execution
from .fp8_projection import NativeFP8Projection
from ..native_library_identity import release_finalizer


class NativePublisherV41Arithmetic:
    @contextmanager
    def execution_scope(self):
        with self._lock:
            self._borrowers=getattr(self,'_borrowers',0)+1
            try:
                self._ensure_open()
                yield self
            finally:
                self._borrowers-=1

    @contextmanager
    def cancellation_scope(self, cancel):
        """Bind cancellation to borrowed provider work on the calling thread."""
        if cancel is not None and not callable(getattr(cancel,'is_set',None)):
            raise ValueError('event-like publisher arithmetic cancellation required')
        seen = set()
        with ExitStack() as scopes:
            old=getattr(self._cancellation,'event',None)
            self._cancellation.event=cancel
            scopes.callback(setattr,self._cancellation,'event',old)
            backend=self._projection_backend
            scope=getattr(backend,'cancellation_scope',None)
            if callable(scope):scopes.enter_context(scope(cancel))
            for name in ('expert_provider', 'engram_provider', 'fp8_provider','embedding_provider'):
                provider = getattr(self.payload, name, None)
                if provider is None or id(provider) in seen:
                    continue
                seen.add(id(provider))
                scope = getattr(provider, 'cancellation_scope', None)
                if callable(scope):
                    scopes.enter_context(scope(cancel))
            yield

    def __init__(self, payload, *, execution_library, mxfp4_library,
                 tokenizer=None, compressed_ids=None, max_bytes=64*1024**2,
                 fp8_library=None, projection_backend=None):
        self._lock = threading.RLock()
        self._cancellation=threading.local()
        self._closed = False
        self._release = self._fp4 = self._fp8 = None
        try:
            self._initialize(payload, execution_library=execution_library,
                mxfp4_library=mxfp4_library, tokenizer=tokenizer,
                compressed_ids=compressed_ids, max_bytes=max_bytes,
                fp8_library=fp8_library, projection_backend=projection_backend)
        except BaseException:
            self.close()
            raise

    def _initialize(self, payload, *, execution_library, mxfp4_library,
                    tokenizer, compressed_ids, max_bytes, fp8_library,
                    projection_backend):
        self.payload, self.max_bytes = payload, max_bytes
        self._projection_backend = projection_backend
        self._projection_identity = None
        if projection_backend is not None:
            for method in ('handles', 'shape', 'linear'):
                if not callable(getattr(projection_backend, method, None)):
                    raise ValueError('projection backend missing '+method)
            self._projection_identity = getattr(projection_backend, 'identity', None)
            if not isinstance(self._projection_identity, str) or not self._projection_identity:
                raise ValueError('identified immutable projection bank required')
        if type(max_bytes) is not int or not 0<max_bytes<2**(8*C.sizeof(C.c_size_t)):
            raise ValueError('positive native arithmetic admission budget required')
        from ..native_library_identity import NativeLibraryIdentity
        self._library_identities=tuple(NativeLibraryIdentity.read(path) for path in
            (execution_library,mxfp4_library,fp8_library) if path is not None)
        self.rules = {rule.sources[0]: rule for rule in payload.contracts}
        if len(self.rules)!=len(payload.contracts):
            raise ValueError('duplicate exact publisher arithmetic source')
        for name in getattr(payload, 'sharded_projections', ()):
            if (projection_backend is None or not projection_backend.handles(name) or
                    projection_backend.shape(name) != self.rules[name].shape):
                raise ValueError('omitted publisher projection requires an exact admitted shard: '+name)
            if name.endswith('.wo_a.weight') and not callable(getattr(projection_backend, 'grouped_linear', None)):
                raise ValueError('sharded wo_a requires grouped projection backend')
        self._lib = C.CDLL(str(execution_library))
        release = _ctypes.FreeLibrary if os.name=='nt' else _ctypes.dlclose
        self._release = weakref.finalize(self,release,self._lib._handle)
        self._closed=False
        self._op = self._lib.oa_v41_execution_op
        self._op.argtypes = [C.c_int]*4+[C.c_double]+[C.c_void_p,C.c_size_t]*4
        self._op.restype = C.c_int
        capability = self._lib.oa_v41_publisher_contract_version
        capability.argtypes, capability.restype = [], C.c_uint32
        if capability()!=1:
            raise ValueError('publisher composition arithmetic ABI required')
        self._hash = self._lib.oa_v41_publisher_hash
        self._hash.argtypes = [C.c_void_p,C.c_void_p,C.c_size_t,C.c_void_p,
                              C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t]
        self._hash.restype = C.c_int
        if payload.expert_provider is not None:
            self._fp4 = NativeMXFP4(library=mxfp4_library,max_bytes=max_bytes)
        self._fp8 = None
        if getattr(payload, 'fp8_provider', None) is not None:
            provider=payload.fp8_provider
            if (not callable(getattr(provider,'lease',None)) or
                    not callable(getattr(provider,'tile_rows',None)) or
                    not isinstance(getattr(provider,'identity',None),str) or not provider.identity):
                raise ValueError('identified tiled FP8 projection provider required')
            self._fp8 = NativeFP8Projection(
                library=execution_library if fp8_library is None else fp8_library,
                max_bytes=max_bytes)
        self._map, self._hash_metadata = None, {}
        eg = payload.profile.engram
        if eg.layers:
            if payload.engram_provider is None:
                raise ValueError('selected publisher Engram row provider required')
            if compressed_ids is None:
                if tokenizer is None:
                    raise ValueError('publisher tokenizer or explicit compressed ID map required')
                compressed_ids = self._compressed_map(tokenizer, payload.profile.vocabulary)
            ids = np.asarray(compressed_ids)
            if (ids.shape!=(payload.profile.vocabulary,) or ids.dtype.kind not in 'iu' or
                    np.any(ids<0) or np.any(ids>=eg.compressed_vocabulary) or
                    np.unique(ids).size!=eg.compressed_vocabulary):
                raise ValueError('compressed tokenizer map differs from publisher vocabulary')
            self._map = np.array(ids,dtype=np.uint64,copy=True)
            self._map.flags.writeable=False
            self._make_hash_metadata()
        providers = (getattr(payload.expert_provider,'identity',None),
                     getattr(payload.engram_provider,'identity',None),
                     getattr(getattr(payload,'embedding_provider',None),'identity',None),
                     getattr(getattr(payload,'fp8_provider',None),'identity',None))
        metadata = tuple((layer,tuple(tuple(int(v) for v in array) for array in arrays))
                         for layer,arrays in self._hash_metadata.items())
        for record in self._library_identities:record.assert_unchanged()
        self.collective_policy_identity=hashlib.sha256(repr(('scalar-fp64-accumulation-bf16-boundaries-v2',
            tuple(record.sha256 for record in self._library_identities),max_bytes)).encode()).hexdigest()
        digest = hashlib.sha256(repr((payload.config_sha256,providers,metadata,
            tuple(record.sha256 for record in self._library_identities),
            'scalar-fp64-accumulation-bf16-boundaries-v2', self._projection_identity,
            'packed-tiled32-ue8m0-v1' if self._fp8 is not None else 'resident-fp32',
            tuple((r.sources,r.shape,r.encoding) for r in payload.contracts))).encode('utf-8')+
            (b'' if self._map is None else self._map.tobytes()))
        for name in sorted(payload.tensors):
            array=payload.tensors[name]
            if (not isinstance(array,np.ndarray) or array.dtype!=np.float32 or
                    not array.flags.c_contiguous or not array.flags.aligned):
                raise ValueError('contiguous FP32 publisher resident tensor required')
            digest.update(repr((name,array.shape,array.dtype.str,array.nbytes)).encode('utf-8'))
            digest.update(memoryview(array).cast('B'))
        self.identity = digest.hexdigest()

    def _kernel(self, op, x, w=(), z=(), *, a=0,b=0,c=0,p=0,n=None):
        with self._lock:
            self._ensure_open()
            counts=[]
            copies=0
            for value in (x,w,z):
                if isinstance(value,np.ndarray):
                    count=value.size
                    if value.dtype!=np.float32 or not value.flags.c_contiguous:
                        copies+=4*count
                elif isinstance(value,(tuple,list)) and all(np.isscalar(v) for v in value):
                    count=len(value)
                    copies+=4*count
                else:
                    raise ValueError('flat native arithmetic transport required')
                counts.append(count)
            output=counts[0] if n is None else n
            if type(output) is not int or output<0:
                raise ValueError('nonnegative native arithmetic output extent required')
            if copies+4*(counts[0]+output)>self.max_bytes:
                raise MemoryError('native arithmetic transport exceeds scratch admission')
            result=V41Execution._kernel(self,op,x,w,z,a=a,b=b,c=c,p=p,n=output)
            self.check_cancelled()
            return result

    @staticmethod
    def _compressed_map(tokenizer, vocabulary):
        # Text normalization is tokenizer preprocessing, never numerical model math.
        from tokenizers import Regex, normalizers
        backend = getattr(tokenizer,'backend_tokenizer',tokenizer)
        marker = '\ue000'
        normalize = normalizers.Sequence([
            normalizers.NFKC(),normalizers.NFD(),normalizers.StripAccents(),normalizers.Lowercase(),
            normalizers.Replace(Regex(r'[ \t\r\n]+'),' '),
            normalizers.Replace(Regex(r'^ $'),marker),normalizers.Strip(),
            normalizers.Replace(marker,' ')])
        classes, ids = {}, []
        for token in range(vocabulary):
            decoded = backend.decode([token],skip_special_tokens=False)
            key = backend.id_to_token(token) if '\ufffd' in decoded else (normalize.normalize_str(decoded) or decoded)
            if key not in classes:
                classes[key] = len(classes)
            ids.append(classes[key])
        return ids

    def _make_hash_metadata(self):
        from sympy import nextprime
        eg = self.payload.profile.engram
        if not 2<=eg.maximum_order<=32 or eg.heads_per_order>4096:
            raise ValueError('publisher Engram geometry exceeds native hash ABI')
        used = set()
        bound = max(1,((2**63-1)//eg.compressed_vocabulary)//2)
        for layer,rows in zip(eg.layers,eg.rows):
            primes=[]
            for _ in range(eg.maximum_order-1):
                cursor=eg.hash_vocabulary-1
                for _ in range(eg.heads_per_order):
                    cursor=int(nextprime(cursor))
                    while cursor in used:
                        cursor=int(nextprime(cursor))
                    used.add(cursor)
                    primes.append(cursor)
            offsets=[]
            end=0
            for prime in primes:
                offsets.append(end)
                end+=prime
            if end>rows:
                raise ValueError('publisher Engram prime bucket ranges exceed table rows')
            # Explicit PCG64 matches publisher default_rng's per-layer contract.
            rng=np.random.Generator(np.random.PCG64(10007*layer))
            multipliers=rng.integers(0,bound,size=eg.maximum_order,dtype=np.int64)*2+1
            self._hash_metadata[layer]=(np.asarray(multipliers,dtype=np.uint64),
                                       np.asarray(primes,dtype=np.uint64),np.asarray(offsets,dtype=np.uint64))

    def cast(self, row):
        self._ensure_open()
        return self._kernel(26,row,a=0)

    def roundtrip(self, row, encoding):
        self._ensure_open()
        modes={'fp8-e4m3-g32-ue8m0':1,'fp4-e2m1-g16-e4m3':2,'fp4-e2m1-g32-ue8m0':3}
        if encoding not in modes:
            raise ValueError('unsupported publisher quantization encoding')
        return self._kernel(26,row,a=modes[encoding])

    def linear(self, source, x):
        with self._lock:
            return self._project(source,x)

    def grouped_linear(self, source, x, *, groups):
        with self._lock:
            self._ensure_open()
            rule = self.rules[source]
            if not source.endswith('.wo_a.weight') or len(rule.shape) != 2:
                raise ValueError('grouped publisher projection requires wo_a')
            rows, cols = rule.shape
            if type(groups) is not int or groups <= 0 or rows % groups:
                raise ValueError('invalid publisher output groups')
            values = np.asarray(x, dtype=np.float32)
            if (values.ndim not in (2, 3) or values.shape[-2:] != (groups, cols) or
                    not values.size or not np.isfinite(values).all()):
                raise ValueError('exact finite grouped publisher input required')
            batch = 1 if values.ndim == 2 else values.shape[0]
            if values.nbytes*3+batch*rows*12 > self.max_bytes:
                raise MemoryError('grouped publisher projection transport exceeds admission')
            backend = self._projection_backend
            if backend is not None and backend.identity != self._projection_identity:
                raise ValueError('projection bank changed after arithmetic admission')
            if backend is not None and backend.handles(source):
                if backend.shape(source) != (rows, cols):
                    raise ValueError('distributed grouped projection geometry mismatch')
                result = np.asarray(backend.grouped_linear(source, values, groups=groups))
            else:
                if self.payload.tensors[source].nbytes*3+values.nbytes*3+batch*rows*12 > self.max_bytes:
                    raise MemoryError('resident grouped publisher weights exceed admission')
                weights = self.cast(self.payload.tensors[source]).reshape(groups, rows//groups, cols)
                if values.ndim == 3:
                    result = np.stack(tuple(np.stack(tuple(
                        V41Execution._linear(self,weights[group],values[batch_index,group])
                        for group in range(groups))) for batch_index in range(batch)))
                else:
                    result = np.stack(tuple(V41Execution._linear(self, weights[g], values[g])
                                            for g in range(groups)))
            expected = (*values.shape[:-2], groups, rows//groups)
            if result.dtype != np.float32 or result.shape != expected or not np.isfinite(result).all():
                raise ValueError('invalid grouped publisher projection output')
            return result.reshape(*values.shape[:-2], rows)

    def _project(self, source, x):
        self._ensure_open()
        rule=self.rules[source]
        if len(rule.shape)!=2:
            raise ValueError('publisher matrix projection required')
        rows,cols=rule.shape
        if any(type(v) is not int or v<=0 for v in (rows,cols)):
            raise ValueError('positive publisher projection geometry required')
        if source.endswith('.wo_a.weight'):
            raise ValueError('wo_a requires group-local projection')
        if 4*(2*cols+2*rows)>self.max_bytes:
            raise MemoryError('projection transport exceeds arithmetic scratch admission')
        if isinstance(x,np.ndarray):
            if x.size!=cols:
                raise ValueError('publisher projection input extent mismatch')
        elif not isinstance(x,(tuple,list)) or len(x)!=cols or any(not np.isscalar(v) for v in x):
            raise ValueError('exact flat publisher projection input required')
        x=np.ascontiguousarray(x,dtype=np.float32).reshape(-1)
        if x.size!=cols:
            raise ValueError('publisher projection input extent mismatch')
        if rule.encoding in ('fp8-e4m3-g32x32','v41-fp4-g32'):
            x=self._kernel(26,x,a=4)
        backend = self._projection_backend
        if backend is not None and backend.identity != self._projection_identity:
            raise ValueError('projection bank changed after arithmetic admission')
        if backend is not None and backend.handles(source):
            if backend.shape(source) != (rows, cols):
                raise ValueError('distributed source geometry differs from publisher contract')
            result = np.asarray(backend.linear(source, x))
            if (result.dtype != np.float32 or result.shape != (rows,) or
                    not np.isfinite(result).all()):
                raise ValueError('projection backend returned invalid publisher output')
        elif rule.encoding=='fp8-e4m3-g32x32' and self._fp8 is not None:
            provider=self.payload.fp8_provider
            if cols%32 or len(rule.sources)!=2:
                raise ValueError('exact tiled32 FP8 source pair required')
            tile_rows=provider.tile_rows(source)
            if (type(tile_rows) is not int or tile_rows<=0 or
                    (tile_rows<rows and tile_rows%32)):
                raise ValueError('FP8 provider requires positive aligned row tiles')
            result=np.empty(rows,np.float32)
            for start in range(0,rows,tile_rows):
                count=min(tile_rows,rows-start)
                with provider.lease(source,row_start=start,row_count=count) as tables:
                    self._fp8.matvec(tables[rule.sources[0]],tables[rule.sources[1]],
                                     x,rows=count,cols=cols,out=result[start:start+count])
        elif rule.encoding=='v41-fp4-g32' and self.payload.expert_provider is not None:
            parts=source.split('.')
            with self.payload.expert_provider.lease(int(parts[1]),int(parts[4]),namespace=parts[0]) as tables:
                result=self._fp4.matvec(tables[rule.sources[0]],
                    tables[rule.sources[1]],x,rows=rows,cols=cols)
        else:
            if source not in self.payload.tensors:
                raise ValueError('missing resident publisher projection or packed provider: '+source)
            matrix=self.payload.tensors[source]
            if (not isinstance(matrix,np.ndarray) or matrix.dtype!=np.float32 or
                    matrix.shape!=(rows,cols) or not matrix.flags.c_contiguous or
                    not matrix.flags.aligned):
                raise ValueError('exact contiguous resident FP32 publisher matrix required')
            result=V41Execution._linear(self,matrix,x)
        # Source router, mHC fn, lm head and ratio>1 compressor are FP32 math.
        fp32=(source=='head.weight' or source.endswith(('_fn','ffn.gate.weight')) or
              ('.attn.compressor.' in source and
               self.payload.profile.layers[int(source.split('.')[1])].compression_ratio>1))
        return result if fp32 else self.cast(result)

    def engram_rows(self, layer, tokens):
        with self._lock:
            return self._engram_rows(layer,tokens)

    def _engram_rows(self, layer, tokens):
        self._ensure_open()
        eg=self.payload.profile.engram
        multipliers,primes,offsets=self._hash_metadata[layer]
        pad=int(self._map[eg.padding_token])
        lookback=np.asarray([int(self._map[tokens[-1-shift]]) if shift<len(tokens) else pad
                             for shift in range(eg.maximum_order)],dtype=np.uint64)
        rows=np.empty(primes.size,dtype=np.uint64)
        status=self._hash(lookback.ctypes.data,multipliers.ctypes.data,lookback.size,
                          primes.ctypes.data,offsets.ctypes.data,eg.heads_per_order,rows.ctypes.data,rows.size)
        if status:
            raise ValueError('native publisher Engram hash failure')
        source=f'layers.{layer}.engram.embed.weight'
        with self.payload.engram_provider.lease_rows(source,tuple(int(row) for row in rows)) as tables:
            table=tables[source]
            if (not isinstance(table,np.ndarray) or table.dtype!=np.float32 or
                    not table.flags.c_contiguous):
                raise ValueError('exact contiguous Engram FP32 rows required')
            if 3*table.nbytes>self.max_bytes:
                raise MemoryError('Engram row transport exceeds arithmetic scratch admission')
            return self.cast(np.array(table,dtype=np.float32,copy=True).reshape(-1))

    def close(self):
        # Caller retains ownership of loader providers and their checkpoint.
        with self._lock:
            if getattr(self,'_borrowers',0):
                raise RuntimeError('publisher execution still borrows arithmetic resources')
            self._closed=True
            if self._fp4 is not None:
                if hasattr(self._fp4,'close'):
                    self._fp4.close()
                else:
                    release_finalizer(self._fp4,'_release')
                self._fp4=None
            if self._fp8 is not None:
                self._fp8.close()
                self._fp8=None
            if self._release is not None:
                release_finalizer(self,'_release')
                self._release=None

    def check_cancelled(self):
        event=getattr(self._cancellation,'event',None)
        if event is not None and event.is_set():
            from ..execution_service import ExecutionCancelled
            raise ExecutionCancelled('publisher arithmetic operation cancelled')

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('native publisher arithmetic closed')
        self.check_cancelled()
        for record in self._library_identities:record.assert_unchanged()
