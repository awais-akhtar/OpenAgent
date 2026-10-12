"""Original Qwen4 logical PLE tables through direct native row callbacks."""
import ctypes as C
import hashlib
import threading
from types import MappingProxyType
from .native_original_projections import NativeOriginalProjections
from .publisher_tensor_reader import PublisherTensorReader,TensorSlice
from .qwen4_ple_store import Qwen4PLEStore


class NativeOriginalPLE:
    """Own native logical-row contexts and their original-shard projection bank.

    The runtime is borrowed and must outlive this owner. Tables retain the bank
    natively until their contexts are released; close retries unfinished release.
    """
    def __init__(self,runtime,provider,*,max_bytes):
        if type(provider) is not Qwen4PLEStore or type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('exact Qwen4 processor table provider and positive native budget required')
        self._lock=threading.RLock();self._closed=False;self._runtime=runtime;self._bank=None;self._tables={}
        self.max_bytes=max_bytes;self.provider_identity=provider.identity;self.config=provider.config
        self.shapes=MappingProxyType(dict(provider.shapes))
        names={name for sources,starts in provider.bindings.values() for name in sources}
        if not names:raise ValueError('occupied native PLE source inventory required')
        table_budget=sum(128+32*len(sources) for sources,starts in provider.bindings.values())
        if table_budget>=max_bytes:raise MemoryError('native logical PLE contexts exceed aggregate admission')
        bindings={name:TensorSlice(name,provider.snapshot.tensors[name].shape) for name in sorted(names)}
        reader=PublisherTensorReader(provider.snapshot,max_work_bytes=max_bytes-table_budget)
        try:
            self._bank=NativeOriginalProjections(runtime,reader,bindings,max_bytes=max_bytes-table_budget,
                reduction='fp64',rows_only=True)
            create=runtime.library.oa_original_rows_create
            create.argtypes=[C.c_void_p,C.POINTER(C.c_uint32),C.c_uint32,C.c_uint64,
                C.POINTER(C.c_void_p),C.c_void_p,C.c_size_t];create.restype=C.c_int
            for layer,(sources,starts) in sorted(provider.bindings.items()):
                indices=(C.c_uint32*len(sources))(*[self._bank._indices[name] for name in sources])
                handle=C.c_void_p();error=C.create_string_buffer(4096)
                if create(self._bank.handle,indices,len(indices),128+32*len(indices),C.byref(handle),error,len(error)):
                    raise RuntimeError(error.value.decode('utf-8',errors='replace'))
                self._tables[layer]=handle
            provider.snapshot.assert_unchanged()
            self.identity=hashlib.sha256(repr((provider.identity,self._bank.identity,max_bytes,
                'native-logical-original-ple-rows-v1')).encode()).hexdigest()
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:
                primary.cleanup=self.close;primary.add_note('native PLE cleanup retained: '+repr(cleanup))
            raise
        finally:reader.close()

    def context(self,layer):
        if self._closed or layer not in self._tables:raise RuntimeError('native PLE context unavailable')
        return self._tables[layer]

    @property
    def callback(self):
        if self._closed:raise RuntimeError('native PLE owner closed')
        return C.cast(self._runtime.library.oa_original_rows_lookup,C.c_void_p)

    def close(self):
        with self._lock:
            self._closed=True
            if self._tables:
                release=self._runtime.library.oa_original_rows_release
                release.argtypes=[C.c_void_p];release.restype=C.c_int
                for layer in tuple(self._tables):
                    if release(self._tables[layer]):raise RuntimeError('native PLE context release remains pending')
                    del self._tables[layer]
            if self._bank is not None:self._bank.close();self._bank=None


__all__=['NativeOriginalPLE']
