"""Explicit callback-free CPU FP32 OLMoE range and edge execution.

Load a caller-built library; no JIT, compilation, downloads or NumPy math
fallback. NumPy supplies owned contiguous buffers, not transformer arithmetic.
"""
import ctypes as C
import _ctypes
import os
from pathlib import Path
import threading
import weakref

import numpy as np


VERSION = 1
PACKED_BLOCK_BYTES = {'q4_0': 18, 'q4_1': 20, 'q5_0': 22,
                      'q5_1': 24, 'q8_0': 34, 'q8_1': 36}
NUMERIC_CLASS = "c-fp32-weights-double-dot-resident-experts-v1"
_U64 = C.c_uint64


class NativeExecutionError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"Native segment executor status {status}")


class Tensor(C.Structure):
    _fields_ = [("data", C.c_void_p), ("count", _U64)]


class RangeConfig(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
               ("version", "heads", "kv_heads", "topk", "normalize_topk", "reserved")] + [
               (name, _U64) for name in ("hidden", "intermediate", "experts", "layers", "capacity")] + [
               ("epsilon", C.c_double), ("theta", C.c_double)]


class RangeLayer(C.Structure):
    _fields_ = [(name, Tensor) for name in
               ("input_norm", "q", "k", "v", "o", "q_norm", "k_norm", "post_norm",
                "router", "gate", "up", "down")]


def _integer(value, name, maximum=(1 << 64) - 1):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return value


def _array(value, shape, limit, name):
    array = np.asarray(value)
    if array.dtype != np.dtype("float32") or array.shape != tuple(shape):
        raise ValueError(f"{name} must be native FP32 with shape {tuple(shape)}")
    if array.nbytes > limit:
        raise NativeExecutionError(5)
    return np.array(array, dtype=np.float32, order="C", copy=True)


def _tensor(array):
    return Tensor(array.ctypes.data, array.size)


class NativeSegmentExecutor:
    """Library facade. C entry points are stateless; sessions serialize mutations."""
    native_range_execution = True
    numeric_class = NUMERIC_CLASS
    upstream_abi_compatible = False

    def __init__(self, library, *, max_bytes=256 * 1024 * 1024):
        self.max_bytes = _integer(max_bytes, "max_bytes", np.iinfo(np.intp).max)
        self.lib = C.CDLL(str(Path(library).resolve(strict=True)))
        release = _ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose
        self._release = weakref.finalize(self, release, self.lib._handle)
        try:
            self._bind_library()
        except BaseException:
            self._release()
            raise

    def _bind_library(self):
        self.lib.oa_executor_version.argtypes = []
        self.lib.oa_executor_version.restype = C.c_uint32
        if self.lib.oa_executor_version() != VERSION:
            raise NativeExecutionError(2)
        self.lib.oa_executor_range.argtypes = [C.POINTER(RangeConfig), C.POINTER(RangeLayer),
            _U64, Tensor, _U64, C.c_void_p, _U64, C.c_void_p, C.c_void_p, _U64,
            C.POINTER(_U64), _U64]
        self.lib.oa_executor_range.restype = C.c_int32
        self.lib.oa_executor_embed.argtypes = [Tensor, _U64, _U64, C.c_void_p, _U64,
            C.c_void_p, _U64, _U64]
        self.lib.oa_executor_embed.restype = C.c_int32
        self.lib.oa_executor_head.argtypes = [Tensor, _U64, _U64, Tensor, Tensor, _U64,
            C.c_double, C.c_void_p, _U64, _U64]
        self.lib.oa_executor_head.restype = C.c_int32
        # Optional additive v1 capability: older range-only libraries still load.
        self._decode = getattr(self.lib,"oa_executor_decode",None)
        if self._decode is not None:
            self._decode.argtypes = [C.POINTER(RangeConfig), C.POINTER(RangeLayer), _U64,
                Tensor, Tensor, Tensor, _U64, C.c_void_p, _U64, C.c_void_p, _U64,
                C.c_void_p, C.c_void_p, _U64, C.POINTER(_U64), _U64]
            self._decode.restype = C.c_int32
        specifications = {
            "_prepare": ("oa_executor_prepare_layer",[C.POINTER(RangeConfig),C.POINTER(RangeLayer),
                Tensor,_U64,C.c_void_p,C.c_void_p,_U64,C.c_void_p,C.c_void_p,_U64,
                C.c_void_p,C.c_void_p,_U64,_U64,_U64]),
            "_packed": ("oa_executor_swiglu_int8_row",[Tensor,_U64,_U64,_U64,
                C.c_void_p,_U64,C.c_void_p,_U64,_U64]),
            "_apply_packed": ("oa_executor_apply_int8_row",[Tensor,_U64,_U64,_U64,
                C.c_void_p,C.c_void_p,_U64,_U64,_U64,_U64,C.c_void_p,_U64,Tensor,
                C.c_void_p,_U64,_U64]),
            "_residual": ("oa_executor_residual_f32",[Tensor,Tensor,C.c_void_p,_U64,_U64]),
        }
        for attribute,(symbol,arguments) in specifications.items():
            function = getattr(self.lib,symbol,None)
            if function is not None:
                function.argtypes,function.restype = arguments,C.c_int32
            setattr(self,attribute,function)
        for attribute,source,symbol in (("_packed_q8","_packed","oa_executor_swiglu_q8_0"),
                                       ("_apply_q8","_apply_packed","oa_executor_apply_q8_0"),
                                       ("_packed_q4","_packed","oa_executor_swiglu_q4_0"),
                                       ("_apply_q4","_apply_packed","oa_executor_apply_q4_0"),
                                       *((f'_packed_{dtype}', '_packed', f'oa_executor_swiglu_{dtype}')
                                         for dtype in ('q4_1', 'q5_0', 'q5_1', 'q8_1')),
                                       *((f'_apply_{dtype}', '_apply_packed', f'oa_executor_apply_{dtype}')
                                         for dtype in ('q4_1', 'q5_0', 'q5_1', 'q8_1'))):
            function = getattr(self.lib,symbol,None)
            if function is not None:
                function.argtypes,function.restype = specifications[source][1],C.c_int32
            setattr(self,attribute,function)

    @staticmethod
    def _check(status):
        if status:
            raise NativeExecutionError(status)

    def create_range(self, layers, *, hidden, intermediate, experts, heads,
                     kv_heads, topk, capacity, epsilon=1e-5, theta=10000.,
                     normalize_topk=True):
        return NativeRange(self, layers, hidden=hidden, intermediate=intermediate,
            experts=experts, heads=heads, kv_heads=kv_heads, topk=topk,
            capacity=capacity, epsilon=epsilon, theta=theta, normalize_topk=normalize_topk)

    def require_leased(self):
        if any(getattr(self,name) is None for name in ("_prepare","_packed","_apply_packed","_residual")):
            raise RuntimeError("native library lacks leased INT8-row phases; rebuild segment_executor.c")

    def require_q8(self):
        self.require_leased()
        if self._packed_q8 is None or self._apply_q8 is None:
            raise RuntimeError("native library lacks Q8_0 exports; rebuild segment_executor.c")

    def q8_swiglu(self,hidden,payload,*,intermediate):
        return self.quantized_swiglu(hidden, payload, intermediate=intermediate, dtype='q8_0')

    def require_q4(self):
        self.require_leased()
        if self._packed_q4 is None or self._apply_q4 is None:
            raise RuntimeError('native library lacks Q4_0 leased exports; rebuild segment_executor.c')

    def q4_swiglu(self, hidden, payload, *, intermediate):
        """Execute GGML Q4_0 expert banks without resident decoded weights."""
        return self.quantized_swiglu(hidden, payload, intermediate=intermediate, dtype='q4_0')

    def require_quantized(self, dtype):
        if type(dtype) is not str or dtype not in PACKED_BLOCK_BYTES:
            raise ValueError('unsupported native packed expert format')
        if dtype == 'q4_0':
            self.require_q4()
        elif dtype == 'q8_0':
            self.require_q8()
        else:
            self.require_leased()
            if (getattr(self, '_packed_'+dtype) is None or
                    getattr(self, '_apply_'+dtype) is None):
                raise RuntimeError('native library lacks '+dtype+' leased exports; rebuild segment_executor.c')

    def quantized_swiglu(self, hidden, payload, *, intermediate, dtype):
        """Run one supported GGML packed expert format through explicit C exports."""
        self.require_quantized(dtype)
        intermediate = _integer(intermediate, 'intermediate')
        x = np.asarray(hidden)
        if x.ndim != 2 or not all(x.shape) or x.shape[1] % 32 or intermediate % 32:
            raise ValueError('nonempty block-aligned FP32 hidden matrix required')
        if x.dtype != np.dtype('float32'):
            raise ValueError('hidden must be native FP32')
        view = memoryview(payload)
        expected = 3*x.shape[1]*intermediate//32*PACKED_BLOCK_BYTES[dtype]
        scratch = 4*(3*x.size+intermediate)
        if not view.c_contiguous or view.nbytes != expected:
            raise ValueError('exact contiguous GGML '+dtype+' expert payload required')
        if expected+scratch > self.max_bytes:
            raise NativeExecutionError(5)
        x = _array(x, x.shape, self.max_bytes, 'hidden')
        if not np.isfinite(x).all():
            raise ValueError('hidden must be finite')
        packed = view.tobytes()
        out = np.empty_like(x)
        function = (self._packed_q4 if dtype == 'q4_0' else self._packed_q8
                    if dtype == 'q8_0' else getattr(self, '_packed_'+dtype))
        self._check(function(_tensor(x), len(x), x.shape[1], intermediate,
            C.c_char_p(packed), len(packed), out.ctypes.data, out.size, self.max_bytes))
        return out

    def apply_quantized(self, hidden, payload, accumulator, ids, weights, *,
                        intermediate, experts, expert, dtype):
        """Add one leased expert's selected routes to an owned accumulator."""
        self.require_quantized(dtype)
        intermediate = _integer(intermediate, 'intermediate')
        experts = _integer(experts, 'experts')
        if type(expert) is not int or not 0 <= expert < experts:
            raise ValueError('valid expert index required')
        x = np.asarray(hidden)
        routes, coefficients = np.asarray(ids), np.asarray(weights)
        if (x.ndim != 2 or not all(x.shape) or x.shape[1] % 32 or intermediate % 32 or
                x.dtype != np.dtype('float32') or
                routes.ndim != 2 or routes.shape[0] != x.shape[0] or
                not 1 <= routes.shape[1] <= experts or routes.dtype.kind not in 'iu' or
                coefficients.shape != routes.shape or coefficients.dtype != np.dtype('float32')):
            raise ValueError('invalid block-aligned hidden matrix or expert routes')
        view = memoryview(payload)
        expected = 3*x.shape[1]*intermediate//32*PACKED_BLOCK_BYTES[dtype]
        scratch = 4*(5*x.size+intermediate+x.shape[1])+24*routes.size
        if not view.c_contiguous or view.nbytes != expected:
            raise ValueError('exact contiguous packed expert payload required')
        if expected+scratch > self.max_bytes:
            raise NativeExecutionError(5)
        if np.any(routes < 0) or np.any(routes >= experts):
            raise ValueError('invalid expert route indices')
        x = _array(x, x.shape, self.max_bytes, 'hidden')
        current = _array(accumulator, x.shape, self.max_bytes, 'accumulator')
        routes = np.array(routes, order='C', copy=True)
        if np.any(routes < 0) or np.any(routes >= experts):
            raise ValueError('invalid owned expert route indices')
        routes = routes.astype(np.uint64, copy=False)
        coefficients = np.array(coefficients, dtype=np.float32, order='C', copy=True)
        if (not np.isfinite(x).all() or not np.isfinite(current).all() or
                np.any(routes >= experts) or not np.isfinite(coefficients).all() or
                np.any(coefficients < 0) or np.any(coefficients > 1)):
            raise ValueError('invalid owned expert apply inputs')
        for route in routes:
            if len(set(map(int,route))) != routes.shape[1]:
                raise ValueError('duplicate experts in an owned route row')
        packed = view.tobytes()
        out = np.empty_like(x)
        function = (self._apply_q4 if dtype == 'q4_0' else self._apply_q8
                    if dtype == 'q8_0' else getattr(self, '_apply_'+dtype))
        self._check(function(_tensor(x), len(x), x.shape[1], intermediate,
            routes.ctypes.data, coefficients.ctypes.data, routes.size, routes.shape[1],
            experts, expert, C.c_char_p(packed), len(packed), _tensor(current),
            out.ctypes.data, out.size, self.max_bytes))
        return out

    def packed_swiglu(self,hidden,payload,*,intermediate):
        """Unweighted native INT8-row expert, without decoded FP32 weight banks."""
        self.require_leased()
        intermediate = _integer(intermediate,"intermediate")
        x = np.asarray(hidden)
        if x.ndim != 2 or not all(x.shape):
            raise ValueError("hidden must be a nonempty FP32 matrix")
        view = memoryview(payload)
        expected = 3*x.shape[1]*intermediate+4*(2*intermediate+x.shape[1])
        if not view.c_contiguous or view.nbytes!=expected:
            raise ValueError("exact contiguous INT8-row expert payload required")
        if expected+4*(3*x.size+intermediate)>self.max_bytes:
            raise NativeExecutionError(5)
        x = _array(x,x.shape,self.max_bytes,"hidden")
        if not np.isfinite(x).all():
            raise ValueError('hidden must be finite')
        packed = view.tobytes()
        out = np.empty_like(x)
        self._check(self._packed(_tensor(x),len(x),x.shape[1],intermediate,
            C.c_char_p(packed),len(packed),out.ctypes.data,out.size,self.max_bytes))
        return out

    def embeddings(self, embedding, token_ids):
        w = np.asarray(embedding)
        if w.ndim != 2 or not all(w.shape):
            raise ValueError("embedding must be a nonempty matrix")
        w = _array(w, w.shape, self.max_bytes, "embedding")
        ids = np.asarray(token_ids)
        if ids.dtype != np.dtype("int64") or ids.ndim != 1 or not ids.size:
            raise ValueError("token_ids must be a nonempty native int64 vector")
        if ids.nbytes > self.max_bytes or ids.size * w.shape[1] * 4 > self.max_bytes:
            raise NativeExecutionError(5)
        ids = np.ascontiguousarray(ids)
        out = np.empty((ids.size, w.shape[1]), np.float32)
        self._check(self.lib.oa_executor_embed(_tensor(w), *w.shape, ids.ctypes.data,
            ids.size, out.ctypes.data, out.size, self.max_bytes))
        return out

    def final_head(self, hidden, norm, weight, *, epsilon=1e-5):
        x, w = np.asarray(hidden), np.asarray(weight)
        if x.ndim != 2 or w.ndim != 2 or not all(x.shape) or not all(w.shape) or x.shape[1] != w.shape[1]:
            raise ValueError("expected matching nonempty hidden and head matrices")
        x = _array(x, x.shape, self.max_bytes, "hidden")
        w = _array(w, w.shape, self.max_bytes, "weight")
        nw = _array(norm, (x.shape[1],), self.max_bytes, "norm")
        if x.shape[0] * w.shape[0] * 4 > self.max_bytes:
            raise NativeExecutionError(5)
        out = np.empty((x.shape[0], w.shape[0]), np.float32)
        self._check(self.lib.oa_executor_head(_tensor(x), *x.shape, _tensor(nw),
            _tensor(w), w.shape[0], float(epsilon), out.ctypes.data, out.size, self.max_bytes))
        return out


class NativeRange:
    """Own immutable weight copies; multiple independent KV sessions are allowed.

    gate/up: [experts, intermediate, hidden]; down: [experts, hidden,
    intermediate]. Q/K normalization is over complete projection width.
    """
    native_range_execution = True
    numeric_class = NUMERIC_CLASS

    def __init__(self, executor, layers, *, hidden, intermediate, experts, heads,
                 kv_heads, topk, capacity, epsilon, theta, normalize_topk):
        self.executor = executor
        for name, value in (("hidden", hidden), ("intermediate", intermediate),
                            ("experts", experts), ("capacity", capacity)):
            _integer(value, name)
        for name, value in (("heads", heads), ("kv_heads", kv_heads), ("topk", topk)):
            _integer(value, name, (1 << 32) - 1)
        if hidden % heads or heads % kv_heads or (hidden // heads) % 2 or topk > experts:
            raise ValueError("invalid GQA, even head dimension or topk")
        if type(normalize_topk) is not bool:
            raise ValueError("normalize_topk must be bool")
        epsilon, theta = float(epsilon), float(theta)
        if not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(theta) or theta <= 0:
            raise ValueError("epsilon and theta must be positive finite")
        layers = list(layers)
        _integer(len(layers), "layers")
        kv = kv_heads * (hidden // heads)
        self.config = RangeConfig(VERSION, heads, kv_heads, topk, int(normalize_topk), 0,
            hidden, intermediate, experts, len(layers), capacity, epsilon, theta)
        shapes = dict(input_norm=(hidden,), q=(hidden, hidden), k=(kv, hidden),
            v=(kv, hidden), o=(hidden, hidden), q_norm=(hidden,), k_norm=(kv,),
            post_norm=(hidden,), router=(experts, hidden),
            gate=(experts, intermediate, hidden), up=(experts, intermediate, hidden),
            down=(experts, hidden, intermediate))
        self._weights = []
        self._layers = (RangeLayer * len(layers))()
        for i, layer in enumerate(layers):
            if set(layer) != set(shapes):
                raise ValueError("layer fields must exactly match native range contract")
            for name, shape in shapes.items():
                array = _array(layer[name], shape, executor.max_bytes, name)
                if not np.isfinite(array).all():
                    raise ValueError(f"nonfinite {name}")
                array.flags.writeable = False
                self._weights.append(array)
                setattr(self._layers[i], name, _tensor(array))
        self.resident_weight_bytes = sum(a.nbytes for a in self._weights)

    def status(self):
        return dict(native_range_execution=True, numeric_class=self.numeric_class,
                    resident_weight_bytes=self.resident_weight_bytes,
                    expert_storage="resident-fp32-banks", sparse_streamed_experts=False,
                    max_bytes_scope="each-tensor-and-aggregate-C-scratch")

    def create_session(self):
        return NativeRangeSession(self)


class NativeRangeSession:
    native_range_execution = True
    numeric_class = NUMERIC_CLASS

    def __init__(self, engine):
        self.engine = engine
        c = engine.config
        shape = (c.layers, c.kv_heads, c.capacity, c.hidden // c.heads)
        if np.prod(shape, dtype=object) * 4 > engine.executor.max_bytes:
            raise NativeExecutionError(5)
        self._keys = np.zeros(shape, np.float32)
        self._values = np.zeros(shape, np.float32)
        self._position = _U64(0)
        self._lock = threading.RLock()
        self._closed = False
        self.kv_bytes = self._keys.nbytes + self._values.nbytes

    @property
    def position(self):
        with self._lock:
            return self._position.value

    def forward(self, hidden):
        with self._lock:
            if self._closed:
                raise RuntimeError("native session is closed")
            engine, c = self.engine, self.engine.config
            x = np.asarray(hidden)
            if x.ndim != 2 or not x.shape[0]:
                raise ValueError("hidden must be nonempty FP32 rows x hidden")
            x = _array(x, (x.shape[0], c.hidden), engine.executor.max_bytes, "hidden")
            out = np.empty_like(x)
            engine.executor._check(engine.executor.lib.oa_executor_range(C.byref(c),
                engine._layers, c.layers, _tensor(x), x.shape[0], out.ctypes.data,
                out.size, self._keys.ctypes.data, self._values.ctypes.data, self._keys.size,
                C.byref(self._position), engine.executor.max_bytes))
            return out

    def reset(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("native session is closed")
            self._keys.fill(0)
            self._values.fill(0)
            self._position.value = 0

    def close(self):
        with self._lock:
            self._closed = True
            self._keys = self._values = None
