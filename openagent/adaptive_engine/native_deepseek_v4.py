"""Length-checked ctypes binding to the original resident V4 C decoder."""

import ctypes as C

import numpy as np

from .independent_deepseek_v4 import GLOBAL_NAMES, KINDS, IndependentDeepseekV4Decoder


class Tensor(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_float)), ("count", C.c_uint64)]


class Config(C.Structure):
    _fields_ = [(name, C.c_int) for name in (
        "h", "a", "d", "r", "q", "g", "o", "i", "j", "k", "e", "top", "m", "u", "iters",
        "window", "csa", "hca", "layers", "vocab", "capacity", "activation")]
    _fields_ += [(name, C.c_float) for name in (
        "eps", "hc_eps", "limit", "route_scale", "main_theta", "compress_theta")]


class Layer(C.Structure):
    _fields_ = [("kind", C.c_int), ("hash", C.c_int), ("w", Tensor*33),
                ("table", C.POINTER(C.c_int64)), ("table_count", C.c_uint64)]


def tensor(value):
    if value is None:
        return Tensor()
    if value.dtype != np.float32 or not value.flags.c_contiguous:
        raise ValueError("native V4 tensors require contiguous FP32")
    return Tensor(value.ctypes.data_as(C.POINTER(C.c_float)), value.size)


class NativeDeepseekV4Decoder(IndependentDeepseekV4Decoder):
    implementation = "compiled-c-deepseek-v4-token-replay"

    def __init__(self, config, tensors, *, library, max_context=128, max_bytes=512*1024*1024):
        super().__init__(config, tensors, max_context=max_context, max_bytes=max_bytes)
        c = self.config
        self._c = Config(c.hidden_size, c.num_attention_heads, c.head_dim, c.qk_rope_head_dim,
                         c.q_lora_rank, c.o_groups, c.o_lora_rank, c.index_head_dim, c.index_n_heads,
                         c.index_topk, c.n_routed_experts, c.num_experts_per_tok, c.moe_intermediate_size,
                         c.hc_mult, c.hc_sinkhorn_iters, c.sliding_window,
                         c.compress_rates[KINDS[1]], c.compress_rates[KINDS[2]], c.num_hidden_layers,
                         c.vocab_size, max_context, int(c.hidden_act == "relu"),
                         c.rms_norm_eps, c.hc_eps, c.swiglu_limit,
                         c.routed_scaling_factor, *self.thetas)
        self.library = C.CDLL(str(library))
        lib = self.library
        lib.oa_deepseek_v4_version.argtypes = []
        lib.oa_deepseek_v4_version.restype = C.c_uint64
        if lib.oa_deepseek_v4_version() != 2:
            raise ValueError("unsupported private V4 ABI")
        lib.oa_deepseek_v4_geometry.argtypes = [C.POINTER(Config)]
        lib.oa_deepseek_v4_geometry.restype = C.c_int
        if lib.oa_deepseek_v4_geometry(C.byref(self._c)):
            raise ValueError("unsupported native V4 geometry")
        lib.oa_deepseek_v4_workspace.argtypes = [C.POINTER(Config), C.c_int]
        lib.oa_deepseek_v4_workspace.restype = C.c_uint64
        lib.oa_deepseek_v4_forward.argtypes = [C.POINTER(Config), C.POINTER(Tensor), C.POINTER(Layer),
                                              C.POINTER(C.c_int64), C.c_int, Tensor, C.c_uint64]
        lib.oa_deepseek_v4_forward.restype = C.c_int
        # Tied embeddings share one immutable allocation in both ABI descriptors.
        self._global = (Tensor*6)(*[tensor(self.weights[n]) for n in GLOBAL_NAMES])
        layers = []
        for i, (kind, mode) in enumerate(zip(c.layer_types, c.mlp_layer_types)):
            prefix = f"model.layers.{i}."
            w = (Tensor*33)(*[tensor(self.weights[prefix+n]) if n else Tensor()
                             for n, _ in self.layer_specs[i]])
            table = self.weights.get(prefix+"mlp.gate.tid2eid")
            pointer = table.ctypes.data_as(C.POINTER(C.c_int64)) if table is not None else None
            layers.append(Layer(KINDS.index(kind), int(mode == "hash_moe"), w,
                                pointer, table.size if table is not None else 0))
        self._layers = (Layer*c.num_hidden_layers)(*layers)

    def _evaluate(self, ids):
        output = np.empty((len(ids), self.config.vocab_size), dtype=np.float32)
        budget = self.max_bytes-self.resident_bytes
        code = self.library.oa_deepseek_v4_forward(
            C.byref(self._c), self._global, self._layers,
            ids.ctypes.data_as(C.POINTER(C.c_int64)), len(ids), tensor(output), budget)
        if code == 4:
            raise ValueError("top-k cutoff tie is outside the exact subset; state not committed")
        if code == 2:
            raise MemoryError("native V4 workspace allocation rejected; state not committed")
        if code == 3:
            raise FloatingPointError("nonfinite native V4 arithmetic; state not committed")
        if code:
            raise RuntimeError(f"native V4 kernel failed ({code}); state not committed")
        return output
