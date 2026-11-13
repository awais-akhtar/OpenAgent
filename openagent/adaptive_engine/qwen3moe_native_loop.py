"""One native call per generated token for the Qwen3-MoE decoder.

The C side (`qwen3moe_token` in native/qwen3moe_kernels.c) runs every layer:
RMS norms, Q8 attention projections, QK norm, RoPE, KV-cache writes, attention,
the router with top-k, the routed experts and the output head. Python supplies
the weight addresses, KV caches and two callbacks. The expert cache stays in
Python: `acquire` leases the routed experts for one layer and `release` returns
them, so the bounded cache and its lease rules are unchanged.
"""

from __future__ import annotations

import ctypes
from contextlib import ExitStack

import numpy as np

from . import qwen3moe_native as native

EXPERT_PARTS = ("gate", "up", "down")


class _Context(ctypes.Structure):
    _fields_ = [
        ("n_layer", ctypes.c_int64), ("n_embd", ctypes.c_int64), ("n_head", ctypes.c_int64),
        ("n_head_kv", ctypes.c_int64), ("head_dim", ctypes.c_int64), ("n_ff", ctypes.c_int64),
        ("n_expert", ctypes.c_int64), ("n_used", ctypes.c_int64), ("ctx", ctypes.c_int64),
        ("vocab", ctypes.c_int64),
        ("eps", ctypes.c_double), ("scale", ctypes.c_double),
        ("output_norm", ctypes.c_int64), ("output_w", ctypes.c_int64), ("rope_inv", ctypes.c_int64),
        ("output_type", ctypes.c_int32), ("gate_type", ctypes.c_int32),
        ("down_type", ctypes.c_int32), ("pad", ctypes.c_int32),
        ("layers", ctypes.c_int64), ("k_cache", ctypes.c_int64), ("v_cache", ctypes.c_int64),
        ("acquire_cb", ctypes.c_int64), ("release_cb", ctypes.c_int64), ("buf", ctypes.c_int64),
    ]


class _Layer(ctypes.Structure):
    _fields_ = [
        ("attn_norm", ctypes.c_int64), ("q_norm", ctypes.c_int64), ("k_norm", ctypes.c_int64),
        ("ffn_norm", ctypes.c_int64), ("router", ctypes.c_int64),
        ("wq", ctypes.c_int64), ("wk", ctypes.c_int64), ("wv", ctypes.c_int64), ("wo", ctypes.c_int64),
        ("tq", ctypes.c_int32), ("tk", ctypes.c_int32), ("tv", ctypes.c_int32), ("to", ctypes.c_int32),
        ("tgate", ctypes.c_int32), ("tup", ctypes.c_int32), ("tdown", ctypes.c_int32), ("tpad", ctypes.c_int32),
    ]


ACQUIRE = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
)
RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_int64)


def _address(array: np.ndarray) -> int:
    return int(array.ctypes.data)


class NativeTokenLoop:
    """Owns the native context for one decoder. Built once; reused for every token."""

    def __init__(self, decoder) -> None:
        self.decoder = decoder
        cfg = decoder.config
        self._keep: list = []
        library = native._load()
        if library is None or not hasattr(library, "qwen3moe_token"):
            raise RuntimeError("native token loop is unavailable")
        self._library = library
        library.qwen3moe_token_init.argtypes = [ctypes.POINTER(_Context)]
        library.qwen3moe_token_init.restype = ctypes.c_int
        library.qwen3moe_token_free.argtypes = [ctypes.POINTER(_Context)]
        library.qwen3moe_token_free.restype = None
        library.qwen3moe_token.argtypes = [
            ctypes.POINTER(_Context), ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
        ]
        library.qwen3moe_token.restype = ctypes.c_int

        layers = (_Layer * cfg.n_layer)()
        for index, layer in enumerate(decoder.layers):
            for field, array in (("attn_norm", layer.attn_norm), ("q_norm", layer.q_norm),
                                 ("k_norm", layer.k_norm), ("ffn_norm", layer.ffn_norm),
                                 ("router", layer.router)):
                contiguous = np.ascontiguousarray(array, dtype=np.float32)
                self._keep.append(contiguous)
                setattr(layers[index], field, _address(contiguous))
            for field, quant in (("wq", layer.wq), ("wk", layer.wk), ("wv", layer.wv), ("wo", layer.wo)):
                buffer = np.frombuffer(quant.raw, dtype=np.uint8)
                self._keep.append(buffer)
                setattr(layers[index], field, _address(buffer))
                setattr(layers[index], "t" + field[1], quant.ggml_type)
            types = decoder._expert_types
            layers[index].tgate = types[f"{index}:gate"][0]
            layers[index].tup = types[f"{index}:up"][0]
            layers[index].tdown = types[f"{index}:down"][0]
        self._keep.append(layers)

        out_norm = np.ascontiguousarray(decoder.output_norm, dtype=np.float32)
        out_buffer = np.frombuffer(decoder.output.raw, dtype=np.uint8)
        inverse = np.ascontiguousarray(decoder._rope_inv, dtype=np.float64)
        self._keep.extend([out_norm, out_buffer, inverse])

        kv_shape = (cfg.n_layer, decoder.context, cfg.n_head_kv, cfg.head_dim)
        self._k = np.zeros(kv_shape, dtype=np.float32)
        self._v = np.zeros(kv_shape, dtype=np.float32)
        self._keep.extend([self._k, self._v])

        ctx = _Context()
        ctx.n_layer = cfg.n_layer
        ctx.n_embd = cfg.n_embd
        ctx.n_head = cfg.n_head
        ctx.n_head_kv = cfg.n_head_kv
        ctx.head_dim = cfg.head_dim
        ctx.n_ff = cfg.n_ff_expert
        ctx.n_expert = cfg.n_expert
        ctx.n_used = cfg.n_expert_used
        ctx.ctx = decoder.context
        ctx.vocab = decoder.output.rows
        ctx.eps = cfg.eps
        ctx.scale = 1.0 / float(np.sqrt(cfg.head_dim))
        ctx.output_norm = _address(out_norm)
        ctx.output_w = _address(out_buffer)
        ctx.rope_inv = _address(inverse)
        ctx.output_type = decoder.output.ggml_type
        ctx.layers = ctypes.addressof(layers)
        ctx.k_cache = _address(self._k)
        ctx.v_cache = _address(self._v)
        self._leases: dict[int, ExitStack] = {}
        self._acquire = ACQUIRE(self._acquire_experts)
        self._release = RELEASE(self._release_experts)
        ctx.acquire_cb = ctypes.cast(self._acquire, ctypes.c_void_p).value
        ctx.release_cb = ctypes.cast(self._release, ctypes.c_void_p).value
        self._ctx = ctx
        if library.qwen3moe_token_init(ctypes.byref(ctx)) != 0:
            raise MemoryError("native token loop could not allocate its buffers")
        self._logits = np.zeros(decoder.output.rows, dtype=np.float32)

    def _acquire_experts(self, layer, experts, count, gate, up, down):
        try:
            stack = ExitStack()
            ids = [int(experts[i]) for i in range(count)]
            views = {}
            for expert in ids:
                for part in EXPERT_PARTS:
                    views[(expert, part)] = stack.enter_context(
                        self.decoder.store.acquire(f"{layer}:{expert}:{part}"))
            self._leases[int(layer)] = (stack, views)
            for i, expert in enumerate(ids):
                gate[i] = np.frombuffer(views[(expert, "gate")], dtype=np.uint8).ctypes.data
                up[i] = np.frombuffer(views[(expert, "up")], dtype=np.uint8).ctypes.data
                down[i] = np.frombuffer(views[(expert, "down")], dtype=np.uint8).ctypes.data
            return 0
        except Exception:  # reported through the native status code
            self._release_experts(layer)
            return -1

    def _release_experts(self, layer):
        held = self._leases.pop(int(layer), None)
        if held is not None:
            held[0].close()

    def forward(self, token: int, position: int) -> np.ndarray:
        decoder = self.decoder
        if not 0 <= position < decoder.context:
            raise ValueError("position exceeds the context window")
        emb = np.ascontiguousarray(decoder._embed(token), dtype=np.float32)
        status = self._library.qwen3moe_token(
            ctypes.byref(self._ctx), emb.ctypes.data, position, self._logits.ctypes.data)
        if status != 0:
            raise RuntimeError(f"native token loop failed with status {status}")
        return self._logits.copy()

    def close(self) -> None:
        self._library.qwen3moe_token_free(ctypes.byref(self._ctx))
