"""Qwen3-MoE decoder over a GGUF file, with routed experts streamed from disk.

The attention and output weights are dequantized once to float16 and kept in
RAM. The routed expert matrices stay in their quantized form inside the GGUF
file and are leased through `DiskExpertStore`, which keeps a bounded RAM cache
and reads misses from disk. Only the eight experts a token routes to are
touched per layer.

The forward pass follows the Qwen3-MoE definition: RMS-normalized attention
with per-head QK-norm and NeoX-style RoPE, grouped-query KV cache, a softmax
router over all experts with top-k renormalization, and SwiGLU experts.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import regex

from .expert_store import DiskExpertStore, ExpertRegion
from . import qwen3moe_native as native
from .qwen3moe_gguf import GGUFError, GGUFFile, _ggml_type, dequantize

ARCHITECTURE = "qwen3moe"
EXPERT_PARTS = ("gate", "up", "down")
QWEN2_PRE_TOKENIZE = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
TOKEN_TYPE_CONTROL = 3


@dataclass(frozen=True)
class Qwen3MoeConfig:
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_expert: int
    n_expert_used: int
    n_ff_expert: int
    rope_base: float
    eps: float
    eos_token_id: int

    @classmethod
    def from_gguf(cls, gguf: GGUFFile) -> "Qwen3MoeConfig":
        meta = gguf.metadata
        if meta.get("general.architecture") != ARCHITECTURE:
            raise GGUFError(f"expected architecture {ARCHITECTURE}, found {meta.get('general.architecture')!r}")

        def need(key: str):
            if key not in meta:
                raise GGUFError(f"GGUF metadata lacks {key}")
            return meta[key]

        config = cls(
            n_layer=int(need(f"{ARCHITECTURE}.block_count")),
            n_embd=int(need(f"{ARCHITECTURE}.embedding_length")),
            n_head=int(need(f"{ARCHITECTURE}.attention.head_count")),
            n_head_kv=int(need(f"{ARCHITECTURE}.attention.head_count_kv")),
            head_dim=int(need(f"{ARCHITECTURE}.attention.key_length")),
            n_expert=int(need(f"{ARCHITECTURE}.expert_count")),
            n_expert_used=int(need(f"{ARCHITECTURE}.expert_used_count")),
            n_ff_expert=int(need(f"{ARCHITECTURE}.expert_feed_forward_length")),
            rope_base=float(need(f"{ARCHITECTURE}.rope.freq_base")),
            eps=float(need(f"{ARCHITECTURE}.attention.layer_norm_rms_epsilon")),
            eos_token_id=int(meta.get("tokenizer.ggml.eos_token_id", 151645)),
        )
        if config.n_head % config.n_head_kv:
            raise GGUFError("attention heads must be a multiple of KV heads")
        if config.n_expert_used > config.n_expert or config.n_expert_used <= 0:
            raise GGUFError("expert_used_count is out of range")
        return config


def cfg_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * scale * weight


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / exp.sum()


@dataclass(frozen=True)
class _Quant:
    """A dense matrix kept in its quantized form inside the GGUF file."""

    ggml_type: int
    rows: int
    cols: int
    raw: object


class _Layer:
    __slots__ = ("attn_norm", "q_norm", "k_norm", "wq", "wk", "wv", "wo", "ffn_norm", "router")

    def __init__(self) -> None:
        self.attn_norm = self.q_norm = self.k_norm = None
        self.wq = self.wk = self.wv = self.wo = None
        self.ffn_norm = self.router = None


class Qwen3MoeDecoder:
    """One Qwen3-MoE GGUF model: resident dense weights plus a streamed expert store."""

    def __init__(self, gguf_path: str | Path, *, context: int = 2048, expert_cache_bytes: int = 4 << 30,
                 io_mode: str = "mmap") -> None:
        self.gguf = GGUFFile.open(gguf_path)
        self.config = Qwen3MoeConfig.from_gguf(self.gguf)
        if context <= 0:
            raise ValueError("context must be positive")
        self.context = int(context)
        self._lock = threading.RLock()
        self.timers = {"attn_proj": 0.0, "attn_core": 0.0, "router": 0.0, "experts": 0.0,
                       "moe": 0.0, "output": 0.0, "forwards": 0}
        self._expert_pool = ThreadPoolExecutor(max_workers=cfg_workers(), thread_name_prefix="qwen-expert")
        self._native_moe = native.moe_layer_available() and native.moe_pool_init(cfg_workers())
        self._last_routing: dict[int, list[int]] = {}
        self._token_loop = None
        self._native_loop_enabled = os.environ.get("OPENAGENT_QWEN3MOE_NATIVE_LOOP", "1") != "0"
        self._prefetch_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qwen-prefetch")
        self.prefetch_enabled = os.environ.get("OPENAGENT_QWEN3MOE_PREFETCH", "0") == "1"
        self._moe_scratch = np.zeros((native.MOE_MAX_TASKS, self.config.n_embd), dtype=np.float32)
        self._moe_ffn = np.zeros((native.MOE_MAX_TASKS, 2, self.config.n_ff_expert), dtype=np.float32)
        self._load_dense()
        self._register_experts(expert_cache_bytes, io_mode)
        self._rope_inv = self.config.rope_base ** (-np.arange(0, self.config.head_dim, 2, dtype=np.float64)
                                                   / self.config.head_dim)
        kv_shape = (self.context, self.config.n_head_kv, self.config.head_dim)
        self._k_cache = [np.zeros(kv_shape, dtype=np.float32) for _ in range(self.config.n_layer)]
        self._v_cache = [np.zeros(kv_shape, dtype=np.float32) for _ in range(self.config.n_layer)]

    # --- loading -----------------------------------------------------------

    def _quant(self, name: str) -> _Quant:
        info = self.gguf.tensors[name]
        if len(info.dims) != 2:
            raise GGUFError(f"{name} is not a matrix")
        return _Quant(info.ggml_type, info.dims[1], info.dims[0], self.gguf.bytes_of(name))

    def _mv(self, matrix: _Quant, x: np.ndarray) -> np.ndarray:
        return native.q8_matvec(matrix.ggml_type, matrix.raw, matrix.rows, matrix.cols, x)

    def _dense(self, name: str, dtype=np.float16) -> np.ndarray:
        info = self.gguf.tensors[name]
        values = dequantize(self.gguf.bytes_of(name), info.ggml_type, info.elements)
        if len(info.dims) == 1:
            return values.astype(np.float32)
        out_dim = info.dims[1]
        return values.reshape(out_dim, info.dims[0]).astype(dtype)

    def _load_dense(self) -> None:
        cfg = self.config
        self.token_embd = self.gguf.tensors["token_embd.weight"]
        self.output_norm = self._dense("output_norm.weight", np.float32)
        if "output.weight" not in self.gguf.tensors:
            raise GGUFError("this decoder needs an untied output.weight")
        self.output = self._quant("output.weight")
        self.layers: list[_Layer] = []
        for index in range(cfg.n_layer):
            prefix = f"blk.{index}."
            layer = _Layer()
            layer.attn_norm = self._dense(prefix + "attn_norm.weight", np.float32)
            layer.q_norm = self._dense(prefix + "attn_q_norm.weight", np.float32)
            layer.k_norm = self._dense(prefix + "attn_k_norm.weight", np.float32)
            layer.wq = self._quant(prefix + "attn_q.weight")
            layer.wk = self._quant(prefix + "attn_k.weight")
            layer.wv = self._quant(prefix + "attn_v.weight")
            layer.wo = self._quant(prefix + "attn_output.weight")
            layer.ffn_norm = self._dense(prefix + "ffn_norm.weight", np.float32)
            layer.router = self._dense(prefix + "ffn_gate_inp.weight", np.float32)
            self.layers.append(layer)
        for index in range(cfg.n_layer):
            for part in EXPERT_PARTS:
                name = f"blk.{index}.ffn_{part}_exps.weight"
                info = self.gguf.tensors[name]
                if len(info.dims) != 3 or info.dims[2] != cfg.n_expert:
                    raise GGUFError(f"{name} does not hold {cfg.n_expert} experts")

    def _expert_shape(self, part: str) -> tuple[int, int]:
        cfg = self.config
        if part == "down":
            return cfg.n_embd, cfg.n_ff_expert
        return cfg.n_ff_expert, cfg.n_embd

    def _register_experts(self, cache_bytes: int, io_mode: str) -> None:
        cfg = self.config
        regions: dict[str, ExpertRegion] = {}
        self._expert_types: dict[str, tuple[int, int]] = {}
        for index in range(cfg.n_layer):
            for part in EXPERT_PARTS:
                name = f"blk.{index}.ffn_{part}_exps.weight"
                info = self.gguf.tensors[name]
                rows, cols = self._expert_shape(part)
                elements = rows * cols
                _, per_block, block_bytes = _ggml_type(info.ggml_type)
                per_expert = elements // per_block * block_bytes
                self._expert_types[f"{index}:{part}"] = (info.ggml_type, elements)
                for expert in range(cfg.n_expert):
                    regions[f"{index}:{expert}:{part}"] = ExpertRegion(
                        path=self.gguf.path.name,
                        offset=info.absolute + expert * per_expert,
                        size=per_expert,
                    )
        self.store = DiskExpertStore(
            self.gguf.path.parent, regions,
            capacity_bytes=max(cache_bytes, max(r.size for r in regions.values())),
            io_mode=io_mode,
            # Routing has already selected the complete current-layer working
            # set. Read its bounded regions concurrently before the native
            # workers begin, so mapped/buffered misses can overlap each other.
            # The store still enforces the configured byte limit.
            prefetch_workers=max(1, min(8, int(os.environ.get("OPENAGENT_QWEN3MOE_PREFETCH_WORKERS", "4")))),
            prefetch_depth=max(1, min(24, int(os.environ.get("OPENAGENT_QWEN3MOE_PREFETCH_DEPTH", "24")))),
        )

    def close(self) -> None:
        self.store.close()
        self.gguf.close()

    # --- kernels -----------------------------------------------------------

    def _embed(self, token: int) -> np.ndarray:
        info = self.token_embd
        cfg = self.config
        if not 0 <= token < info.dims[1]:
            raise ValueError(f"token {token} is outside the vocabulary")
        _, per_block, block_bytes = _ggml_type(info.ggml_type)
        row_bytes = cfg.n_embd // per_block * block_bytes
        start = info.absolute + token * row_bytes
        raw = self.gguf.bytes_at(start, row_bytes)
        return dequantize(raw, info.ggml_type, cfg.n_embd)

    def _expert_matvec(self, layer: int, expert: int, part: str, x: np.ndarray) -> np.ndarray:
        ggml_type, _ = self._expert_types[f"{layer}:{part}"]
        rows, cols = self._expert_shape(part)
        with self.store.acquire(f"{layer}:{expert}:{part}") as view:
            return native.matvec(ggml_type, view, rows, cols, x, split=False)

    def _moe_native(self, layer_index: int, experts: list[int], weights: np.ndarray, h: np.ndarray) -> np.ndarray:
        from contextlib import ExitStack

        cfg = self.config
        gate_type = self._expert_types[f"{layer_index}:gate"][0]
        up_type = self._expert_types[f"{layer_index}:up"][0]
        down_type = self._expert_types[f"{layer_index}:down"][0]
        with ExitStack() as stack:
            views = {}
            for expert in experts:
                for part in EXPERT_PARTS:
                    views[(expert, part)] = stack.enter_context(
                        self.store.acquire(f"{layer_index}:{expert}:{part}"))
            gate = [np.frombuffer(views[(e, "gate")], dtype=np.uint8).ctypes.data for e in experts]
            up = [np.frombuffer(views[(e, "up")], dtype=np.uint8).ctypes.data for e in experts]
            down = [np.frombuffer(views[(e, "down")], dtype=np.uint8).ctypes.data for e in experts]
            out = np.empty(cfg.n_embd, dtype=np.float32)
            native.moe_layer(gate, up, down, gate_type, up_type, down_type, cfg.n_embd, cfg.n_ff_expert,
                             weights, h.astype(np.float32, copy=False), self._moe_scratch,
                             self._moe_ffn, out)
        return out

    def _touch_experts(self, layer_index: int, experts: list[int]) -> None:
        """Speculative read-ahead: fault the predicted experts into the cache in the background.

        The next layer's router is not known yet, so the previous token's routing for
        that layer is used. Each page is touched here so the read happens off the
        compute path. Speculative reads hold no lease and never fail a request.
        """
        for expert in experts:
            for part in EXPERT_PARTS:
                try:
                    with self.store.acquire(f"{layer_index}:{expert}:{part}") as view:
                        np.frombuffer(view, dtype=np.uint8)[::4096].sum()
                except (KeyError, RuntimeError, MemoryError, InterruptedError, ValueError):
                    return

    def _one_expert(self, layer_index: int, expert: int, h: np.ndarray) -> np.ndarray:
        gate = self._expert_matvec(layer_index, expert, "gate", h)
        up = self._expert_matvec(layer_index, expert, "up", h)
        hidden = _silu(gate) * up
        return self._expert_matvec(layer_index, expert, "down", hidden)

    def _moe(self, layer_index: int, layer: _Layer, h: np.ndarray) -> np.ndarray:
        cfg = self.config
        timers = self.timers
        clock = time.perf_counter()
        probs = _softmax(layer.router @ h)
        top = np.argpartition(-probs, cfg.n_expert_used - 1)[: cfg.n_expert_used]
        weights = probs[top]
        weights = weights / weights.sum()
        timers["router"] += time.perf_counter() - clock
        clock = time.perf_counter()
        experts = top.tolist()
        self._last_routing[layer_index] = experts
        if self._native_moe:
            out = self._moe_native(layer_index, experts, weights.astype(np.float32), h)
            timers["experts"] += time.perf_counter() - clock
            return out
        outputs = list(self._expert_pool.map(lambda e: self._one_expert(layer_index, e, h), experts))
        out = np.zeros(cfg.n_embd, dtype=np.float32)
        for weight, contribution in zip(weights.tolist(), outputs):
            out += weight * contribution
        timers["experts"] += time.perf_counter() - clock
        return out

    def _rope(self, x: np.ndarray, position: int) -> np.ndarray:
        half = self.config.head_dim // 2
        angle = position * self._rope_inv
        cos = np.cos(angle).astype(np.float32)
        sin = np.sin(angle).astype(np.float32)
        first, second = x[:, :half], x[:, half:]
        return np.concatenate([first * cos - second * sin, second * cos + first * sin], axis=1)

    # --- forward -----------------------------------------------------------

    def _ensure_native_loop(self):
        if self._token_loop is None and self._native_loop_enabled:
            try:
                from .qwen3moe_native_loop import NativeTokenLoop

                self._token_loop = NativeTokenLoop(self)
            except (RuntimeError, MemoryError, OSError):
                self._native_loop_enabled = False
        return self._token_loop

    def forward(self, token: int, position: int) -> np.ndarray:
        loop = self._ensure_native_loop()
        if loop is not None and os.environ.get("OPENAGENT_QWEN3MOE_PYTHON_FORWARD", "0") != "1":
            return loop.forward(token, position)
        return self._forward_python(token, position)

    def _forward_python(self, token: int, position: int) -> np.ndarray:
        cfg = self.config
        if not 0 <= position < self.context:
            raise ValueError("position exceeds the context window")
        group = cfg.n_head // cfg.n_head_kv
        timers = self.timers
        timers["forwards"] += 1
        x = self._embed(token)
        for index, layer in enumerate(self.layers):
            clock = time.perf_counter()
            if self.prefetch_enabled and index + 1 < len(self.layers):
                predicted = self._last_routing.get(index + 1)
                if predicted:
                    self._prefetch_pool.submit(self._touch_experts, index + 1, predicted)
            h = _rms_norm(x, layer.attn_norm, cfg.eps)
            q = self._mv(layer.wq, h).reshape(cfg.n_head, cfg.head_dim)
            k = self._mv(layer.wk, h).reshape(cfg.n_head_kv, cfg.head_dim)
            v = self._mv(layer.wv, h).reshape(cfg.n_head_kv, cfg.head_dim)
            timers["attn_proj"] += time.perf_counter() - clock
            clock = time.perf_counter()
            q = _rms_norm(q, layer.q_norm, cfg.eps)
            k = _rms_norm(k, layer.k_norm, cfg.eps)
            q = self._rope(q, position)
            k = self._rope(k, position)
            self._k_cache[index][position] = k
            self._v_cache[index][position] = v
            keys = self._k_cache[index][: position + 1]
            values = self._v_cache[index][: position + 1]
            attended = np.empty((cfg.n_head, cfg.head_dim), dtype=np.float32)
            scale = 1.0 / math.sqrt(cfg.head_dim)
            for kv in range(cfg.n_head_kv):
                heads = q[kv * group:(kv + 1) * group]
                scores = heads @ keys[:, kv, :].T * scale
                scores = scores - scores.max(axis=1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(axis=1, keepdims=True)
                attended[kv * group:(kv + 1) * group] = probs @ values[:, kv, :]
            timers["attn_core"] += time.perf_counter() - clock
            clock = time.perf_counter()
            x = x + self._mv(layer.wo, attended.reshape(-1))
            timers["attn_proj"] += time.perf_counter() - clock
            clock = time.perf_counter()
            h2 = _rms_norm(x, layer.ffn_norm, cfg.eps)
            x = x + self._moe(index, layer, h2)
            timers["moe"] += time.perf_counter() - clock
        clock = time.perf_counter()
        x = _rms_norm(x, self.output_norm, cfg.eps)
        logits = self._mv(self.output, x)
        timers["output"] += time.perf_counter() - clock
        return logits

    def generate(self, prompt_ids: list[int], *, max_new_tokens: int) -> list[int]:
        """Greedy decoding. Stops at the end-of-turn token or the context limit."""
        return list(self.generate_iter(prompt_ids, max_new_tokens=max_new_tokens))

    def generate_iter(self, prompt_ids: list[int], *, max_new_tokens: int, cancelled=None):
        """Yield greedy tokens one at a time as each forward pass completes.

        `cancelled` is polled before every forward pass; a true value stops the
        stream. Closing the generator (or breaking out of it) releases the lock.
        """
        if not prompt_ids:
            raise ValueError("prompt is empty")
        if len(prompt_ids) + max_new_tokens > self.context:
            raise ValueError("prompt and output exceed the context window")
        if cancelled is not None and not callable(cancelled):
            raise TypeError("cancelled must be callable")
        with self._lock:
            logits = None
            for position, token in enumerate(prompt_ids):
                if cancelled is not None and cancelled():
                    return
                logits = self.forward(token, position)
            position = len(prompt_ids)
            for _ in range(max_new_tokens):
                if cancelled is not None and cancelled():
                    return
                token = int(np.argmax(logits))
                if token == self.config.eos_token_id:
                    return
                yield token
                if cancelled is not None and cancelled():
                    return
                logits = self.forward(token, position)
                position += 1


# --- tokenizer ---------------------------------------------------------------

def _bytes_to_unicode() -> dict[int, str]:
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, (chr(c) for c in chars)))


class ByteLevelBPE:
    """GPT-2 style byte-level BPE, built from the GGUF tokenizer metadata."""

    def __init__(self, gguf: GGUFFile) -> None:
        meta = gguf.metadata
        if meta.get("tokenizer.ggml.model") != "gpt2":
            raise GGUFError("only byte-level BPE tokenizers are supported")
        tokens = meta.get("tokenizer.ggml.tokens")
        merges = meta.get("tokenizer.ggml.merges")
        if not isinstance(tokens, list) or not isinstance(merges, list):
            raise GGUFError("GGUF lacks tokenizer vocabulary or merges")
        self.id_of = {token: index for index, token in enumerate(tokens)}
        self.token_of = tokens
        types = meta.get("tokenizer.ggml.token_type") or []
        self.specials = sorted(
            (tokens[i] for i, kind in enumerate(types) if kind == TOKEN_TYPE_CONTROL and i < len(tokens)),
            key=len, reverse=True,
        )
        self.special_pattern = (
            regex.compile("(" + "|".join(regex.escape(s) for s in self.specials) + ")") if self.specials else None
        )
        self.ranks = {}
        for rank, merge in enumerate(merges):
            left, _, right = merge.partition(" ")
            self.ranks[(left, right)] = rank
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._cache: dict[str, list[int]] = {}

    def _bpe(self, word: str) -> list[str]:
        parts = list(word)
        while len(parts) > 1:
            best = None
            best_rank = None
            for index in range(len(parts) - 1):
                rank = self.ranks.get((parts[index], parts[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = index, rank
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in QWEN2_PRE_TOKENIZE.findall(text):
            cached = self._cache.get(piece)
            if cached is None:
                mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
                cached = []
                for symbol in self._bpe(mapped):
                    if symbol not in self.id_of:
                        raise GGUFError(f"tokenizer has no id for symbol {symbol!r}")
                    cached.append(self.id_of[symbol])
                if len(self._cache) < 100000:
                    self._cache[piece] = cached
            ids.extend(cached)
        return ids

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return self._encode_plain(text)
        ids: list[int] = []
        for part in self.special_pattern.split(text):
            if not part:
                continue
            if part in self.id_of and part in self.specials:
                ids.append(self.id_of[part])
            else:
                ids.extend(self._encode_plain(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.token_of[i] for i in ids)
        data = bytes(self.byte_decoder.get(ch, ord("?")) for ch in text)
        return data.decode("utf-8", errors="replace")


def chat_prompt(messages: list[dict[str, str]]) -> str:
    """Qwen chat format: each message is a role-tagged block, then the assistant turn opens."""
    parts = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role not in {"system", "user", "assistant"}:
            role = "user"
        parts.append(f"<|im_start|>{role}\n{message.get('content', '')}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)
