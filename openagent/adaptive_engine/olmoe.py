"""Independent OLMoE checkpoint contract for OpenAgent's sparse runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .expert_packing import (
    ExpertLayerBinding,
    ExpertTensorBinding,
    pack_safetensors_directory_int8,
    _read_float_matrix,
    _scan_directory,
)
from .expert_runtime import SparseExpertManifest, SparseExpertRuntime


@dataclass(frozen=True)
class OlmoeConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    norm_topk_prob: bool
    eos_token_id: int

    @classmethod
    def read(cls, model_directory: str | Path) -> "OlmoeConfig":
        path = Path(model_directory).resolve(strict=True) / "config.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read OLMoE config: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("model_type") != "olmoe":
            raise ValueError("checkpoint is not an OLMoE model")
        if type(raw.get("norm_topk_prob", False)) is not bool:
            raise ValueError("norm_topk_prob must be a boolean")
        config = cls(
            hidden_size=_positive_int(raw, "hidden_size"),
            intermediate_size=_positive_int(raw, "intermediate_size"),
            num_hidden_layers=_positive_int(raw, "num_hidden_layers"),
            num_attention_heads=_positive_int(raw, "num_attention_heads"),
            num_key_value_heads=_positive_int(raw, "num_key_value_heads"),
            num_experts=_positive_int(raw, "num_experts"),
            num_experts_per_tok=_positive_int(raw, "num_experts_per_tok"),
            vocab_size=_positive_int(raw, "vocab_size"),
            max_position_embeddings=_positive_int(raw, "max_position_embeddings"),
            rms_norm_eps=_positive_float(raw, "rms_norm_eps"),
            rope_theta=_positive_float(raw, "rope_theta"),
            norm_topk_prob=bool(raw.get("norm_topk_prob", False)),
            eos_token_id=_nonnegative_int(raw, "eos_token_id", default=50279),
        )
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("attention heads must be divisible by key/value heads")
        if config.num_experts_per_tok > config.num_experts:
            raise ValueError("num_experts_per_tok exceeds num_experts")
        if (config.hidden_size // config.num_attention_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if config.eos_token_id >= config.vocab_size:
            raise ValueError("eos_token_id must be inside the vocabulary")
        return config

    def expert_bindings(self) -> tuple[ExpertLayerBinding, ...]:
        return tuple(
            ExpertLayerBinding(
                id=f"model.layers.{layer}",
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                experts_used=self.num_experts_per_tok,
                experts=tuple(
                    ExpertTensorBinding(
                        id=str(expert),
                        gate=f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                        up=f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                        down=f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                    )
                    for expert in range(self.num_experts)
                ),
            )
            for layer in range(self.num_hidden_layers)
        )


def pack_olmoe_experts(
    model_directory: str | Path,
    destination: str | Path,
    *,
    model_id: str = "allenai/OLMoE-1B-7B-0125-Instruct",
    progress: Callable[[int, int, str, str], None] | None = None,
    storage_destinations: dict[str, str | Path] | None = None,
    dtype: str = "int8-row",
    lattice_contract=None,
    iq3_grid=None,
    max_conversion_bytes: int = 1024*1024*1024,
) -> SparseExpertManifest:
    root = Path(model_directory).resolve(strict=True)
    config = OlmoeConfig.read(root)
    fingerprint = olmoe_metadata_fingerprint(root)
    return pack_safetensors_directory_int8(
        root,
        destination,
        model_id=model_id,
        architecture="olmoe",
        fingerprint=fingerprint,
        layers=config.expert_bindings(),
        progress=progress,
        storage_destinations=storage_destinations,
        dtype=dtype,
        lattice_contract=lattice_contract,
        iq3_grid=iq3_grid,
        max_conversion_bytes=max_conversion_bytes,
    )


def olmoe_metadata_fingerprint(model_directory: str | Path) -> str:
    """Bind generated artifacts to config, index and exact shard sizes."""
    root = Path(model_directory).resolve(strict=True)
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        path = root / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError("OLMoE checkpoint has no SafeTensors shards")
    for shard in shards:
        digest.update(shard.name.encode("utf-8"))
        digest.update(shard.stat().st_size.to_bytes(8, "little"))
    return "sha256:" + digest.hexdigest()


def dense_execution_fingerprint(tensors) -> str:
    """Hash loaded dense values without duplicating contiguous FP32 buffers."""
    digest = hashlib.sha256()
    for name, value in tensors:
        encoded = json.dumps([name, list(value.shape)], separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(memoryview(np.ascontiguousarray(value, dtype="<f4")).cast("B"))
    return digest.hexdigest()


def validate_olmoe_tensors(config: OlmoeConfig, tensors: dict) -> None:
    """Check the complete tensor contract without loading weight payloads."""
    hidden = config.hidden_size
    kv_width = config.num_key_value_heads * (hidden // config.num_attention_heads)
    expected = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "lm_head.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        for suffix, shape in {
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
            "self_attn.q_proj.weight": (hidden, hidden),
            "self_attn.k_proj.weight": (kv_width, hidden),
            "self_attn.v_proj.weight": (kv_width, hidden),
            "self_attn.o_proj.weight": (hidden, hidden),
            "self_attn.q_norm.weight": (hidden,),
            "self_attn.k_norm.weight": (kv_width,),
            "mlp.gate.weight": (config.num_experts, hidden),
        }.items():
            expected[prefix + suffix] = shape
        for expert in range(config.num_experts):
            base = prefix + f"mlp.experts.{expert}."
            expected[base + "gate_proj.weight"] = (config.intermediate_size, hidden)
            expected[base + "up_proj.weight"] = (config.intermediate_size, hidden)
            expected[base + "down_proj.weight"] = (hidden, config.intermediate_size)
    if set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))
        extra = sorted(set(tensors) - set(expected))
        raise ValueError(f"OLMoE tensor names differ: missing={missing}, unexpected={extra}")
    for name, shape in expected.items():
        metadata = tensors[name][1]
        if tuple(metadata["shape"]) != shape:
            raise ValueError(f"OLMoE tensor {name} shape differs: expected {shape}")
        if metadata["dtype"] not in {"F32", "F16", "BF16"}:
            raise ValueError(f"OLMoE tensor {name} has unsupported dtype")


class OlmoeModel:
    """Complete NumPy OLMoE decoder using OpenAgent's streamed expert tier.

    Dense tensors remain resident in FP32. Routed experts use the independently
    generated row-wise INT8, opt-in INT4 or Q8_0 container. This backend favors explicit
    correctness and evidence over native-kernel throughput.
    """

    def __init__(
        self,
        model_directory: str | Path,
        manifest: SparseExpertManifest | str | Path,
        *,
        expert_cache_bytes: int,
        context_tokens: int | None = None,
        io_mode: str = "mmap",
        storage_roots: dict[str, str | Path] | None = None,
        prefetch_workers: int = 2,
        prefetch_depth: int = 8,
        kernels=None,
        preferred_kernel: str | None = None,
    ) -> None:
        self.root = Path(model_directory).resolve(strict=True)
        self.config = OlmoeConfig.read(self.root)
        self.manifest = SparseExpertManifest.read(manifest) if isinstance(manifest, (str, Path)) else manifest
        if self.manifest.architecture != "olmoe":
            raise ValueError("expert manifest is not for OLMoE")
        expected_fingerprint = olmoe_metadata_fingerprint(self.root)
        if self.manifest.fingerprint != expected_fingerprint:
            raise ValueError("expert manifest does not match the configured OLMoE checkpoint")
        if len(self.manifest.layers) != self.config.num_hidden_layers:
            raise ValueError("expert manifest layer count does not match OLMoE config")
        for index, layer in enumerate(self.manifest.layers):
            if (
                layer.id != f"model.layers.{index}"
                or layer.hidden_size != self.config.hidden_size
                or layer.intermediate_size != self.config.intermediate_size
                or layer.experts_used != self.config.num_experts_per_tok
                or set(layer.regions) != {str(i) for i in range(self.config.num_experts)}
                or layer.dtype not in {"int8-row", "int2-row", "int3-grouped-g64", "int4-row", "int4-grouped-g64", "mxfp4-e2m1-g32-ue8m0", "q8_0", "q8_1", "q4_0", "q4_1", "q5_0", "q5_1", "fp8-e4m3-b128-f32"}
            ):
                raise ValueError(f"expert manifest layer {index} does not match OLMoE config")
            if any(not region.sha256 for region in layer.regions.values()):
                raise ValueError("OLMoE expert regions require SHA-256 content hashes")
        self.context_tokens = self.config.max_position_embeddings if context_tokens is None else context_tokens
        if (type(self.context_tokens) is not int or self.context_tokens <= 0
                or self.context_tokens > self.config.max_position_embeddings):
            raise ValueError("context_tokens exceeds the checkpoint limit")
        self._index = _scan_directory(self.root)
        validate_olmoe_tensors(self.config, self._index)
        self.embed = self._tensor("model.embed_tokens.weight", (self.config.vocab_size, self.config.hidden_size))
        self.lm_head = self._tensor("lm_head.weight", (self.config.vocab_size, self.config.hidden_size))
        self.final_norm = self._tensor("model.norm.weight", (self.config.hidden_size,))
        self.layers = [self._load_layer(index) for index in range(self.config.num_hidden_layers)]
        from .expert_store import DEFAULT_EXPERT_KERNELS

        registry = kernels or DEFAULT_EXPERT_KERNELS
        identity = [{
            "id": layer.id, "dtype": layer.dtype,
            "hidden": layer.hidden_size, "intermediate": layer.intermediate_size,
            "experts": {key: {"sha256": region.sha256, "size": region.size}
                        for key, region in layer.regions.items()},
            "backend": registry.resolve(dtype=layer.dtype, preferred=preferred_kernel,
                                        allow_fallback=preferred_kernel is None).info().name,
        } for layer in self.manifest.layers]
        dense = [("embeddings", self.embed), ("lm_head", self.lm_head), ("final_norm", self.final_norm)]
        dense.extend((f"layer.{index}.{key}", value)
                     for index, layer in enumerate(self.layers) for key, value in sorted(layer.items()))
        self._state_identity = hashlib.sha256(json.dumps({
            "experts": identity, "dense": dense_execution_fingerprint(dense),
        }, sort_keys=True).encode()).hexdigest()
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self._k = [np.zeros((self.config.num_key_value_heads, self.context_tokens, head_dim), dtype=np.float32)
                   for _ in range(self.config.num_hidden_layers)]
        self._v = [np.zeros_like(value) for value in self._k]
        self.position = 0
        self._tokenizer = None
        # Validate identity and allocate KV before opening stores and worker threads.
        self.experts = SparseExpertRuntime(
            self.root,
            self.manifest,
            capacity_bytes=expert_cache_bytes,
            io_mode=io_mode,
            storage_roots=storage_roots,
            prefetch_workers=prefetch_workers,
            prefetch_depth=prefetch_depth,
            kernels=kernels,
            preferred_kernel=preferred_kernel,
            allow_kernel_fallback=preferred_kernel is None,
        )

    def _tensor(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = _read_float_matrix(self._index, name)
        if value.shape != shape:
            raise ValueError(f"tensor {name} has shape {value.shape}; expected {shape}")
        flat = value.reshape(-1)
        for offset in range(0, flat.size, 65536):
            if not np.isfinite(flat[offset:offset + 65536]).all():
                raise ValueError(f"tensor {name} must contain only finite weights")
        return np.ascontiguousarray(value, dtype=np.float32)

    def _load_layer(self, index: int) -> dict[str, np.ndarray]:
        prefix = f"model.layers.{index}."
        hidden = self.config.hidden_size
        kv_width = self.config.num_key_value_heads * (hidden // self.config.num_attention_heads)
        return {
            "input_norm": self._tensor(prefix + "input_layernorm.weight", (hidden,)),
            "post_norm": self._tensor(prefix + "post_attention_layernorm.weight", (hidden,)),
            "q": self._tensor(prefix + "self_attn.q_proj.weight", (hidden, hidden)),
            "k": self._tensor(prefix + "self_attn.k_proj.weight", (kv_width, hidden)),
            "v": self._tensor(prefix + "self_attn.v_proj.weight", (kv_width, hidden)),
            "o": self._tensor(prefix + "self_attn.o_proj.weight", (hidden, hidden)),
            "q_norm": self._tensor(prefix + "self_attn.q_norm.weight", (hidden,)),
            "k_norm": self._tensor(prefix + "self_attn.k_norm.weight", (kv_width,)),
            "router": self._tensor(prefix + "mlp.gate.weight", (self.config.num_experts, hidden)),
        }

    def reset(self) -> None:
        self.position = 0
        for key, value in zip(self._k, self._v):
            key.fill(0)
            value.fill(0)

    def forward(self, token_ids) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or not ids.size:
            raise ValueError("token_ids must be a non-empty vector")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("token id is outside the OLMoE vocabulary")
        if self.position + ids.size > self.context_tokens:
            raise ValueError("OLMoE context capacity exceeded")
        x = self.embed[ids].copy()
        start = self.position
        for layer_index, layer in enumerate(self.layers):
            normalized = _rms_norm(x, layer["input_norm"], self.config.rms_norm_eps)
            x += self._attention(layer_index, layer, normalized, start)
            normalized = _rms_norm(x, layer["post_norm"], self.config.rms_norm_eps)
            probabilities = _softmax(normalized @ layer["router"].T)
            order = np.argsort(-probabilities, axis=1, kind="stable")[:, :self.config.num_experts_per_tok]
            weights = np.take_along_axis(probabilities, order, axis=1)
            if self.config.norm_topk_prob:
                weights /= np.sum(weights, axis=1, keepdims=True)
            routes = [
                [(str(expert), float(weight)) for expert, weight in zip(row_ids, row_weights)]
                for row_ids, row_weights in zip(order, weights)
            ]
            x += self.experts.execute_batch(f"model.layers.{layer_index}", normalized, routes)
        self.position += ids.size
        final = _rms_norm(x[-1:], self.final_norm, self.config.rms_norm_eps)[0]
        return self.lm_head @ final

    def _attention(self, layer_index: int, layer: dict[str, np.ndarray], x: np.ndarray, start: int) -> np.ndarray:
        config = self.config
        sequence = x.shape[0]
        head_dim = config.hidden_size // config.num_attention_heads
        q = _rms_norm(x @ layer["q"].T, layer["q_norm"], config.rms_norm_eps)
        k = _rms_norm(x @ layer["k"].T, layer["k_norm"], config.rms_norm_eps)
        v = x @ layer["v"].T
        q = q.reshape(sequence, config.num_attention_heads, head_dim)
        k = k.reshape(sequence, config.num_key_value_heads, head_dim)
        v = v.reshape(sequence, config.num_key_value_heads, head_dim)
        positions = np.arange(start, start + sequence, dtype=np.float32)
        q = _rope(q, positions, config.rope_theta)
        k = _rope(k, positions, config.rope_theta)
        self._k[layer_index][:, start:start + sequence] = np.transpose(k, (1, 0, 2))
        self._v[layer_index][:, start:start + sequence] = np.transpose(v, (1, 0, 2))
        groups = config.num_attention_heads // config.num_key_value_heads
        keys = np.repeat(self._k[layer_index][:, :start + sequence], groups, axis=0)
        values = np.repeat(self._v[layer_index][:, :start + sequence], groups, axis=0)
        context = np.empty_like(q)
        scale = np.float32(1.0 / np.sqrt(head_dim))
        for row in range(sequence):
            available = start + row + 1
            queries = q[row]
            scores = np.einsum("hd,htd->ht", queries, keys[:, :available]) * scale
            attention = _softmax(scores)
            context[row] = np.einsum("ht,htd->hd", attention, values[:, :available])
        return context.reshape(sequence, config.hidden_size) @ layer["o"].T

    def generate(self, prompt: str, *, max_tokens: int = 64, temperature: float = 0.0) -> str:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        ids = self.encode(prompt)
        generated = self.generate_ids(ids, max_tokens=max_tokens, temperature=temperature)
        return self.decode(generated)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        return list(self._get_tokenizer().encode(text).ids)

    def decode(self, token_ids) -> str:
        ids = [int(value) for value in token_ids]
        if any(value < 0 or value >= self.config.vocab_size for value in ids):
            raise ValueError("token id is outside the OLMoE vocabulary")
        return self._get_tokenizer().decode(ids)

    def generate_ids(self, prompt_ids, *, max_tokens: int = 64, temperature: float = 0.0) -> list[int]:
        """Generate token IDs directly for deterministic checks and integrations."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        return list(self.generate_id_stream(prompt_ids, max_tokens=max_tokens, temperature=temperature))

    def generate_id_stream(self, prompt_ids, *, max_tokens: int = 64, temperature: float = 0.0):
        """Yield verified target-model tokens while retaining incremental KV state."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        ids = np.asarray(prompt_ids, dtype=np.int64)
        if ids.ndim != 1 or not ids.size:
            raise ValueError("prompt_ids must be a non-empty vector")
        self.reset()
        yield from self.continue_id_stream(ids, max_tokens=max_tokens, temperature=temperature)

    def continue_id_stream(self, input_ids, *, max_tokens: int = 64, temperature: float = 0.0):
        """Append input IDs to the current KV state and stream target tokens."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim != 1 or not ids.size:
            raise ValueError("input_ids must be a non-empty vector")
        logits = self.forward(ids)
        rng = np.random.default_rng()
        for _ in range(max_tokens):
            if temperature <= 0:
                token = int(np.argmax(logits))
            else:
                probabilities = _softmax(logits / np.float32(temperature))
                token = int(rng.choice(len(probabilities), p=probabilities))
            if token == self.config.eos_token_id:
                break
            yield token
            logits = self.forward([token])

    def snapshot_state(self) -> dict[str, Any]:
        """Copy only committed KV positions for a resumable local session."""
        return {
            "format": "openagent.olmoe-state.v2",
            "fingerprint": self.manifest.fingerprint,
            "execution_identity": self._state_identity,
            "position": self.position,
            "keys": [value[:, :self.position].copy() for value in self._k],
            "values": [value[:, :self.position].copy() for value in self._v],
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("format") != "openagent.olmoe-state.v2":
            raise ValueError("unsupported OLMoE state format")
        if state.get("fingerprint") != self.manifest.fingerprint:
            raise ValueError("OLMoE state belongs to a different checkpoint")
        if state.get("execution_identity") != self._state_identity:
            raise ValueError("OLMoE state belongs to different model weights or precision/backend")
        position = state.get("position")
        keys = state.get("keys")
        values = state.get("values")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            or position > self.context_tokens
            or not isinstance(keys, list)
            or not isinstance(values, list)
            or len(keys) != len(self._k)
            or len(values) != len(self._v)
        ):
            raise ValueError("OLMoE state metadata is invalid")
        validated_keys = []
        validated_values = []
        for target, source in zip(self._k, keys):
            array = np.asarray(source, dtype=np.float32)
            if array.shape != target[:, :position].shape or not np.isfinite(array).all():
                raise ValueError("OLMoE key state has invalid shape or values")
            validated_keys.append(array.copy())
        for target, source in zip(self._v, values):
            array = np.asarray(source, dtype=np.float32)
            if array.shape != target[:, :position].shape or not np.isfinite(array).all():
                raise ValueError("OLMoE value state has invalid shape or values")
            validated_values.append(array.copy())
        self.reset()
        for target, array in zip(self._k, validated_keys):
            target[:, :position] = array
        for target, array in zip(self._v, validated_values):
            target[:, :position] = array
        self.position = position

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError("the tokenizers package is required for OLMoE text generation") from exc
            self._tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        return self._tokenizer

    def status(self) -> dict[str, Any]:
        return {
            "architecture": "olmoe",
            "position": self.position,
            "context_tokens": self.context_tokens,
            "dense_dtype": "float32",
            "expert_runtime": self.experts.status(),
        }

    def close(self) -> None:
        self.experts.close()

    def __enter__(self) -> "OlmoeModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _rms_norm(values: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    scale = np.reciprocal(np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + epsilon))
    return values * scale * weight


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _rope(values: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    head_dim = values.shape[-1]
    if head_dim % 2:
        raise ValueError("OLMoE head dimension must be even for RoPE")
    half = head_dim // 2
    inverse = np.power(np.float32(theta), -2.0 * np.arange(half, dtype=np.float32) / head_dim)
    angle = positions[:, None] * inverse[None, :]
    cosine = np.cos(angle)[:, None, :]
    sine = np.sin(angle)[:, None, :]
    first = values[..., :half]
    second = values[..., half:]
    return np.concatenate((first * cosine - second * sine, second * cosine + first * sine), axis=-1)


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"OLMoE config field {key} must be a positive integer")
    return value


def _positive_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"OLMoE config field {key} must be positive")
    return float(value)


def _nonnegative_int(payload: dict[str, Any], key: str, *, default: int) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"OLMoE config field {key} must be a nonnegative integer")
    return value
