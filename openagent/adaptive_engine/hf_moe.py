"""Optional, local-only Transformers MoE inference with per-expert disk offload.

Requires torch, transformers (architecture support varies by version), accelerate,
and safetensors. No package imports occur until loading. Parameters are built on
meta, then loaded shard-by-shard with temporary CPU-state-dict disk offload.
Peak RAM still includes resident dense weights and checkpoint-shard working data;
an unsharded checkpoint does not provide a bounded small-shard memory guarantee.
Only CPU or an explicit CUDA index is supported, with all dense tensors resident
on that device. No router, attention, expert, or generation code is replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping


SUPPORTED_MODEL_TYPES = frozenset({"qwen2_moe", "qwen3_moe", "mixtral", "olmoe",
                                   "qwen3_next", "glm4_moe", "deepseek_v3"})
_QWEN_MODEL_TYPES = frozenset({"qwen2_moe", "qwen3_moe", "qwen3_next"})
_DENSE_PREFIX_TYPES = frozenset({"glm4_moe", "deepseek_v3"})


def _expert_count_key(model_type: str) -> str:
    if model_type == "mixtral":
        return "num_local_experts"
    return "n_routed_experts" if model_type in _DENSE_PREFIX_TYPES else "num_experts"


def require_hf_moe_execution_admission(manifest: "HFMoeManifest") -> tuple[dict, ...]:
    """Require a complete Phase 1 route before Transformers loads checkpoint data.

    Manifest and expert-placement inspection remain available to tooling, but
    this loader is not currently represented by an exact family/backend/format
    row. In particular, an HF model allowlist is not P02 admission.
    """
    if type(manifest) is not HFMoeManifest:
        raise TypeError("validated local HF MoE manifest required")
    from .source_scope import (require_source_family_backend_admission,
                               source_family_for_identity)
    try:
        family = source_family_for_identity(manifest.model_type)
    except ValueError as error:
        raise ValueError(
            f"P02 HF MoE execution is restricted: {manifest.model_type} "
            "is not a catalogued Phase 1 family identity") from error
    try:
        return require_source_family_backend_admission(
            family=family, model_type=manifest.model_type,
            backend="transformers-expert-offload", arithmetic="fp32")
    except ValueError as error:
        raise ValueError(
            f"P02 HF MoE execution is restricted for {manifest.model_type}: "
            "no admitted Transformers/Accelerate loader-to-consumer route") from error


@dataclass(frozen=True)
class HFMoeManifest:
    model_directory: Path
    model_type: str
    config_json: str

    @classmethod
    def read(cls, directory: str | Path) -> "HFMoeManifest":
        root = Path(directory).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("checkpoint must be a local directory")
        raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("model_type") not in SUPPORTED_MODEL_TYPES:
            raise ValueError("unsupported sparse MoE model_type")
        if raw.get("auto_map") or raw.get("quantization_config"):
            raise ValueError("custom code and quantized checkpoints are unsupported")
        expert_key = _expert_count_key(raw["model_type"])
        keys = ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
                expert_key, "num_experts_per_tok")
        if raw["model_type"] in _QWEN_MODEL_TYPES | _DENSE_PREFIX_TYPES:
            keys += ("moe_intermediate_size",)
        if raw["model_type"] in _DENSE_PREFIX_TYPES:
            keys += ("n_shared_experts", "n_group", "topk_group")
        if raw["model_type"] == "deepseek_v3":
            keys += ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim")
            if raw.get("q_lora_rank") is not None:
                keys += ("q_lora_rank",)
        if raw["model_type"] == "qwen3_next":
            keys += ("head_dim", "linear_key_head_dim", "linear_value_head_dim",
                     "linear_num_key_heads", "linear_num_value_heads", "linear_conv_kernel_dim",
                     "shared_expert_intermediate_size")
        for key in keys:
            value = raw.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if raw["num_experts_per_tok"] > raw[expert_key]:
            raise ValueError("num_experts_per_tok exceeds expert count")
        if raw["model_type"] != "deepseek_v3" and "head_dim" not in raw and raw["hidden_size"] % raw["num_attention_heads"]:
            raise ValueError("hidden_size must be divisible by attention heads")
        if raw["num_attention_heads"] % raw["num_key_value_heads"]:
            raise ValueError("attention heads must be divisible by key/value heads")
        if raw["model_type"] in _DENSE_PREFIX_TYPES:
            prefix = raw.get("first_k_dense_replace")
            if type(prefix) is not int or not 0 <= prefix < raw["num_hidden_layers"]:
                raise ValueError("first_k_dense_replace must leave at least one sparse layer")
            if (raw[expert_key] % raw["n_group"] or raw[expert_key] // raw["n_group"] < 2
                    or raw["topk_group"] > raw["n_group"]
                    or raw["num_experts_per_tok"] > raw["topk_group"] * (raw[expert_key] // raw["n_group"])):
                raise ValueError("invalid grouped expert routing geometry")
        if raw["model_type"] == "deepseek_v3" and raw["qk_rope_head_dim"] % 2:
            raise ValueError("qk_rope_head_dim must be even")
        if raw["model_type"] == "deepseek_v3" and raw["num_key_value_heads"] != raw["num_attention_heads"]:
            raise ValueError("deepseek_v3 MLA requires equal attention and key/value head counts")
        if raw["model_type"] == "qwen3_next":
            kinds = raw.get("layer_types")
            if (not isinstance(kinds, list) or len(kinds) != raw["num_hidden_layers"]
                    or any(kind not in ("linear_attention", "full_attention") for kind in kinds)
                    or not {"linear_attention", "full_attention"}.issubset(kinds)):
                raise ValueError("qwen3_next requires an exact hybrid layer_types list")
            if raw["linear_num_value_heads"] % raw["linear_num_key_heads"]:
                raise ValueError("linear value heads must be divisible by key heads")
        if raw["model_type"] in _QWEN_MODEL_TYPES:
            step = raw.get("decoder_sparse_step", 1)
            dense_layers = raw.get("mlp_only_layers", [])
            if type(step) is not int or step <= 0:
                raise ValueError("decoder_sparse_step must be a positive integer")
            if not isinstance(dense_layers, list) or any(
                type(index) is not int or not 0 <= index < raw["num_hidden_layers"]
                for index in dense_layers
            ):
                raise ValueError("invalid mlp_only_layers")
        return cls(root, raw["model_type"], json.dumps(raw, sort_keys=True))


def expert_module_names(manifest: HFMoeManifest) -> tuple[str, ...]:
    """Return layer-major routed expert owners from a manifest validated by read().

    Uses the captured config, without rereading files or importing model libraries.
    Shared experts and dense-prefix/MLP-only layers are never placement targets.
    """
    raw = json.loads(manifest.config_json)
    model_type = manifest.model_type
    count = raw[_expert_count_key(model_type)]
    layers = range(raw["num_hidden_layers"])
    if model_type in _DENSE_PREFIX_TYPES:
        layers = range(raw["first_k_dense_replace"], raw["num_hidden_layers"])
    elif model_type in _QWEN_MODEL_TYPES:
        step = raw.get("decoder_sparse_step", 1)
        dense_layers = set(raw.get("mlp_only_layers", []))
        layers = [layer for layer in layers if layer not in dense_layers and (layer + 1) % step == 0]
    block = "block_sparse_moe" if model_type == "mixtral" else "mlp"
    return tuple(f"model.layers.{layer}.{block}.experts.{expert}"
                 for layer in layers for expert in range(count))


def _device(value: str, torch: Any) -> Any:
    if not isinstance(value, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", value):
        raise ValueError("resident_device must be cpu or an explicit cuda:N")
    device = torch.device(value)
    if device.type == "cuda" and (
        not torch.cuda.is_available() or device.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device unavailable: {value}")
    return device


def _validate_checkpoint(root: Path, model: Any) -> Path:
    """Validate headers without materializing checkpoint tensors or trusting an index."""
    from safetensors import safe_open
    from accelerate.utils import find_tied_parameters

    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if single.exists() == index.exists():
        raise ValueError("require exactly one local SafeTensors file or shard index")
    weight_map = None
    if index.exists():
        raw = json.loads(index.read_text(encoding="utf-8"))
        weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("invalid SafeTensors shard index")
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in weight_map.items()):
            raise ValueError("invalid SafeTensors weight_map")
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = [single.name]
    shapes = {}
    locations = {}
    for filename in filenames:
        path = (root / filename).resolve(strict=True)
        if path.parent != root or path.suffix != ".safetensors":
            raise ValueError("shards must be local SafeTensors files inside checkpoint")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in shapes:
                    raise ValueError(f"duplicate checkpoint tensor: {key}")
                if weight_map is not None and weight_map.get(key) != filename:
                    raise ValueError(f"shard index disagrees with tensor: {key}")
                shapes[key] = tuple(handle.get_slice(key).get_shape())
                locations[key] = path
    if weight_map is not None and set(weight_map) != set(shapes):
        raise ValueError("shard index has missing or extra tensors")
    expected = model.state_dict()
    unexpected = set(shapes) - set(expected)
    missing = set(expected) - set(shapes)
    for group in find_tied_parameters(model):
        present = [key for key in group if key in shapes]
        if present:
            missing.difference_update(group)
        # Multiple serialized aliases must agree, not silently overwrite each other.
        for alias in present[1:]:
            import torch
            with safe_open(str(locations[present[0]]), framework="pt", device="cpu") as first, \
                    safe_open(str(locations[alias]), framework="pt", device="cpu") as second:
                left, right = first.get_slice(present[0]), second.get_slice(alias)
                if left.get_shape() != right.get_shape() or left.get_dtype() != right.get_dtype():
                    raise ValueError("inconsistent serialized tied weights")
                for row in range(left.get_shape()[0]):
                    if not torch.equal(left[row:row + 1], right[row:row + 1]):
                        raise ValueError("inconsistent serialized tied weights")
    mismatched = [key for key in shapes if key in expected and shapes[key] != tuple(expected[key].shape)]
    if missing or unexpected or mismatched:
        raise ValueError(f"checkpoint mismatch: missing={sorted(missing)}, "
                         f"unexpected={sorted(unexpected)}, shapes={sorted(mismatched)}")
    return index if weight_map is not None else single


class HFMoeAdapter:
    """Inference-only wrapper; use forward/generate or the unchanged library model.

    expert_placement maps exact expert module names to 'disk' or resident_device.
    Omitted experts stay resident. A fresh offload subdirectory avoids stale data.
    Model calls must be serialized by the caller; Accelerate hooks are mutable.
    """

    def __init__(self, model: Any, manifest: HFMoeManifest, resident_device: Any,
                 expert_names: tuple[str, ...], placement: Mapping[str, str],
                 offload_directory: Path | None):
        self.model = model
        self.manifest = manifest
        self.resident_device = resident_device
        self.expert_names = expert_names
        self.placement = dict(placement)
        self.offload_directory = offload_directory

    @classmethod
    def load(cls, model_directory: str | Path, *, resident_device: str = "cpu",
             expert_placement: Mapping[str, str] | None = None,
             offload_directory: str | Path | None = None) -> "HFMoeAdapter":
        manifest = HFMoeManifest.read(model_directory)
        require_hf_moe_execution_admission(manifest)
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        except ImportError as exc:
            raise ImportError("HF MoE requires torch, transformers, accelerate, safetensors") from exc
        device = _device(resident_device, torch)
        raw = json.loads(manifest.config_json)
        config = AutoConfig.for_model(manifest.model_type, **{
            key: value for key, value in raw.items() if key != "model_type"
        })
        model_class = AutoModelForCausalLM._model_mapping[type(config)]
        if raw.get("architectures") not in (None, [model_class.__name__]):
            raise ValueError("unrecognized architecture declaration")
        config._attn_implementation = "eager"
        # Nonpersistent rotary buffers must be real: they are absent from checkpoints.
        with init_empty_weights(include_buffers=False):
            model = model_class(config)
        model.tie_weights()
        if (manifest.model_directory / "generation_config.json").exists():
            model.generation_config = GenerationConfig.from_pretrained(
                str(manifest.model_directory), local_files_only=True,
            )
        modules = dict(model.named_modules())
        expert_names = []
        count = raw[_expert_count_key(manifest.model_type)]
        for name, module in modules.items():
            if name.endswith(".experts"):
                block = "block_sparse_moe" if manifest.model_type == "mixtral" else "mlp"
                if not re.fullmatch(rf"model\.layers\.[0-9]+\.{block}\.experts", name):
                    raise ValueError("unrecognized sparse expert module path")
                if not isinstance(module, torch.nn.ModuleList) or len(module) != count:
                    raise ValueError("unsupported expert layout: individually callable modules required")
                expert_names.extend(f"{name}.{index}" for index in range(count))
        if not expert_names:
            raise ValueError("no individually callable sparse experts found")
        expected_experts = set(expert_module_names(manifest))
        if set(expert_names) != expected_experts:
            raise ValueError("sparse expert layout does not match config manifest")
        placement = dict(expert_placement or {})
        if set(placement) - set(expert_names):
            raise ValueError("placement must name exact sparse expert modules only")
        if any(value not in ("disk", resident_device) for value in placement.values()):
            raise ValueError("expert placement must be disk or resident_device")
        disk_names = [name for name, value in placement.items() if value == "disk"]
        if disk_names and offload_directory is None:
            raise ValueError("disk experts require offload_directory")
        checkpoint = _validate_checkpoint(manifest.model_directory, model)
        folder = None
        if disk_names:
            parent = Path(offload_directory).resolve()
            if parent == manifest.model_directory or manifest.model_directory in parent.parents:
                raise ValueError("offload_directory must be outside the checkpoint")
            parent.mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix="hf-moe-", dir=parent))
        resolved = {name: placement.get(name, resident_device) for name in expert_names}
        # A resident ancestor plus disk descendants makes Accelerate's ancestor
        # hook try to materialize meta expert weights. Use disjoint tensor owners.
        device_map = dict(resolved)
        for name, module in modules.items():
            if any(name == expert or name.startswith(expert + ".") for expert in expert_names):
                continue
            if list(module.parameters(recurse=False)) or list(module.buffers(recurse=False)):
                device_map[name] = resident_device
        try:
            model = load_checkpoint_and_dispatch(
                model, str(checkpoint), device_map=device_map,
                offload_folder=str(folder) if folder else None,
                offload_state_dict=True, offload_buffers=True, strict=True,
                dtype=torch.float32,
            )
            model.eval().requires_grad_(False)
            adapter = cls(model, manifest, device, tuple(expert_names), resolved, folder)
            adapter.validate_placement()
            return adapter
        except Exception as exc:
            if folder is not None and folder.resolve().parent == parent:
                try:
                    shutil.rmtree(folder)
                except OSError as cleanup_error:
                    exc.add_note(f"Owned offload directory cleanup failed: {cleanup_error}")
            raise

    def validate_placement(self) -> None:
        """Verify disk expert tensors are meta and all other tensors resident."""
        for name, tensor in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            owner = next((expert for expert in self.expert_names if name.startswith(expert + ".")), None)
            expected = "meta" if owner and self.placement[owner] == "disk" else self.resident_device
            if str(tensor.device) != str(expected):
                raise ValueError(f"unexpected tensor placement: {name}: {tensor.device}")

    def forward(self, **inputs: Any) -> Any:
        import torch
        with torch.inference_mode():
            return self.model(**inputs)

    def generate(self, **inputs: Any) -> Any:
        import torch
        with torch.inference_mode():
            return self.model.generate(**inputs)
