"""Optional gateway provider for configured, local library MoE checkpoints."""

import importlib.util
import os
from pathlib import Path
import threading
import time

from ..model_gateway import BaseProvider, ModelResult, ProviderConfigurationError, ProviderUnavailableError
from .expert_packing import _scan_directory
from .hf_moe import HFMoeAdapter, HFMoeManifest
from .hf_moe_cli import expert_placement


class HFMoeProvider(BaseProvider):
    id = "hf-moe-direct"
    label = "Local Transformers MoE"
    local = True
    network = False
    privacy_warning = "Uses configured local checkpoint files and expert offload storage; no downloads."

    def __init__(self):
        self.model_path = os.environ.get("OPENAGENT_HF_MOE_MODEL_DIR", "")
        self.offload_path = os.environ.get("OPENAGENT_HF_MOE_OFFLOAD_DIR", "")
        self.device = os.environ.get("OPENAGENT_HF_MOE_DEVICE", "cpu")
        self.default_model = Path(self.model_path).name or "local-hf-moe"
        self.max_tokens = int(os.environ.get("OPENAGENT_HF_MOE_MAX_TOKENS", "128"))
        if self.max_tokens <= 0:
            raise ProviderConfigurationError("HF MoE max tokens must be positive")
        self._adapter = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def available(self):
        if not self.model_path or not all(importlib.util.find_spec(name) for name in (
            "torch", "transformers", "accelerate", "safetensors",
        )):
            return False
        try:
            manifest = HFMoeManifest.read(self.model_path)
            _scan_directory(manifest.model_directory)
            return (manifest.model_directory / "tokenizer_config.json").is_file()
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def complete(self, messages, *, model=None):
        if model not in (None, self.default_model):
            raise ProviderConfigurationError("Requested model differs from the configured HF MoE checkpoint")
        if not self.available():
            raise ProviderUnavailableError("Local HF MoE dependencies, weights or tokenizer are unavailable")
        started = time.perf_counter()
        with self._lock:
            if self._adapter is None:
                from transformers import AutoTokenizer

                manifest = HFMoeManifest.read(self.model_path)
                placement = expert_placement(manifest, self.device, disk=bool(self.offload_path))
                tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                    local_files_only=True, trust_remote_code=False)
                adapter = HFMoeAdapter.load(self.model_path, resident_device=self.device,
                    expert_placement=placement, offload_directory=self.offload_path or None)
                self._tokenizer, self._adapter = tokenizer, adapter
            if self._tokenizer.chat_template:
                prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"
            inputs = self._tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
            length = inputs["input_ids"].shape[1]
            limit = getattr(self._adapter.model.config, "max_position_embeddings", None)
            if limit is not None and length + self.max_tokens > limit:
                raise ProviderConfigurationError("Prompt and requested output exceed model context capacity")
            inputs = {key: value.to(self._adapter.resident_device) for key, value in inputs.items()}
            output = self._adapter.generate(**inputs, do_sample=False, max_new_tokens=self.max_tokens)
            text = self._tokenizer.decode(output[0, length:], skip_special_tokens=True)
            if not text.strip():
                raise ProviderUnavailableError("Local HF MoE completed without text")
            metadata = {"network": False, "offline": True, "engine": "transformers-expert-offload",
                        "resident_device": str(self._adapter.resident_device),
                        "disk_experts": sum(value == "disk" for value in self._adapter.placement.values()),
                        "streamed": False, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
        return ModelResult(self.id, self.default_model, text, metadata)
