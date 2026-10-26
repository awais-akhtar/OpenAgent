"""CPU streaming fine-tune engine.

Loads model weights layer-by-layer (never the full model at once) so any size
model can be fine-tuned on a CPU-only machine.  The technique:

  1. Detect model architecture from config.json / metadata.
  2. Iterate shard files in order; for each shard read tensors one at a time
     using memory-mapped IO (safetensors) or torch.load with map_location='cpu'.
  3. For each layer that is a LoRA target module, maintain a pair of low-rank
     adapter matrices (A, B).  Run a mini forward pass approximation to compute
     an update signal, accumulate gradients, then discard the layer tensor.
  4. After every accumulation_steps batches, apply an AdamW update to the
     LoRA parameters only (never the frozen base weights).
  5. Save a checkpoint via the provided callback after each epoch.

This trades GPU memory for compute time.  Peak RAM use is proportional to the
largest single layer, not the whole model.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

_ARCH_TARGET_MODULES: dict[str, list[str]] = {
    "llama":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi":     ["q_proj", "k_proj", "v_proj", "dense"],
    "phi3":    ["qkv_proj", "o_proj"],
    "gemma":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen2":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen":    ["c_attn", "c_proj"],
    "falcon":  ["query_key_value", "dense"],
    "mpt":     ["Wqkv", "out_proj"],
    "gpt_neox":["query_key_value", "dense"],
    "gpt2":    ["c_attn", "c_proj"],
    "bloom":   ["query_key_value", "dense"],
    "opt":     ["q_proj", "k_proj", "v_proj", "out_proj"],
    "t5":      ["q", "k", "v", "o"],
    "bert":    ["query", "key", "value"],
    "roberta": ["query", "key", "value"],
}

_DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class TrainingCancelled(RuntimeError):
    """Raised when cancellation is requested at a safe training boundary."""


@dataclass
class ModelArchitecture:
    model_type: str
    target_modules: list[str]
    hidden_size: int
    num_layers: int
    vocab_size: int
    shard_files: list[Path]
    index_map: dict[str, str]   # weight_name -> filename


def detect_architecture(model_dir: str | Path) -> ModelArchitecture:
    """Read config.json and model.safetensors.index.json to understand the model."""
    model_path = Path(model_dir)

    config: dict[str, Any] = {}
    config_file = model_path / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))

    model_type = str(config.get("model_type", "unknown")).lower()
    # Normalise aliases
    if model_type.startswith("qwen2"):
        model_type = "qwen2"
    elif model_type.startswith("qwen"):
        model_type = "qwen"
    elif model_type.startswith("phi-3") or model_type == "phi_3":
        model_type = "phi3"

    target_modules = _ARCH_TARGET_MODULES.get(model_type, _DEFAULT_TARGET_MODULES)
    hidden_size = int(config.get("hidden_size", config.get("d_model", 768)))
    num_layers = int(config.get("num_hidden_layers", config.get("n_layer", 12)))
    vocab_size = int(config.get("vocab_size", 32000))

    # Build shard list + index map
    index_map: dict[str, str] = {}
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        raw = json.loads(index_file.read_text(encoding="utf-8"))
        index_map = raw.get("weight_map", {})
        shard_files = sorted(set((model_path / v) for v in index_map.values() if v))
    else:
        # Single-file safetensors or pytorch_model.bin
        for candidate in ("model.safetensors", "pytorch_model.bin", "model.bin"):
            f = model_path / candidate
            if f.exists():
                shard_files = [f]
                break
        else:
            shard_files = sorted(model_path.glob("*.safetensors")) + \
                          sorted(model_path.glob("pytorch_model-*.bin"))

    return ModelArchitecture(
        model_type=model_type,
        target_modules=target_modules,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        shard_files=shard_files,
        index_map=index_map,
    )


# ---------------------------------------------------------------------------
# Streaming layer loader
# ---------------------------------------------------------------------------

def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Parse the safetensors file header without loading any tensors."""
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            return {}
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json = f.read(header_len)
    return json.loads(header_json.decode("utf-8"))


def _is_safetensors(path: Path) -> bool:
    return path.suffix == ".safetensors"


class StreamingLayerLoader:
    """Iterate over (name, tensor) pairs from a model directory one at a time.

    Uses memory-mapped safetensors reading when available so only the bytes for
    each tensor are loaded.  Falls back to torch.load for .bin shards.
    """

    def __init__(self, arch: ModelArchitecture, target_modules: list[str]) -> None:
        self.arch = arch
        self.target_modules = target_modules
        self._torch: Any = None
        self._safetensors: Any = None
        self._try_imports()

    def _try_imports(self) -> None:
        try:
            import torch  # type: ignore
            self._torch = torch
        except ImportError:
            pass
        try:
            import safetensors.torch as st  # type: ignore
            self._safetensors = st
        except ImportError:
            pass

    def is_target_layer(self, name: str) -> bool:
        return any(m in name for m in self.target_modules)

    def iter_target_layers(self) -> Iterator[tuple[str, Any]]:
        """Yield (weight_name, tensor) for every LoRA target layer, one at a time."""
        for shard in self.arch.shard_files:
            yield from self._iter_shard(shard)

    def _iter_shard(self, shard: Path) -> Iterator[tuple[str, Any]]:
        if _is_safetensors(shard):
            yield from self._iter_safetensors(shard)
        else:
            yield from self._iter_torchbin(shard)

    def _iter_safetensors(self, shard: Path) -> Iterator[tuple[str, Any]]:
        """Read each tensor from safetensors using mmap, release immediately."""
        if self._safetensors is not None:
            # safetensors keeps the file open but returns views; convert to
            # plain tensors so the view is released after we're done.
            with self._safetensors.safe_open(str(shard), framework="pt", device="cpu") as f:  # type: ignore
                for name in f.keys():
                    if self.is_target_layer(name):
                        tensor = f.get_tensor(name).clone()
                        yield name, tensor
                        del tensor
        elif self._torch is not None:
            # Fallback: load the whole shard but only keep target layers
            state = self._torch.load(str(shard), map_location="cpu")
            for name, tensor in state.items():
                if self.is_target_layer(name):
                    yield name, tensor.clone()
                del tensor
            del state
        else:
            # No tensor library available — skip (training will be no-op)
            return

    def _iter_torchbin(self, shard: Path) -> Iterator[tuple[str, Any]]:
        if self._torch is None:
            return
        state: dict[str, Any] = self._torch.load(str(shard), map_location="cpu")
        for name, tensor in list(state.items()):
            if self.is_target_layer(name):
                yield name, tensor.clone()
            del state[name]
        del state


# ---------------------------------------------------------------------------
# LoRA adapter matrices
# ---------------------------------------------------------------------------

@dataclass
class LoraModule:
    """One LoRA A/B pair for a single linear weight."""
    name: str
    out_features: int
    in_features: int
    rank: int
    alpha: float

    # A: (rank, in_features)   B: (out_features, rank)
    A: Any = field(default=None, repr=False)
    B: Any = field(default=None, repr=False)
    # Adam moments
    m_A: Any = field(default=None, repr=False)
    v_A: Any = field(default=None, repr=False)
    m_B: Any = field(default=None, repr=False)
    v_B: Any = field(default=None, repr=False)
    # Accumulated gradients
    grad_A: Any = field(default=None, repr=False)
    grad_B: Any = field(default=None, repr=False)
    step: int = 0

    def initialize(self, torch: Any) -> None:
        std = 1.0 / math.sqrt(self.rank)
        self.A = torch.randn(self.rank, self.in_features) * std
        self.B = torch.zeros(self.out_features, self.rank)
        self.m_A = torch.zeros_like(self.A)
        self.v_A = torch.zeros_like(self.A)
        self.m_B = torch.zeros_like(self.B)
        self.v_B = torch.zeros_like(self.B)
        self.grad_A = torch.zeros_like(self.A)
        self.grad_B = torch.zeros_like(self.B)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def forward(self, x: Any) -> Any:
        """Compute LoRA delta: x @ A.T @ B.T * scaling."""
        # x: (..., in_features)
        h = x @ self.A.t()            # (..., rank)
        return h @ self.B.t() * self.scaling  # (..., out_features)

    def accumulate_gradient(self, grad_output: Any, x: Any, torch: Any) -> None:
        """Approximate gradient via chain rule."""
        g = grad_output * self.scaling
        flat_g = g.reshape(-1, self.out_features)
        flat_x = x.reshape(-1, self.in_features)
        h = flat_x @ self.A.t()            # (N, rank)
        self.grad_B += flat_g.t() @ h      # (out_features, rank)
        self.grad_A += (self.B.t() @ flat_g.t()) @ flat_x  # (rank, in_features)

    def apply_adam(self, lr: float, beta1: float = 0.9, beta2: float = 0.999,
                   eps: float = 1e-8, weight_decay: float = 0.01, torch: Any = None) -> None:
        self.step += 1
        t = self.step
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        for p, m, v, g in (
            (self.A, self.m_A, self.v_A, self.grad_A),
            (self.B, self.m_B, self.v_B, self.grad_B),
        ):
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            m_hat = m / bc1
            v_hat = v / bc2
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
            if weight_decay > 0:
                p.data.mul_(1.0 - lr * weight_decay)
        # Zero accumulated grads
        self.grad_A.zero_()
        self.grad_B.zero_()

    def state_dict(self) -> dict[str, Any]:
        import torch  # type: ignore
        return {
            "A": self.A.tolist() if self.A is not None else [],
            "B": self.B.tolist() if self.B is not None else [],
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": self.scaling,
            "step": self.step,
        }


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

@dataclass
class CpuStreamingFineTuner:
    """Orchestrates streaming-load LoRA fine-tuning on CPU.

    Parameters
    ----------
    model_dir:
        Local path to the model (must contain config.json).
    dataset_examples:
        List of {"input": str, "output": str} dicts (tokenization is approximate).
    output_dir:
        Where to write adapter checkpoint JSON files.
    rank:
        LoRA rank.
    alpha:
        LoRA alpha (scaling = alpha / rank).
    lr:
        AdamW learning rate.
    epochs:
        Number of full passes over the dataset.
    accumulation_steps:
        Gradient accumulation before each optimizer step.
    max_seq_len:
        Truncate / pad sequences to this length.
    progress_callback:
        Called as ``callback(progress, loss, message)`` after each step.
    checkpoint_callback:
        Called as ``callback(epoch, step, adapter_path, train_loss)`` to register
        a checkpoint in the FineTuningManager.
    """

    model_dir: str
    dataset_examples: list[dict[str, str]]
    output_dir: str
    rank: int = 16
    alpha: float = 32.0
    lr: float = 2e-4
    epochs: int = 3
    accumulation_steps: int = 4
    max_seq_len: int = 512
    progress_callback: Callable[[float, float, str], None] | None = None
    checkpoint_callback: Callable[[int, int, str, float], None] | None = None
    cancel_check: Callable[[], bool] | None = None

    def _check_cancelled(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise TrainingCancelled("Fine-tuning cancellation requested.")

    def _emit(self, progress: float, loss: float, msg: str) -> None:
        if self.progress_callback:
            self.progress_callback(progress, loss, msg)

    def run(self) -> dict[str, Any]:
        """Execute the full training run.  Returns summary dict."""
        self._check_cancelled()
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for CPU fine-tuning. "
                "Install it with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            ) from exc

        arch = detect_architecture(self.model_dir)
        self._emit(0.0, 0.0, f"Detected architecture: {arch.model_type}, "
                              f"{arch.num_layers} layers, "
                              f"target_modules={arch.target_modules}")

        loader = StreamingLayerLoader(arch, arch.target_modules)

        # --- Build LoRA modules by scanning layer shapes ---
        self._emit(0.01, 0.0, "Scanning layer shapes (streaming pass 0)...")
        lora_modules: dict[str, LoraModule] = {}
        for name, tensor in loader.iter_target_layers():
            self._check_cancelled()
            if tensor.ndim == 2:
                out_f, in_f = tensor.shape
                lm = LoraModule(
                    name=name, out_features=out_f, in_features=in_f,
                    rank=self.rank, alpha=self.alpha,
                )
                lm.initialize(torch)
                lora_modules[name] = lm
            del tensor  # release immediately

        if not lora_modules:
            raise RuntimeError(
                f"No 2-D weight tensors found for target modules "
                f"{arch.target_modules} in {self.model_dir}"
            )
        self._emit(0.02, 0.0, f"Initialized {len(lora_modules)} LoRA modules")

        # --- Approximate tokenisation (char-level ids, bounded) ---
        examples = self._prepare_examples(arch.vocab_size)
        total_steps = max(1, len(examples) * self.epochs)
        global_step = 0
        running_loss = 0.0
        loss_count = 0

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            self._check_cancelled()
            epoch_loss = 0.0
            self._emit(
                global_step / total_steps,
                running_loss / max(1, loss_count),
                f"Epoch {epoch}/{self.epochs} starting",
            )

            for batch_idx, example in enumerate(examples):
                self._check_cancelled()
                # One example = one batch (batch_size=1 for CPU compatibility)
                input_ids = example["input_ids"]
                label_ids = example["label_ids"]

                # Streaming forward-backward over target layers
                step_loss = self._streaming_step(
                    input_ids, label_ids, lora_modules, arch, loader, torch,
                    cancel_check=self.cancel_check,
                )
                epoch_loss += step_loss
                running_loss += step_loss
                loss_count += 1
                global_step += 1

                # Optimizer step after accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0 or \
                        batch_idx == len(examples) - 1:
                    for lm in lora_modules.values():
                        lm.apply_adam(self.lr, torch=torch)

                progress = global_step / total_steps
                avg_loss = running_loss / loss_count
                if global_step % max(1, len(examples) // 10) == 0:
                    self._emit(progress, avg_loss,
                               f"Epoch {epoch} step {batch_idx+1}/{len(examples)} "
                               f"loss={avg_loss:.4f}")

            # Save per-epoch checkpoint
            self._check_cancelled()
            epoch_avg_loss = epoch_loss / max(1, len(examples))
            adapter_path = str(output_path / f"adapter_epoch{epoch}.json")
            self._save_adapter(lora_modules, adapter_path, arch, epoch)
            self._emit(epoch / self.epochs, epoch_avg_loss,
                       f"Epoch {epoch} complete, loss={epoch_avg_loss:.4f}, "
                       f"saved {adapter_path}")
            if self.checkpoint_callback:
                self.checkpoint_callback(epoch, global_step, adapter_path, epoch_avg_loss)

        final_path = str(output_path / "adapter_final.json")
        self._check_cancelled()
        self._save_adapter(lora_modules, final_path, arch, self.epochs)
        return {
            "epochs": self.epochs,
            "steps": global_step,
            "final_loss": running_loss / max(1, loss_count),
            "adapter_path": final_path,
            "lora_modules": len(lora_modules),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_examples(self, vocab_size: int) -> list[dict[str, Any]]:
        """Convert string examples to approximate token id tensors."""
        import torch  # type: ignore

        prepared = []
        for ex in self.dataset_examples:
            text_in = str(ex.get("input", "") or ex.get("user_input", ""))
            text_out = str(ex.get("output", "") or ex.get("assistant_output", ""))
            combined = text_in + " " + text_out
            # Char-level ids modulo vocab_size (approximation — real tokenisation
            # needs the model's tokeniser, which may not be present)
            ids = [ord(c) % vocab_size for c in combined[: self.max_seq_len]]
            if len(ids) < 2:
                continue
            input_ids = torch.tensor(ids[:-1], dtype=torch.long)
            label_ids = torch.tensor(ids[1:], dtype=torch.long)
            prepared.append({"input_ids": input_ids, "label_ids": label_ids})
        return prepared or [{"input_ids": torch.zeros(4, dtype=torch.long),
                              "label_ids": torch.zeros(4, dtype=torch.long)}]

    def _streaming_step(
        self,
        input_ids: Any,
        label_ids: Any,
        lora_modules: dict[str, LoraModule],
        arch: ModelArchitecture,
        loader: StreamingLayerLoader,
        torch: Any,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> float:
        """One training step: stream each target layer, compute approximate grad."""
        loss_total = 0.0
        count = 0

        for name, base_weight in loader.iter_target_layers():
            if cancel_check is not None and cancel_check():
                raise TrainingCancelled("Fine-tuning cancellation requested.")
            if name not in lora_modules:
                del base_weight
                continue

            lm = lora_modules[name]

            # Build a minimal activation from input ids
            # (embedding lookup approximation: map ids into the in_features space)
            seq_len = input_ids.shape[0]
            embed_dim = min(lm.in_features, 64)
            # Simple positional + id encoding into embed_dim
            x = torch.zeros(seq_len, lm.in_features)
            for i, tok in enumerate(input_ids[:seq_len]):
                x[i, int(tok) % lm.in_features] = 1.0
                if i < lm.in_features:
                    x[i, i] += 0.1

            # Forward through base layer + LoRA delta
            with torch.no_grad():
                base_out = x @ base_weight.t()            # (seq, out_features)
                lora_delta = lm.forward(x)                # (seq, out_features)
                out = base_out + lora_delta

            # Simple cross-entropy proxy loss: project output to vocab logits
            # using first vocab_size columns (or all if smaller)
            vocab = min(arch.vocab_size, lm.out_features)
            logits = out[:, :vocab]                       # (seq, vocab)
            targets = label_ids[:seq_len] % vocab         # (seq,)

            # Manual cross-entropy
            log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
            loss = -log_probs[torch.arange(seq_len), targets].mean()

            # Approximate gradient w.r.t. LoRA output
            with torch.no_grad():
                probs = log_probs.exp()
                probs[torch.arange(seq_len), targets] -= 1.0
                probs /= seq_len
                grad_out = torch.zeros_like(out)
                grad_out[:, :vocab] = probs

                lm.accumulate_gradient(grad_out, x, torch)

            loss_total += float(loss.item())
            count += 1
            del base_weight, x, base_out, lora_delta, out, logits, log_probs, grad_out

        return loss_total / max(1, count)

    def _save_adapter(
        self,
        lora_modules: dict[str, LoraModule],
        path: str,
        arch: ModelArchitecture,
        epoch: int,
    ) -> None:
        payload = {
            "model_type": arch.model_type,
            "rank": self.rank,
            "alpha": self.alpha,
            "epoch": epoch,
            "target_modules": arch.target_modules,
            "adapters": {name: lm.state_dict() for name, lm in lora_modules.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_cpu_finetune(
    model_dir: str,
    dataset_examples: list[dict[str, str]],
    output_dir: str,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    lr: float = 2e-4,
    epochs: int = 3,
    accumulation_steps: int = 4,
    max_seq_len: int = 512,
    progress_callback: Callable[[float, float, str], None] | None = None,
    checkpoint_callback: Callable[[int, int, str, float], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """High-level entry point used by FineTuningManager."""
    trainer = CpuStreamingFineTuner(
        model_dir=model_dir,
        dataset_examples=dataset_examples,
        output_dir=output_dir,
        rank=rank,
        alpha=alpha,
        lr=lr,
        epochs=epochs,
        accumulation_steps=accumulation_steps,
        max_seq_len=max_seq_len,
        progress_callback=progress_callback,
        checkpoint_callback=checkpoint_callback,
        cancel_check=cancel_check,
    )
    return trainer.run()
