"""Bounded genuine worker verification using a locally initialized tiny model.

No pretrained weights or network requests: this proves the implemented worker's
parameter/checkpoint lifecycle, not model quality or full-family LoRA fidelity.
Run with the existing PyTorch-capable Python environment.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def create_tiny_model(directory: Path) -> dict[str, object]:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(config)
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory, safe_serialization=True)
    return {"architecture": "llama", "parameters": sum(p.numel() for p in model.parameters()),
            "origin": "locally initialized random test weights; not pretrained", "downloaded_bytes": 0}


def verify() -> dict[str, object]:
    from openagent.fine_tuning import FineTuningManager
    from openagent.storage import OpenAgentStore

    with TemporaryDirectory(prefix="oa-tiny-finetune-") as temporary:
        root = Path(temporary)
        model = root / "tiny-model"
        identity = create_tiny_model(model)
        weights = model / "model.safetensors"
        original_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        manager = FineTuningManager(OpenAgentStore(root / "runtime"))
        dataset = manager.create_dataset("Tiny worker verification", project="default")
        manager.add_example(dataset["id"], "A", "B", project="default")
        manager.add_example(dataset["id"], "C", "D", project="default")
        job = manager.create_job(dataset["id"], str(model), project="default", config={
            "epochs": 1, "rank": 2, "alpha": 4, "lr": 0.001,
            "max_seq_len": 8, "accumulation_steps": 1,
        })
        manager.run_job_cpu_streaming(job["id"], str(model), project="default")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            recorded = manager.get_job(job["id"], project="default")
            if recorded["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        else:
            manager.cancel_queued_job(job["id"], project="default")
            raise AssertionError("Bounded tiny training did not terminate within 30 seconds")
        if recorded["status"] != "completed":
            raise AssertionError(f"Actual worker failed: {recorded}")
        checkpoints = manager.list_checkpoints(job_id=job["id"], project="default")
        assert checkpoints, "Worker completion did not persist a checkpoint"
        checkpoint = Path(checkpoints[0]["adapter_path"])
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        matrices = payload["adapters"]
        assert matrices, "No adapter matrices were produced"
        assert any(abs(value) > 0 for adapter in matrices.values() for row in adapter["B"] for value in row), "LoRA B parameters remained at their zero initialization"
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == original_hash, "Frozen base weights changed"
        restored = FineTuningManager(OpenAgentStore(root / "runtime"))
        assert restored.get_job(job["id"], project="default")["status"] == "completed"
        assert restored.list_checkpoints(job_id=job["id"], project="default"), "Checkpoint disappeared on reload"
        return {"status": "PASS", "model": identity, "job_status": recorded["status"],
                "checkpoints": len(checkpoints), "adapter_modules": len(matrices),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "base_weights_unchanged": True, "adapter_parameters_updated": True,
                "persistence_reload_verified": True,
                "limitation": "Existing approximate character-token/layer-proxy worker; not full Llama task-loss training or quality validation"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
