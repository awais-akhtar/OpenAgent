"""Inspect local Transformers MoE placement; execution requires P02 route admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .hf_moe import (HFMoeAdapter, HFMoeManifest, expert_module_names,
                     require_hf_moe_execution_admission)


def expert_placement(manifest: HFMoeManifest, resident_device: str, *, disk: bool) -> dict[str, str]:
    return {
        name: "disk" if disk else resident_device
        for name in expert_module_names(manifest)
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--offload-directory", type=Path)
    parser.add_argument("--resident-device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--chat-template", action="store_true")
    args = parser.parse_args(argv)
    if not args.prompt.strip() or args.max_tokens <= 0:
        parser.error("prompt and positive max-tokens are required")
    try:
        manifest = HFMoeManifest.read(args.model)
        require_hf_moe_execution_admission(manifest)
        from transformers import AutoTokenizer

        placement = expert_placement(manifest, args.resident_device, disk=args.offload_directory is not None)
        adapter = HFMoeAdapter.load(args.model, resident_device=args.resident_device,
                                  expert_placement=placement, offload_directory=args.offload_directory)
        tokenizer = AutoTokenizer.from_pretrained(str(manifest.model_directory),
                                                 local_files_only=True, trust_remote_code=False)
        text = args.prompt
        if args.chat_template:
            text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                 tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False)
        limit = json.loads(manifest.config_json).get("max_position_embeddings")
        if limit is not None and inputs["input_ids"].shape[1] + args.max_tokens > limit:
            raise ValueError("prompt and output exceed checkpoint context capacity")
        inputs = {key: value.to(adapter.resident_device) for key, value in inputs.items()}
        output = adapter.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        generated = output[0, inputs["input_ids"].shape[1]:]
        print(json.dumps({"text": tokenizer.decode(generated, skip_special_tokens=True),
                          "model_type": manifest.model_type,
                          "resident_device": str(adapter.resident_device),
                          "disk_experts": sum(value == "disk" for value in adapter.placement.values()),
                          "generated_tokens": len(generated), "external_calls": False}), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
