"""Locked reference run of the same Qwen3-Coder GGUF through Ollama's llama.cpp backend.

Records prompt-processing and generation rates from Ollama's own timing fields,
the output token IDs' text, and whether the model was unloaded before the cold run.
Usage: python scripts/bench_ollama_reference.py --label ollama_ref
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openagent.adaptive_engine.qwen3moe_decoder import chat_prompt  # noqa: E402

PROMPT = "Write a Python function that adds two numbers and explain it briefly."
URL = "http://127.0.0.1:11434/api/generate"


def call(payload: dict) -> tuple[dict, float]:
    request = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={"content-type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def summarize(body: dict, wall: float) -> dict:
    eval_s = body.get("eval_duration", 0) / 1e9
    prompt_s = body.get("prompt_eval_duration", 0) / 1e9
    load_s = body.get("load_duration", 0) / 1e9
    return {
        "prompt_tokens": body.get("prompt_eval_count"),
        "prompt_tok_per_s": round(body.get("prompt_eval_count", 0) / prompt_s, 3) if prompt_s else None,
        "generated_tokens": body.get("eval_count"),
        "gen_tok_per_s": round(body.get("eval_count", 0) / eval_s, 3) if eval_s else None,
        "load_seconds": round(load_s, 3),
        "wall_seconds": round(wall, 3),
        "text_preview": body.get("response", "")[:160],
        "response": body.get("response", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()
    options = {"temperature": 0, "num_predict": args.tokens, "num_ctx": 1024, "seed": 1}
    base = {"model": "qwen3-coder:30b", "prompt": chat_prompt([{"role": "user", "content": PROMPT}]),
            "raw": True, "stream": False, "options": options}
    call({"model": "qwen3-coder:30b", "prompt": "", "keep_alive": 0, "stream": False})
    cold_body, cold_wall = call(base)
    warm_body, warm_wall = call(base)
    result = {
        "label": args.label,
        "engine": "ollama (llama.cpp backend)",
        "model": "qwen3-coder:30b",
        "tokens_requested": args.tokens,
        "cold": summarize(cold_body, cold_wall),
        "warm": summarize(warm_body, warm_wall),
        "warm_matches_cold": cold_body.get("response") == warm_body.get("response"),
    }
    out = ROOT / "scripts" / "bench_results" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"cold gen {result['cold']['gen_tok_per_s']} tok/s, warm gen {result['warm']['gen_tok_per_s']} tok/s, "
          f"warm prompt {result['warm']['prompt_tok_per_s']} tok/s, tokens {result['warm']['generated_tokens']}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
