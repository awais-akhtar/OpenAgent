"""Benchmark the Qwen3-MoE direct decoder on a fixed GGUF, prompt and length.

Usage:
    python scripts/bench_qwen3moe.py --gguf PATH --cache-gb 4 --label baseline

Runs one cold generation (fresh decoder, cold expert cache) and one warm
generation on the same decoder, then writes a JSON record to
scripts/bench_results/<label>.json so each change can be compared with the
baseline. Times are measured in the decoder's own stage timers; process and
system memory come from Windows APIs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openagent.adaptive_engine.qwen3moe_decoder import (  # noqa: E402
    ByteLevelBPE,
    Qwen3MoeDecoder,
    chat_prompt,
)

PROMPT = "Write a Python function that adds two numbers and explain it briefly."
RESULTS_DIR = ROOT / "scripts" / "bench_results"


class _ProcessMemory(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def process_memory() -> dict:
    counters = _ProcessMemory()
    counters.cb = ctypes.sizeof(_ProcessMemory)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemory), wintypes.DWORD]
    psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return {
        "working_set_gb": round(counters.WorkingSetSize / 2**30, 3),
        "peak_working_set_gb": round(counters.PeakWorkingSetSize / 2**30, 3),
        "private_commit_gb": round(counters.PagefileUsage / 2**30, 3),
        "page_faults": int(counters.PageFaultCount),
    }


def system_memory() -> dict:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(status))
    return {
        "load_percent": int(status.dwMemoryLoad),
        "total_gb": round(status.ullTotalPhys / 2**30, 2),
        "used_gb": round((status.ullTotalPhys - status.ullAvailPhys) / 2**30, 2),
    }


def run(decoder: Decoder, tokenizer: ByteLevelBPE, prompt_ids: list[int], tokens: int) -> dict:
    decoder.timers = {key: 0.0 for key in decoder.timers}
    decoder.timers["forwards"] = 0
    stats_before = decoder.store.status()
    memory_before = process_memory()
    started = time.perf_counter()
    produced = decoder.generate(prompt_ids, max_new_tokens=tokens)
    elapsed = time.perf_counter() - started
    stats_after = decoder.store.status()
    timers = dict(decoder.timers)
    forwards = max(1, int(timers.pop("forwards")))
    attributed = sum(timers.get(key, 0.0) for key in ("attn_proj", "attn_core", "router", "experts", "output"))
    return {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(produced),
        "token_ids": produced,
        "text_preview": tokenizer.decode(produced)[:160],
        "wall_seconds": round(elapsed, 3),
        "tok_per_s": round(len(produced) / elapsed, 3) if elapsed else 0.0,
        "ms_per_forward": round(1000 * elapsed / forwards, 2),
        "stage_ms_per_forward": {
            key: round(1000 * value / forwards, 3) for key, value in timers.items() if key != "moe"
        },
        "other_ms_per_forward": round(1000 * max(0.0, elapsed - attributed) / forwards, 3),
        "expert_cache": {
            "hits": stats_after["hits"] - stats_before["hits"],
            "misses": stats_after["misses"] - stats_before["misses"],
            "hit_rate": round(
                (stats_after["hits"] - stats_before["hits"])
                / max(1, (stats_after["hits"] - stats_before["hits"]) + (stats_after["misses"] - stats_before["misses"])),
                4,
            ),
            "read_seconds": round(stats_after["read_seconds"] - stats_before["read_seconds"], 3),
            "resident_gb": round(stats_after["resident_bytes"] / 2**30, 3),
        },
        "process_memory_before": memory_before,
        "process_memory_after": process_memory(),
        "system_memory_after": system_memory(),
    }


Decoder = Qwen3MoeDecoder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", required=True, help="path to the Qwen3-MoE GGUF file")
    parser.add_argument("--cache-gb", type=float, default=4.0, help="expert cache limit in GiB")
    parser.add_argument("--tokens", type=int, default=256, help="generated tokens per pass")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--label", required=True, help="name of the result file")
    parser.add_argument("--io-mode", default="mmap", choices=["mmap", "buffered", "direct", "uring"])
    args = parser.parse_args()

    decoder = Qwen3MoeDecoder(args.gguf, context=args.context, expert_cache_bytes=int(args.cache_gb * 2**30),
                              io_mode=args.io_mode)
    tokenizer = ByteLevelBPE(decoder.gguf)
    prompt_ids = tokenizer.encode(chat_prompt([{"role": "user", "content": PROMPT}]))
    cold = run(decoder, tokenizer, prompt_ids, args.tokens)
    warm = run(decoder, tokenizer, prompt_ids, args.tokens)
    result = {
        "label": args.label,
        "gguf": Path(args.gguf).name,
        "prompt": PROMPT,
        "cache_gb": args.cache_gb,
        "io_mode": args.io_mode,
        "tokens_requested": args.tokens,
        "platform": platform.platform(),
        "cpu_logical": os.cpu_count(),
        "cold": cold,
        "warm": warm,
        "warm_matches_cold_tokens": warm["token_ids"] == cold["token_ids"],
    }
    decoder.close()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULTS_DIR / f"{args.label}.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("label", "cache_gb")}), file=sys.stderr)
    print(f"cold {cold['tok_per_s']} tok/s, warm {warm['tok_per_s']} tok/s, warm ms/forward {warm['ms_per_forward']}")
    print(f"saved {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
