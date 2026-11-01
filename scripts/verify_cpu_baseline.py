"""Run the bundled small OpenVINO model through OpenAgent's real CPU gateway.

This is a local product smoke check, not a Phase 1 family-route receipt. It
never downloads or converts a model and never enables an external fallback.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1] /
                        "models/openvino/qwen2.5-1.5b-instruct-int4-ov")
    parser.add_argument("--runtime-python", type=Path, default=Path(__file__).resolve().parents[1] /
                        ".accelerator-venv/Scripts/python.exe")
    parser.add_argument("--max-model-gib", type=float, default=4.0)
    parser.add_argument("--max-sampled-rss-gib", type=float, default=12.0)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    runtime_python = args.runtime_python.resolve()
    required = ("config.json", "openvino_model.xml", "openvino_model.bin", "tokenizer.json")
    if not model_dir.is_dir() or any(not (model_dir / name).is_file() for name in required):
        parser.error("a complete local OpenVINO model directory is required; nothing is downloaded")
    if not runtime_python.is_file():
        parser.error("an existing OpenVINO GenAI Python runtime is required; nothing is installed")
    if args.max_model_gib <= 0 or args.max_sampled_rss_gib <= 0:
        parser.error("positive model and sampled-memory budgets are required")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2":
        parser.error("this baseline checks a Qwen2 OpenVINO model, not a MoE route")
    model_bytes = sum(path.stat().st_size for path in model_dir.iterdir() if path.is_file())
    if model_bytes > args.max_model_gib * 1024**3:
        parser.error("local model exceeds the configured storage budget")

    import psutil
    from openagent.adaptive_engine.device_runtime import DirectHardwareClient
    from openagent.adaptive_engine.hardware import HardwareProfiler
    from openagent.model_gateway import ModelGateway

    os.environ["OPENAGENT_MODEL_FALLBACK"] = "none"
    profiler = HardwareProfiler(model_dir.parent, accelerator_python=str(runtime_python))
    client = DirectHardwareClient(profiler)
    gateway = ModelGateway(direct_hardware=client)
    gateway.providers["openvino-direct"].configure(
        model_path=str(model_dir), device="CPU", model_id="phase1-cpu-small")
    stop = threading.Event()
    peak_rss = 0

    def sample() -> None:
        nonlocal peak_rss
        while not stop.wait(0.02):
            process = client._process  # diagnostic only; owned by DirectHardwareClient
            if process is None:
                continue
            try:
                root = psutil.Process(process.pid)
                processes = [root, *root.children(recursive=True)]
                rss = sum(item.memory_info().rss for item in processes if item.is_running())
                peak_rss = max(peak_rss, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    started = time.perf_counter()
    try:
        result = gateway.complete(
            [{"role": "user", "content": "Say hello in one short sentence."}],
            provider_id="openvino-direct", max_output_tokens=16)
        chunks = list(gateway.stream_complete(
            [{"role": "user", "content": "Say the word ready."}],
            provider_id="openvino-direct", max_output_tokens=16))
        streamed = "".join(chunk.content for chunk in chunks)
        if (result.provider != "openvino-direct" or not result.content.strip() or
                not chunks or not chunks[-1].done or not streamed.strip() or
                any(chunk.provider != "openvino-direct" for chunk in chunks)):
            raise RuntimeError("CPU gateway completion or streaming was incomplete")
    finally:
        stop.set()
        watcher.join(timeout=2)
        client.close()

    report = {
        "kind": "local-cpu-product-smoke-not-family-route-receipt",
        "model_type": config["model_type"],
        "backend": "openvino-direct",
        "device": "CPU",
        "model_bytes": model_bytes,
        "sampled_process_tree_peak_rss_bytes": peak_rss,
        "sampled_rss_limit_bytes": int(args.max_sampled_rss_gib * 1024**3),
        "complete_text": result.content,
        "stream_text": streamed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(report, sort_keys=True))
    if peak_rss == 0 or peak_rss > report["sampled_rss_limit_bytes"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
