"""Measure one local OpenVINO GenAI model on selected hardware devices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openagent.adaptive_engine.runtime import AdaptiveExecutionEngine
from openagent.storage import OpenAgentStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--data-dir", default=".tmp-device-benchmark")
    parser.add_argument("--devices", default="gpu:openvino,npu:openvino")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    engine = AdaptiveExecutionEngine(OpenAgentStore(Path(args.data_dir).expanduser()))
    results = []
    try:
        for device_id in [item.strip() for item in args.devices.split(",") if item.strip()]:
            model_id = "hardware-smoke-" + device_id.replace(":", "-").replace(".", "-")
            try:
                loaded = engine.load_direct_llm(
                    str(model_path), device_id=device_id, project="hardware-check", model_id=model_id
                )
                generated = engine.generate_direct(
                    model_id,
                    prompt="Reply with exactly: OpenAgent hardware verified.",
                    generation={"max_new_tokens": args.max_new_tokens},
                    project="hardware-check",
                )
                results.append({"device_id": device_id, "loaded": loaded, "generated": generated})
                engine.unload_direct_model(model_id, project="hardware-check")
            except Exception as exc:  # noqa: BLE001 - smoke output must retain every device result
                results.append({"device_id": device_id, "error": str(exc)})
    finally:
        engine.close()
    print(json.dumps({"model_path": str(model_path), "results": results}, indent=2))
    return 1 if not any("generated" in item for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
