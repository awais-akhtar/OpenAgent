"""Actual isolated HTTP server with a tiny local PyTorch checkpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.support.tiny_finetune_verification import create_tiny_model
from openagent.fine_tuning import FineTuningManager
from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


if __name__ == "__main__":
    os.environ["OPENAGENT_AUTH_ENFORCED"] = "0"
    with TemporaryDirectory(prefix="oa-tiny-model-http-") as temporary:
        root = Path(temporary)
        model = root / "tiny-model"
        identity = create_tiny_model(model)
        store = OpenAgentStore(root / "runtime")
        manager = FineTuningManager(store)
        dataset = manager.create_dataset("Tiny actual worker", project="default")
        manager.add_example(dataset["id"], "A", "B", project="default")
        manager.add_example(dataset["id"], "C", "D", project="default")
        server = build_http_server("127.0.0.1", 0, store, warm_openvino=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print("READY:" + json.dumps({"origin": f"http://127.0.0.1:{server.server_address[1]}",
              "modelDir": str(model), "datasetId": dataset["id"], "identity": identity}), flush=True)
        try:
            sys.stdin.readline()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
