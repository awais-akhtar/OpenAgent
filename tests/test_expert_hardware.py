"""Opt-in differential execution on actual OpenVINO devices."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.worker import HardwareWorker


@unittest.skipUnless(os.environ.get("OPENAGENT_TEST_EXPERT_DEVICES"), "hardware devices must be explicitly selected")
class ExpertHardwareTests(unittest.TestCase):
    def test_worker_matches_reference_on_selected_devices(self):
        worker = HardwareWorker()
        rng = np.random.default_rng(47)
        hidden, intermediate = 8, 16
        weights = rng.normal(0, .2, 3 * hidden * intermediate).astype(np.float32)
        vector = rng.normal(0, .2, hidden).astype(np.float32)
        for device in os.environ["OPENAGENT_TEST_EXPERT_DEVICES"].split(","):
            device = device.strip().upper()
            for dtype in ("float32", "float16", "int8-row"):
                with self.subTest(device=device, dtype=dtype), tempfile.TemporaryDirectory() as temp:
                    if dtype == "int8-row":
                        quantized = np.clip(np.rint(weights * 64), -127, 127).astype(np.int8)
                        scales = np.full(2 * intermediate + hidden, 1 / 64, dtype="<f4")
                        payload = quantized.tobytes() + scales.tobytes()
                    else:
                        payload = weights.astype("<f4" if dtype == "float32" else "<f2").tobytes()
                    path = Path(temp) / "expert.bin"
                    path.write_bytes(payload)
                    identifier = f"probe-{device.replace('.', '-')}-{dtype}"
                    loaded = worker.load_expert({
                        "expert_id": identifier, "path": str(path), "offset": 0,
                        "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                        "hidden_size": hidden, "intermediate_size": intermediate,
                        "dtype": dtype, "device": device,
                    })
                    try:
                        result = worker.run_expert({"expert_id": identifier, "inputs": vector.tolist()})
                        self.assertTrue(any(str(value).upper().startswith(device) for value in loaded["execution_devices"]))
                        reference = NumPyRoutedSwiGLUBackend().execute(
                            MappingPayloadSource({"a": payload}), vector, [("a", 1.0)],
                            hidden_size=hidden, intermediate_size=intermediate, dtype=dtype,
                        )
                        np.testing.assert_allclose(result["output"], reference, atol=2e-4, rtol=2e-3)
                    finally:
                        worker.models.pop(identifier, None)
