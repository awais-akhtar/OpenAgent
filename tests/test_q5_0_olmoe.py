"""Q5_0 checkpoint and gateway execution against independently decoded weights."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.expert_kernels import (
    ExpertKernelRegistry, MappingPayloadSource, NumPyRoutedSwiGLUBackend, _lease,
)
from openagent.adaptive_engine.native_q5_0_expert import NativeQ5_0ExpertBackend
from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_cli import _kernel_resources, build_parser
from openagent.adaptive_engine.q5_0_expert import Q5_0ExpertBackend, pack_expert, payload_size
from tests.test_q8_0_olmoe import checkpoint


def decode(payload, hidden, intermediate):
    cursor = 0
    matrices = []
    for rows, columns in ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate)):
        matrix = np.empty((rows, columns), np.float32)
        for row in range(rows):
            for column in range(0, columns, 32):
                delta, high = struct.unpack_from("<eI", payload, cursor)
                for lane in range(32):
                    byte = payload[cursor + 6 + lane % 16]
                    low = byte & 15 if lane < 16 else byte >> 4
                    code = low + (((high >> lane) & 1) << 4) - 16
                    matrix[row, column + lane] = np.float32(delta * code)
                cursor += 22
        matrices.append(matrix)
    return matrices


class DecodedOracle(Q5_0ExpertBackend):
    NAME = "q5_0-test-decoded"

    def info(self):
        return replace(super().info(), name=self.NAME)

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size,
                      dtype="q5_0"):
        blobs = {}
        for routes in route_batch:
            for key, _ in routes:
                if key not in blobs:
                    with _lease(source, key) as payload:
                        matrices = decode(payload, hidden_size, intermediate_size)
                    blobs[key] = b"".join(matrix.astype("<f4").tobytes() for matrix in matrices)
        return NumPyRoutedSwiGLUBackend().execute_batch(
            MappingPayloadSource(blobs), inputs, route_batch, hidden_size=hidden_size,
            intermediate_size=intermediate_size, dtype="float32")


class Q5OlmoeTests(unittest.TestCase):
    def test_checkpoint_hashes_packing_logits_and_cached_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.q5", dtype="q5_0")
            for layer in manifest.layers:
                self.assertEqual(layer.dtype, "q5_0")
                for key, region in layer.regions.items():
                    with (root / region.path).open("rb") as stream:
                        stream.seek(region.offset)
                        blob = stream.read(region.size)
                    self.assertEqual(region.size, payload_size(64, 96))
                    self.assertEqual(hashlib.sha256(blob).hexdigest(), region.sha256)
                    matrices = [tensors[f"{layer.id}.mlp.experts.{key}.{role}_proj.weight"]
                                for role in ("gate", "up", "down")]
                    self.assertEqual(blob, pack_expert(*matrices))
            backends = [Q5_0ExpertBackend()]
            if NativeQ5_0ExpertBackend().info().available:
                backends.append(NativeQ5_0ExpertBackend())
            for backend in backends:
                registry = ExpertKernelRegistry()
                registry.register(backend, replace=True)
                with OlmoeModel(root, manifest, kernels=registry,
                                preferred_kernel=backend.NAME,
                                expert_cache_bytes=payload_size(64, 96)) as actual, OlmoeModel(
                                    root, manifest, kernels=ExpertKernelRegistry([DecodedOracle()]),
                                    preferred_kernel=DecodedOracle.NAME,
                                    expert_cache_bytes=payload_size(64, 96)) as expected:
                    for tokens in ([1, 4, 7], [3], [9, 2]):
                        np.testing.assert_allclose(actual.forward(tokens), expected.forward(tokens),
                                                   rtol=8e-5, atol=8e-6)

    def test_explicit_cli_and_gateway_compiled_selection(self):
        backend = NativeQ5_0ExpertBackend()
        if not backend.info().available:
            self.skipTest("active optional Numba required")
        from openagent.model_gateway import OlmoeDirectProvider

        parsed = build_parser().parse_args(["pack", "model", "--dtype", "q5_0"])
        self.assertEqual(parsed.dtype, "q5_0")
        registry, resource = _kernel_resources(Path("."), "q5_0-numba", 1)
        self.assertIsNone(resource)
        self.assertTrue(registry.resolve(dtype="q5_0", preferred="q5_0-numba",
                                         allow_fallback=False).info().capabilities.compiled)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "gateway.q5", dtype="q5_0")
            path = root / "manifest.json"
            manifest.write(path)
            environment = {"OPENAGENT_OLMOE_MODEL_DIR": str(root),
                           "OPENAGENT_OLMOE_MANIFEST": str(path),
                           "OPENAGENT_OLMOE_KERNEL": "q5_0-numba",
                           "OPENAGENT_OLMOE_CONTEXT": "16",
                           "OPENAGENT_OLMOE_EXPERT_CACHE_GB": "0.001",
                           "OPENAGENT_OLMOE_IO_MODE": "mmap",
                           "OPENAGENT_OLMOE_STORAGE_ROOTS": "",
                           "OPENAGENT_OLMOE_PREFETCH_WORKERS": "1",
                           "OPENAGENT_OLMOE_PREFETCH_DEPTH": "1"}
            with patch.dict(os.environ, environment), OlmoeModel(
                    root, manifest, kernels=ExpertKernelRegistry([backend]),
                    preferred_kernel="q5_0-numba", context_tokens=16,
                    expert_cache_bytes=int(.001 * 1024**3)) as direct:
                provider = OlmoeDirectProvider()
                try:
                    runtime = provider._load()
                    self.assertIs(type(runtime.experts.kernels.resolve(
                        dtype="q5_0", preferred="q5_0-numba", allow_fallback=False)),
                        NativeQ5_0ExpertBackend)
                    for tokens in ([1, 4], [7], [3, 9]):
                        np.testing.assert_array_equal(runtime.forward(tokens), direct.forward(tokens))
                finally:
                    if provider._runtime is not None:
                        provider._runtime.close()


if __name__ == "__main__":
    unittest.main()
