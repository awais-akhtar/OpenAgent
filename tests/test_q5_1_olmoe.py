"""Q5_1 model integration with independent affine block decoding."""
from dataclasses import replace
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
from openagent.adaptive_engine.native_q5_1_expert import NativeQ5_1ExpertBackend
from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_cli import _kernel_resources, main
from openagent.adaptive_engine.q5_1_expert import Q5_1ExpertBackend, pack_expert, payload_size
from tests.test_q8_0_olmoe import checkpoint


class DecodedOracle(Q5_1ExpertBackend):
    NAME = "q5_1-test-decoded"

    def info(self):
        return replace(super().info(), name=self.NAME)

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size,
                      dtype="q5_1"):
        blobs = {}
        for routes in route_batch:
            for key, _ in routes:
                if key in blobs:
                    continue
                cursor, matrices = 0, []
                with _lease(source, key) as payload:
                    for rows, columns in ((intermediate_size, hidden_size),
                                          (intermediate_size, hidden_size),
                                          (hidden_size, intermediate_size)):
                        matrix = np.empty((rows, columns), np.float32)
                        for row in range(rows):
                            for column in range(0, columns, 32):
                                delta, minimum, high = struct.unpack_from("<eeI", payload, cursor)
                                for lane in range(32):
                                    byte = payload[cursor + 8 + lane % 16]
                                    low = byte & 15 if lane < 16 else byte >> 4
                                    code = low + (((high >> lane) & 1) << 4)
                                    matrix[row, column + lane] = np.float32(delta * code + minimum)
                                cursor += 24
                        matrices.append(matrix)
                blobs[key] = b"".join(matrix.astype("<f4").tobytes() for matrix in matrices)
        return NumPyRoutedSwiGLUBackend().execute_batch(
            MappingPayloadSource(blobs), inputs, route_batch, hidden_size=hidden_size,
            intermediate_size=intermediate_size, dtype="float32")


class Q5AffineOlmoeTests(unittest.TestCase):
    def test_packing_bytes_cached_logits_and_format_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.q51", dtype="q5_1")
            for layer in manifest.layers:
                for key, region in layer.regions.items():
                    with (root / region.path).open("rb") as stream:
                        stream.seek(region.offset)
                        blob = stream.read(region.size)
                    self.assertEqual(region.size, payload_size(64, 96))
                    matrices = [tensors[f"{layer.id}.mlp.experts.{key}.{role}_proj.weight"]
                                for role in ("gate", "up", "down")]
                    self.assertEqual(blob, pack_expert(*matrices))
            backends = [Q5_1ExpertBackend()]
            if NativeQ5_1ExpertBackend().info().available:
                backends.append(NativeQ5_1ExpertBackend())
            for backend in backends:
                registry = ExpertKernelRegistry()
                registry.register(backend, replace=True)
                with OlmoeModel(root, manifest, kernels=registry, preferred_kernel=backend.NAME,
                                expert_cache_bytes=payload_size(64, 96)) as actual, OlmoeModel(
                                    root, manifest, kernels=ExpertKernelRegistry([DecodedOracle()]),
                                    preferred_kernel=DecodedOracle.NAME,
                                    expert_cache_bytes=payload_size(64, 96)) as expected:
                    for tokens in ([1, 4], [7], [3, 9]):
                        np.testing.assert_allclose(actual.forward(tokens), expected.forward(tokens),
                                                   rtol=8e-5, atol=8e-6)

    def test_gateway_and_cli_defaults_preserve_other_formats(self):
        from contextlib import redirect_stdout
        import io
        from openagent.model_gateway import OlmoeDirectProvider

        if not NativeQ5_1ExpertBackend().info().available:
            self.skipTest("active optional Numba required")
        registry, resource = _kernel_resources(Path("."), "q5_1-numba", 1)
        self.assertIsNone(resource)
        self.assertTrue(registry.resolve(dtype="q5_1", preferred="q5_1-numba",
                                         allow_fallback=False).info().capabilities.compiled)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            protected = root / "openagent-experts.q5"
            protected.write_bytes(b"other-format")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["pack", str(root), "--dtype", "q5_1"]), 0)
            self.assertEqual(protected.read_bytes(), b"other-format")
            path = root / "openagent-experts.q51.manifest.json"
            with patch.dict(os.environ, {"OPENAGENT_OLMOE_MODEL_DIR": str(root),
                    "OPENAGENT_OLMOE_MANIFEST": str(path), "OPENAGENT_OLMOE_KERNEL": "q5_1-numba",
                    "OPENAGENT_OLMOE_CONTEXT": "16", "OPENAGENT_OLMOE_EXPERT_CACHE_GB": "0.001",
                    "OPENAGENT_OLMOE_IO_MODE": "mmap", "OPENAGENT_OLMOE_STORAGE_ROOTS": ""}):
                provider = OlmoeDirectProvider()
                try:
                    loaded = provider._load()
                    backend = loaded.experts.kernels.resolve(dtype="q5_1", preferred="q5_1-numba",
                                                             allow_fallback=False)
                    self.assertIs(type(backend), NativeQ5_1ExpertBackend)
                    with OlmoeModel(root, path, kernels=ExpertKernelRegistry([NativeQ5_1ExpertBackend()]),
                                    preferred_kernel="q5_1-numba", context_tokens=16,
                                    expert_cache_bytes=int(.001 * 1024**3)) as direct:
                        np.testing.assert_array_equal(loaded.forward([1, 4]), direct.forward([1, 4]))
                finally:
                    if provider._runtime is not None:
                        provider._runtime.close()


if __name__ == "__main__":
    unittest.main()
