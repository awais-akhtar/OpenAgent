"""Block FP8 full-model execution with an independent test-only unpacker."""
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.expert_kernels import (
    ExpertKernelRegistry, MappingPayloadSource, NumPyRoutedSwiGLUBackend, _lease,
)
from openagent.adaptive_engine.fp8_block_expert import (
    DTYPE, matrix_payload_size, pack_expert, payload_size,
)
from openagent.adaptive_engine.native_fp8_block_expert import NativeFP8BlockExpertBackend
from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.olmoe_cli import _kernel_resources
from tests.test_fp8_block_expert import decode_matrix
from tests.test_q8_0_olmoe import checkpoint


class DecodedOracle(NativeFP8BlockExpertBackend):
    NAME = "fp8-test-decoded"

    def info(self):
        info = super().info()
        return replace(info, name=self.NAME, provider="test-only", available=True,
                       unavailable_reason="", capabilities=replace(info.capabilities, compiled=False))

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
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
                        count = matrix_payload_size(rows, columns)
                        matrices.append(decode_matrix(payload[cursor:cursor + count], rows, columns))
                        cursor += count
                blobs[key] = b"".join(matrix.astype("<f4").tobytes() for matrix in matrices)
        return NumPyRoutedSwiGLUBackend().execute_batch(
            MappingPayloadSource(blobs), inputs, route_batch, hidden_size=hidden_size,
            intermediate_size=intermediate_size, dtype="float32")


@unittest.skipUnless(NativeFP8BlockExpertBackend().info().available, "active optional Numba required")
class FP8ModelIntegrationTests(unittest.TestCase):
    def test_packer_hashes_projection_scales_and_full_cached_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.fp8", dtype=DTYPE)
            for layer in manifest.layers:
                self.assertEqual(layer.dtype, DTYPE)
                for key, region in layer.regions.items():
                    with (root / region.path).open("rb") as stream:
                        stream.seek(region.offset)
                        blob = stream.read(region.size)
                    self.assertEqual(len(blob), payload_size(64, 96))
                    self.assertEqual(hashlib.sha256(blob).hexdigest(), region.sha256)
                    matrices = [tensors[f"{layer.id}.mlp.experts.{key}.{role}_proj.weight"]
                                for role in ("gate", "up", "down")]
                    self.assertEqual(blob, pack_expert(*matrices))
            with OlmoeModel(root, manifest, kernels=ExpertKernelRegistry([NativeFP8BlockExpertBackend()]),
                            preferred_kernel=NativeFP8BlockExpertBackend.NAME,
                            expert_cache_bytes=payload_size(64, 96)) as actual, OlmoeModel(
                                root, manifest, kernels=ExpertKernelRegistry([DecodedOracle()]),
                                preferred_kernel=DecodedOracle.NAME,
                                expert_cache_bytes=payload_size(64, 96)) as expected:
                for tokens in ([1, 4], [7], [3, 9]):
                    np.testing.assert_allclose(actual.forward(tokens), expected.forward(tokens),
                                               atol=8e-6, rtol=8e-5)

    def test_actual_gateway_and_cli_select_compiled_backend(self):
        from openagent.model_gateway import OlmoeDirectProvider

        registry, resource = _kernel_resources(Path("."), NativeFP8BlockExpertBackend.NAME, 1)
        self.assertIsNone(resource)
        self.assertTrue(registry.resolve(dtype=DTYPE, preferred=NativeFP8BlockExpertBackend.NAME,
                                         allow_fallback=False).info().capabilities.compiled)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "experts.fp8", dtype=DTYPE)
            path = root / "manifest.json"
            manifest.write(path)
            with patch.dict(os.environ, {"OPENAGENT_OLMOE_MODEL_DIR": str(root),
                    "OPENAGENT_OLMOE_MANIFEST": str(path),
                    "OPENAGENT_OLMOE_KERNEL": NativeFP8BlockExpertBackend.NAME,
                    "OPENAGENT_OLMOE_CONTEXT": "16", "OPENAGENT_OLMOE_EXPERT_CACHE_GB": "0.001",
                    "OPENAGENT_OLMOE_IO_MODE": "mmap", "OPENAGENT_OLMOE_STORAGE_ROOTS": ""}):
                provider = OlmoeDirectProvider()
                try:
                    runtime = provider._load()
                    self.assertIs(type(runtime.experts.kernels.resolve(dtype=DTYPE,
                        preferred=NativeFP8BlockExpertBackend.NAME, allow_fallback=False)),
                        NativeFP8BlockExpertBackend)
                    self.assertTrue(np.isfinite(runtime.forward([1, 4])).all())
                finally:
                    if provider._runtime is not None:
                        provider._runtime.close()


if __name__ == "__main__":
    unittest.main()
