"""Synthetic real SafeTensors packing and model logits, not checkpoint parity."""

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.expert_kernels import (ExpertKernelRegistry, MappingPayloadSource,
                                                     NumPyRoutedSwiGLUBackend, _lease)
from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.q4_1_expert import Q4_1ExpertBackend, pack_expert, payload_size
from openagent.adaptive_engine.native_q4_1_expert import NativeQ4_1ExpertBackend, _compiled
from openagent.adaptive_engine.olmoe_cli import main, _kernel_resources
from openagent.adaptive_engine.expert_runtime import SparseExpertManifest
from tests.test_q8_0_olmoe import checkpoint
from tests.test_q4_1_expert import scalar_decode


class DecodedOracle(Q4_1ExpertBackend):
    """Test-only full decode into the established float32 reference executor."""

    NAME = "q4_1-test-decoded"

    def info(self):
        return replace(super().info(), name=self.NAME)

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype="q4_1"):
        blobs = {}
        for routes in route_batch:
            for key, _ in routes:
                if key not in blobs:
                    with _lease(source, key) as view:
                        matrices = scalar_decode(view, hidden_size, intermediate_size)
                    blobs[key] = b"".join(m.astype("<f4").tobytes() for m in matrices)
        return NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource(blobs), inputs,
                     route_batch, hidden_size=hidden_size, intermediate_size=intermediate_size, dtype="float32")


class Q4OlmoeTests(unittest.TestCase):
    def test_real_packer_and_decoded_float32_model_logit_differential(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = checkpoint(root)
            packed = pack_olmoe_experts(root, root / "experts.q41", dtype="q4_1")
            for layer in packed.layers:
                self.assertEqual(layer.dtype, "q4_1")
                for key, region in layer.regions.items():
                    with (root / region.path).open("rb") as stream:
                        stream.seek(region.offset)
                        blob = stream.read(region.size)
                    self.assertEqual(region.size, payload_size(64, 96))
                    self.assertEqual(hashlib.sha256(blob).hexdigest(), region.sha256)
                    matrices = [tensors[f"{layer.id}.mlp.experts.{key}.{role}_proj.weight"]
                                for role in ("gate", "up", "down")]
                    self.assertEqual(blob, pack_expert(*matrices))
            candidates = [Q4_1ExpertBackend()]
            if _compiled:
                candidates.append(NativeQ4_1ExpertBackend())
            for backend in candidates:
                registry = ExpertKernelRegistry()
                registry.register(backend, replace=True)
                with OlmoeModel(root, packed, kernels=registry, preferred_kernel=backend.NAME,
                                expert_cache_bytes=payload_size(64, 96), prefetch_workers=1,
                                prefetch_depth=1) as actual, OlmoeModel(
                                    root, packed, expert_cache_bytes=payload_size(64, 96),
                                    kernels=ExpertKernelRegistry([DecodedOracle()]),
                                    preferred_kernel=DecodedOracle.NAME, prefetch_workers=1,
                                    prefetch_depth=1) as expected:
                    for tokens in ([1, 4, 7], [3], [9, 2]):
                        np.testing.assert_allclose(actual.forward(tokens), expected.forward(tokens),
                                                   rtol=8e-5, atol=8e-6)

    @unittest.skipUnless(_compiled, "active optional Numba required")
    def test_gateway_real_load_compiled_backend_and_direct_logits(self):
        from openagent.model_gateway import OlmoeDirectProvider

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            manifest = pack_olmoe_experts(root, root / "gateway.q41", dtype="q4_1")
            manifest_path = root / "gateway.manifest.json"
            manifest.write(manifest_path)
            environment = {
                "OPENAGENT_OLMOE_MODEL_DIR": str(root),
                "OPENAGENT_OLMOE_MANIFEST": str(manifest_path),
                "OPENAGENT_OLMOE_KERNEL": "q4_1-numba",
                "OPENAGENT_OLMOE_CONTEXT": "16",
                "OPENAGENT_OLMOE_EXPERT_CACHE_GB": "0.001",
                "OPENAGENT_OLMOE_IO_MODE": "mmap",
                "OPENAGENT_OLMOE_STORAGE_ROOTS": "",
                "OPENAGENT_OLMOE_PREFETCH_WORKERS": "1",
                "OPENAGENT_OLMOE_PREFETCH_DEPTH": "1",
            }
            registry = ExpertKernelRegistry()
            registry.register(NativeQ4_1ExpertBackend(), replace=True)
            with OlmoeModel(root, manifest, kernels=registry, preferred_kernel="q4_1-numba",
                            expert_cache_bytes=int(.001 * 1024 ** 3), context_tokens=16,
                            prefetch_workers=1, prefetch_depth=1) as direct, patch.dict(os.environ, environment):
                provider = OlmoeDirectProvider()
                try:
                    loaded = provider._load()
                    self.assertIs(loaded, provider._load())
                    selected = loaded.experts.kernels.resolve(dtype="q4_1", preferred="q4_1-numba",
                                                              allow_fallback=False)
                    self.assertIs(type(selected), NativeQ4_1ExpertBackend)
                    self.assertTrue(selected.info().capabilities.compiled)
                    for tokens in ([1, 4], [7], [3, 9]):
                        np.testing.assert_array_equal(loaded.forward(tokens), direct.forward(tokens))
                finally:
                    if provider._runtime is not None:
                        provider._runtime.close()
                        provider._runtime = None

    def test_cli_safe_defaults_and_bidirectional_format_preservation(self):
        formats = (
            ("int8-row", ("openagent-experts.i8", "openagent-experts.manifest.json")),
            ("int4-row", ("openagent-experts.i4", "openagent-experts.i4.manifest.json")),
            ("q8_0", ("openagent-experts.q8", "openagent-experts.q8.manifest.json")),
            ("q4_0", ("openagent-experts.q4", "openagent-experts.q4.manifest.json")),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            originals = {}
            for dtype, names in formats:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["pack", str(root), "--dtype", dtype]), 0)
                originals.update({name: (root / name).read_bytes() for name in names})
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["pack", str(root), "--dtype", "q4_1"]), 0)
            names = ("openagent-experts.q41", "openagent-experts.q41.manifest.json")
            q41 = {name: (root / name).read_bytes() for name in names}
            manifest = SparseExpertManifest.read(root / names[1])
            self.assertTrue(manifest.layers)
            self.assertTrue(all(layer.dtype == "q4_1" for layer in manifest.layers))
            self.assertTrue(all(region.path == names[0] and region.size == payload_size(64, 96)
                                for layer in manifest.layers for region in layer.regions.values()))
            for layer in manifest.layers:
                for region in layer.regions.values():
                    blob = q41[names[0]][region.offset:region.offset + region.size]
                    self.assertEqual(hashlib.sha256(blob).hexdigest(), region.sha256)
            for name, original in originals.items():
                self.assertEqual((root / name).read_bytes(), original)
            for dtype, _ in formats:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["pack", str(root), "--dtype", dtype]), 0)
                for name, original in q41.items():
                    self.assertEqual((root / name).read_bytes(), original)

    def test_cli_reference_and_compiled_selectors(self):
        reference = _kernel_resources(Path("."), "q4_1-reference", .001)[0].resolve(
            dtype="q4_1", preferred="q4_1-reference", allow_fallback=False)
        self.assertIs(type(reference), Q4_1ExpertBackend)
        self.assertFalse(reference.info().capabilities.compiled)
        if _compiled:
            compiled = _kernel_resources(Path("."), "q4_1-numba", .001)[0].resolve(
                dtype="q4_1", preferred="q4_1-numba", allow_fallback=False)
            self.assertIs(type(compiled), NativeQ4_1ExpertBackend)
            self.assertTrue(compiled.info().capabilities.compiled)


if __name__ == "__main__":
    unittest.main()
