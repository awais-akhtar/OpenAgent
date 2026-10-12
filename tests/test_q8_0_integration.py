"""Unmocked matrix conversion, registry, manifest and disk-store execution."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.expert_kernels import (
    ExpertKernelRegistry, KernelBackendInfo, MappingPayloadSource,
    NumPyRoutedSwiGLUBackend, RoutedSwiGLUKernelBackend,
)
from openagent.adaptive_engine.expert_runtime import ExpertLayer, SparseExpertManifest, SparseExpertRuntime
from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion, routed_swiglu_batch
from openagent.adaptive_engine.q8_0_conversion import convert_experts
from openagent.adaptive_engine.q8_0_expert import Q8_0ExpertBackend, payload_size
from tests.test_q8_0_expert import scalar_decode


class Q8_0IntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.h, self.i = 64, 32
        rng = np.random.default_rng(202)
        self.experts = {key: tuple(rng.normal(0, .1, shape).astype(np.float32)
                                  for shape in ((self.i, self.h), (self.i, self.h), (self.h, self.i)))
                        for key in ("a", "b", "c")}
        self.options = dict(hidden_size=self.h, intermediate_size=self.i, model_id="matrix-fixture",
                            architecture="swiglu-fixture", fingerprint="test-input-v1", experts_used=2)

    def convert(self):
        return convert_experts(self.experts, self.root / "experts.q8", block_count=1, **self.options)

    def oracle(self, manifest, inputs, routes):
        payloads = {}
        for key, region in manifest.layers[0].regions.items():
            with (self.root / region.path).open("rb") as stream:
                stream.seek(region.offset)
                raw = stream.read(region.size)
            payloads[key] = b"".join(a.astype("<f4").tobytes() for a in scalar_decode(raw, self.h, self.i))
        return NumPyRoutedSwiGLUBackend().execute_batch(
            MappingPayloadSource(payloads), inputs, routes, hidden_size=self.h,
            intermediate_size=self.i, dtype="float32")

    def test_registry_info_selection_and_original_defaults(self):
        registry = ExpertKernelRegistry()
        backend = registry.resolve(dtype="q8_0", batch=True)
        self.assertIsInstance(backend, Q8_0ExpertBackend)
        self.assertIsInstance(backend, RoutedSwiGLUKernelBackend)
        info = backend.info()
        self.assertIsInstance(info, KernelBackendInfo)
        self.assertEqual(info.capabilities.payload_dtypes, frozenset({"q8_0"}))
        self.assertEqual(info.priority, -11000)
        self.assertFalse(info.capabilities.compiled)
        self.assertFalse(info.capabilities.native_simd)
        self.assertIs(registry.resolve(dtype="q8_0", preferred=info.name, allow_fallback=False), backend)
        self.assertIs(registry.resolve(dtype="q8_0", preferred="numpy-reference"), backend)
        with self.assertRaises(RuntimeError):
            registry.resolve(dtype="q8_0", preferred="numpy-reference", allow_fallback=False)
        for dtype in ("float32", "float16", "int8-row"):
            self.assertEqual(registry.resolve(dtype=dtype).info().name, "numpy-reference")
        self.assertEqual(registry.resolve(dtype="int4-row").info().name, "packed-int4-reference")
        self.assertNotIn("q8_0", NumPyRoutedSwiGLUBackend().info().capabilities.payload_dtypes)

    def test_converter_hashes_exact_offsets_and_manifest_roundtrip(self):
        manifest = self.convert()
        parsed = SparseExpertManifest.read(manifest.write(self.root / "experts.json"))
        self.assertEqual(parsed, manifest)
        layer = parsed.layers[0]
        self.assertEqual(layer.dtype, "q8_0")
        size = 3 * self.h * self.i // 32 * 34
        self.assertEqual(payload_size(self.h, self.i), size)
        raw = (self.root / "experts.q8").read_bytes()
        self.assertEqual(len(raw), 3 * size)
        for index, region in enumerate(layer.regions.values()):
            self.assertEqual((region.offset, region.size), (index * size, size))
            self.assertEqual(region.sha256, hashlib.sha256(raw[region.offset:region.offset + size]).hexdigest())

    def test_manifest_rejects_wrong_lengths_dimensions_and_unknown_dtype(self):
        manifest = self.convert()
        layer = manifest.layers[0]
        region = layer.regions["a"]
        for size in (region.size - 1, region.size + 1, 3 * self.h * self.i):
            with self.assertRaisesRegex(ValueError, "expected"):
                replace(layer, experts_used=1, regions={"a": replace(region, size=size)}).validate()
        for h, i in ((33, 32), (64, 31), (32, 32), (True, 32), (64.0, 32)):
            with self.assertRaises(ValueError):
                replace(layer, hidden_size=h, intermediate_size=i).validate()
        path = manifest.write(self.root / "bad.json")
        data = json.loads(path.read_text())
        data["layers"][0]["experts"]["a"]["size"] += 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "expected"):
            SparseExpertManifest.read(path)
        with self.assertRaisesRegex(ValueError, "dtype"):
            replace(layer, dtype="unknown").validate()

    def test_real_disk_store_registry_batch_with_one_expert_capacity(self):
        manifest = self.convert()
        rng = np.random.default_rng(19)
        inputs = rng.normal(0, .2, (4, self.h)).astype(np.float32)
        routes = [[("a", .3), ("b", .7)], [("c", 1)], [("b", 0)], [("c", .4), ("a", .6)]]
        expected = self.oracle(manifest, inputs, routes)
        size = payload_size(self.h, self.i)
        for mode in ("buffered", "mmap"):
            with DiskExpertStore(self.root, dict(manifest.layers[0].regions), capacity_bytes=size, io_mode=mode) as store:
                actual = routed_swiglu_batch(store, inputs, routes, hidden_size=self.h,
                                             intermediate_size=self.i, dtype="q8_0",
                                             preferred_backend="q8_0-numpy")
                status = store.status()
                self.assertEqual(status["active_leases"], 0)
                self.assertLessEqual(status["resident_bytes"], size)
                self.assertEqual(status["checksums_verified"], 3)
                self.assertGreater(status["evictions"], 0)
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)
            self.assertTrue(store.status()["closed"])

    def test_sparse_runtime_single_batch_default_and_explicit_registry(self):
        manifest = self.convert()
        inputs = np.full((2, self.h), .1, np.float32)
        routes = [[("a", .2), ("b", .8)], [("c", 1)]]
        expected = self.oracle(manifest, inputs, routes)
        for mode, registry in (("buffered", None), ("mmap", ExpertKernelRegistry())):
            with SparseExpertRuntime(self.root, manifest, capacity_bytes=payload_size(self.h, self.i),
                                     io_mode=mode, kernels=registry, preferred_kernel="q8_0-numpy") as runtime:
                actual = runtime.execute_batch("block.0", inputs, routes)
                single = runtime.execute("block.0", inputs[0], routes[0])
                self.assertEqual(runtime.store.status()["active_leases"], 0)
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)
            np.testing.assert_array_equal(single, actual[0])

    def test_store_corruption_releases_leases_and_closes(self):
        for mode in ("buffered", "mmap"):
            # Invalid scale with a correct content hash reaches backend validation.
            raw = bytes(payload_size(32, 32) - 34) + struct.pack("<e", float("nan")) + bytes(32)
            path = self.root / (mode + ".q8")
            path.write_bytes(raw)
            region = ExpertRegion(path.name, 0, len(raw), hashlib.sha256(raw).hexdigest())
            layer = ExpertLayer("block.0", 32, 32, 1, "q8_0", "swiglu", {"a": region})
            manifest = SparseExpertManifest("invalid", "fixture", "invalid-scale", (layer,))
            with SparseExpertRuntime(self.root, manifest, capacity_bytes=len(raw), io_mode=mode) as runtime:
                with self.assertRaisesRegex(ValueError, "scales"):
                    runtime.execute("block.0", np.ones(32), [("a", 1)])
                self.assertEqual(runtime.store.status()["active_leases"], 0)
            self.assertTrue(runtime.store.status()["closed"])

    def test_buffered_disk_checksum_rejected_before_execution(self):
        manifest = self.convert()
        region = manifest.layers[0].regions["a"]
        region = replace(region, sha256="0" * 64)
        with DiskExpertStore(self.root, {"a": region}, capacity_bytes=region.size) as store:
            with self.assertRaisesRegex(ValueError, "checksum"):
                ExpertKernelRegistry().resolve(dtype="q8_0").execute(
                    store, np.ones(self.h), [("a", 1)], hidden_size=self.h, intermediate_size=self.i)
            self.assertEqual(store.status()["active_leases"], 0)

    def test_converter_failure_does_not_publish_or_overwrite(self):
        output = self.root / "bad.q8"
        cases = ({}, {"a": self.experts["a"][:2]}, {"bad/id": self.experts["a"]},
                 {"a": self.experts["a"], "b": (np.ones((1, 1)),) * 3},
                 {"a": (np.full((self.i, self.h), np.nan), *self.experts["a"][1:])})
        for experts in cases:
            with self.assertRaises(ValueError):
                convert_experts(experts, output, **self.options)
            self.assertFalse(output.exists())
            self.assertEqual(list(self.root.glob(".q8_0-*")), [])
        with self.assertRaises(ValueError):
            convert_experts(self.experts, output, **dict(self.options, model_id=""))
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".q8_0-*")), [])
        output.write_bytes(b"keep existing")
        with self.assertRaises(FileExistsError):
            convert_experts(self.experts, output, **self.options)
        self.assertEqual(output.read_bytes(), b"keep existing")


if __name__ == "__main__":
    unittest.main()
