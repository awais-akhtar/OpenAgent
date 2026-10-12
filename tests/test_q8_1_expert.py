"""Independent struct oracle and actual compiled Q8_1 execution checks."""

from contextlib import contextmanager
from pathlib import Path
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import q8_1_expert as scalar
from openagent.adaptive_engine import native_q8_1_expert as native
from openagent.adaptive_engine.expert_kernels import (
    ExpertKernelRegistry, MappingPayloadSource, NumPyRoutedSwiGLUBackend,
)
from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion


def oracle(payload, hidden, intermediate):
    matrices = []
    position = 0
    for rows, columns in ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate)):
        values = []
        for _ in range(rows * columns // 32):
            delta = struct.unpack_from("<e", payload, position)[0]
            codes = struct.unpack_from("<32b", payload, position + 4)
            values.extend(np.float32(delta * code) for code in codes)
            position += 36
        matrices.append(np.asarray(values, np.float32).reshape(rows, columns))
    return tuple(matrices)


def backends():
    result = [scalar.Q8_1ExpertBackend()]
    if native._compiled is not None:
        result.append(native.NativeQ8_1ExpertBackend())
    return result


class ObservedSource:
    def __init__(self, payload):
        self.payload = payload
        self.active = self.maximum = self.calls = 0

    @contextmanager
    def acquire(self, key):
        self.active += 1
        self.calls += 1
        self.maximum = max(self.maximum, self.active)
        try:
            yield self.payload
        finally:
            self.active -= 1


class Q8_1Tests(unittest.TestCase):
    def test_handbuilt_signed_codes_and_metadata_differential(self):
        rng = np.random.default_rng(81023)
        for hidden, intermediate in ((32, 32), (64, 96), (96, 64)):
            payload = b"".join(struct.pack("<ee", float(rng.choice([-.002, .003, 0., 2 ** -24])), .5)
                               + rng.integers(-128, 128, 32, dtype=np.int8).tobytes()
                               for _ in range(3 * hidden * intermediate // 32))
            matrices = oracle(payload, hidden, intermediate)
            for actual, expected in zip(scalar.decode_expert(payload, hidden, intermediate), matrices):
                np.testing.assert_array_equal(actual, expected)
            floating = b"".join(matrix.astype("<f4").tobytes() for matrix in matrices)
            vectors = rng.normal(0, .1, (3, hidden)).astype(np.float32)
            routes = [[("a", .3)], [("a", 0)], [("a", 1)]]
            arguments = dict(hidden_size=hidden, intermediate_size=intermediate)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(
                MappingPayloadSource({"a": floating}), vectors, routes, dtype="float32", **arguments)
            for backend in backends():
                source = ObservedSource(payload)
                actual = backend.execute_batch(source, vectors, routes, **arguments)
                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-8)
                self.assertEqual((source.active, source.maximum), (0, 1))

    def test_packing_sum_and_error(self):
        matrix = np.random.default_rng(4).normal(0, .1, (4, 64))
        payload = scalar.pack_matrix(matrix)
        for offset in range(0, len(payload), 36):
            delta, total, *codes = struct.unpack_from("<ee32b", payload, offset)
            self.assertEqual(total, float(np.float16(delta * sum(codes))))
        for value in (np.nan, np.inf, 1e10, 65504.):
            with self.assertRaises(ValueError):
                scalar.pack_matrix(np.full((1, 32), value))
        for shape in ((1, 31), (0, 32)):
            with self.assertRaises(ValueError):
                scalar.pack_matrix(np.zeros(shape))

    def test_malformed_metadata_lengths_and_overflow_release(self):
        payload = scalar.pack_expert(*[np.full((32, 32), .1)] * 3)
        args = dict(hidden_size=32, intermediate_size=32)
        for backend in backends():
            for offset in (0, 2, len(payload) - 36, len(payload) - 34):
                for bits in (0x7c00, 0xfc00, 0x7e00):
                    invalid = bytearray(payload)
                    struct.pack_into("<H", invalid, offset, bits)
                    source = ObservedSource(invalid)
                    with self.assertRaisesRegex(ValueError, "finite"):
                        backend.execute(source, np.ones(32), [("a", 0)], **args)
                    self.assertEqual(source.active, 0)
            for invalid in (payload[:-1], payload + b"a", memoryview(payload)[::2]):
                source = ObservedSource(invalid)
                with self.assertRaises(ValueError):
                    backend.execute(source, np.ones(32), [("a", 1)], **args)
                self.assertEqual(source.active, 0)
            source = ObservedSource(payload)
            with self.assertRaises(ArithmeticError):
                backend.execute(source, np.full(32, 1e30), [("a", 1)], **args)
            self.assertEqual(source.active, 0)

    def test_scratch_and_registry(self):
        for backend in backends():
            registry = ExpertKernelRegistry()
            registry.register(backend, replace=True)
            self.assertIs(registry.resolve(dtype="q8_1", preferred=backend.NAME, allow_fallback=False), backend)
            source = ObservedSource(bytes(scalar.payload_size(32, 32)))
            with self.assertRaises(MemoryError):
                type(backend)(max_scratch_bytes=127).execute(source, np.zeros(32), [("a", 1)],
                                                          hidden_size=32, intermediate_size=32)
            self.assertEqual(source.calls, 0)

    def test_packer_and_cli_manifest(self):
        from contextlib import redirect_stdout
        import io
        from openagent.adaptive_engine.olmoe_cli import main, _kernel_resources
        from openagent.adaptive_engine.expert_runtime import SparseExpertManifest
        from tests.test_q8_0_olmoe import checkpoint

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = checkpoint(root)
            protected = root / "openagent-experts.q8"
            protected.write_bytes(b"previous format")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["pack", str(root), "--dtype", "q8_1"]), 0)
            self.assertEqual(protected.read_bytes(), b"previous format")
            manifest = SparseExpertManifest.read(root / "openagent-experts.q81.manifest.json")
            for layer in manifest.layers:
                self.assertEqual(layer.dtype, "q8_1")
                layer.validate()
                for key, region in layer.regions.items():
                    with (root / region.path).open("rb") as stream:
                        stream.seek(region.offset)
                        payload = stream.read(region.size)
                    matrices = [tensors[f"{layer.id}.mlp.experts.{key}.{role}_proj.weight"]
                                for role in ("gate", "up", "down")]
                    self.assertEqual(payload, scalar.pack_expert(*matrices))
            registry, resource = _kernel_resources(root, "q8_1-reference", 1)
            self.assertIsNone(resource)
            self.assertTrue(registry.resolve(dtype="q8_1", preferred="q8_1-reference").info().available)

    def test_disabled_jit_fails_explicitly(self):
        command = """
from openagent.adaptive_engine.native_q8_1_expert import NativeQ8_1ExpertBackend
backend = NativeQ8_1ExpertBackend()
assert not backend.info().available
try:
    backend.execute_batch(None, [], [], hidden_size=32, intermediate_size=32)
except RuntimeError:
    pass
else:
    raise AssertionError('silent fallback')
"""
        result = subprocess.run([sys.executable, "-c", command],
                                env=dict(os.environ, NUMBA_DISABLE_JIT="1"),
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_mmap_and_retained_nested_failure(self):
        payload = scalar.pack_expert(*[np.full((32, 32), .1)] * 3)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "weights").write_bytes(payload)
            for backend in backends():
                with DiskExpertStore(directory, {"a": ExpertRegion("weights", 0, len(payload))},
                                     capacity_bytes=len(payload), io_mode="mmap") as store:
                    retained = []
                    def fail(view, *arguments):
                        borrowed = np.frombuffer(view, np.uint8)
                        assert borrowed.size
                        try:
                            raise ValueError("inner")
                        except ValueError as cause:
                            retained.append(cause)
                            raise RuntimeError("outer") from cause
                    module, attribute = (native, "_compiled") if isinstance(backend, native.NativeQ8_1ExpertBackend) else (scalar, "_kernel")
                    with patch.object(module, attribute, fail):
                        try:
                            backend.execute(store, np.ones(32), [("a", 1)], hidden_size=32, intermediate_size=32)
                        except RuntimeError as error:
                            retained.append(error)
                    self.assertEqual(len(retained), 2)
                    self.assertEqual(store.status()["active_leases"], 0)
                    store.close()

    @unittest.skipUnless(native._compiled is not None, "optional Numba required")
    def test_actual_compiled_execution(self):
        backend = native.NativeQ8_1ExpertBackend()
        payload = scalar.pack_expert(*[np.zeros((32, 32))] * 3)
        backend.execute(MappingPayloadSource({"a": payload}), np.ones(32), [("a", 1)],
                        hidden_size=32, intermediate_size=32)
        self.assertTrue(native._compiled.nopython_signatures)
        self.assertFalse(native._compiled.targetoptions["fastmath"])
        for signature in native._compiled.signatures:
            self.assertNotIn("NRT_MemInfo_alloc", native._compiled.inspect_llvm(signature))
        with patch.object(native, "_compiled", None):
            with self.assertRaises(RuntimeError):
                backend.execute(None, np.zeros(32), [("a", 1)], hidden_size=32, intermediate_size=32)
