"""Owned MXFP4 bytes: partial rows, compiled execution and mmap cleanup."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import mxfp4_expert as scalar
from openagent.adaptive_engine import native_mxfp4_expert as native
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion
from tests.test_extended_packed_olmoe import mxfp_oracle
from tests.test_q8_1_expert import ObservedSource


def backends():
    result = [scalar.MXFP4ExpertBackend()]
    if native._compiled_loop is not None:
        result.append(native.NativeMXFP4ExpertBackend())
    return result


class MXFP4IntegrationTests(unittest.TestCase):
    def test_disabled_jit_native_refuses_execution(self):
        code = """
from openagent.adaptive_engine.native_mxfp4_expert import NativeMXFP4ExpertBackend
backend = NativeMXFP4ExpertBackend()
assert not backend.info().available
try:
    backend.execute_batch(None, [], [], hidden_size=5, intermediate_size=5)
except RuntimeError:
    pass
else:
    raise AssertionError('disabled native backend silently executed')
"""
        process = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, NUMBA_DISABLE_JIT="1"),
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
    def test_handbuilt_all_nibbles_partial_rows_and_signed_zero(self):
        for hidden, intermediate in ((7, 5), (67, 71)):
            payload = bytearray()
            for rows, columns in ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate)):
                for row in range(rows):
                    for start in range(0, columns, 32):
                        payload.append(119 + row % 3)
                        codes = [((start + lane + row) % 16) if start + lane < columns else 0 for lane in range(32)]
                        payload.extend(codes[lane] | codes[lane + 1] << 4 for lane in range(0, 32, 2))
            matrices = mxfp_oracle(payload, hidden, intermediate)
            floating = b"".join(matrix.astype("<f4").tobytes() for matrix in matrices)
            vectors = np.random.default_rng(4).normal(0, .1, (3, hidden)).astype(np.float32)
            routes = [[("a", .3)], [("a", 0)], [("a", 1)]]
            args = dict(hidden_size=hidden, intermediate_size=intermediate)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource({"a": floating}),
                                                               vectors, routes, dtype="float32", **args)
            for backend in backends():
                source = ObservedSource(payload)
                actual = backend.execute_batch(source, vectors, routes, **args)
                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-8)
                self.assertEqual((source.active, source.maximum), (0, 1))

    def test_malformed_scale_padding_overflow_scratch_and_lengths(self):
        payload = scalar.pack_expert(np.full((5, 7), .1), np.full((5, 7), .1), np.full((7, 5), .1))
        malformed = []
        for offset in (0, len(payload) - 17):
            invalid = bytearray(payload)
            invalid[offset] = 255
            malformed.append(invalid)
        # Column 7 is outside the seven-column gate row.
        invalid = bytearray(payload)
        invalid[4] |= 16
        malformed.append(invalid)
        malformed.extend((payload[:-1], payload + b"a", memoryview(payload)[::2]))
        overflow = bytearray(payload)
        overflow[0] = 254
        overflow[1] = 7  # 6 * 2^127 is not a finite FP32 weight.
        args = dict(hidden_size=7, intermediate_size=5)
        for backend in backends():
            for invalid in malformed:
                source = ObservedSource(invalid)
                with self.assertRaises(ValueError):
                    backend.execute(source, np.ones(7), [("a", 0)], **args)
                self.assertEqual(source.active, 0)
            source = ObservedSource(overflow)
            with self.assertRaises(ArithmeticError):
                backend.execute(source, np.zeros(7), [("a", 0)], **args)
            self.assertEqual(source.active, 0)
            source = ObservedSource(payload)
            with self.assertRaises(MemoryError):
                type(backend)(max_scratch_bytes=19).execute(source, np.zeros(7), [("a", 1)], **args)
            self.assertEqual(source.calls, 0)

    def test_mmap_chained_compiler_exception_releases_all_borrowers(self):
        payload = scalar.pack_expert(*[np.full((5, 5), .1)] * 3)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "weights").write_bytes(payload)
            for backend in backends():
                with DiskExpertStore(directory, {"a": ExpertRegion("weights", 0, len(payload))},
                                     capacity_bytes=len(payload), io_mode="mmap") as store:
                    retained = []
                    def fail(raw, *args):
                        borrowed = raw[::2]
                        assert borrowed.size
                        try:
                            raise ValueError("inner")
                        except ValueError as cause:
                            retained.append(cause)
                            raise RuntimeError("outer") from cause
                    module, attribute = (native, "_compiled_loop") if isinstance(backend, native.NativeMXFP4ExpertBackend) else (scalar, "_loop")
                    with patch.object(module, attribute, fail):
                        try:
                            backend.execute(store, np.ones(5), [("a", 1)], hidden_size=5, intermediate_size=5)
                        except RuntimeError as error:
                            retained.append(error)
                    self.assertEqual(len(retained), 2)
                    self.assertEqual(store.status()["active_leases"], 0)
                    store.close()

    @unittest.skipUnless(native._compiled_loop is not None, "optional Numba required")
    def test_actual_compiled_no_numeric_allocations(self):
        payload = scalar.pack_expert(*[np.zeros((5, 5))] * 3)
        native.NativeMXFP4ExpertBackend().execute(MappingPayloadSource({"a": payload}), np.ones(5),
                                                [("a", 1)], hidden_size=5, intermediate_size=5)
        self.assertTrue(native._compiled_loop.nopython_signatures)
        for signature in native._compiled_loop.signatures:
            self.assertNotIn("NRT_MemInfo_alloc", native._compiled_loop.inspect_llvm(signature))
