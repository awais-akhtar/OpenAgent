"""Handbuilt bit streams, compiled differentials and lease failure paths."""

from pathlib import Path
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import lowbit_expert as scalar
from openagent.adaptive_engine import native_lowbit_expert as native
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion
from tests.test_q8_1_expert import ObservedSource


def oracle(payload, hidden, intermediate, dtype):
    bits = 2 if dtype == "int2-row" else 3
    shapes = ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))
    scales_offset = sum(rows * ((columns * bits + 7) // 8) for rows, columns in shapes)
    codes_offset = 0
    matrices = []
    for rows, columns in shapes:
        stride = (columns * bits + 7) // 8
        group = columns if bits == 2 else 64
        groups = (columns + group - 1) // group
        matrix = np.empty((rows, columns), np.float32)
        for row in range(rows):
            row_bits = int.from_bytes(payload[codes_offset + row * stride:codes_offset + (row + 1) * stride], "little")
            for column in range(columns):
                code = (row_bits >> (column * bits)) & ((1 << bits) - 1)
                if code >= 1 << (bits - 1):
                    code -= 1 << bits
                scale = struct.unpack_from("<f", payload, scales_offset + (row * groups + column // group) * 4)[0]
                matrix[row, column] = np.float32(code * scale)
        matrices.append(matrix)
        codes_offset += rows * stride
        scales_offset += rows * groups * 4
    return matrices


def backends(dtype):
    result = [scalar.LowBitExpertBackend(dtype)]
    if native._compiled is not None:
        result.append(native.NativeLowBitExpertBackend(dtype))
    return result


class LowBitTests(unittest.TestCase):
    def test_scalar_import_without_numba_and_disabled_native(self):
        code = """
import importlib.abc
import sys
class NoNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numba' or fullname.startswith('numba.'):
            raise ModuleNotFoundError('optional library intentionally unavailable')
sys.meta_path.insert(0, NoNumba())
import numpy as np
from openagent.adaptive_engine.lowbit_expert import LowBitExpertBackend, pack_expert
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource
for dtype in ('int2-row', 'int3-grouped-g64'):
    payload = pack_expert(*[np.zeros((5,5))] * 3, dtype=dtype)
    result = LowBitExpertBackend(dtype).execute(MappingPayloadSource({'a':payload}), np.ones(5),
        [('a',1)], hidden_size=5, intermediate_size=5)
    assert np.all(result == 0)
from openagent.adaptive_engine.native_lowbit_expert import NativeLowBitExpertBackend
assert not NativeLowBitExpertBackend().info().available
"""
        process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        disabled = """
from openagent.adaptive_engine.native_lowbit_expert import NativeLowBitExpertBackend
assert not NativeLowBitExpertBackend().info().available
try:
    NativeLowBitExpertBackend().execute_batch(None, [], [], hidden_size=5, intermediate_size=5)
except RuntimeError:
    pass
else:
    raise AssertionError('disabled native backend silently executed')
"""
        process = subprocess.run([sys.executable, "-c", disabled], env=dict(os.environ, NUMBA_DISABLE_JIT="1"),
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
    def test_handbuilt_all_codes_and_cross_byte_odd_rows(self):
        rng = np.random.default_rng(323)
        for dtype in (scalar.INT2, scalar.INT3):
            bits = 2 if dtype == scalar.INT2 else 3
            for hidden, intermediate in ((5, 9), (67, 71)):
                codes, scales = bytearray(), bytearray()
                for rows, columns in ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate)):
                    group = columns if bits == 2 else 64
                    for _ in range(rows):
                        word = sum(int(code) << (bits * column) for column, code in
                                   enumerate(rng.integers(0, 1 << bits, columns)))
                        codes.extend(word.to_bytes((columns * bits + 7) // 8, "little"))
                    for _ in range(rows * ((columns + group - 1) // group)):
                        scales.extend(struct.pack("<f", float(rng.choice([-.002, .003, 0., 2 ** -149]))))
                payload = bytes(codes + scales)
                matrices = oracle(payload, hidden, intermediate, dtype)
                floating = b"".join(matrix.astype("<f4").tobytes() for matrix in matrices)
                x = rng.normal(0, .1, (3, hidden)).astype(np.float32)
                routes = [[("a", .2)], [("a", 0)], [("a", 1)]]
                arguments = dict(hidden_size=hidden, intermediate_size=intermediate)
                expected = NumPyRoutedSwiGLUBackend().execute_batch(
                    MappingPayloadSource({"a": floating}), x, routes, dtype="float32", **arguments)
                for backend in backends(dtype):
                    source = ObservedSource(payload)
                    actual = backend.execute_batch(source, x, routes, **arguments)
                    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=1e-10)
                    self.assertEqual((source.active, source.maximum), (0, 1))

    def test_packing_bound_and_partial_groups(self):
        rng = np.random.default_rng(141)
        for dtype in (scalar.INT2, scalar.INT3):
            matrices = [rng.normal(0, .1, shape) for shape in ((71, 67), (71, 67), (67, 71))]
            payload = scalar.pack_expert(*matrices, dtype=dtype)
            self.assertEqual(len(payload), scalar.payload_size(67, 71, dtype=dtype))
            decoded = oracle(payload, 67, 71, dtype)
            for original, actual in zip(matrices, decoded):
                _, scales = scalar.pack_matrix(original, dtype=dtype)
                expanded = np.repeat(scales, original.shape[1] if dtype == scalar.INT2 else 64, axis=1)[:, :original.shape[1]]
                self.assertTrue(np.all(np.abs(original - actual) <= expanded / 2 + 1e-7))
            for value in (np.nan, np.inf):
                with self.assertRaises(ValueError):
                    scalar.pack_matrix(np.full((2, 5), value), dtype=dtype)

    def test_metadata_padding_lengths_overflow_and_scratch(self):
        for dtype in (scalar.INT2, scalar.INT3):
            payload = scalar.pack_expert(*[np.full((5, 5), .1)] * 3, dtype=dtype)
            bits = 2 if dtype == scalar.INT2 else 3
            stride = (5 * bits + 7) // 8
            codes_end = 15 * stride
            invalid = bytearray(payload)
            invalid[stride - 1] |= 128
            bad_scale = bytearray(payload)
            struct.pack_into("<f", bad_scale, codes_end, np.nan)
            for backend in backends(dtype):
                for malformed in (invalid, bad_scale, payload[:-1], payload + b"a"):
                    source = ObservedSource(malformed)
                    with self.assertRaises(ValueError):
                        backend.execute(source, np.ones(5), [("a", 0)], hidden_size=5, intermediate_size=5)
                    self.assertEqual(source.active, 0)
                source = ObservedSource(payload)
                with self.assertRaises(ArithmeticError):
                    backend.execute(source, np.full(5, 1e30), [("a", 1)], hidden_size=5, intermediate_size=5)
                with self.assertRaises(MemoryError):
                    type(backend)(dtype, max_scratch_bytes=19).execute(source, np.zeros(5), [("a", 1)], hidden_size=5, intermediate_size=5)
                self.assertEqual(source.active, 0)

    def test_retained_mmap_compiler_exception(self):
        for dtype in (scalar.INT2, scalar.INT3):
            payload = scalar.pack_expert(*[np.ones((5, 5))] * 3, dtype=dtype)
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "weights").write_bytes(payload)
                for backend in backends(dtype):
                    with DiskExpertStore(directory, {"a": ExpertRegion("weights", 0, len(payload))},
                                         capacity_bytes=len(payload), io_mode="mmap") as store:
                        retained = []
                        def fail(raw, *arguments):
                            view = raw[::2]
                            assert view.size
                            try:
                                raise ValueError("inner")
                            except ValueError as cause:
                                retained.append(cause)
                                raise RuntimeError("outer") from cause
                        module, attribute = (native, "_compiled") if isinstance(backend, native.NativeLowBitExpertBackend) else (scalar, "_kernel")
                        with patch.object(module, attribute, fail):
                            try:
                                backend.execute(store, np.ones(5), [("a", 1)], hidden_size=5, intermediate_size=5)
                            except RuntimeError as error:
                                retained.append(error)
                        self.assertEqual(len(retained), 2)
                        self.assertEqual(store.status()["active_leases"], 0)
                        store.close()

    @unittest.skipUnless(native._compiled is not None, "optional Numba required")
    def test_actual_nopython_no_allocation(self):
        for dtype in (scalar.INT2, scalar.INT3):
            payload = scalar.pack_expert(*[np.ones((5, 5))] * 3, dtype=dtype)
            native.NativeLowBitExpertBackend(dtype).execute(MappingPayloadSource({"a": payload}), np.ones(5),
                [("a", 1)], hidden_size=5, intermediate_size=5)
        self.assertTrue(native._compiled.nopython_signatures)
        for signature in native._compiled.signatures:
            self.assertNotIn("NRT_MemInfo_alloc", native._compiled.inspect_llvm(signature))
