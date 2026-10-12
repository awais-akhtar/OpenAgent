"""Caller E8 equations, orthogonal transforms and actual native byte execution."""

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import e8_lattice_expert as scalar
from openagent.adaptive_engine import native_e8_lattice_expert as native
from openagent.adaptive_engine.expert_kernels import ExpertKernelRegistry, MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion
from tests.test_q8_1_expert import ObservedSource


def decoded_oracle(payload, hidden, intermediate, contract):
    position, matrices = 0, []
    for rows, columns in ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate)):
        matrix = np.empty((rows, columns), np.float32)
        for row in range(rows):
            for start in range(0, columns, 8):
                index, scale = struct.unpack_from("<Hf", payload, position)
                rotated = contract.codebook[index].astype(np.float64) * scale
                matrix[row, start:start + 8] = rotated @ contract.transform.astype(np.float64)
                position += 6
        matrices.append(matrix)
    return matrices


def contract():
    return scalar.LatticeContract(scalar.root_codebook(), scalar.hadamard_transform())


def backends(configuration):
    result = [scalar.E8LatticeExpertBackend(configuration)]
    if native._compiled is not None:
        result.append(native.NativeE8LatticeExpertBackend(configuration))
    return result


class E8LatticeTests(unittest.TestCase):
    def test_root_geometry_and_immutable_content_addressed_contract(self):
        book = scalar.root_codebook()
        self.assertEqual(book.shape, (241, 8))
        np.testing.assert_array_equal(np.sum(book[1:] ** 2, axis=1), 2)
        doubled = book * 2
        self.assertTrue(np.all((doubled.sum(axis=1) % 4) == 0))
        first = contract()
        second = scalar.LatticeContract(book[::-1], first.transform)
        third = scalar.LatticeContract(book, np.eye(8))
        self.assertNotEqual(first.identity, second.identity)
        self.assertNotEqual(first.identity, third.identity)
        self.assertEqual(first.identity, contract().identity)
        with self.assertRaises(ValueError):
            first.codebook.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            first.transform = np.eye(8)
        self.assertEqual(first.metadata()["contract_sha256"], first.identity)
        for invalid in (np.ones((2, 7)), np.full((1, 8), .25), np.array([[1., 0, 0, 0, 0, 0, 0, 0]]),
                        np.full((1, 8), np.nan), np.zeros((2, 8))):
            with self.assertRaises(ValueError):
                scalar.LatticeContract(invalid, first.transform)
        with self.assertRaises(ValueError):
            scalar.LatticeContract(book, np.ones((8, 8)))

    def test_handbuilt_indices_signed_scales_rotation_and_compiled_differential(self):
        configuration = contract()
        rng = np.random.default_rng(841)
        for hidden, intermediate in ((8, 8), (16, 24), (24, 16)):
            payload = b"".join(struct.pack("<Hf", int(rng.integers(0, 241)),
                                           float(rng.choice([-.025, .03, 0., 2 ** -149])))
                               for _ in range(3 * hidden * intermediate // 8))
            floating = b"".join(matrix.astype("<f4").tobytes() for matrix in
                                decoded_oracle(payload, hidden, intermediate, configuration))
            vectors = rng.normal(0, .1, (3, hidden)).astype(np.float32)
            routes = [[("a", .3)], [("a", 0)], [("a", 1)]]
            args = dict(hidden_size=hidden, intermediate_size=intermediate)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource({"a": floating}),
                                                               vectors, routes, dtype="float32", **args)
            for backend in backends(configuration):
                source = ObservedSource(payload)
                actual = backend.execute_batch(source, vectors, routes, **args)
                np.testing.assert_allclose(actual, expected, rtol=6e-5, atol=2e-10)
                self.assertEqual((source.active, source.maximum), (0, 1))

    def test_packing_exact_lattice_direction_with_identity_transform(self):
        configuration = scalar.LatticeContract(scalar.root_codebook(), np.eye(8))
        matrix = np.tile(configuration.codebook[1] * .25, (8, 1))
        payload = scalar.pack_expert(matrix, matrix, matrix, configuration)
        self.assertEqual(len(payload), scalar.payload_size(8, 8))
        for actual in decoded_oracle(payload, 8, 8, configuration):
            np.testing.assert_array_equal(actual, matrix)
        with self.assertRaises(ValueError):
            scalar.pack_matrix(np.full((8, 8), np.nan), configuration)
        with self.assertRaises(ValueError):
            scalar.payload_size(8, 9)

    def test_registry_names_bind_caller_tables(self):
        first = scalar.E8LatticeExpertBackend(contract())
        second = scalar.E8LatticeExpertBackend(scalar.LatticeContract(scalar.root_codebook(), np.eye(8)))
        self.assertNotEqual(first.NAME, second.NAME)
        registry = ExpertKernelRegistry([first, second])
        self.assertIs(registry.resolve(dtype=scalar.DTYPE, preferred=first.NAME, allow_fallback=False), first)
        with self.assertRaises(ValueError):
            registry.resolve(dtype=scalar.DTYPE)
        with self.assertRaises(KeyError):
            registry.resolve(dtype=scalar.DTYPE, preferred="unknown-contract", allow_fallback=True)
        with self.assertRaises(AttributeError):
            first.contract = second.contract
        from openagent.adaptive_engine.expert_runtime import ExpertLayer
        with self.assertRaises(ValueError):
            ExpertLayer("layer", 8, 8, 1, scalar.DTYPE, "swiglu", {"a": ExpertRegion("weights", 0, 144)}).validate()

    def test_malformed_index_scale_length_overflow_and_scratch_release(self):
        configuration = contract()
        payload = scalar.pack_expert(*[np.ones((8, 8))] * 3, configuration)
        malformed = []
        for position in (0, len(payload) - 6):
            bad_index = bytearray(payload)
            struct.pack_into("<H", bad_index, position, 241)
            malformed.append(bad_index)
            bad_scale = bytearray(payload)
            struct.pack_into("<f", bad_scale, position + 2, np.nan)
            malformed.append(bad_scale)
        malformed.extend((payload[:-1], payload + b"a", memoryview(payload)[::2]))
        for backend in backends(configuration):
            for invalid in malformed:
                source = ObservedSource(invalid)
                with self.assertRaises(ValueError):
                    backend.execute(source, np.zeros(8), [("a", 0)], hidden_size=8, intermediate_size=8)
                self.assertEqual(source.active, 0)
            source = ObservedSource(payload)
            with self.assertRaises(ArithmeticError):
                backend.execute(source, np.full(8, 1e30), [("a", 1)], hidden_size=8, intermediate_size=8)
            with self.assertRaises(MemoryError):
                type(backend)(configuration, max_scratch_bytes=31).execute(source, np.ones(8), [("a", 1)],
                                                                          hidden_size=8, intermediate_size=8)
            self.assertEqual(source.active, 0)

    def test_retained_chained_mmap_exception(self):
        configuration = contract()
        payload = scalar.pack_expert(*[np.zeros((8, 8))] * 3, configuration)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "weights").write_bytes(payload)
            for backend in backends(configuration):
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
                    module, attribute = (native, "_compiled") if isinstance(backend, native.NativeE8LatticeExpertBackend) else (scalar, "_kernel")
                    with patch.object(module, attribute, fail):
                        try:
                            backend.execute(store, np.ones(8), [("a", 1)], hidden_size=8, intermediate_size=8)
                        except RuntimeError as error:
                            retained.append(error)
                    self.assertEqual(len(retained), 2)
                    self.assertEqual(store.status()["active_leases"], 0)
                    store.close()

    @unittest.skipUnless(native._compiled is not None, "optional Numba required")
    def test_actual_nopython_and_disabled_jit(self):
        configuration = contract()
        payload = scalar.pack_expert(*[np.zeros((8, 8))] * 3, configuration)
        native.NativeE8LatticeExpertBackend(configuration).execute(MappingPayloadSource({"a": payload}),
            np.ones(8), [("a", 1)], hidden_size=8, intermediate_size=8)
        self.assertTrue(native._compiled.nopython_signatures)
        for signature in native._compiled.signatures:
            self.assertNotIn("NRT_MemInfo_alloc", native._compiled.inspect_llvm(signature))
        code = """
from openagent.adaptive_engine.e8_lattice_expert import LatticeContract, root_codebook, hadamard_transform
from openagent.adaptive_engine.native_e8_lattice_expert import NativeE8LatticeExpertBackend
b = NativeE8LatticeExpertBackend(LatticeContract(root_codebook(), hadamard_transform()))
assert not b.info().available
try:
    b.execute_batch(None, [], [], hidden_size=8, intermediate_size=8)
except RuntimeError:
    pass
else:
    raise AssertionError('native silently executed')
"""
        process = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, NUMBA_DISABLE_JIT="1"),
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
