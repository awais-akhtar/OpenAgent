import math
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from openagent.adaptive_engine import native_fp8_block_expert as native
from openagent.adaptive_engine.fp8_block_expert import pack_expert, matrix_payload_size, FP8BlockExpertBackend
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion
from tests.test_fp8_block_expert import oracle, decode_matrix


@unittest.skipUnless(native.NativeFP8BlockExpertBackend().info().available, "Numba unavailable")
class NativeTests(unittest.TestCase):
    def test_codes_scales_compilation(self):
        for code in range(256):
            actual, expected = native._code(code), oracle(code)
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertEqual(actual, expected)
                self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
        for bits in (0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0x7fc00000):
            actual = native._scale(np.frombuffer(struct.pack("<I", bits), np.uint8), 0)
            expected = struct.unpack("<f", struct.pack("<I", bits))[0]
            if math.isfinite(expected):
                self.assertEqual(actual, expected)
                self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))
            else:
                self.assertTrue(math.isnan(actual))
        self.assertTrue(native._code.nopython_signatures)

    def test_differential_edge_blocks(self):
        rng = np.random.default_rng(991)
        for h, i in ((3, 5), (129, 131), (131, 129)):
            matrices = [rng.normal(0, .02, shape).astype(np.float32) for shape in ((i,h), (i,h), (h,i))]
            raw = pack_expert(*matrices)
            offset, decoded = 0, []
            for rows, cols in ((i,h), (i,h), (h,i)):
                size = matrix_payload_size(rows, cols)
                decoded.append(decode_matrix(raw[offset:offset+size], rows, cols))
                offset += size
            x = rng.normal(size=(3,h)).astype(np.float32)
            routes = [[("a", .3)], [("a", 0)], [("a", 2)]]
            args = dict(hidden_size=h, intermediate_size=i)
            actual = native.NativeFP8BlockExpertBackend().execute_batch(MappingPayloadSource({"a":raw}), x, routes, **args)
            expected = NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource({"a":b"".join(a.astype("<f4").tobytes() for a in decoded)}), x, routes, dtype="float32", **args)
            np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-8)
            reference = FP8BlockExpertBackend().execute_batch(MappingPayloadSource({"a":raw}), x, routes, **args)
            np.testing.assert_allclose(reference, actual, rtol=5e-5, atol=1e-8)
        self.assertTrue(native._compiled_loop.nopython_signatures)
        for signature in native._compiled_loop.signatures:
            self.assertNotIn("NRT_MemInfo_alloc", native._compiled_loop.inspect_llvm(signature))

    def test_mmap_errors_retained_cleanup(self):
        raw = pack_expert(*[np.ones((3,3), np.float32)]*3)
        backend = native.NativeFP8BlockExpertBackend()
        args = dict(hidden_size=3, intermediate_size=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for damaged in (raw[:-1], raw+b"\0", bytes([127])+raw[1:], raw[:9]+struct.pack("<f", math.inf)+raw[13:], raw):
                (root/"weights").write_bytes(damaged)
                store = DiskExpertStore(root, {"a":ExpertRegion("weights",0,len(damaged))}, capacity_bytes=len(damaged), io_mode="mmap")
                retained = None
                try:
                    backend.execute(store, np.full(3, np.finfo(np.float32).max) if damaged == raw else np.ones(3), [("a",0 if damaged != raw else 1)], **args)
                except (ValueError, ArithmeticError) as error:
                    retained = error
                self.assertIsNotNone(retained)
                self.assertEqual(store.status()["active_leases"], 0)
                store.close()
            (root/"weights").write_bytes(raw)
            store = DiskExpertStore(root, {"a":ExpertRegion("weights",0,len(raw))}, capacity_bytes=len(raw), io_mode="mmap")
            def fail(raw, *args):
                borrowed = raw[::2]
                try:
                    raise ValueError("inner")
                except ValueError as cause:
                    raise RuntimeError("compiler") from cause
            with patch.object(native, "_compiled_loop", fail):
                try:
                    backend.execute(store, np.ones(3), [("a",1)], **args)
                except RuntimeError as error:
                    retained = error
            self.assertIsNone(retained.__context__)
            store.close()

    def test_limits(self):
        source = MappingPayloadSource({"a":pack_expert(*[np.ones((3,3))]*3)})
        with self.assertRaises(MemoryError):
            native.NativeFP8BlockExpertBackend(max_scratch_bytes=11).execute(source,np.ones(3),[("a",1)],hidden_size=3,intermediate_size=3)
        with self.assertRaises(ValueError):
            native.NativeFP8BlockExpertBackend(scale_encoding="ue8m0")

    def test_handbuilt_scales_and_checksums(self):
        backend = native.NativeFP8BlockExpertBackend()
        args = dict(hidden_size=2, intermediate_size=2)
        for scale in (-.01, 0., -0., 2**-149):
            raw = (bytes([126, 254, 1, 129]) + struct.pack("<f", scale))*3
            matrices = [decode_matrix(raw[j*8:j*8+8], 2, 2) for j in range(3)]
            x = np.array([.01, -.02], np.float32)
            expected = NumPyRoutedSwiGLUBackend().execute(MappingPayloadSource({"a":b"".join(a.astype("<f4").tobytes() for a in matrices)}),x,[("a",1)],dtype="float32",**args)
            actual = backend.execute(MappingPayloadSource({"a":raw}),x,[("a",1)],**args)
            np.testing.assert_allclose(actual,expected,rtol=5e-5,atol=1e-15)
        raw = pack_expert(*[np.ones((2,2))]*3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"weights").write_bytes(raw)
            for mode in ("mmap", "buffered"):
                for checksum in (hashlib.sha256(raw).hexdigest(), "0"*64):
                    store = DiskExpertStore(root,{"a":ExpertRegion("weights",0,len(raw),sha256=checksum)},capacity_bytes=len(raw),io_mode=mode)
                    if checksum == "0"*64:
                        with self.assertRaisesRegex(ValueError,"checksum"):
                            backend.execute(store,np.ones(2),[("a",1)],**args)
                    else:
                        backend.execute(store,np.ones(2),[("a",1)],**args)
                    self.assertEqual(store.status()["active_leases"],0)
                    store.close()


class UnavailableTests(unittest.TestCase):
    def test_no_fallback(self):
        code = "from openagent.adaptive_engine.native_fp8_block_expert import NativeFP8BlockExpertBackend as B; b=B(); assert not b.info().available\ntry: b.execute_batch(None, [], [], hidden_size=1, intermediate_size=1)\nexcept RuntimeError: pass\nelse: raise AssertionError('fallback')"
        result = subprocess.run([sys.executable,"-c",code], env=dict(os.environ, NUMBA_DISABLE_JIT="1"), capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
