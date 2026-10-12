import math
import struct
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from openagent.adaptive_engine import fp8_block_expert as fp8
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource, NumPyRoutedSwiGLUBackend
from openagent.adaptive_engine.expert_store import DiskExpertStore, ExpertRegion


def oracle(code):
    sign = -1 if code >= 128 else 1
    magnitude = code % 128
    exponent, mantissa = divmod(magnitude, 8)
    if magnitude == 127:
        return math.nan
    return sign * (mantissa / 512 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7))


def decode_matrix(raw, rows, cols):
    out = np.empty((rows, cols), np.float32)
    for row in range(rows):
        for col in range(cols):
            scale = struct.unpack_from("<f", raw, rows*cols + 4*((row//128)*((cols+127)//128)+col//128))[0]
            out[row, col] = oracle(raw[row*cols+col]) * scale
    return out


class FormatTests(unittest.TestCase):
    def test_all_codes(self):
        for code in range(256):
            actual, expected = fp8.decode_code(code), oracle(code)
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertEqual(actual, expected)
                self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))

    def test_optional_torch_oracle(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch unavailable")
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("torch float8 unavailable")
        values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
        np.testing.assert_equal(values, [fp8.decode_code(c) for c in range(256)])

    def test_pack_edges_and_validation(self):
        a = np.arange(129*131, dtype=np.float32).reshape(129, 131) / 1000
        raw = fp8.pack_matrix(a)
        self.assertEqual(len(raw), 129*131+16)
        decoded = decode_matrix(raw, 129, 131)
        np.testing.assert_allclose(decoded, a, rtol=.065, atol=.002)
        self.assertEqual(fp8.payload_size(131, 129), 3*129*131+48)
        for encoding in ("ue8m0", "FP32", None):
            with self.assertRaises(ValueError):
                fp8.pack_matrix(a, scale_encoding=encoding)
        for a in (np.array([[math.nan]]), np.array([[math.inf]]), np.array([[1e100]]), np.empty((0, 2))):
            with self.assertRaises(ValueError):
                fp8.pack_matrix(a)

    def test_ties_signed_zero_extremes(self):
        values = np.array([[448, 1.0625, 1.1875, -0.0]], np.float32)
        raw = fp8.pack_matrix(values)
        self.assertEqual(list(raw[:4]), [126, 56, 58, 128])
        for value in (np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1))):
            raw = fp8.pack_matrix(np.array([[value]], np.float32))
            self.assertTrue(np.isfinite(decode_matrix(raw, 1, 1)).all())


class ReferenceTests(unittest.TestCase):
    def test_execution_without_jit_and_metadata(self):
        backend = fp8.FP8BlockExpertBackend()
        self.assertEqual(backend.NAME, "fp8-e4m3-b128-f32-reference")
        self.assertTrue(backend.info().available)
        self.assertFalse(backend.info().capabilities.compiled)
        rng = np.random.default_rng(12)
        shapes = ((5,3),(5,3),(3,5))
        raw = fp8.pack_expert(*[rng.normal(0,.1,s) for s in shapes])
        matrices, offset = [], 0
        for rows, cols in shapes:
            size = fp8.matrix_payload_size(rows, cols)
            matrices.append(decode_matrix(raw[offset:offset+size],rows,cols))
            offset += size
        args = dict(hidden_size=3, intermediate_size=5)
        x = rng.normal(size=(2,3)).astype(np.float32)
        routes = [[("a",.25),("b",.75)],[("b",0)]]
        expected = NumPyRoutedSwiGLUBackend().execute_batch(MappingPayloadSource({key:b"".join(a.astype("<f4").tobytes() for a in matrices) for key in ("a","b")}),x,routes,dtype="float32",**args)
        source = MappingPayloadSource({"a":raw,"b":raw})
        actual = backend.execute_batch(source,x,routes,**args)
        np.testing.assert_allclose(actual,expected,rtol=5e-5,atol=1e-8)
        np.testing.assert_array_equal(backend.execute(source,x[0],routes[0],**args),actual[0])
        self.assertEqual(backend.execute_batch(source,np.empty((0,3)),[],**args).shape,(0,3))
        with self.assertRaises(MemoryError):
            fp8.FP8BlockExpertBackend(max_scratch_bytes=19).execute(source,x[0],routes[0],**args)
        for routes in ([],[("a",-1)],[("a",1e100)],[("a",1),("a",1)]):
            with self.assertRaises(ValueError):
                backend.execute(source,x[0],routes,**args)
        code = "from openagent.adaptive_engine.fp8_block_expert import *; from openagent.adaptive_engine.expert_kernels import MappingPayloadSource; import numpy as np; b=FP8BlockExpertBackend(); assert b.info().available; raw=pack_expert(*[np.ones((1,1))]*3); assert np.isfinite(b.execute(MappingPayloadSource({'a':raw}),[1],[('a',1)],hidden_size=1,intermediate_size=1)).all()"
        result = subprocess.run([sys.executable,"-c",code],env=dict(os.environ,NUMBA_DISABLE_JIT="1"),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_retained_failure_mmap_cleanup(self):
        raw = fp8.pack_expert(*[np.ones((2,2))]*3)
        backend = fp8.FP8BlockExpertBackend()
        args = dict(hidden_size=2,intermediate_size=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for damaged in (raw[:-1], raw+b"\0", bytes([255])+raw[1:],raw[:4]+struct.pack("<f",math.nan)+raw[8:],raw):
                (root/"weights").write_bytes(damaged)
                store = DiskExpertStore(root,{"a":ExpertRegion("weights",0,len(damaged))},capacity_bytes=len(damaged),io_mode="mmap")
                retained = None
                try:
                    backend.execute(store,np.full(2,np.finfo(np.float32).max) if damaged == raw else np.ones(2),[("a",1 if damaged == raw else 0)],**args)
                except (ValueError,ArithmeticError) as error:
                    retained = error
                self.assertIsNotNone(retained)
                self.assertEqual(store.status()["active_leases"],0)
                store.close()
            (root/"weights").write_bytes(raw)
            store = DiskExpertStore(root,{"a":ExpertRegion("weights",0,len(raw))},capacity_bytes=len(raw),io_mode="mmap")
            def fail(raw,*args):
                borrowed = raw[::2]
                try:
                    raise ValueError("inner")
                except ValueError as cause:
                    raise RuntimeError("reference failure") from cause
            with patch.object(fp8,"_reference_loop",fail):
                try:
                    backend.execute(store,np.ones(2),[("a",1)],**args)
                except RuntimeError as error:
                    retained = error
            self.assertIsNone(retained.__context__)
            self.assertIsNone(retained.__cause__)
            store.close()
