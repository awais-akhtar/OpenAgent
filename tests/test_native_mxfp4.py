"""Compiled C execution, independent equations, and adversarial ABI checks."""

import ctypes as C
import gc
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.native_mxfp4 import NativeMXFP4, decode_bytes


def compile_library(directory):
    source = (
        Path(__file__).resolve().parents[1] / "openagent/adaptive_engine/native_mxfp4.c"
    )
    output = Path(directory) / ("mxfp4.dll" if os.name == "nt" else "mxfp4.so")
    if os.name == "nt":
        finder = (
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            / "Microsoft Visual Studio/Installer/vswhere.exe"
        )
        location = subprocess.check_output(
            [
                str(finder),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
        if not location:
            raise RuntimeError("MSVC required")
        batch = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
        args = [
            "cl",
            "/nologo",
            "/TC",
            "/W4",
            "/WX",
            "/O2",
            "/fp:strict",
            "/LD",
            str(source),
            "/link",
            "/OUT:" + str(output),
        ]
        command = (
            'cmd.exe /d /s /c ""'
            + str(batch)
            + '" && '
            + subprocess.list2cmdline(args)
            + '"'
        )
    else:
        cc = shutil.which("cc")
        if not cc:
            raise RuntimeError("C compiler required; not a passing skip")
        command = [
            cc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(output),
            "-lm",
        ]
    subprocess.run(command, cwd=directory, check=True, capture_output=True)
    return output


class TestNativeMXFP4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = compile_library(cls.temp.name)
        cls.decoder = NativeMXFP4(library=cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.decoder._release()
        gc.collect()
        cls.temp.cleanup()

    def test_all_codes_and_scales(self):
        codes = np.tile(np.arange(16, dtype=np.uint8), (255, 2))
        packed = codes[:, ::2] | (codes[:, 1::2] << 4)
        scales = np.arange(255, dtype=np.uint8)[:, None]
        # Only combinations whose exact result fits FP32 are admitted.
        safe = scales[:, 0] <= 252
        actual = self.decoder.decode(packed[safe], scales[safe], rows=253, cols=32)
        magnitude = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
        expected = np.ldexp(
            magnitude[codes[safe] & 7] * np.where(codes[safe] & 8, -1, 1),
            scales[safe].astype(int) - 127,
        ).astype(np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        # Largest scale remains legal for small values, including negative zero.
        p = np.full((1, 16), 0x81, np.uint8)
        a = self.decoder.decode(p, np.array([[254]], np.uint8), rows=1, cols=32)
        self.assertEqual(a[0, 0], np.float32(2.0**126))
        self.assertTrue(np.signbit(a[0, 1]))

    def test_torch_oracle(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch unavailable")
        rng = np.random.default_rng(731)
        p = rng.integers(0, 256, (3, 48), dtype=np.uint8)
        s = rng.integers(100, 140, (3, 3), dtype=np.uint8)
        b = torch.from_numpy(p.astype(np.int64))
        codes = torch.stack((b % 16, b // 16), dim=-1).reshape(3, 96)
        e, m = (codes // 2) % 4, codes % 2
        v = torch.where(e == 0, m.double() / 2, (1 + m.double() / 2) * 2.0 ** (e - 1))
        v *= torch.where(codes >= 8, -1, 1)
        v *= torch.repeat_interleave(
            2.0 ** (torch.from_numpy(s.astype(np.int64)) - 127), 32, dim=1
        )
        np.testing.assert_array_equal(
            self.decoder.decode(p, s, rows=3, cols=96), v.float().numpy()
        )

    def test_atomicity_canaries_and_alias(self):
        p, s = np.full((2, 32), 0x32, np.uint8), np.full((2, 2), 127, np.uint8)
        guard = np.full(132, -17, np.float32)
        out = guard[2:-2].reshape(2, 64)
        self.decoder.decode(p, s, rows=2, cols=64, out=out)
        np.testing.assert_array_equal(guard[[0, 1, -2, -1]], -17)
        before = guard.tobytes()
        p[-1, -1] = 0x77
        for bad in (255, 254):
            s[-1, -1] = bad
            with self.assertRaises(ValueError):
                self.decoder.decode(p, s, rows=2, cols=64, out=out)
            self.assertEqual(guard.tobytes(), before)
        s[:] = 127
        aliased = out.view(np.uint8).reshape(-1)[:64].reshape(2, 32)
        with self.assertRaisesRegex(ValueError, "aliases"):
            self.decoder.decode(aliased, s, rows=2, cols=64, out=out)
        self.assertEqual(guard.tobytes(), before)

    def test_direct_abi_bounds_and_budget(self):
        p, s, out = (
            np.zeros(16, np.uint8),
            np.array([127], np.uint8),
            np.full(32, 19, np.float32),
        )
        args = [p.ctypes.data, 16, s.ctypes.data, 1, 1, 32, out.ctypes.data, 128, 145]
        for index, value, status in (
            (0, 0, 1),
            (1, 15, 1),
            (3, 2, 1),
            (4, 0, 1),
            (5, 31, 1),
            (7, 127, 1),
            (8, 144, 4),
            (4, C.c_size_t(-1).value, 1),
            (6, s.ctypes.data, 2),
            (6, p.ctypes.data + 1, 2),
            (0, C.c_size_t(-1).value - 1, 1),
        ):
            bad = args.copy()
            bad[index] = value
            self.assertEqual(self.decoder._decode(*bad), status)
            np.testing.assert_array_equal(out, 19)
        self.assertEqual(self.decoder._decode(*args), 0)
        np.testing.assert_array_equal(out, 0)

    def test_wrapper_contract(self):
        p, s = np.zeros((1, 16), np.uint8), np.zeros((1, 1), np.uint8)
        self.assertEqual(decode_bytes(1, 32), 145)
        with self.assertRaises(MemoryError):
            NativeMXFP4(library=self.path, max_bytes=144).decode(p, s, rows=1, cols=32)
        for r, c in ((True, 32), (1, 33), (0, 32), (2**64, 32)):
            with self.assertRaises(ValueError):
                decode_bytes(r, c)
        for bad in (p.astype(np.int8), p[:, ::2], p.reshape(-1)):
            with self.assertRaises(ValueError):
                self.decoder.decode(bad, s, rows=1, cols=32)
        out = np.zeros((1, 32), np.float32)
        out.flags.writeable = False
        with self.assertRaises(ValueError):
            self.decoder.decode(p, s, rows=1, cols=32, out=out)

    def test_direct_matvec_equations_and_no_dense_decode(self):
        from unittest.mock import patch
        rng = np.random.default_rng(802)
        packed = rng.integers(0, 256, (7, 48), dtype=np.uint8)
        scales = rng.integers(123, 131, (7, 3), dtype=np.uint8)
        x = rng.normal(size=96).astype(np.float32)
        codebook = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
        codes = np.empty((7, 96), np.uint8)
        codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
        weights = (codebook[codes] * np.repeat(np.exp2(scales.astype(int) - 127), 32, axis=1)).astype(np.float32)
        expected = (weights.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
        with patch.object(self.decoder, 'decode', side_effect=AssertionError('dense expansion')):
            actual = self.decoder.matvec(packed, scales, x, rows=7, cols=96)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_matvec_atomicity_budget_alias_and_bad_inputs(self):
        packed = np.full((2, 16), 0x22, np.uint8)
        scales = np.full((2, 1), 127, np.uint8)
        x = np.ones(32, np.float32)
        out = np.full(2, 19, np.float32)
        required = 32 + 2 + 128 + 16
        model = NativeMXFP4(library=self.path, max_bytes=required)
        np.testing.assert_array_equal(model.matvec(packed, scales, x, rows=2, cols=32), 32)
        scales[1, 0] = 255
        with self.assertRaises(ValueError):
            model.matvec(packed, scales, x, rows=2, cols=32, out=out)
        np.testing.assert_array_equal(out, 19)
        scales[:] = 127
        with self.assertRaises(ValueError):
            model.matvec(packed, scales, x, rows=2, cols=32, out=x[:2])
        np.testing.assert_array_equal(x, 1)
        model.max_bytes = required - 1
        with self.assertRaises(MemoryError):
            model.matvec(packed, scales, x, rows=2, cols=32, out=out)
        model.max_bytes = True
        with self.assertRaises(ValueError):
            model.matvec(packed, scales, x, rows=2, cols=32, out=out)
        model._release()


if __name__ == "__main__":
    unittest.main()
