"""Strict compiled local-codec tests; PyTorch is a format-only external oracle."""

import ctypes as C
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import math
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TORCH_ORACLE_SKIP_REASON = (
    "optional external E4M3 oracle unavailable: Torch is not installed; "
    "strict acceptance requires the accelerator environment and zero skips"
)


class Quant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        source = ROOT / "openagent/adaptive_engine/native_deepseek_v41_quant.c"
        output = Path(cls.tmp.name) / ("quant.dll" if os.name == "nt" else "quant.so")
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
            env = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
            args = [
                "cl",
                "/nologo",
                "/W4",
                "/WX",
                "/O2",
                "/LD",
                str(source),
                "/link",
                "/OUT:" + str(output),
            ]
            command = (
                'cmd.exe /d /s /c ""'
                + str(env)
                + '" && '
                + subprocess.list2cmdline(args)
                + '"'
            )
        else:
            command = [
                shutil.which("cc") or "cc",
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
        result = subprocess.run(
            command, cwd=cls.tmp.name, capture_output=True, text=True, timeout=120
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        spec = importlib.util.spec_from_file_location(
            "isolated_v41_quant",
            ROOT / "openagent/adaptive_engine/native_deepseek_v41_quant.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module
        cls.output = output
        cls.codec = module.NativeDeepSeekV41Quant(output)
        cls.lib = cls.codec.library
        cls.addClassCleanup(cls.codec.close)

    def test_library_path_handshake_and_lifetime(self):
        self.assertEqual(self.lib.oa_v41_quant_abi_version(), 1)
        self.assertEqual(self.codec.library_path, self.output.resolve())
        with mock.patch.object(self.module.C, "CDLL") as load:
            for path in [self.output.parent, self.output.parent / "absent.dll"]:
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.module.NativeDeepSeekV41Quant(path)
            load.assert_not_called()
        # Real additional load and unload, with a spy forwarding to the OS.
        release = self.module._release_library
        with mock.patch.object(
            self.module, "_release_library", wraps=release
        ) as unload:
            instance = self.module.NativeDeepSeekV41Quant(self.output)
            handle = instance.library._handle
            ref = weakref.ref(instance)
            del instance
            gc.collect()
            self.assertIsNone(ref())
            unload.assert_called_once_with(handle)
            instance = self.module.NativeDeepSeekV41Quant(self.output)
            instance.close()
            instance.close()
            with self.assertRaises(RuntimeError):
                instance.decode(bytes(9), 1, 16)
            del instance
            gc.collect()
            self.assertEqual(unload.call_count, 2)
        for missing, version in [(True, 1), (False, 2)]:
            fake = mock.Mock()
            fake._handle = 123
            if missing:
                del fake.oa_v41_quant_abi_version
            else:
                fake.oa_v41_quant_abi_version.return_value = version
            with (
                mock.patch.object(self.module.C, "CDLL", return_value=fake),
                mock.patch.object(self.module, "_release_library") as unload,
            ):
                with self.assertRaisesRegex(ValueError, "ABI"):
                    self.module.NativeDeepSeekV41Quant(self.output)
                gc.collect()
                unload.assert_called_once_with(123)

    def test_close_waits_for_active_codec_call(self):
        instance = self.module.NativeDeepSeekV41Quant(self.output)
        entered, finish, closing = threading.Event(), threading.Event(), threading.Event()
        original = instance.library.oa_v41_quant_encode

        def held(*args):
            entered.set()
            if not finish.wait(5):
                raise RuntimeError('test worker timed out')
            return original(*args)

        def close():
            closing.set()
            instance.close()

        try:
            with mock.patch.object(instance.library, 'oa_v41_quant_encode', side_effect=held):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    encoding = pool.submit(instance.encode, [1] * 16, 1, 16)
                    try:
                        self.assertTrue(entered.wait(5))
                        closing_future = pool.submit(close)
                        self.assertTrue(closing.wait(5))
                        self.assertFalse(closing_future.done())
                        self.assertTrue(instance._finalizer.alive)
                    finally:
                        finish.set()
                    self.assertEqual(len(encoding.result(timeout=5)), 9)
                    closing_future.result(timeout=5)
            self.assertFalse(instance._finalizer.alive)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                instance.decode(bytes(9), 1, 16)
        finally:
            finish.set()
            instance.close()

    def test_reentrant_iterator_cannot_unload_library(self):
        instance = self.module.NativeDeepSeekV41Quant(self.output)
        def values():
            with self.assertRaisesRegex(RuntimeError, 'active operation'):
                instance.close()
            yield from [1] * 16
        try:
            packed = instance.encode(values(), 1, 16, scales=bytes([56]))
            self.assertEqual(instance.decode(packed, 1, 16), (1,) * 16)
        finally:
            instance.close()

    def test_scale_and_packed_iterator_close_and_exception_unwind(self):
        instance = self.module.NativeDeepSeekV41Quant(self.output)
        def guarded(values):
            with self.assertRaisesRegex(RuntimeError, 'active operation'):
                instance.close()
            yield from values
        def broken():
            raise ValueError('conversion failed')
            yield 0
        try:
            packed = instance.encode([1] * 16, 1, 16, scales=guarded([56]))
            self.assertEqual(instance.decode(guarded(packed), 1, 16), (1,) * 16)
            with self.assertRaisesRegex(ValueError, 'conversion failed'):
                instance.encode(broken(), 1, 16)
            self.assertEqual(instance._active, 0)
            self.assertEqual(instance.decode(packed, 1, 16), (1,) * 16)
        finally:
            instance.close()

    def test_bools_and_bounded_consumption(self):
        for rows, channels, budget in [
            (True, 16, 65536),
            (1, True, 65536),
            (1, 16, True),
            (False, 16, 65536),
            (1, 16, False),
        ]:
            with self.assertRaises(ValueError):
                self.codec.encode([], rows, channels, budget=budget)
            with self.assertRaises(ValueError):
                self.codec.decode(b"", rows, channels, budget=budget)
        consumed = []

        def endless():
            while True:
                consumed.append(1)
                yield 0

        with self.assertRaises(ValueError):
            self.codec.encode(endless(), 1, 16)
        self.assertEqual(len(consumed), 17)
        consumed.clear()
        with self.assertRaises(ValueError):
            self.codec.decode(endless(), 1, 16)
        self.assertEqual(len(consumed), 10)
        consumed.clear()
        with self.assertRaises(ValueError):
            self.codec.encode([0] * 16, 1, 16, scales=endless())
        self.assertEqual(len(consumed), 2)

        class NoCopy(bytearray):
            def __bytes__(self):
                raise AssertionError("must reject extent before copying")

        with self.assertRaises(ValueError):
            self.codec.decode(NoCopy(1000000), 1, 16)
        with self.assertRaises(ValueError):
            self.codec.encode([0] * 16, 1, 16, scales=NoCopy(1000000))
        packed = self.codec.encode(iter([1] * 16), 1, 16, scales=iter([56]))
        self.assertEqual(self.codec.decode(memoryview(packed), 1, 16), (1,) * 16)

    def test_e2_codes_signed_zero_subnormal_and_ties(self):
        levels = [0.0, 0.5, 1, 1.5, 2, 3, 4, 6]
        values = levels + [-v for v in levels]
        p = self.codec.encode(values, 1, 16, scales=b"\x38")
        self.assertEqual(p, bytes([56, 16, 50, 84, 118, 152, 186, 220, 254]))
        self.assertEqual(self.codec.decode(p, 1, 16), tuple(values))
        self.assertEqual(math.copysign(1, self.codec.decode(p, 1, 16)[8]), -1)
        for i in range(7):
            midpoint = (levels[i] + levels[i + 1]) / 2
            expected = i if i % 2 == 0 else i + 1
            for sign in (1, -1):
                packed = self.codec.encode(
                    [sign * midpoint] * 16, 1, 16, scales=b"\x38"
                )
                self.assertEqual(packed[1] & 15, expected | (8 if sign < 0 else 0))
                for delta, code in [(-(2**-20), i), (2**-20, i + 1)]:
                    packed = self.codec.encode(
                        [sign * (midpoint + delta)] * 16, 1, 16, scales=b"\x38"
                    )
                    self.assertEqual(packed[1] & 15, code | (8 if sign < 0 else 0))
        for v in (1e30, -1e30):
            self.assertEqual(
                abs(self.codec.decode(self.codec.encode([v] * 16, 1, 16), 1, 16)[0]),
                2688,
            )

    def test_external_torch_e4m3_oracle(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest(TORCH_ORACLE_SKIP_REASON)
        # Import only in tests. No public implementation source copied or translated.
        import torch

        print("Independent E4M3 oracle: torch", torch.__version__)
        values = (
            torch.arange(127, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .float()
            .tolist()
        )
        for code, value in enumerate(values):
            packed = bytes([code] + [0x77] * 8)
            self.assertEqual(self.codec.decode(packed, 1, 16), (value * 6,) * 16)
        probes = values + [(a + b) / 2 for a, b in zip(values, values[1:])]
        for target in probes:
            x = (C.c_float * 16)(*([target * 6] * 16))
            out = (C.c_uint8 * 1)()
            self.assertEqual(self.lib.oa_v41_quant_scales(x, 16, 1, 16, out, 1, 1), 0)
            expected = (
                torch.tensor(target).to(torch.float8_e4m3fn).view(torch.uint8).item()
            )
            self.assertEqual(out[0], max(1, expected) if target else 0)

    def test_missing_torch_has_explicit_optional_skip(self):
        with mock.patch.object(importlib.util, "find_spec", return_value=None) as find:
            with self.assertRaises(unittest.SkipTest) as skipped:
                self.test_external_torch_e4m3_oracle()
            self.assertEqual(str(skipped.exception), TORCH_ORACLE_SKIP_REASON)
            find.assert_called_once_with("torch")

    def test_geometry_chunks_and_wrapper_rejection(self):
        for rows, channels in [(1, 16), (3, 48), (32, 512)]:
            values = [((i % 37) - 18) / 8 for i in range(rows * channels)]
            whole = self.codec.encode(values, rows, channels)
            chunks = b"".join(
                self.codec.encode(
                    values[r * channels : (r + 1) * channels], 1, channels
                )
                for r in range(rows)
            )
            self.assertEqual(whole, chunks)
            decoded = self.codec.decode(whole, rows, channels)
            self.assertEqual(len(decoded), len(values))
            self.assertEqual(
                self.codec.encode(decoded, rows, channels, scales=whole[::9]), whole
            )
        for r, c in [(0, 16), (33, 16), (1, 15), (1, 513), (2**32 + 1, 16)]:
            with self.assertRaises(ValueError):
                self.codec.encode([], r, c)
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                self.codec.encode([bad] * 16, 1, 16)
        with self.assertRaises(ValueError):
            self.codec.decode(b"", 1, 16)
        with self.assertRaises(ValueError):
            self.codec.encode([0] * 16, 1, 16, budget=-1)

    def test_native_transactional_extents_budget_and_canaries(self):
        x = (C.c_float * 32)(*([1] * 32))
        scales = (C.c_uint8 * 2)(56, 56)
        guard = (C.c_uint8 * 20)(*([165] * 20))
        out = C.byref(guard, 1)

        def encode(**kw):
            args = dict(x=x, nx=32, s=scales, ns=2, r=1, c=32, o=out, no=18, b=18)
            args.update(kw)
            return self.lib.oa_v41_quant_encode(*args.values())

        for kw, status in [
            (dict(nx=31), 1),
            (dict(ns=3), 1),
            (dict(no=17), 1),
            (dict(r=0), 1),
            (dict(c=17), 1),
            (dict(x=None), 1),
            (dict(o=None), 1),
            (dict(x=C.byref(x, 1)), 1),
            (dict(o=C.c_void_p(C.c_size_t(-1).value - 1)), 1),
            (dict(b=17), 2),
        ]:
            self.assertEqual(encode(**kw), status)
            self.assertEqual(bytes(guard), bytes([165] * 20))
        for bad in (float("nan"), float("inf"), -float("inf")):
            x[31] = bad
            self.assertEqual(encode(), 3)
            self.assertEqual(bytes(guard), bytes([165] * 20))
        x[31] = 1
        for code in (0, 127, 128, 255):
            scales[1] = code
            self.assertEqual(encode(), 3)
            self.assertEqual(bytes(guard), bytes([165] * 20))
        scales[1] = 56
        self.assertEqual(encode(), 0)
        self.assertEqual((guard[0], guard[19]), (165, 165))
        target = (C.c_float * 34)(*([-123] * 34))
        before = bytes(target)
        packed = (C.c_uint8 * 18).from_buffer_copy(bytes(guard)[1:19])
        for budget, count, status in [(127, 32, 2), (128, 31, 1)]:
            self.assertEqual(
                self.lib.oa_v41_quant_decode(
                    packed, 18, 1, 32, C.byref(target, 4), count, budget
                ),
                status,
            )
            self.assertEqual(bytes(target), before)
        packed[9] = 127
        self.assertEqual(
            self.lib.oa_v41_quant_decode(
                packed, 18, 1, 32, C.byref(target, 4), 32, 128
            ),
            3,
        )
        self.assertEqual(bytes(target), before)
        for invalid in range(127, 256):
            packed[9] = invalid
            self.assertEqual(
                self.lib.oa_v41_quant_decode(
                    packed, 18, 1, 32, C.byref(target, 4), 32, 128
                ),
                3,
            )
            self.assertEqual(bytes(target), before)
        packed[9] = 56
        for ptr, count, rows, channels, dest, status in [
            (None, 18, 1, 32, C.byref(target, 4), 1),
            (packed, 17, 1, 32, C.byref(target, 4), 1),
            (packed, 18, 33, 32, C.byref(target, 4), 1),
            (packed, 18, 1, 32, None, 1),
            (packed, 18, 1, 32, C.byref(target, 1), 1),
        ]:
            self.assertEqual(
                self.lib.oa_v41_quant_decode(ptr, count, rows, channels, dest, 32, 128),
                status,
            )
            self.assertEqual(bytes(target), before)
        self.assertEqual(
            self.lib.oa_v41_quant_decode(
                packed, 18, 1, 32, C.byref(target, 4), 32, 128
            ),
            0,
        )
        self.assertEqual((target[0], target[33]), (-123, -123))
        scale_out = (C.c_uint8 * 2)(99, 99)
        x[31] = float("nan")
        self.assertEqual(self.lib.oa_v41_quant_scales(x, 32, 1, 32, scale_out, 2, 2), 3)
        self.assertEqual(bytes(scale_out), b"cc")
        x[31] = 1
        for nx, rows, channels, ns, budget, status in [
            (31, 1, 32, 2, 2, 1),
            (32, 1, 32, 1, 2, 1),
            (32, 1, 32, 2, 1, 2),
            (32, 0, 32, 2, 2, 1),
        ]:
            self.assertEqual(
                self.lib.oa_v41_quant_scales(
                    x, nx, rows, channels, scale_out, ns, budget
                ),
                status,
            )
            self.assertEqual(bytes(scale_out), b"cc")

    def test_zero_scale_and_input_output_alias(self):
        packed = self.codec.encode([-0.0] * 16, 1, 16)
        self.assertEqual(packed, bytes([0] + [136] * 8))
        self.assertTrue(
            all(math.copysign(1, v) < 0 for v in self.codec.decode(packed, 1, 16))
        )
        x = (C.c_float * 16)(*([1] * 16))
        scale = (C.c_uint8 * 1)(56)
        self.assertEqual(
            self.lib.oa_v41_quant_encode(x, 16, scale, 1, 1, 16, x, 9, 9), 0
        )
        self.assertEqual(C.string_at(x, 9), bytes([56] + [34] * 8))


if __name__ == "__main__":
    unittest.main()
