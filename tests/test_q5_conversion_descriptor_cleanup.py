"""Descriptor ownership regressions for packed Q5 converters."""
import importlib
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class Q5ConversionDescriptorCleanupTests(unittest.TestCase):
    def test_scalar_and_native_kernels_preserve_signed_finite_scales(self):
        from openagent.adaptive_engine.q5_0_expert import _scalar_expert as scalar_q50
        from openagent.adaptive_engine.q5_1_expert import _scalar_expert as scalar_q51
        from openagent.adaptive_engine.native_q5_0_expert import _kernel as native_q50
        from openagent.adaptive_engine.native_q5_1_expert import _kernel as native_q51

        input_vector = np.ones(32, dtype=np.float32)
        for scalar, native, payload in (
                (scalar_q50, native_q50,
                 struct.pack('<eI16B', -0.25, 0, *([0] * 16)) * 96),
                (scalar_q51, native_q51,
                 struct.pack('<eeI16B', -0.25, 0.75, 0, *([0] * 16)) * 96)):
            with self.subTest(kernel=scalar.__module__):
                activation = np.zeros(32, dtype=np.float32)
                output = np.zeros(32, dtype=np.float32)
                self.assertEqual(scalar(memoryview(payload), input_vector, 32, 32, 1.0,
                                        activation, output), 0)
                activation = np.zeros(32, dtype=np.float32)
                output = np.zeros(32, dtype=np.float32)
                self.assertEqual(native(np.frombuffer(payload, np.uint8),
                    input_vector, 32, 32, 1.0, activation, output), 0)
                self.assertTrue(np.isfinite(output).all())

    def test_compiled_native_kernels_preserve_signed_finite_scales(self):
        from openagent.adaptive_engine import native_q5_0_expert, native_q5_1_expert
        from openagent.adaptive_engine.q5_0_expert import _scalar_expert as scalar_q50
        from openagent.adaptive_engine.q5_1_expert import _scalar_expert as scalar_q51

        kernels = (native_q5_0_expert._compiled_kernel,
                   native_q5_1_expert._compiled_kernel)
        scalars = (scalar_q50, scalar_q51)
        if any(kernel is None for kernel in kernels):
            self.skipTest('Numba compiled Q5 kernels are unavailable')
        inputs = np.ones(32, dtype=np.float32)
        for kernel, scalar, payload in (
                (kernels[0], scalars[0], struct.pack('<eI16B', -0.25, 0, *([0] * 16)) * 96),
                (kernels[1], scalars[1], struct.pack('<eeI16B', -0.25, 0.75, 0, *([0] * 16)) * 96)):
            with self.subTest(kernel=kernel):
                activation = np.zeros(32, dtype=np.float32)
                output = np.zeros(32, dtype=np.float32)
                status = kernel(np.frombuffer(payload, dtype=np.uint8), inputs,
                    32, 32, np.float32(1), activation, output)
                self.assertEqual(status, 0)
                self.assertTrue(np.isfinite(activation).all())
                self.assertTrue(np.isfinite(output).all())
                scalar_activation = np.zeros(32, dtype=np.float32)
                scalar_output = np.zeros(32, dtype=np.float32)
                self.assertEqual(scalar(memoryview(payload), inputs, 32, 32, 1.0,
                                        scalar_activation, scalar_output), 0)
                np.testing.assert_allclose(activation, scalar_activation, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(output, scalar_output, rtol=1e-6, atol=1e-6)

    def test_fdopen_failure_closes_descriptor_and_removes_staging_file(self):
        matrices = [np.full((32, 32), 0.1, dtype=np.float32) for _ in range(3)]
        real_mkstemp = tempfile.mkstemp
        for module_name in ('q5_0_conversion', 'q5_1_conversion'):
            with self.subTest(converter=module_name), tempfile.TemporaryDirectory() as directory:
                converter = importlib.import_module(
                    'openagent.adaptive_engine.' + module_name)
                opened = {}

                def capture_descriptor(*args, **kwargs):
                    descriptor, name = real_mkstemp(*args, **kwargs)
                    opened.update(descriptor=descriptor, path=Path(name))
                    return descriptor, name

                with patch.object(converter.tempfile, 'mkstemp',
                                  side_effect=capture_descriptor), \
                     patch.object(converter.os, 'fdopen',
                                  side_effect=OSError('fdopen failed')):
                    with self.assertRaisesRegex(OSError, 'fdopen failed'):
                        converter.convert_experts({'a': matrices},
                            Path(directory) / 'packed.bin',
                            hidden_size=32, intermediate_size=32)

                with self.assertRaises(OSError):
                    os.fstat(opened['descriptor'])
                self.assertFalse(opened['path'].exists())
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
