"""Real compiled C primitive checks, plus exact Qwen3.5 decoder oracle."""
import os
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from openagent.adaptive_engine.native_mixtral_delta import CompiledGatedDeltaKernel


def compile_kernel(directory):
    source = Path(__file__).resolve().parents[1] / 'openagent/adaptive_engine/native_mixtral_delta.c'
    output = Path(directory) / ('delta.dll' if os.name == 'nt' else 'delta.so')
    if os.name == 'nt':
        vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
        if not vswhere.exists():
            raise unittest.SkipTest('MSVC discovery unavailable')
        location = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
            'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
        if not location:
            raise unittest.SkipTest('MSVC unavailable')
        batch = Path(location) / 'VC/Auxiliary/Build/vcvars64.bat'
        arguments = ['cl', '/nologo', '/W4', '/WX', '/O2', '/fp:precise', '/LD', str(source), '/link', '/OUT:'+str(output)]
        command = 'cmd.exe /d /s /c ""'+str(batch)+'" && '+subprocess.list2cmdline(arguments)+'"'
    else:
        compiler = shutil.which('cc')
        if compiler is None:
            raise unittest.SkipTest('C compiler unavailable')
        command = [compiler, '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC', str(source), '-o', str(output), '-lm']
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return CompiledGatedDeltaKernel(output)


class CompiledDeltaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        try:
            cls.kernel = compile_kernel(cls.directory.name)
        except Exception:
            cls.directory.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        # Windows keeps loaded DLLs mapped; release before removing test artifacts.
        if os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            ctypes.windll.kernel32.FreeLibrary(cls.kernel.library._handle)
        cls.directory.cleanup()

    def test_recurrent_equation_and_rejection_atomicity(self):
        rng = np.random.default_rng(109)
        state = rng.normal(size=(2, 3, 4, 6)).astype(np.float32)
        reference = state.copy()
        for _ in range(8):
            q, k = [rng.normal(size=(2, 3, 4)).astype(np.float32) for _ in range(2)]
            k /= np.linalg.norm(k, axis=-1, keepdims=True)
            v = rng.normal(size=(2, 3, 6)).astype(np.float32)
            beta, decay = [rng.uniform(size=(2, 3)).astype(np.float32) for _ in range(2)]
            reference *= decay[..., None, None]
            error = v-np.einsum('bhk,bhkv->bhv', k, reference)
            reference += k[..., None]*(error*beta[..., None])[..., None, :]
            expected = np.einsum('bhk,bhkv->bhv', q, reference)
            actual = self.kernel.step(q, k, v, beta, decay, state)
            np.testing.assert_allclose(state, reference, atol=5e-7, rtol=2e-6)
            np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=2e-6)
        before = state.copy()
        beta[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.kernel.step(q, k, v, beta, decay, state)
        np.testing.assert_array_equal(state, before)

    def test_compiled_qwen35_hybrid_against_transformers(self):
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'transformers')):
            self.skipTest('optional exact numerical oracle unavailable')
        from tests.test_native_mixtral_qwen35 import Qwen35NumericalTests
        from openagent.adaptive_engine.independent_qwen35 import IndependentQwen35TextDecoder
        import torch
        suite = Qwen35NumericalTests()
        oracle, numpy_decoder = suite.fixture()
        compiled = IndependentQwen35TextDecoder(oracle.config.to_dict(), numpy_decoder.weights, delta_kernel=self.kernel)
        self.assertFalse(compiled.native_range_execution)
        self.assertEqual(compiled.implementation, 'numpy+c-gated-delta-step')
        ids = torch.tensor([[1, 4, 8, 3, 2]])
        with torch.inference_mode():
            suite.compare(compiled.forward(ids.numpy()), oracle(ids, output_hidden_states=True))
            compiled.reset()
            past = None
            for start, end in ((0, 3), (3, 4), (4, 5)):
                expected = oracle(ids[:, start:end], past_key_values=past, output_hidden_states=True)
                past = expected.past_key_values
                suite.compare(compiled.forward(ids[:, start:end].numpy()), expected)
                for i in (0, 2):
                    np.testing.assert_allclose(compiled.cache[i][1], past.layers[i].recurrent_states[0].numpy(), atol=2e-6)


if __name__ == '__main__':
    unittest.main()
