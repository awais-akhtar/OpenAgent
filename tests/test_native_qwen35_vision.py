"""Compiled vertical subset against installed official Transformers 5.16.1."""
import gc
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.native_qwen35_vision import NativeQwen35Vision
from tests.test_native_qwen35 import build as build_text
from tests import test_native_mixtral_qwen35 as text_oracle


def build_vision(directory):
    source = Path(__file__).resolve().parents[1]/'openagent/adaptive_engine/native_qwen35_vision.c'
    output = Path(directory)/('qwen35_vision.dll' if os.name == 'nt' else 'qwen35_vision.so')
    if os.name == 'nt':
        finder = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
        if not finder.exists():
            raise unittest.SkipTest('MSVC unavailable')
        location = subprocess.check_output([str(finder), '-latest', '-products', '*', '-requires',
            'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
        if not location:
            raise unittest.SkipTest('MSVC unavailable')
        batch = Path(location)/'VC/Auxiliary/Build/vcvars64.bat'
        args = ['cl', '/nologo', '/W4', '/WX', '/O2', '/fp:precise', '/LD', str(source), '/link', '/OUT:'+str(output)]
        command = 'cmd.exe /d /s /c ""'+str(batch)+'" && '+subprocess.list2cmdline(args)+'"'
    else:
        compiler = shutil.which('cc')
        if not compiler:
            raise unittest.SkipTest('C compiler unavailable')
        command = [compiler, '-std=c99', '-Wall', '-Wextra', '-Werror', '-O2', '-shared', '-fPIC', str(source), '-o', str(output), '-lm']
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return output


class Qwen35VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.text_path = build_text(cls.directory.name)
        cls.vision_path = build_vision(cls.directory.name)
        cls.owners = []

    @classmethod
    def tearDownClass(cls):
        cls.owners.clear()
        gc.collect()
        cls.directory.cleanup()

    def fixture(self, depth=1):
        import torch
        import transformers
        from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
        self.assertEqual(transformers.__version__, '5.16.1')
        text, _ = text_oracle.Qwen35NumericalTests().fixture(481)
        config = Qwen3_5MoeConfig(text_config=text.config.to_dict(),
            vision_config=dict(NativeQwen35Vision._geometry, depth=depth), image_token_id=35, video_token_id=34,
            vision_start_token_id=32, vision_end_token_id=33)
        config._attn_implementation = 'eager'
        torch.manual_seed(482)
        oracle = Qwen3_5MoeForConditionalGeneration(config).eval()
        weights = {k: v.detach().numpy() for k, v in oracle.state_dict().items()}
        native = NativeQwen35Vision(config.to_dict(), weights,
            library=self.vision_path, text_library=self.text_path)
        self.owners.append(native)
        return oracle, native, weights

    def test_official_image_text_prefill_cached_and_positions(self):
        import torch
        oracle, native, _ = self.fixture()
        ids = torch.tensor([[1, 32, 35, 33, 7]])
        types = (ids == 35).long()
        grid = torch.tensor([[1, 2, 2]])
        pixels = torch.tensor([[.2, -.4, .7], [.9, .1, -.8], [-.6, .3, .5], [.4, -.2, -.1]])
        with torch.inference_mode():
            visual = oracle.model.visual(pixels, grid_thw=grid).pooler_output
            encoded = np.empty(24, np.float32)
            from openagent.adaptive_engine.native_qwen35 import tensor
            self.assertEqual(native._encode(tensor(pixels.numpy()), native._bindings, len(native._bindings), tensor(encoded)), 0)
            np.testing.assert_allclose(encoded, visual.numpy()[0], atol=2e-6, rtol=5e-5)
            positions, delta = oracle.model.get_rope_index(ids, image_grid_thw=grid, mm_token_type_ids=types)
            np.testing.assert_array_equal(positions.numpy(), np.broadcast_to(np.arange(5), (3, 1, 5)))
            np.testing.assert_array_equal(delta.numpy(), [[0]])
            expected = oracle(ids, pixel_values=pixels, image_grid_thw=grid,
                mm_token_type_ids=types, output_hidden_states=True, output_attentions=True)
            with (patch('openagent.adaptive_engine.independent_qwen35.IndependentQwen35TextDecoder.forward',
                        side_effect=AssertionError('reference decoder fallback')),
                  patch('numpy.exp', side_effect=AssertionError('NumPy model math')),
                  patch('numpy.einsum', side_effect=AssertionError('NumPy model math')),
                  patch('numpy.matmul', side_effect=AssertionError('NumPy model math')),
                  patch('numpy.dot', side_effect=AssertionError('NumPy model math')),
                  patch('numpy.sqrt', side_effect=AssertionError('NumPy model math')),
                  patch('torch.nn.Module.__call__', side_effect=AssertionError('Torch runtime delegation'))):
                actual = native.forward(ids.numpy(), pixel_values=pixels.numpy(), image_grid_thw=grid.numpy(),
                                        mm_token_type_ids=types.numpy())
            text_oracle.Qwen35NumericalTests().compare(actual, expected)
            past = expected.past_key_values
            for token in (8, 9, 2):
                next_ids = torch.tensor([[token]])
                expected = oracle(next_ids, past_key_values=past, output_hidden_states=True, output_attentions=True)
                with (patch('numpy.einsum', side_effect=AssertionError('NumPy cached math')),
                      patch('torch.nn.Module.__call__', side_effect=AssertionError('Torch cached delegation')),
                      patch('openagent.adaptive_engine.independent_qwen35.IndependentQwen35TextDecoder.forward',
                            side_effect=AssertionError('reference cached decoder fallback'))):
                    actual = native.forward(next_ids.numpy())
                text_oracle.Qwen35NumericalTests().compare(actual, expected)
                past = expected.past_key_values
            native.reset()
            text_oracle.Qwen35NumericalTests().compare(native.forward([1, 3, 7]),
                oracle(torch.tensor([[1, 3, 7]]), output_hidden_states=True, output_attentions=True))

    def test_fail_closed_and_transactional(self):
        oracle, native, weights = self.fixture()
        pixels = np.ones((4, 3), np.float32)
        for kwargs in (
            dict(pixel_values=pixels, image_grid_thw=[[2, 2, 2]]),
            dict(pixel_values=pixels, image_grid_thw=[[1, 4, 2]]),
            dict(pixel_values=pixels.astype(np.float64), image_grid_thw=[[1, 2, 2]]),
            dict(pixel_values=np.full((4, 3), np.nan, np.float32), image_grid_thw=[[1, 2, 2]]),
            dict(pixel_values=pixels, image_grid_thw=[[1, 2, 2]], mm_token_type_ids=[[0, 0, 0]]),
            dict(video_grid_thw=[[1, 2, 2]]), {},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                native.forward([1, 35, 7], **kwargs)
            self.assertEqual(native.decoder.position, 0)
        config = oracle.config.to_dict()
        for key, value in [('depth', 5), ('patch_size', 2), ('temporal_patch_size', 2), ('hidden_size', 16)]:
            bad = dict(config, vision_config=dict(config['vision_config'], **{key: value}))
            with self.assertRaises(ValueError):
                NativeQwen35Vision(bad, weights, library=self.vision_path, text_library=self.text_path)
        with self.assertRaises(ValueError):
            NativeQwen35Vision(config, weights, library=self.vision_path, text_library=self.text_path, max_bytes=1024)
        native.forward([1, 35, 7], pixel_values=pixels, image_grid_thw=[[1, 2, 2]])
        position = native.decoder.position
        with self.assertRaises(ValueError):
            native.forward([35], pixel_values=pixels, image_grid_thw=[[1, 2, 2]])
        self.assertEqual(native.decoder.position, position)

    def test_depth_boundaries_and_active_transformer_weights(self):
        import torch
        from openagent.adaptive_engine.native_qwen35 import tensor
        pixels = np.array([[.2, -.4, .7], [.9, .1, -.8], [-.6, .3, .5], [.4, -.2, -.1]], np.float32)
        for depth in (0, 2, 4):
            with self.subTest(depth=depth), torch.inference_mode():
                oracle, native, _ = self.fixture(depth)
                output = np.empty(24, np.float32)
                self.assertEqual(native._encode(tensor(pixels), native._bindings,
                    len(native._bindings), tensor(output)), 0)
                expected = oracle.model.visual(torch.from_numpy(pixels),
                    grid_thw=torch.tensor([[1, 2, 2]])).pooler_output
                np.testing.assert_allclose(output, expected.numpy()[0], atol=2e-6, rtol=5e-5)
        oracle, native, weights = self.fixture()
        changed = dict(weights)
        changed['model.visual.blocks.0.attn.proj.bias'] = np.arange(8, dtype=np.float32)
        alternate = NativeQwen35Vision(oracle.config.to_dict(), changed,
            library=self.vision_path, text_library=self.text_path)
        self.owners.append(alternate)
        original, output = np.empty(24, np.float32), np.empty(24, np.float32)
        for model, target in ((native, original), (alternate, output)):
            self.assertEqual(model._encode(tensor(pixels), model._bindings,
                len(model._bindings), tensor(target)), 0)
        self.assertGreater(float(np.max(np.abs(output-original))), 1e-4)

    def test_block_tensor_errors_and_numerical_failure_rollback(self):
        from openagent.adaptive_engine.native_qwen35 import Tensor, tensor
        oracle, native, weights = self.fixture()
        config = oracle.config.to_dict()
        key = 'model.visual.blocks.0.attn.qkv.weight'
        for bad in (np.zeros((24, 7), np.float32), np.full((24, 8), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                NativeQwen35Vision(config, dict(weights, **{key: bad}),
                    library=self.vision_path, text_library=self.text_path)
        missing = dict(weights)
        del missing[key]
        with self.assertRaises(ValueError):
            NativeQwen35Vision(config, missing, library=self.vision_path, text_library=self.text_path)
        bindings = (Tensor*len(native._bindings))(*native._bindings)
        output = np.full(24, -17, np.float32)
        pixels = np.ones((4, 3), np.float32)
        bindings[13] = Tensor(bindings[13].data, 191)
        self.assertNotEqual(native._encode(tensor(pixels), bindings, len(bindings), tensor(output)), 0)
        bindings[13] = Tensor(None, 192)
        self.assertNotEqual(native._encode(tensor(pixels), bindings, len(bindings), tensor(output)), 0)
        huge = np.full((24, 8), np.finfo(np.float32).max, np.float32)
        bindings[13] = tensor(huge)
        self.assertNotEqual(native._encode(tensor(pixels), bindings, len(bindings), tensor(output)), 0)
        np.testing.assert_array_equal(output, np.full(24, -17, np.float32))

    def test_reset_original_weights_new_image_and_pure_text_oracle(self):
        import torch
        oracle, native, weights = self.fixture()
        base = native.decoder
        library = base.library
        ids = torch.tensor([[1, 32, 35, 33, 7]])
        grid = torch.tensor([[1, 2, 2]])
        types = (ids == 35).long()
        original = base.weights['model.embed_tokens.weight'].copy()
        for owner in (*native.weights, *base.weights.values()):
            with self.assertRaises(ValueError):
                owner.flags.writeable = True
        with self.assertRaises(AttributeError):
            native.weights = ()
        # Caller mutation cannot change the pinned vision or text tensors.
        for a in weights.values():
            a.fill(0)
        # Recreate an untouched oracle after mutating its state-dict views.
        oracle, other, _ = self.fixture()
        with torch.inference_mode():
            for pixels in (torch.ones((4, 3)), torch.tensor([[.2, -.4, .7], [.9, .1, -.8], [-.6, .3, .5], [.4, -.2, -.1]])):
                expected = oracle(ids, pixel_values=pixels, image_grid_thw=grid,
                    mm_token_type_ids=types, output_hidden_states=True, output_attentions=True)
                with patch('ctypes.CDLL', side_effect=AssertionError('candidate must share DLL')):
                    actual = native.forward(ids.numpy(), pixel_values=pixels.numpy(), image_grid_thw=grid.numpy())
                text_oracle.Qwen35NumericalTests().compare(actual, expected)
                self.assertIs(native.decoder.library, library)
                self.assertFalse(np.array_equal(native.decoder.weights['model.embed_tokens.weight'][35], original[35]))
                candidate = native.decoder
                native.reset()
                self.assertIs(native.decoder, base)
                self.assertEqual(candidate.position, 0)
                np.testing.assert_array_equal(native.decoder.weights['model.embed_tokens.weight'], original)
                text_oracle.Qwen35NumericalTests().compare(native.forward([1, 3, 7]),
                    oracle(torch.tensor([[1, 3, 7]]), output_hidden_states=True, output_attentions=True))
                native.reset()
        self.assertIs(other.decoder.library, other._base_decoder.library)

    def test_candidate_failure_and_library_finalizer_lifetime(self):
        import _ctypes
        from openagent.adaptive_engine.native_qwen35_vision import _own_library
        _, native, _ = self.fixture()
        original = native.decoder
        with (patch('openagent.adaptive_engine.native_qwen35.NativeQwen35TextDecoder.forward',
                    side_effect=ValueError('injected execution failure')),
              patch('ctypes.CDLL', side_effect=AssertionError('unexpected DLL acquisition')),
              self.assertRaises(ValueError)):
            native.forward([1, 35, 7], pixel_values=np.ones((4, 3), np.float32), image_grid_thw=[[1, 2, 2]])
        self.assertIs(native.decoder, original)
        self.assertEqual(original.position, 0)

        class LibraryOwner:
            _handle = 123

        release_name = 'FreeLibrary' if hasattr(_ctypes, 'FreeLibrary') else 'dlclose'
        with patch.object(_ctypes, release_name) as release:
            owner = LibraryOwner()
            _own_library(owner)
            alias = owner
            del owner
            gc.collect()
            release.assert_not_called()
            del alias
            gc.collect()
            release.assert_called_once_with(123)

    def test_c_abi_rejects_invalid_extents_without_output_commit(self):
        _, native, _ = self.fixture()
        from openagent.adaptive_engine.native_qwen35 import Tensor, tensor
        output = np.full(24, -17, np.float32)
        pixels = np.ones((4, 3), np.float32)
        count = len(native._bindings)
        self.assertNotEqual(native._encode(Tensor(pixels.ctypes.data, 11), native._bindings, count, tensor(output)), 0)
        self.assertNotEqual(native._encode(tensor(pixels), native._bindings, 8, tensor(output)), 0)
        self.assertNotEqual(native._encode(tensor(pixels), native._bindings, count, Tensor(output.ctypes.data, 23)), 0)
        pixels[0, 0] = np.inf
        self.assertNotEqual(native._encode(tensor(pixels), native._bindings, count, tensor(output)), 0)
        np.testing.assert_array_equal(output, np.full(24, -17, np.float32))
        positions = np.full(3, -17, np.int64)
        self.assertNotEqual(native._positions(positions.ctypes.data, 3, 31, 32), 0)
        np.testing.assert_array_equal(positions, [-17, -17, -17])


if __name__ == '__main__':
    unittest.main()
