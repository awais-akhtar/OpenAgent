"""Restore caller precision settings after comparison success or failure."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class ValidationPrecisionScopeTests(unittest.TestCase):
    def test_precision_state_restored_on_exception(self):
        import torch
        from openagent.adaptive_engine.wholemodel_validation.adapters import RandomState
        original = (torch.get_float32_matmul_precision(),
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled())
        scope = RandomState(97, torch)
        with self.assertRaisesRegex(RuntimeError, 'comparison failed'):
            with scope.activate():
                self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
                self.assertFalse(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                raise RuntimeError('comparison failed')
        self.assertEqual((torch.get_float32_matmul_precision(),
                          torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                          torch.are_deterministic_algorithms_enabled(),
                          torch.is_deterministic_algorithms_warn_only_enabled()), original)

    def test_independent_rng_scopes_preserve_caller_rng(self):
        import torch
        from openagent.adaptive_engine.wholemodel_validation.adapters import RandomState
        before = torch.random.get_rng_state().clone()
        first, second = RandomState(73, torch), RandomState(73, torch)
        with first.activate():
            expected = torch.rand(16)
        with second.activate():
            actual = torch.rand(16)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
