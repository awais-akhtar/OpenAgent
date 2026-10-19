"""Phase 1 A02 — Native family factory verification.

Tests that all priority model factories are implemented and importable.
Source-level verification; full end-to-end requires real checkpoints (A10).
"""

import unittest


class Phase1A02FactoriesTest(unittest.TestCase):
    """Verify A02 factories exist and are callable."""

    def test_all_factories_importable(self):
        """A02: All 5 priority model factories are importable and callable."""
        from openagent.adaptive_engine.execution_factories import (
            create_qwen35_native_session,
            create_qwen4_native_session,
            create_glm5_native_session,
            create_glm_dsa_native_session,
            create_deepseek_v41_native_session
        )

        factories = [
            create_qwen35_native_session,
            create_qwen4_native_session,
            create_glm5_native_session,
            create_glm_dsa_native_session,
            create_deepseek_v41_native_session,
        ]

        for factory in factories:
            self.assertTrue(callable(factory), f"{factory.__name__} is not callable")
            self.assertIsNotNone(factory.__doc__, f"{factory.__name__} missing docstring")
            self.assertIn('a02', factory.__doc__.lower(), f"{factory.__name__} not marked A02")

    def test_native_checkpoint_loader_exists(self):
        """A02: native_checkpoint loader function exists."""
        from openagent.adaptive_engine import native_checkpoint
        self.assertTrue(hasattr(native_checkpoint, 'read_native_text_checkpoint'))


if __name__ == '__main__':
    unittest.main()
