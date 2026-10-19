"""The public Segment/Edge ABI accepts only exact cancellation wire values."""
import unittest
from pathlib import Path
from unittest.mock import Mock

from openagent.adaptive_engine.native_family_public_text import _Cancellation
from openagent.adaptive_engine.segment_edge_runtime import _cancel_requested


class SegmentEdgeCancellationContractTests(unittest.TestCase):
    def test_exact_zero_and_one_are_forwarded_with_user_data(self):
        for value, expected in ((0, False), (1, True)):
            callback = Mock(return_value=value)
            self.assertIs(_cancel_requested(callback, "request"), expected)
            callback.assert_called_once_with("request")

    def test_non_wire_values_are_rejected(self):
        for value in (-1, 2, True, False, None, "1"):
            with self.subTest(value=value):
                callback = Mock(return_value=value)
                with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
                    _cancel_requested(callback, None)

    def test_missing_callback_means_not_cancelled(self):
        self.assertIs(_cancel_requested(None, None), False)

    def test_python_cancel_predicate_is_normalized_before_abi_validation(self):
        self.assertIs(_Cancellation(lambda: True).is_set(), True)
        self.assertIs(_Cancellation(lambda: False).is_set(), False)

    def test_native_qwen4_public_callbacks_all_use_the_checked_guard(self):
        root = Path(__file__).resolve().parents[1] / "openagent" / "adaptive_engine" / "native"
        adapters = (root / "qwen4_public_adapters.c").read_text(encoding="utf-8")
        sessions = (root / "qwen4_public_session.c").read_text(encoding="utf-8")
        internal = (root / "qwen4_public_internal.h").read_text(encoding="utf-8")

        self.assertIn("oa_q4_cancel_guard", internal)
        self.assertIn("static inline int oa_q4_checked_cancel(void *opaque)", internal)
        self.assertIn("value!=0&&value!=1", internal)
        self.assertIn("if(guard->invalid)return 1", internal)
        self.assertNotIn("r->cancel(r->cancel_data)", adapters)
        self.assertNotIn("r->cancel(r->cancel_data)", sessions)
        self.assertIn("r->cancel?oa_q4_checked_cancel:NULL", adapters)
        self.assertIn("r->cancel?oa_q4_checked_cancel:NULL", sessions)


if __name__ == "__main__":
    unittest.main()
