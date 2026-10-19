import copy
from contextlib import contextmanager
import threading
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.execution_service import ExecutionCancelled
from openagent.adaptive_engine.multimodal_session import MultimodalSessionExecutor
from openagent.adaptive_engine.qwen4_multimodal import Qwen4MultimodalDecoder
from openagent.adaptive_engine.v41_execution.speculative import AutoregressiveDraftProposer


class _DraftModel:
    def __init__(self):
        self.history = ()
        self.logits = None
        self.cancel = None
        self.cancel_source = None
        self.cancel_scope_seen = False

    @contextmanager
    def cancellation_scope(self, cancel):
        old = self.cancel
        self.cancel = cancel
        self.cancel_scope_seen = True
        try:
            yield self
        finally:
            self.cancel = old

    def forward(self, tokens):
        self.history += tuple(tokens)
        self.logits = np.asarray([1.0, 0.0, -1.0])
        if self.cancel_source is not None:
            self.cancel_source.set()
        return np.tile(self.logits, (len(tokens), 1))

    def snapshot(self):
        return (self.history, None if self.logits is None else self.logits.copy())

    def restore(self, state):
        self.history, logits = state
        self.logits = None if logits is None else logits.copy()

    def release_snapshot(self, _state):
        pass

    def reset(self):
        self.history, self.logits = (), None


class _FailingResetDraftModel(_DraftModel):
    def __init__(self):
        super().__init__()
        self.fail_reset = True

    def reset(self):
        # Simulate a native reset that mutates part of its physical state and
        # then reports a failure.
        self.history = ()
        if self.fail_reset:
            self.fail_reset = False
            raise RuntimeError('native reset failed after partial mutation')
        self.logits = None


class SpeculativeCancellationTest(unittest.TestCase):
    def test_failed_draft_reset_quarantines_until_a_later_reset_succeeds(self):
        model = _FailingResetDraftModel()
        proposer = AutoregressiveDraftProposer(model, vocabulary=3, context=8)

        with self.assertRaisesRegex(RuntimeError, 'partial mutation'):
            proposer.reset()

        self.assertTrue(proposer._failed)
        with self.assertRaisesRegex(RuntimeError, 'requires reset or successful restore'):
            proposer((0,), 1)

        proposer.reset()
        self.assertFalse(proposer._failed)
        self.assertEqual(proposer.history, ())
        self.assertIsNone(proposer.logits)

    def test_autoregressive_draft_cancellation_restores_model_and_rng(self):
        model = _DraftModel()
        rng = np.random.default_rng(9)
        proposer = AutoregressiveDraftProposer(
            model, vocabulary=3, context=8, rng=rng)
        event = threading.Event()
        model.cancel_source = event
        rng_before = copy.deepcopy(rng.bit_generator.state)

        with proposer.cancellation_scope(event), self.assertRaises(ExecutionCancelled):
            proposer((0,), 1)

        self.assertTrue(model.cancel_scope_seen)
        self.assertEqual(model.history, ())
        self.assertIsNone(model.logits)
        self.assertEqual(proposer.history, ())
        self.assertIsNone(proposer.logits)
        self.assertEqual(rng.bit_generator.state, rng_before)

    def test_conditional_wrapper_rechecks_cancellation_after_forward(self):
        event = threading.Event()
        decoder = object.__new__(Qwen4MultimodalDecoder)

        def forward(_decoder, _tokens, *, cancelled=None, **_media):
            event.set()
            return {"logits": np.zeros((1, 1, 4), dtype=np.float32)}

        with patch.object(Qwen4MultimodalDecoder, "forward", forward), \
                patch.object(Qwen4MultimodalDecoder, "close", lambda _decoder: None):
            executor = MultimodalSessionExecutor(decoder, max_media_bytes=1024)
            with executor.cancellation_scope(event), self.assertRaises(ExecutionCancelled):
                executor.forward([1])
            with self.assertRaises(ExecutionCancelled):
                executor.forward_media([1], {"pixel_values": np.ones((1,), dtype=np.float32)}, event)
            executor.close()


if __name__ == '__main__':
    unittest.main()
