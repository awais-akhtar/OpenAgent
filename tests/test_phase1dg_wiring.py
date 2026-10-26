from __future__ import annotations

import unittest

import numpy as np

from openagent.adaptive_engine.cpu_kernel_adapter import Phase1DOptimizedSession
from openagent.adaptive_engine.grammar_speculative_decoder import GrammarSpeculativeDecoder
from openagent.adaptive_engine.aot_kernel_adapter import Phase1EOptimizedSession
from openagent.adaptive_engine.draft_model_adapter import Phase1GOptimizedSession
from openagent.adaptive_engine.draft_model import (
    PrefixCacheDraftModel,
    SmallModelDraftAdapter,
)


class _Session:
    def __init__(self, base_session=None, *, verifier=True):
        self.base_session = base_session
        if verifier:
            self.batch_logits = lambda _context, count: np.zeros((count, 4), dtype=np.float32)


class Phase1DSpeculativeWiringTests(unittest.TestCase):
    def test_phase1d_requested_speculation_fails_closed_without_verifier(self):
        with self.assertRaisesRegex(ValueError, "callable batch_logits verifier"):
            Phase1DOptimizedSession(_Session(verifier=False), enable_speculative_decode=True)

    def test_phase1d_grammar_cannot_be_changed_after_close(self):
        class ClosableSession(_Session):
            def close(self):
                pass

        session = Phase1DOptimizedSession(
            ClosableSession(),
            enable_speculative_placement=False,
            enable_speculative_decode=True,
        )
        session.close()
        with self.assertRaisesRegex(RuntimeError, "session is closed"):
            session.set_grammar("json")

    def test_phase1g_installs_draft_on_the_decoder_attribute_it_reads(self):
        phase1d = _Session()
        phase1d._spec_decoder = GrammarSpeculativeDecoder(
            phase1d.batch_logits,
            4,
            {index: str(index) for index in range(4)},
        )
        phase1e = _Session(base_session=phase1d)

        phase1g = Phase1GOptimizedSession(
            phase1e,
            vocab_size=4,
            enable_prefix_cache=False,
            enable_speculative_decode=True,
        )

        self.assertIs(phase1d._spec_decoder.draft_fn, phase1g._draft_selector.draft_fn)

    def test_phase1g_rejects_unwired_decoder_instead_of_silently_falling_back(self):
        with self.assertRaisesRegex(ValueError, "requires an enabled Phase 1D decoder"):
            Phase1GOptimizedSession(
                _Session(),
                vocab_size=4,
                enable_prefix_cache=False,
                enable_speculative_decode=True,
            )

    def test_phase1g_rejects_draft_target_vocabulary_mismatch(self):
        phase1d = _Session()
        phase1d._spec_decoder = GrammarSpeculativeDecoder(
            phase1d.batch_logits,
            4,
            {index: str(index) for index in range(4)},
        )
        with self.assertRaisesRegex(ValueError, "vocabulary does not match"):
            Phase1GOptimizedSession(
                _Session(base_session=phase1d),
                vocab_size=5,
                enable_prefix_cache=False,
                enable_speculative_decode=True,
            )

    def test_small_model_draft_receives_cancellation_and_does_not_swallow_it(self):
        from threading import Event

        cancelled = Event()
        received = []
        cancelled_fn = cancelled.is_set

        class DraftSession:
            def generate_single(self, _tokens, *, cancelled=None):
                received.append(cancelled)
                raise InterruptedError("draft backend cancelled")

        adapter = SmallModelDraftAdapter(DraftSession(), vocab_size=2)
        with self.assertRaisesRegex(InterruptedError, "draft backend cancelled"):
            adapter([1], cancelled=cancelled_fn)
        self.assertEqual(len(received), 1)
        self.assertIs(received[0], cancelled_fn)

    def test_prefix_draft_forwards_cancellation_to_its_fallback(self):
        received = []

        def fallback(_tokens, *, cancelled=None):
            received.append(cancelled)
            return np.zeros(2, dtype=np.float32)

        adapter = PrefixCacheDraftModel(vocab_size=2, fallback_fn=fallback)
        adapter([1], cancelled=lambda: False)
        self.assertEqual(len(received), 1)
        self.assertIsNotNone(received[0])

    def test_close_waits_for_active_wrapped_stream_to_release_its_session(self):
        from threading import Event, Thread

        closed = Event()
        close_started = Event()

        class ActiveSession:
            def generate(self, *args, **kwargs):
                return []

            def stream(self, *args, **kwargs):
                yield 1
                yield 2

            def close(self):
                closed.set()

        phase1d = Phase1DOptimizedSession(
            ActiveSession(), enable_speculative_placement=False,
        )
        phase1e = Phase1EOptimizedSession(phase1d, enable_arc_gpu=False)
        phase1g = Phase1GOptimizedSession(
            phase1e, vocab_size=4, enable_speculative_decode=False,
        )
        stream = phase1g.stream()
        self.assertEqual(next(stream), 1)

        def close_session():
            close_started.set()
            phase1g.close()

        closer = Thread(target=close_session)
        closer.start()
        self.assertTrue(close_started.wait(1))
        self.assertFalse(closed.wait(0.05), "session closed while stream still owned it")

        stream.close()
        closer.join(timeout=1)
        self.assertFalse(closer.is_alive(), "close did not resume after stream released ownership")
        self.assertTrue(closed.is_set())


if __name__ == "__main__":
    unittest.main()
