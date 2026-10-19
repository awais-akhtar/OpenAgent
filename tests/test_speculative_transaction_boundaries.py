import copy
import threading
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.v41_execution.speculative import (
    AutoregressiveDraftProposer,
    DraftProposal,
    SpeculativeDecoder,
)
from openagent.adaptive_engine.execution_service import ExecutionCancelled
from openagent.adaptive_engine.publisher_v41_cli import _sample


class _Target:
    def __init__(self):
        self.history = ()
        self.logits = None
        self.after_forward = None

    def forward(self, tokens):
        self.history += tuple(tokens)
        self.logits = np.asarray([0.0, 10.0, 0.0])
        if self.after_forward is not None:
            self.after_forward()
        return np.tile(self.logits, (len(tokens), 1))

    def snapshot(self):
        return self.history, None if self.logits is None else self.logits.copy()

    def restore(self, state):
        self.history, logits = state
        self.logits = None if logits is None else logits.copy()

    def release_snapshot(self, _state):
        pass

    def reset(self):
        self.history, self.logits = (), None


class _Proposer:
    def __init__(self, tokens):
        self.tokens = tuple(tokens)
        self.rng = np.random.default_rng(91)
        self.history = ()
        self.after_propose = None

    def snapshot(self):
        return self.history

    def restore(self, state):
        self.history = state

    def release_snapshot(self, _state):
        pass

    def reset(self):
        self.history = ()

    def __call__(self, history, count):
        self.rng.random()  # model proposal-side RNG use during the transaction
        tokens = self.tokens[:count]
        self.history = tuple(history) + tokens
        probabilities = np.zeros((len(tokens), 3), dtype=np.float64)
        for row, token in enumerate(tokens):
            probabilities[row, token] = 1.0
        if self.after_propose is not None:
            self.after_propose()
        return DraftProposal(tokens, probabilities)


class _UnprintableFailure(Exception):
    def __str__(self):
        raise AssertionError('failure stringification must not replace the primary exception')

    def __repr__(self):
        raise AssertionError('failure representation must not replace the primary exception')


class SpeculativeTransactionBoundaryTests(unittest.TestCase):
    def _decoder(self, *, context=8, tokens=(1, 1), draft_length=2):
        target = _Target()
        proposer = _Proposer(tokens)
        decoder = SpeculativeDecoder(
            target, proposer, vocabulary=3, context=context, temperature=0,
            draft_length=draft_length, rng=np.random.default_rng(13))
        decoder.prefill([0])
        return decoder, target, proposer

    def test_full_accept_includes_one_bonus_without_rng_drift(self):
        decoder, target, proposer = self._decoder()
        target_rng = copy.deepcopy(decoder.rng.bit_generator.state)

        result = next(decoder.generate(3))

        self.assertEqual(result.tokens, (1, 1, 1))
        self.assertEqual((result.accepted, result.proposed, result.rejected), (2, 2, False))
        self.assertEqual(target.history, (0, 1, 1, 1))
        self.assertEqual(decoder.rng.bit_generator.state, target_rng)
        decoder.close()

    def test_eos_and_context_limit_suppress_bonus_tokens(self):
        decoder, target, _proposer = self._decoder()
        result = next(decoder.generate(3, stop_tokens=(1,)))
        self.assertEqual(result.tokens, (1,))
        self.assertEqual((result.accepted, result.proposed), (1, 2))
        self.assertEqual(target.history, (0, 1))
        decoder.close()

        decoder, target, _proposer = self._decoder(context=2)
        result = next(decoder.generate(3))
        self.assertEqual(result.tokens, (1,))
        self.assertEqual((result.accepted, result.proposed), (1, 1))
        self.assertEqual(target.history, (0, 1))
        decoder.close()

    def test_rejected_proposal_restores_draft_rng_and_replays_target_choice(self):
        decoder, target, proposer = self._decoder(tokens=(2,))
        draft_rng = copy.deepcopy(proposer.rng.bit_generator.state)

        result = next(decoder.generate(1))

        self.assertEqual(result.tokens, (1,))
        self.assertEqual((result.accepted, result.proposed, result.rejected), (0, 1, True))
        self.assertEqual(target.history, (0, 1))
        self.assertEqual(proposer.rng.bit_generator.state, draft_rng)
        decoder.close()

    def test_silent_draft_restore_miss_after_rejection_quarantines_split_owner(self):
        decoder, target, proposer = self._decoder(tokens=(2,))
        proposer.restore = lambda _state: None

        with self.assertRaisesRegex(ValueError, 'proposer history differs from rollback receipt'):
            decoder.step()

        self.assertEqual(decoder.history, (0,))
        self.assertEqual(target.history, (0,))
        self.assertNotEqual(proposer.history, ())
        self.assertTrue(decoder._quarantined)
        decoder.close()

    def test_stochastic_accept_and_bonus_advance_only_the_target_draw_sequence(self):
        target = _Target()
        proposer = _Proposer((1, 1))
        target_rng = np.random.default_rng(13)
        decoder = SpeculativeDecoder(target, proposer, vocabulary=3, context=8,
            temperature=1.0, draft_length=2, rng=target_rng)
        decoder.prefill([0])
        initial = copy.deepcopy(target_rng.bit_generator.state)
        expected = np.random.default_rng()
        expected.bit_generator.state = copy.deepcopy(initial)
        probabilities = np.exp(np.asarray([0.0, 10.0, 0.0]) - 10.0)
        probabilities /= probabilities.sum()
        expected.random()  # first proposal acceptance
        expected.random()  # second proposal acceptance
        expected.choice(3, p=probabilities)  # target bonus token

        result = decoder.step()

        self.assertEqual(result.tokens, (1, 1, 1))
        self.assertEqual((result.accepted, result.proposed, result.rejected), (2, 2, False))
        self.assertEqual(target_rng.bit_generator.state, expected.bit_generator.state)
        self.assertEqual(proposer.history, (0, 1, 1))
        decoder.close()

    def test_stochastic_rejection_keeps_target_draws_and_rewinds_draft_rng(self):
        target = _Target()
        proposer = _Proposer((2,))
        target_rng = np.random.default_rng(13)
        decoder = SpeculativeDecoder(target, proposer, vocabulary=3, context=8,
            temperature=1.0, draft_length=1, rng=target_rng)
        decoder.prefill([0])
        draft_rng_before = copy.deepcopy(proposer.rng.bit_generator.state)
        expected = np.random.default_rng()
        expected.bit_generator.state = copy.deepcopy(target_rng.bit_generator.state)
        probabilities = np.exp(np.asarray([0.0, 10.0, 0.0]) - 10.0)
        probabilities /= probabilities.sum()
        expected.random()  # target rejects the low-probability draft token
        residual = probabilities.copy()
        residual[2] = 0.0  # q assigns probability one to token 2
        residual /= residual.sum()
        expected.choice(3, p=residual)

        result = decoder.step()

        self.assertEqual(result.tokens, (1,))
        self.assertEqual((result.accepted, result.proposed, result.rejected), (0, 1, True))
        self.assertEqual(target.history, (0, 1))
        self.assertEqual(target_rng.bit_generator.state, expected.bit_generator.state)
        self.assertEqual(proposer.history, ())
        self.assertEqual(proposer.rng.bit_generator.state, draft_rng_before)
        decoder.close()

    def test_cancellation_after_proposer_mutation_rolls_back_both_owners_and_rngs(self):
        decoder, target, proposer = self._decoder()
        target_before = target.history
        target_rng_before = copy.deepcopy(decoder.rng.bit_generator.state)
        draft_rng_before = copy.deepcopy(proposer.rng.bit_generator.state)
        event = threading.Event()
        proposer.after_propose = event.set

        with self.assertRaisesRegex(ExecutionCancelled, 'speculative transaction cancelled'):
            decoder.step(cancelled=event.is_set)

        self.assertEqual(target.history, target_before)
        self.assertEqual(proposer.history, ())
        self.assertEqual(decoder.history, (0,))
        self.assertEqual(decoder.rng.bit_generator.state, target_rng_before)
        self.assertEqual(proposer.rng.bit_generator.state, draft_rng_before)
        decoder.close()

    def test_cancellation_after_target_verification_rolls_back_both_owners_and_rngs(self):
        decoder, target, proposer = self._decoder()
        target_before = target.history
        target_rng_before = copy.deepcopy(decoder.rng.bit_generator.state)
        draft_rng_before = copy.deepcopy(proposer.rng.bit_generator.state)
        event = threading.Event()
        target.after_forward = event.set

        with self.assertRaisesRegex(ExecutionCancelled, 'speculative transaction cancelled'):
            decoder.step(cancelled=event.is_set)

        self.assertEqual(target.history, target_before)
        self.assertEqual(proposer.history, ())
        self.assertEqual(decoder.history, (0,))
        self.assertEqual(decoder.rng.bit_generator.state, target_rng_before)
        self.assertEqual(proposer.rng.bit_generator.state, draft_rng_before)
        decoder.close()

    def test_unprintable_target_rollback_failure_preserves_cancellation_and_restores_draft(self):
        decoder, target, proposer = self._decoder()
        event = threading.Event()
        target.after_forward = event.set
        target.restore = lambda _state: (_ for _ in ()).throw(_UnprintableFailure())

        with self.assertRaisesRegex(ExecutionCancelled, 'speculative transaction cancelled'):
            decoder.step(cancelled=event.is_set)

        self.assertTrue(decoder._quarantined)
        self.assertEqual(proposer.history, ())
        decoder.close()

    def test_unprintable_secondary_reset_failure_preserves_first_reset_error(self):
        decoder, target, proposer = self._decoder()
        target.reset = lambda: (_ for _ in ()).throw(ValueError('target reset failed'))
        proposer.reset = lambda: (_ for _ in ()).throw(_UnprintableFailure())

        with self.assertRaisesRegex(ValueError, 'target reset failed') as raised:
            decoder.reset()

        self.assertEqual(raised.exception.args, ('target reset failed',))
        self.assertTrue(decoder._quarantined)
        decoder.close()

    def test_temperature_overflow_and_nonfinite_values_fail_as_validation_errors(self):
        invalid = (10**1000, float('inf'), float('nan'), True, -1)
        for temperature in invalid:
            with self.subTest(temperature=type(temperature).__name__):
                with self.assertRaisesRegex(ValueError, 'temperature'):
                    AutoregressiveDraftProposer(
                        _Target(), vocabulary=3, context=8, temperature=temperature)
                with self.assertRaisesRegex(ValueError, 'temperature'):
                    SpeculativeDecoder(
                        _Target(), _Proposer((1,)), vocabulary=3, context=8,
                        temperature=temperature)
                with self.assertRaisesRegex(ValueError, 'temperature'):
                    _sample(np.asarray([0.0, 1.0, 0.0]), temperature,
                            np.random.default_rng(2))

    def test_prefill_requires_a_concrete_target_rollback_receipt(self):
        decoder, target, _proposer = self._decoder()
        target.reset()
        decoder.history, decoder.next_logits = (), None
        target.snapshot = lambda: None

        with patch.object(target, 'forward', side_effect=AssertionError('must not mutate')):
            with self.assertRaisesRegex(ValueError, 'target snapshot.*rollback receipt'):
                decoder.prefill([0])

        decoder.close()

    def test_prefill_quarantines_when_target_restore_silently_misses_boundary(self):
        decoder, target, _proposer = self._decoder()
        target.reset()
        decoder.history, decoder.next_logits = (), None
        event = threading.Event()
        target.after_forward = event.set
        target.restore = lambda _state: None

        with self.assertRaisesRegex(ExecutionCancelled, 'speculative transaction cancelled'):
            decoder.prefill([0], cancelled=event.is_set)

        self.assertEqual(decoder.history, ())
        self.assertIsNone(decoder.next_logits)
        self.assertEqual(target.history, (0,))
        self.assertTrue(decoder._quarantined)
        decoder.close()

    def test_step_requires_concrete_draft_receipt_before_target_mutation(self):
        decoder, target, proposer = self._decoder()
        proposer.snapshot = lambda: None
        before = target.history

        with self.assertRaisesRegex(ValueError, 'draft snapshot.*rollback receipt'):
            decoder.step()

        self.assertEqual(target.history, before)
        decoder.close()

    def test_restore_requires_current_owner_receipts_before_changing_state(self):
        decoder, target, _proposer = self._decoder()
        receipt = decoder.snapshot()
        before = target.history
        target.snapshot = lambda: None

        with self.assertRaisesRegex(ValueError, 'target snapshot.*rollback receipt'):
            decoder.restore(receipt)

        self.assertEqual(target.history, before)
        decoder.release_snapshot(receipt)
        decoder.close()

    def test_snapshot_restore_reconciles_target_draft_and_both_rngs(self):
        target = _Target()
        proposer = _Proposer((1, 1))
        target_rng = np.random.default_rng(31)
        decoder = SpeculativeDecoder(target, proposer, vocabulary=3, context=8,
            temperature=1.0, draft_length=2, rng=target_rng)
        decoder.prefill([0])
        receipt = decoder.snapshot()

        decoder.step()
        self.assertNotEqual(target_rng.bit_generator.state, receipt.rng_state)
        self.assertNotEqual(proposer.rng.bit_generator.state, receipt.proposal_rng_state)
        decoder.restore(receipt)

        self.assertEqual(decoder.history, receipt.history)
        self.assertEqual(target.history, receipt.target[0])
        self.assertEqual(proposer.history, receipt.draft)
        self.assertEqual(target_rng.bit_generator.state, receipt.rng_state)
        self.assertEqual(proposer.rng.bit_generator.state, receipt.proposal_rng_state)
        decoder.release_snapshot(receipt)
        decoder.close()

    def test_restore_rejects_silent_draft_miss_and_rolls_back_both_owners(self):
        decoder, target, proposer = self._decoder()
        receipt = decoder.snapshot()
        decoder.step()
        committed = target.history
        committed_draft = proposer.history
        original_restore = proposer.restore
        calls = []

        def miss_once(state):
            calls.append(state)
            if len(calls) == 1:
                return None
            return original_restore(state)

        proposer.restore = miss_once
        with self.assertRaisesRegex(ValueError,
                                    'speculative proposer history differs from rollback receipt'):
            decoder.restore(receipt)

        self.assertEqual(target.history, committed)
        self.assertEqual(proposer.history, committed_draft)
        self.assertEqual(decoder.history, committed)
        self.assertFalse(decoder._quarantined)
        self.assertEqual(len(calls), 2)
        decoder.release_snapshot(receipt)
        decoder.close()

    def test_autoregressive_proposer_rejects_null_physical_model_receipt(self):
        model = _Target()
        model.snapshot = lambda: None
        proposer = AutoregressiveDraftProposer(model, vocabulary=3, context=8)

        with self.assertRaisesRegex(ValueError, 'draft model snapshot.*rollback receipt'):
            proposer.snapshot()

        proposer.close()


if __name__ == '__main__':
    unittest.main()
