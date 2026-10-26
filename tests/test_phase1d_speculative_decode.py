from __future__ import annotations

import unittest
import copy
import math

import numpy as np

from openagent.adaptive_engine.grammar_speculative_decoder import (
    GrammarFSM,
    GrammarMode,
    DraftToken,
    GrammarSpeculativeDecoder,
    SpeculativeDraftBuffer,
    SpeculativeVerifier,
    TokenMask,
)


class Phase1DSpeculativeGrammarTests(unittest.TestCase):
    def test_generation_request_validation_is_independent_of_token_budget(self) -> None:
        decoder = GrammarSpeculativeDecoder(
            main_verify_fn=lambda _context, count: np.zeros((count, 2), dtype=np.float32),
            vocab_size=2,
            id_to_text={0: "a", 1: "b"},
        )

        for invoke in (
            lambda **kwargs: decoder.generate([], max_new_tokens=0, **kwargs),
            lambda **kwargs: list(decoder.stream([], max_new_tokens=0, **kwargs)),
        ):
            with self.subTest(invoke=invoke), self.assertRaisesRegex(
                ValueError, "stop token"
            ):
                invoke(stop_token_id=2)

        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            decoder.generate([], max_new_tokens=-1)
        with self.assertRaisesRegex(ValueError, "context IDs"):
            list(decoder.stream([2], max_new_tokens=0))
        self.assertEqual(decoder.metrics()["spec_total_generated"], 0)

    def test_vocabulary_mask_probes_without_advancing_target_fsm(self) -> None:
        fsm = GrammarFSM(GrammarMode.JSON)
        logits = np.zeros(4, dtype=np.float32)

        masked = TokenMask(4).apply_fsm(
            logits, fsm, {0: "{", 1: "[", 2: "x", 3: " "},
        )

        self.assertEqual(fsm.stats()["fsm_steps"], 0)
        self.assertEqual(fsm.stats()["fsm_rejects"], 0)
        self.assertGreaterEqual(int(np.count_nonzero(masked > -1e8)), 2)
        self.assertTrue(fsm.step("{"))
        self.assertEqual(fsm.stats()["fsm_steps"], 1)

    def test_draft_fsm_copy_does_not_advance_verifier_boundary(self) -> None:
        target_fsm = GrammarFSM(GrammarMode.JSON)
        draft_fsm = target_fsm.clone()
        buffer = SpeculativeDraftBuffer()

        buffer.fill(
            lambda _ids: np.asarray([1.0], dtype=np.float32),
            [], 1, draft_fsm, TokenMask(1), {0: "{"},
        )

        self.assertEqual(len(buffer.tokens()), 1)
        self.assertEqual(draft_fsm.stats()["fsm_steps"], 1)
        self.assertEqual(target_fsm.stats()["fsm_steps"], 0)
        self.assertEqual(target_fsm.stats()["fsm_rejects"], 0)
        np.testing.assert_array_equal(buffer.tokens()[0].probabilities, np.ones(1))

    def test_single_valid_grammar_token_is_not_unmasked(self) -> None:
        fsm = GrammarFSM(GrammarMode.JSON)
        logits = np.asarray([0.0, 10.0], dtype=np.float32)

        masked = TokenMask(2).apply_fsm(logits, fsm, {0: "{"})

        self.assertEqual(float(masked[0]), 0.0)
        self.assertLess(float(masked[1]), -1e8)
        self.assertEqual(fsm.stats()["fsm_steps"], 0)

    def test_empty_grammar_vocabulary_fails_closed(self) -> None:
        fsm = GrammarFSM(GrammarMode.JSON)

        with self.assertRaisesRegex(ValueError, "no token"):
            TokenMask(1).apply_fsm(np.zeros(1, dtype=np.float32), fsm, {0: "x"})

    def test_unknown_grammar_mode_is_rejected_instead_of_becoming_unconstrained(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported grammar mode"):
            GrammarFSM("json-ish")

    def test_invalid_regex_is_rejected_instead_of_falling_back_to_free(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid regex grammar pattern"):
            GrammarFSM(GrammarMode.REGEX, "[")

    def test_json_fsm_accepts_nested_values_and_rejects_bad_punctuation(self) -> None:
        fsm = GrammarFSM(GrammarMode.JSON)
        for piece in ('{', '"key"', ':', '[1,', '2,true,null]', '}'):
            self.assertTrue(fsm.step(piece), piece)
        self.assertTrue(fsm.is_complete())
        self.assertFalse(fsm.step(" false"))

        malformed = GrammarFSM(GrammarMode.JSON)
        for piece in ('{', '"key"'):
            self.assertTrue(malformed.step(piece))
        self.assertFalse(malformed.step(" 1"))  # object key requires a colon
        self.assertTrue(malformed.step(":"))  # rejected token was transactional
        self.assertTrue(malformed.step("1}"))
        self.assertTrue(malformed.is_complete())

    def test_json_number_prefixes_and_terminal_state_are_exact(self) -> None:
        for valid_number in ("0", "-1", "2.5", "1e+3"):
            fsm = GrammarFSM(GrammarMode.JSON)
            self.assertTrue(fsm.step(valid_number))
            self.assertTrue(fsm.is_complete(), valid_number)
        for invalid_number in ("01",):
            fsm = GrammarFSM(GrammarMode.JSON)
            self.assertFalse(fsm.step(invalid_number), invalid_number)
            self.assertFalse(fsm.is_complete(), invalid_number)
        for incomplete_number in ("-", "1.", "1e+"):
            fsm = GrammarFSM(GrammarMode.JSON)
            self.assertTrue(fsm.step(incomplete_number), incomplete_number)
            self.assertFalse(fsm.is_complete(), incomplete_number)

    def test_json_stop_token_is_masked_until_document_is_complete(self) -> None:
        mask = TokenMask(2)
        fsm = GrammarFSM(GrammarMode.JSON)
        logits = np.asarray([10.0, 0.0], dtype=np.float32)
        masked = mask.apply_fsm(logits, fsm, {0: "", 1: "{"}, stop_token_id=0)
        self.assertLess(float(masked[0]), -1e8)
        self.assertEqual(float(masked[1]), 0.0)

        self.assertTrue(fsm.step("{}"))
        masked = mask.apply_fsm(logits, fsm, {0: "", 1: "{"}, stop_token_id=0)
        self.assertEqual(float(masked[0]), 10.0)

    def test_free_mode_still_rejects_wrong_target_or_draft_vocabulary_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured vocabulary"):
            TokenMask(2).apply_fsm(np.zeros(3, dtype=np.float32), GrammarFSM(), {})

    def test_free_mode_rejects_nonfinite_unmasked_logits(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured vocabulary"):
            TokenMask(2).apply_fsm(
                np.asarray([0.0, np.nan], dtype=np.float32), GrammarFSM(), {},
            )

    def test_target_cancellation_propagates_without_committing_grammar_state(self) -> None:
        from threading import Event

        cancellation = Event()

        def verify(_context, _count, *, cancelled=None):
            cancellation.set()
            if cancelled is not None and cancelled():
                raise InterruptedError("target cancelled")
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        decoder = GrammarSpeculativeDecoder(
            main_verify_fn=verify,
            vocab_size=2,
            id_to_text={0: "{", 1: "x"},
        )
        decoder.set_grammar(GrammarMode.JSON)

        with self.assertRaisesRegex(InterruptedError, "target cancelled"):
            decoder.generate([], max_new_tokens=1, cancelled=cancellation.is_set)

        self.assertEqual(decoder._fsm.stats()["fsm_steps"], 0)
        self.assertEqual(decoder.metrics()["spec_total_generated"], 0)

    def test_stream_cancellation_preserves_only_published_grammar_prefix(self) -> None:
        from threading import Event

        cancellation = Event()
        decoder = GrammarSpeculativeDecoder(
            main_verify_fn=lambda _context, _count: np.zeros((1, 1), dtype=np.float32),
            vocab_size=1,
            id_to_text={0: "x"},
        )
        stream = decoder.stream([], max_new_tokens=2, cancelled=cancellation.is_set)

        self.assertEqual(next(stream), 0)
        cancellation.set()
        with self.assertRaises(InterruptedError):
            next(stream)

        self.assertEqual(decoder._fsm.stats()["fsm_steps"], 1)
        self.assertEqual(decoder.metrics()["spec_total_generated"], 1)

    def test_stream_owns_decoder_state_until_closed(self) -> None:
        from threading import Event, Thread

        decoder = GrammarSpeculativeDecoder(
            main_verify_fn=lambda _context, _count: np.zeros((1, 1), dtype=np.float32),
            vocab_size=1,
            id_to_text={0: "x"},
        )
        stream = decoder.stream([], max_new_tokens=2)
        self.assertEqual(next(stream), 0)

        setter_started = Event()
        grammar_changed = Event()

        def change_grammar() -> None:
            setter_started.set()
            decoder.set_grammar(GrammarMode.JSON)
            grammar_changed.set()

        setter = Thread(target=change_grammar)
        setter.start()
        self.assertTrue(setter_started.wait(1), "grammar update thread did not start")
        self.assertFalse(grammar_changed.wait(0.05), "grammar changed during an active stream")

        stream.close()
        setter.join(timeout=1)
        self.assertFalse(setter.is_alive(), "grammar update remained blocked after stream close")
        self.assertTrue(grammar_changed.is_set())
        self.assertEqual(decoder._fsm.mode, GrammarMode.JSON)

    def test_speculative_acceptance_stops_at_eos(self) -> None:
        verifier = SpeculativeVerifier()
        fsm = GrammarFSM()

        accepted, _ = verifier.verify(
            np.zeros((2, 1), dtype=np.float32),
            [DraftToken(0, 0.0, np.ones(1)), DraftToken(0, 0.0, np.ones(1))],
            [], fsm, TokenMask(1), {0: "<eos>"}, stop_token_id=0,
        )

        self.assertEqual(accepted, [0])
        # The stop token is a control token, not grammar content.
        self.assertEqual(fsm.stats()["fsm_steps"], 0)

    def test_rejection_samples_from_full_target_minus_draft_distribution(self) -> None:
        verifier = SpeculativeVerifier()
        fsm = GrammarFSM()
        token = DraftToken(0, math.log(0.9), np.asarray([0.9, 0.1]), "a")

        accepted, _ = verifier.verify(
            np.log(np.asarray([[0.1, 0.9]])), [token], [], fsm,
            TokenMask(2), {0: "a", 1: "b"}, rng=np.random.default_rng(0),
        )

        self.assertEqual(accepted, [1])
        self.assertEqual(verifier.stats()["spec_rejected"], 1)

    def test_low_probability_valid_draft_uses_exact_acceptance_rule(self) -> None:
        verifier = SpeculativeVerifier()
        token = DraftToken(0, math.log(0.9), np.asarray([0.9, 0.1]), "rare")
        acceptance_probability = 0.01 / 0.9
        seed = next(
            value for value in range(10000)
            if np.random.default_rng(value).random() < acceptance_probability
        )

        accepted, _ = verifier.verify(
            np.log(np.asarray([[0.01, 0.99]])), [token], [], GrammarFSM(),
            TokenMask(2), {0: "rare", 1: "common"}, rng=np.random.default_rng(seed),
        )

        self.assertEqual(accepted, [0])
        self.assertEqual(verifier.stats()["spec_accepted"], 1)
        self.assertEqual(verifier.stats()["spec_rejected"], 0)

    def test_cancelled_verification_rolls_back_fsm_and_rng(self) -> None:
        verifier = SpeculativeVerifier()
        fsm = GrammarFSM()
        rng = np.random.default_rng(82)
        rng_before = copy.deepcopy(rng.bit_generator.state)
        checks = 0

        def cancel_after_first_candidate():
            nonlocal checks
            checks += 1
            return checks >= 2

        with self.assertRaises(InterruptedError):
            verifier.verify(
                np.zeros((2, 1), dtype=np.float32),
                [DraftToken(0, 0.0, np.ones(1)), DraftToken(0, 0.0, np.ones(1))],
                [], fsm, TokenMask(1), {0: "x"},
                cancelled=cancel_after_first_candidate, rng=rng,
            )

        self.assertEqual(fsm.stats()["fsm_steps"], 0)
        self.assertEqual(rng.bit_generator.state, rng_before)
        self.assertEqual(verifier.stats()["spec_accepted"], 0)

    def test_all_accepted_draft_emits_target_bonus_when_budget_allows(self) -> None:
        requested_rows = []

        def verify(_context, count):
            requested_rows.append(count)
            return np.zeros((count, 1), dtype=np.float32)

        decoder = GrammarSpeculativeDecoder(
            main_verify_fn=verify,
            vocab_size=1,
            id_to_text={0: "x"},
            draft_fn=lambda _context: np.zeros(1, dtype=np.float32),
            draft_len=2,
        )

        self.assertEqual(decoder.generate([], max_new_tokens=3), [0, 0, 0])
        self.assertEqual(requested_rows, [3])

    def test_json_grammar_commits_bonus_token_without_stop_id(self) -> None:
        def verify(_context, count):
            rows = np.asarray(
                [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]],
                dtype=np.float32,
            )
            return rows[:count]

        def draft(context):
            return np.asarray(
                [10.0, -10.0, -10.0] if not context else [-10.0, 10.0, -10.0],
                dtype=np.float32,
            )

        decoder = GrammarSpeculativeDecoder(
            main_verify_fn=verify,
            vocab_size=3,
            id_to_text={0: "[", 1: "1", 2: "]"},
            draft_fn=draft,
            draft_len=2,
        )
        decoder.set_grammar(GrammarMode.JSON)

        self.assertEqual(decoder.generate([], max_new_tokens=3), [0, 1, 2])
        self.assertTrue(decoder._fsm.is_complete())
        self.assertEqual(decoder._fsm.stats()["fsm_steps"], 3)

    def test_full_draft_distributions_require_explicit_bounded_memory(self) -> None:
        with self.assertRaisesRegex(MemoryError, "configured budget"):
            GrammarSpeculativeDecoder(
                main_verify_fn=lambda _context, _count: np.zeros((1, 8), dtype=np.float32),
                vocab_size=8,
                id_to_text={index: str(index) for index in range(8)},
                draft_fn=lambda _context: np.zeros(8, dtype=np.float32),
                draft_len=2,
                max_draft_bytes=64,
            )

    def test_seeded_speculative_target_and_draft_rngs_are_reproducible(self) -> None:
        def verify(_context, count):
            return np.tile(np.asarray([0.0, 1.0, -1.0], dtype=np.float32), (count, 1))

        def draft(_context):
            return np.asarray([0.0, 1.0, -1.0], dtype=np.float32)

        first = GrammarSpeculativeDecoder(
            verify, 3, {0: "a", 1: "b", 2: "c"}, draft_fn=draft, draft_len=2, seed=731,
        )
        second = GrammarSpeculativeDecoder(
            verify, 3, {0: "a", 1: "b", 2: "c"}, draft_fn=draft, draft_len=2, seed=731,
        )

        self.assertEqual(first.generate([], max_new_tokens=12), second.generate([], max_new_tokens=12))

    def test_cancelled_generation_does_not_commit_target_rng(self) -> None:
        def verify(_context, _count):
            return np.asarray([[0.0, 1.0, -1.0]], dtype=np.float32)

        interrupted = GrammarSpeculativeDecoder(
            verify, 3, {0: "a", 1: "b", 2: "c"}, seed=917,
        )
        control = GrammarSpeculativeDecoder(
            verify, 3, {0: "a", 1: "b", 2: "c"}, seed=917,
        )
        checks = 0

        def cancel_after_sample():
            nonlocal checks
            checks += 1
            return checks >= 5

        with self.assertRaises(InterruptedError):
            interrupted.generate([], max_new_tokens=1, cancelled=cancel_after_sample)

        self.assertEqual(interrupted.generate([], max_new_tokens=1), control.generate([], max_new_tokens=1))


if __name__ == "__main__":
    unittest.main()
