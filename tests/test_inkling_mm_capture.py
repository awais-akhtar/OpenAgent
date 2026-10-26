import numpy as np
from types import SimpleNamespace
import unittest
from threading import RLock
from unittest.mock import Mock

from openagent.adaptive_engine.device_capture import CaptureError
from openagent.adaptive_engine.inkling_mm_capture import _route_ids, _sample_verify


class _FakeMTP:
    depth = 2


class _FakeCandidate:
    components = SimpleNamespace(text_config={"vocab_size": 3, "unpadded_vocab_size": 3})
    mtp = _FakeMTP()

    def __init__(self, proposal_rows, target_rows):
        self.proposal_rows = np.asarray(proposal_rows, dtype=np.float64)
        self.target_rows = [np.asarray(row, dtype=np.float64) for row in target_rows]
        self.forward_count = 0
        self.seen_rng = None

    def snapshot(self):
        return (self.forward_count,)

    def restore(self, receipt):
        self.forward_count = receipt[0]

    def release_snapshot(self, receipt):
        return None

    def draft(self, ids, hidden, *, rng, return_distributions, cancelled=None):
        self.seen_rng = rng
        probs = self.proposal_rows
        tokens = []
        for row in probs:
            tokens.append(int(rng.choice(len(row), p=row)))
        return tuple(tokens), np.log(probs), probs.copy()

    def forward(self, ids, *, cancelled=None):
        row = self.target_rows[min(self.forward_count + 1, len(self.target_rows) - 1)]
        self.forward_count += 1
        return SimpleNamespace(logits=np.log(row)[None, None, :],
                               last_hidden_state=np.zeros((1, 1, 1), dtype=np.float32))


def _output(probabilities):
    return SimpleNamespace(logits=np.log(np.asarray(probabilities))[None, None, :],
                           last_hidden_state=np.zeros((1, 2, 1), dtype=np.float32))


class InklingMMSamplerTests(unittest.TestCase):
    def test_eos_is_not_forwarded_and_has_no_bonus(self):
        q=np.asarray([[.999,.0005,.0005],[.999,.0005,.0005]])
        candidate=_FakeCandidate(q,[q[0],q[0]])
        candidate.components=SimpleNamespace(text_config={'vocab_size':3,'unpadded_vocab_size':3,'eos_token_id':0})
        result=_sample_verify(candidate,[1,2],_output(q[0]),
                              np.random.default_rng(1),np.random.default_rng(2))
        self.assertEqual(result['finish_reason'],'eos')
        self.assertEqual(result['accepted'],1)
        self.assertEqual(result['bonus'],0)
        self.assertEqual(candidate.forward_count,0)

    def test_context_limit_stops_proposals_and_rolls_physical_state_back(self):
        class Bounded(_FakeCandidate):
            context_tokens=3
            @property
            def position(self):return 2+self.forward_count
        q=np.asarray([[.8,.1,.1],[.1,.8,.1]])
        candidate=Bounded(q,[q[0],q[1],q[0]])
        result=_sample_verify(candidate,[1,2],_output(q[0]),
                              np.random.default_rng(11),np.random.default_rng(29))
        self.assertEqual(result['accepted'],1)
        self.assertEqual(result['finish_reason'],'context')
        self.assertEqual(result['bonus'],0)
        self.assertEqual(candidate.position,2)

    def test_cleanup_failure_restores_both_random_streams(self):
        q=np.asarray([[.8,.1,.1],[.1,.8,.1]])
        candidate=_FakeCandidate(q,[q[0],q[1],q[0]])
        candidate.release_snapshot=Mock(side_effect=OSError('release failed'))
        target=np.random.default_rng(11);draft=np.random.default_rng(29)
        before=(target.bit_generator.state,draft.bit_generator.state)
        with self.assertRaisesRegex(RuntimeError,'release remains pending'):
            _sample_verify(candidate,[1,2],_output(q[0]),target,draft)
        self.assertEqual((target.bit_generator.state,draft.bit_generator.state),before)
        self.assertEqual(candidate.forward_count,0)

    def test_mtp_input_embedding_vocabulary_can_include_non_output_media_tokens(self):
        from openagent.adaptive_engine.inkling_mtp import InklingMTP
        from openagent.adaptive_engine.independent_media_math import NumpyMediaMath
        owner=object.__new__(InklingMTP)
        owner._device=False;owner._lock=RLock();owner._closed=False;owner._busy=False;owner._poisoned=False
        owner.depth=1;owner.max_work_bytes=1024**2;owner.hidden_first=True;owner.chain_norm=False
        owner._config={'hidden_size':1,'vocab_size':4,'unpadded_vocab_size':2,
                       'rms_norm_eps':1e-6,'logits_mup_width_multiplier':1.0}
        owner._globals={'model.embed_tokens.weight':np.ones((4,1),np.float32),
                        'lm_head.weight':np.ones((4,1),np.float32)}
        owner.weights={'layers.0.enorm.weight':np.ones(1,np.float32),
                       'layers.0.hnorm.weight':np.ones(1,np.float32),
                       'layers.0.eh_proj.weight':np.ones((1,2),np.float32)}
        decoder=SimpleNamespace(forward=Mock(return_value=SimpleNamespace(
            last_hidden_state=np.ones((1,1,1),np.float32))))
        owner.stages=[SimpleNamespace(decoder=decoder)];owner.ops=NumpyMediaMath()
        owner.snapshot=Mock(return_value=object());owner._release_internal_snapshot=Mock()
        tokens,rows=owner.draft(np.array([[3]],np.int64),np.ones((1,1,1),np.float32))
        self.assertEqual(rows.shape,(1,2))
        self.assertLess(tokens[0],2)
        np.testing.assert_array_equal(decoder.forward.call_args.args[0],[[3]])
    def test_dense_layers_emit_empty_expert_rows_and_reject_fake_routes(self):
        routing = np.zeros((1, 2, 2, 4), dtype=np.float32)
        routing[:, :, 1, :2] = np.asarray([[[0.8, 0.2], [0.3, 0.7]]])
        config = {"n_routed_experts": 2, "num_experts_per_tok": 1,
                  "num_hidden_layers": 2, "mlp_layer_types": ["dense", "sparse"]}
        rows = _route_ids(routing, config)
        self.assertEqual(rows[0], [[], []])
        self.assertEqual(rows[1], [[0], [1]])
        routing[0, 0, 0, 0] = 0.1
        with self.assertRaises(CaptureError):
            _route_ids(routing, config)

    def test_uses_separate_target_and_draft_streams_and_bonus(self):
        q = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        candidate = _FakeCandidate(q, [q[0], q[1], q[0]])
        target_rng = np.random.default_rng(11)
        draft_rng = np.random.default_rng(29)
        target_before = target_rng.bit_generator.state
        draft_before = draft_rng.bit_generator.state

        result = _sample_verify(candidate, [1, 2], _output(q[0]), target_rng, draft_rng)

        self.assertIs(candidate.seen_rng, draft_rng)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["rejected"], 0)
        self.assertEqual(result["bonus"], 1)
        self.assertEqual(len(result["verified_tokens"]), 3)
        self.assertNotEqual(target_rng.bit_generator.state, target_before)
        self.assertNotEqual(draft_rng.bit_generator.state, draft_before)

    def test_stops_at_first_rejection_and_corrects(self):
        q = np.asarray([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        # The draft stream selects token 0, but the target assigns it very
        # little mass. Pick a deterministic seed that rejects the proposal.
        for seed in range(100):
            candidate = _FakeCandidate(q, [[0.01, 0.89, 0.10], q[1], q[0]])
            target_rng = np.random.default_rng(seed)
            draft_rng = np.random.default_rng(7)
            result = _sample_verify(candidate, [1, 2], _output([0.01, 0.89, 0.10]),
                                    target_rng, draft_rng)
            if result["rejected"]:
                break
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(result["bonus"], 0)
        self.assertEqual(len(result["verified_tokens"]), 1)
        self.assertEqual(len(result["draft_tokens"]), 2)
