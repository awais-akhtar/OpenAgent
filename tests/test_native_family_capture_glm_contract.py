"""Exercise capture helpers with GLM-shaped dict/tensor output, no model weights."""
import unittest

import numpy as np

from openagent.adaptive_engine.native_family_capture import _cache_digests, _greedy


class _Contract:
    n_routed_experts = 3
    num_experts_per_tok = 2


class _GlmResult:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _GlmTextCandidate:
    contract = _Contract()
    config = {"eos_token_id": 2}

    def __init__(self):
        self.calls = 0

    def forward(self, token_ids, *, capture_routing=False):
        self.calls += 1
        self.assert_routing = capture_routing
        return {
            "logits": _GlmResult([[0.0, 9.0, -1.0]]),
            "routing_weights": _GlmResult([[[[0.0, 0.25, 0.75]]]]),
        }


class _Qwen4DeviceCandidate:
    config = {"num_experts": 3, "num_experts_per_tok": 2, "eos_token_id": 2}

    def forward(self, token_ids, *, capture_routing=False):
        self.assert_routing = capture_routing
        return {
            "logits": _GlmResult([[0.0, 9.0, -1.0]]),
            "routing_weights": _GlmResult([[[[0.0, 0.25, 0.75]]]]),
        }


class _InklingDeviceCandidate:
    config = {"n_routed_experts": 3, "num_experts_per_tok": 2, "eos_token_id": 2}

    def forward(self, token_ids):
        return {
            "logits": _GlmResult([[0.0, 9.0, -1.0]]),
            "routing_weights": _GlmResult([[[[0.0, 0.25, 0.75]]]]),
        }


class Glm5NextTextContract:
    n_routed_experts = 3
    num_experts_per_tok = 2
    eos_token_id = 2


class _NativeGlmReferenceCandidate:
    config = Glm5NextTextContract()

    def forward(self, token_ids, *, capture_routing=False):
        self.input_shape = np.asarray(token_ids).shape
        self.capture_routing = capture_routing
        return {
            "logits": _GlmResult([[0.0, 9.0, -1.0]]),
            "routing_weights": _GlmResult([[[[0.0, 0.25, 0.75]]]]),
        }


class NativeFamilyCaptureGlmContractTests(unittest.TestCase):
    def test_glm_dict_logits_routing_and_contract_are_captured(self):
        candidate = _GlmTextCandidate()
        tokens, logits, finish, routes = _greedy(
            candidate, [1], 1, eos_token_id=2, capture_routing=True)

        self.assertEqual(tokens, [1])
        self.assertEqual(finish, "length")
        self.assertTrue(candidate.assert_routing)
        self.assertEqual(len(logits), 1)
        self.assertEqual(routes, [[[2, 1]]])

    def test_cache_observation_accepts_device_tensor_protocol(self):
        class CacheOwner:
            cache = None
            _cache = ((_GlmResult([[1.0, 2.0]]),),)

        total_bytes, digests = _cache_digests(CacheOwner(), 1)
        self.assertEqual(total_bytes, 8)
        self.assertEqual(len(digests), 1)
        self.assertEqual(len(digests[0]), 64)

    def test_qwen4_device_greedy_captures_selected_experts(self):
        candidate = _Qwen4DeviceCandidate()
        tokens, _logits, _finish, routes = _greedy(
            candidate, [1], 1, capture_routing=True)
        self.assertEqual(tokens, [1])
        self.assertTrue(candidate.assert_routing)
        self.assertEqual(routes, [[[2, 1]]])

    def test_inkling_device_routing_needs_no_capture_keyword(self):
        candidate = _InklingDeviceCandidate()
        tokens, _logits, _finish, routes = _greedy(
            candidate, [1], 1, capture_routing=True, routing_keyword=False)
        self.assertEqual(tokens, [1])
        self.assertEqual(routes, [[[2, 1]]])

    def test_glm_native_comparator_receives_exact_batch_one_abi(self):
        candidate = _NativeGlmReferenceCandidate()
        tokens, _logits, _finish, routes = _greedy(
            candidate, [1, 0], 1, capture_routing=True)
        self.assertEqual(tokens, [1])
        self.assertEqual(candidate.input_shape, (1, 2))
        self.assertTrue(candidate.capture_routing)
        self.assertEqual(routes, [[[2, 1]]])


if __name__ == "__main__":
    unittest.main()
