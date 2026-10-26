"""Public publisher text ABI request ownership regressions."""
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from openagent.adaptive_engine.native_family_public_text import NativeFamilyPublicTextDecoder


class NativeFamilyPublicTextLifecycleTests(unittest.TestCase):
    @staticmethod
    def decoder(execution):
        decoder = object.__new__(NativeFamilyPublicTextDecoder)
        decoder._lock = RLock()
        decoder._closed = False
        decoder._executing = False
        decoder.execution = execution
        decoder.owner = SimpleNamespace(hidden_size=2, vocabulary=4,
                                        transport_width=2, streams=1)
        decoder.family = 'inkling_text'
        decoder.max_tokens = 8
        decoder._transfer_limit = 1024
        decoder._host_limit = 8192
        decoder._stream_wire = False
        return decoder

    def test_snapshot_release_failure_rolls_back_published_native_kv_state(self):
        receipt = object()
        execution = SimpleNamespace(
            position=0,
            max_prefill_tokens=1,
            snapshot=Mock(return_value=receipt),
            forward=Mock(return_value=(np.zeros((1, 4), dtype=np.float32),
                                       np.ones((1, 2), dtype=np.float32))),
            restore=Mock(),
            release_snapshot=Mock(side_effect=OSError('snapshot file close failed')),
        )
        decoder = self.decoder(execution)

        with self.assertRaisesRegex(OSError, 'snapshot file close failed'):
            decoder.forward(np.asarray([[1]], dtype=np.int32))

        execution.restore.assert_called_once_with(receipt)

    def test_inkling_route_weights_are_returned_from_executed_segment_ranges(self):
        route_weights = np.asarray(
            [[[0.75, 0.25, 0.0]], [[0.1, 0.9, 0.0]]], dtype=np.float32)
        execution = SimpleNamespace(
            position=0,
            max_prefill_tokens=2,
            snapshot=Mock(return_value=object()),
            forward=Mock(return_value=(
                np.zeros((2, 4), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
                route_weights,
            )),
            restore=Mock(),
            release_snapshot=Mock(return_value=None),
        )
        decoder = self.decoder(execution)
        decoder.routing_observation_supported = True
        decoder.contract = {"n_routed_experts": 2, "n_shared_experts": 1}

        result = decoder.forward(np.asarray([[1, 2]], dtype=np.int32), capture_routing=True)

        self.assertEqual(result.routing_weights.shape, (1, 2, 1, 3))
        np.testing.assert_array_equal(result.routing_weights[0], route_weights)
        self.assertTrue(execution.forward.call_args.kwargs["capture_routes"])
        self.assertEqual(execution.forward.call_args.kwargs["route_width"], 3)
        execution.release_snapshot.assert_called_once()

    def test_public_status_reports_closed_decoder_as_lifecycle_error(self):
        decoder = self.decoder(SimpleNamespace())
        decoder._closed = True
        decoder.execution = None

        with self.assertRaisesRegex(RuntimeError, 'closed native public text decoder'):
            decoder.status()

    def test_glm_routes_use_routed_expert_width_without_inkling_shared_bank(self):
        routes = np.asarray([[[0.25, 0.75]]], dtype=np.float32)
        execution = SimpleNamespace(
            position=0, max_prefill_tokens=1, snapshot=Mock(return_value=object()),
            forward=Mock(return_value=(np.zeros((1, 4), np.float32),
                                       np.ones((1, 2), np.float32), routes)),
            restore=Mock(), release_snapshot=Mock(),
        )
        decoder = self.decoder(execution)
        decoder.family = 'glm5_next_text'
        decoder.contract = SimpleNamespace(n_routed_experts=2, n_shared_experts=1)
        decoder.routing_observation_supported = True
        result = decoder.forward(np.asarray([[1]], dtype=np.int32), capture_routing=True)
        self.assertEqual(execution.forward.call_args.kwargs['route_width'], 2)
        np.testing.assert_array_equal(result.routing_weights, routes[None])

    def test_missing_glm_observer_rejects_before_state_capture(self):
        execution = SimpleNamespace(snapshot=Mock())
        decoder = self.decoder(execution)
        decoder.family = 'glm5_next_text'
        decoder.routing_observation_supported = False
        with self.assertRaisesRegex(ValueError, 'no executed route observation'):
            decoder.forward(np.asarray([[1]], dtype=np.int32), capture_routing=True)
        execution.snapshot.assert_not_called()

    def test_unprintable_rollback_failure_does_not_mask_forward_failure(self):
        class UnprintableFailure(Exception):
            def __str__(self):
                raise AssertionError('rollback error formatting must not run')

            def __repr__(self):
                raise AssertionError('rollback error formatting must not run')

        receipt = object()
        execution = SimpleNamespace(
            position=0, max_prefill_tokens=1,
            snapshot=Mock(return_value=receipt),
            forward=Mock(side_effect=ValueError('forward failed')),
            restore=Mock(side_effect=UnprintableFailure()),
            release_snapshot=Mock(return_value=None),
        )
        decoder = self.decoder(execution)

        with self.assertRaisesRegex(ValueError, 'forward failed') as caught:
            decoder.forward(np.asarray([[1]], dtype=np.int32))

        self.assertTrue(any('UnprintableFailure: failure details omitted' in note
                            for note in caught.exception.__notes__))


if __name__ == '__main__':
    unittest.main()
