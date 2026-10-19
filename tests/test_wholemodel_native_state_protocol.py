import unittest

import numpy as np

from openagent.adaptive_engine.wholemodel_validation.adapters import NativeExecutorAdapter


class _SnapshotStateExecutor:
    def __init__(self):
        self.position = 0
        self.state = np.zeros(2, dtype=np.float32)
        self.closed = False

    def forward(self, token_ids):
        self.position += len(token_ids)
        self.state += np.asarray(token_ids, dtype=np.float32).sum()
        logits = np.zeros((len(token_ids), 8), dtype=np.float32)
        logits[:, 1] = 1.0
        return logits

    def snapshot_state(self):
        return {'position': self.position, 'state': self.state.copy()}

    def restore_state(self, snapshot):
        self.position = snapshot['position']
        self.state = snapshot['state'].copy()

    def close(self):
        self.closed = True


class WholemodelNativeStateProtocolTest(unittest.TestCase):
    def test_snapshot_state_pair_supports_cache_observation_and_restore(self):
        metadata = {
            'num_layers': 1,
            'vocabulary_size': 8,
            'cache_schema': {'position': 'global', 'layer.0.state': 0},
        }

        def read_cache(context):
            return {
                'position': np.asarray(context.cache['position'], dtype=np.int64),
                'layer.0.state': context.cache['state'],
            }

        adapter = NativeExecutorAdapter(
            lambda *, seed: _SnapshotStateExecutor(), metadata,
            cache_reader=read_cache,
            cache_getter=lambda executor: executor.snapshot_state())
        session = adapter.new_session(seed=7)
        try:
            first = session.forward([1])
            np.testing.assert_array_equal(first.cache['layer.0.state'], [1, 1])
            boundary = session.snapshot()
            session.forward([2])
            session.restore(boundary)
            np.testing.assert_array_equal(
                session.forward([3]).cache['layer.0.state'], [4, 4])
            session.release_snapshot(boundary)
        finally:
            executor = session.executor
            session.close()
        self.assertTrue(executor.closed)


if __name__ == '__main__':
    unittest.main()
