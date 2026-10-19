import unittest

from openagent.adaptive_engine.k3_execution.batch import K3BatchExecution


class _UnprintableFailure(RuntimeError):
    def __str__(self):
        raise AssertionError('cleanup diagnostics must not stringify failures')


class _Model:
    def __init__(self, name, failure=None):
        self.name = name
        self.failure = failure
        self.restore_calls = 0
        self.release_calls = 0
        self.close_calls = 0
        self.max_bytes = 8

    def restore(self, _snapshot):
        self.restore_calls += 1
        if self.failure is not None:
            raise self.failure

    def release_snapshot(self, _snapshot):
        self.release_calls += 1
        if self.failure is not None:
            raise self.failure

    def close(self):
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


class K3BatchCleanupTests(unittest.TestCase):
    def make_batch(self):
        return K3BatchExecution(lambda: None, max_sessions=4,
            max_model_bytes=64, max_batch_bytes=64, max_snapshot_bytes=64)

    def test_rollback_attempts_every_slot_when_error_cannot_be_formatted(self):
        batch = self.make_batch()
        models = (_Model('first', _UnprintableFailure()),
                  _Model('second', _UnprintableFailure()))
        primary = RuntimeError('generation failed')

        batch._restore_all(models, (object(), object()), primary)

        self.assertEqual(tuple(model.restore_calls for model in models), (1, 1))
        self.assertTrue(batch._closed)
        self.assertIn('_UnprintableFailure', '\n'.join(getattr(primary, '__notes__', ())))

    def test_snapshot_release_attempts_every_slot_and_retains_retry_owners(self):
        batch = self.make_batch()
        models = (_Model('first', _UnprintableFailure()),
                  _Model('second', _UnprintableFailure()))
        primary = RuntimeError('transaction failed')

        batch._release_captured(models, (object(), object()), primary)

        self.assertEqual(tuple(model.release_calls for model in models), (1, 1))
        self.assertEqual(len(batch._pending), 2)
        self.assertIn('_UnprintableFailure', '\n'.join(getattr(primary, '__notes__', ())))

    def test_close_continues_teardown_after_unprintable_slot_failure(self):
        batch = self.make_batch()
        first = _Model('first', _UnprintableFailure())
        second = _Model('second')
        batch._models = {'first': first, 'second': second}
        batch._reserved = 16

        with self.assertRaisesRegex(RuntimeError, 'K3 batch cleanup remains pending') as caught:
            batch.close()

        self.assertEqual(first.close_calls, 1)
        self.assertEqual(second.close_calls, 1)
        self.assertEqual(tuple(batch._models), ('first',))
        self.assertIn('_UnprintableFailure', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
