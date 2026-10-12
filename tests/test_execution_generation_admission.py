"""Concurrent generation admission and callback reentrancy contracts."""
import threading
import unittest

from openagent.adaptive_engine.execution_service import (
    ExecutionCancelled, ExecutionLimits, NativeExecutionService, _Session,
)


class GenerationAdmissionTests(unittest.TestCase):
    def make_service(self):
        return NativeExecutionService(lambda: None, ExecutionLimits(
            vocabulary=8, context=32, max_new_tokens=8, max_sessions=2,
            max_concurrent_generations=1))

    def test_cancelled_waiter_releases_admission_accounting(self):
        service = self.make_service()
        session = _Session(None)
        entered, finished = threading.Event(), threading.Event()
        cancel = threading.Event()
        errors = []
        def wait():
            entered.set()
            try:
                with service._generation_slot(session, cancel):
                    errors.append(AssertionError('cancelled waiter was admitted'))
            except ExecutionCancelled:
                pass
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()
        with service._generation_slot(session, None):
            thread = threading.Thread(target=wait)
            thread.start()
            self.assertTrue(entered.wait(2))
            cancel.set()
            self.assertTrue(finished.wait(2))
            thread.join(2)
            self.assertEqual(service.runtime_status()['active_generations'], 1)
        self.assertFalse(errors)
        self.assertEqual(service.runtime_status()['waiting_generations'], 0)
        self.assertEqual(service.runtime_status()['active_generations'], 0)

    def test_reentrant_callback_fails_without_deadlocking(self):
        service = self.make_service()
        session = _Session(None)
        with service._generation_slot(session, None):
            with self.assertRaisesRegex(RuntimeError, 'reentrant generation'):
                with service._generation_slot(session, None):
                    self.fail('nested generation admitted')
        with service._generation_slot(session, None):
            self.assertEqual(service.runtime_status()['active_generations'], 1)

    def test_exception_releases_slot(self):
        service = self.make_service()
        session = _Session(None)
        with self.assertRaisesRegex(ValueError, 'request failed'):
            with service._generation_slot(session, None):
                raise ValueError('request failed')
        with service._generation_slot(session, None):
            self.assertEqual(service.runtime_status()['active_generations'], 1)

    def test_shutdown_rejects_admission(self):
        service = self.make_service()
        service.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            with service._generation_slot(_Session(None), None):
                self.fail('closed service admitted generation')
