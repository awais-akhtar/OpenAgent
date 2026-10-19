import unittest
from unittest.mock import patch

from openagent.adaptive_engine import execution_factories


class UnprintableCleanupError(RuntimeError):
    def __repr__(self):
        raise RuntimeError('repr must not be called')

    def __str__(self):
        raise RuntimeError('str must not be called')


class PublicFactoryCleanupTests(unittest.TestCase):
    def test_public_abi_composition_failure_closes_backing_service(self):
        primary=ValueError('public adapter construction failed')
        class Backing:
            closes=0
            def close(self):self.closes+=1
        backing=Backing()
        with patch.object(execution_factories,'compose_public_publisher_service',
                          side_effect=primary):
            with self.assertRaises(ValueError) as caught:
                execution_factories._compose_public_service_or_close(backing,runtime_library='unused')
        self.assertIs(caught.exception,primary)
        self.assertEqual(backing.closes,1)

    def test_public_abi_cleanup_failure_preserves_primary_and_retry(self):
        primary=ValueError('public adapter construction failed')
        class Backing:
            closes=0
            def close(self):
                self.closes+=1
                raise OSError('cleanup must be retried')
        backing=Backing()
        with patch.object(execution_factories,'compose_public_publisher_service',
                          side_effect=primary):
            with self.assertRaises(ValueError) as caught:
                execution_factories._compose_public_service_or_close(backing,runtime_library='unused')
        self.assertIs(caught.exception,primary)
        self.assertEqual(backing.closes,1)
        self.assertTrue(callable(caught.exception.cleanup))
        self.assertTrue(any('cleanup retained' in note for note in caught.exception.__notes__))

    def test_unprintable_cleanup_failure_does_not_mask_public_factory_error(self):
        primary=ValueError('public adapter construction failed')
        class Backing:
            def close(self):raise UnprintableCleanupError()
        backing=Backing()
        with patch.object(execution_factories,'compose_public_publisher_service',
                          side_effect=primary):
            with self.assertRaisesRegex(ValueError,'public adapter construction failed') as caught:
                execution_factories._compose_public_service_or_close(backing,runtime_library='unused')
        self.assertIs(caught.exception,primary)
        self.assertTrue(any('UnprintableCleanupError: failure details omitted' in note
                            for note in caught.exception.__notes__))


if __name__=='__main__':
    unittest.main()
