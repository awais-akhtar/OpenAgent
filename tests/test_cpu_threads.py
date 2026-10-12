import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from openagent.adaptive_engine.cpu_threads import blas_thread_scope


class BlasScopeTests(unittest.TestCase):
    def test_unconfigured_scope_has_no_dependency(self):
        with patch.dict(sys.modules, {'threadpoolctl': None}), blas_thread_scope():
            pass

    def test_invalid_limits_rejected(self):
        for value in (0, -1, True, 1.5, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                with blas_thread_scope(value):
                    pass

    def test_scope_restores_after_execution_failure(self):
        module = ModuleType('threadpoolctl')
        module.threadpool_limits = MagicMock()
        with patch.dict(sys.modules, {'threadpoolctl': module}), self.assertRaises(RuntimeError):
            with blas_thread_scope(1):
                raise RuntimeError('execution failed')
        module.threadpool_limits.assert_called_once_with(limits=1, user_api='blas')
        module.threadpool_limits.return_value.__exit__.assert_called_once()

    def test_missing_optional_dependency_is_explicit(self):
        with patch.dict(sys.modules, {'threadpoolctl': None}), self.assertRaisesRegex(RuntimeError, 'engine extra'):
            with blas_thread_scope(1):
                pass
