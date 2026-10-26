"""Mock-only lifecycle regressions for the opt-in K3 native ABI owner."""
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine import native_kimi_k3_decoder as decoder


class _Function:
    def __call__(self, *_args):
        return 0


class _Library:
    _handle = 73
    oa_kimi_k3_decode = _Function()


def _fixture(config):
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in decoder.tensor_shapes(config).items()
    }


class NativeKimiK3ABIOwnerTests(unittest.TestCase):
    def setUp(self):
        self.config = decoder.KimiK3DenseMLAConfig(
            hidden=4, layers=1, heads=1, nope=2, rope=1, value=2,
            qrank=2, kvrank=2, intermediate=4, block=1, vocab=8, capacity=4,
        )

    def test_close_is_idempotent_and_closed_calls_fail(self):
        with patch.object(decoder.C, 'CDLL', return_value=_Library()), \
                patch.object(decoder, '_release_native_library') as release:
            model = decoder.NativeKimiK3DenseMLA('fixture.dll', self.config, _fixture(self.config))
            model.close()
            model.close()

            release.assert_called_once_with(_Library._handle)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                model.snapshot()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                model.forward([1])

    def test_close_can_retry_after_native_release_failure(self):
        with patch.object(decoder.C, 'CDLL', return_value=_Library()), \
                patch.object(decoder, '_release_native_library', side_effect=[OSError('busy'), None]) as release:
            model = decoder.NativeKimiK3DenseMLA('fixture.dll', self.config, _fixture(self.config))
            with self.assertRaisesRegex(OSError, 'busy'):
                model.close()
            self.assertIsNotNone(model._function)
            model.close()
            release.assert_has_calls([unittest.mock.call(_Library._handle)] * 2)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                model.snapshot()

    def test_symbol_binding_failure_releases_loaded_library(self):
        class MissingSymbolLibrary:
            _handle = 74

        with patch.object(decoder.C, 'CDLL', return_value=MissingSymbolLibrary()), \
                patch.object(decoder, '_release_native_library') as release:
            with self.assertRaises(AttributeError):
                decoder.NativeKimiK3DenseMLA('fixture.dll', self.config, _fixture(self.config))
            release.assert_called_once_with(MissingSymbolLibrary._handle)


if __name__ == '__main__':
    unittest.main()
