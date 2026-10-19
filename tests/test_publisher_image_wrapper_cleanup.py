from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.publisher_media_processor import PublisherMediaProcessor


class _TemporaryPILImage:
    def __init__(self, converted=None, *, fail=False):
        self.converted = converted
        self.fail = fail
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def convert(self, mode):
        if self.fail:
            raise RuntimeError('injected RGB conversion failure')
        if mode != 'RGB':
            raise AssertionError('unexpected image conversion mode')
        return self.converted

    def close(self):
        self.closed = True


class PublisherImageWrapperCleanupTests(unittest.TestCase):
    def _processor(self):
        processor = object.__new__(PublisherMediaProcessor)
        processor.max_pixels = 100
        processor.max_bytes = 1_000_000
        return processor

    def _pil_module(self, temporary):
        image_api = SimpleNamespace(
            Image=type('Image', (), {}),
            fromarray=lambda _value: temporary,
        )
        module = ModuleType('PIL')
        module.Image = image_api
        return module

    def test_numpy_image_conversion_closes_borrowed_pil_wrapper(self):
        converted = SimpleNamespace(closed=False)
        temporary = _TemporaryPILImage(converted)
        module = self._pil_module(temporary)
        with patch.dict('sys.modules', {'PIL': module}):
            result = self._processor()._image(np.zeros((2, 3, 3), dtype=np.uint8))

        self.assertIs(result, converted)
        self.assertTrue(temporary.closed)
        self.assertFalse(converted.closed)

    def test_numpy_image_conversion_failure_still_closes_borrowed_wrapper(self):
        temporary = _TemporaryPILImage(fail=True)
        module = self._pil_module(temporary)
        with patch.dict('sys.modules', {'PIL': module}):
            with self.assertRaisesRegex(RuntimeError, 'injected RGB conversion failure'):
                self._processor()._image(np.zeros((2, 3, 3), dtype=np.uint8))

        self.assertTrue(temporary.closed)


if __name__ == '__main__':
    unittest.main()
