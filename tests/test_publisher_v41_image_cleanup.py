from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.v41_execution.image_input import PublisherImageProcessor


class _UnprintableCloseError(Exception):
    def __str__(self):
        raise AssertionError('cleanup error formatting must not replace primary failure')

    def __repr__(self):
        raise AssertionError('cleanup error representation must not replace primary failure')


class _PILImage:
    size = (2, 2)
    n_frames = 1

    def __init__(self, converted):
        self.converted = converted
        self.context_closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.context_closed = True

    def convert(self, mode):
        if mode != 'RGB':
            raise AssertionError('unexpected conversion mode')
        return self.converted


class _OwnedImage:
    def __init__(self, *, pixels=None):
        self.pixels = pixels
        self.close_called = False

    def close(self):
        self.close_called = True
        raise _UnprintableCloseError()

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.pixels, dtype=dtype)


class PublisherV41ImageCleanupTests(unittest.TestCase):
    def test_unprintable_multiple_close_errors_preserve_processing_failure(self):
        image = _OwnedImage()
        resized = _OwnedImage(pixels=np.full((2, 2, 3), 127, dtype=np.uint8))
        opened = _PILImage(image)
        image_api = SimpleNamespace(
            Image=type('Image', (), {}),
            Resampling=SimpleNamespace(BICUBIC=3),
            open=lambda _stream: opened,
            fromarray=lambda _array: opened,
        )
        pil = ModuleType('PIL')
        pil.Image = image_api
        pil.ImageOps = SimpleNamespace(pad=lambda *_args, **_kwargs: resized)
        processor = PublisherImageProcessor(
            patch_size=1,
            downsample_ratio=1,
            max_tokens=20,
            min_pixels=0,
            max_width_height_ratio=None,
            max_bytes=10_000,
            cast=lambda _values: (_ for _ in ()).throw(RuntimeError('primary processing failure')),
        )

        with patch.dict('sys.modules', {'PIL': pil}):
            with self.assertRaisesRegex(RuntimeError, 'primary processing failure') as raised:
                processor.prepare(b'encoded-image', start=0)

        self.assertEqual(raised.exception.args, ('primary processing failure',))
        self.assertTrue(resized.close_called)
        self.assertTrue(image.close_called)
        self.assertTrue(opened.context_closed)


if __name__ == '__main__':
    unittest.main()
