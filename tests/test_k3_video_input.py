import unittest
from unittest.mock import patch
import weakref

import numpy as np

from openagent.adaptive_engine.k3_execution import video_input
from openagent.adaptive_engine.k3_execution.video_input import K3PreparedMedia, K3VideoInput


class K3VideoInputTests(unittest.TestCase):
    def test_owned_image_cleanup_attempts_every_owner_when_exception_is_unprintable(self):
        class UnprintableCleanupError(Exception):
            def __str__(self):
                raise RuntimeError('cleanup formatter failed')

        closed = []

        class ImageOwner:
            def __init__(self, name, error):
                self.name = name
                self.error = error

            def close(self):
                closed.append(self.name)
                if self.error is not None:
                    raise self.error

        first = UnprintableCleanupError()
        owners = (ImageOwner('first', first),
                  ImageOwner('second', RuntimeError('second cleanup')),
                  ImageOwner('third', None))

        with self.assertRaises(UnprintableCleanupError) as caught:
            video_input._close_owned_images(owners)

        self.assertIs(caught.exception, first)
        self.assertEqual(closed, ['first', 'second', 'third'])
        self.assertTrue(any('RuntimeError: failure details omitted' in note
                            for note in getattr(first, '__notes__', ())))

    def test_owned_image_cleanup_never_masks_primary_failure(self):
        class UnprintableCleanupError(Exception):
            def __str__(self):
                raise RuntimeError('cleanup formatter failed')

        closed = []

        class ImageOwner:
            def close(self):
                closed.append(True)
                raise UnprintableCleanupError()

        primary = ValueError('media preprocessing failed')
        video_input._close_owned_images((ImageOwner(), ImageOwner()), primary)

        self.assertEqual(closed, [True, True])
        self.assertTrue(any('UnprintableCleanupError: failure details omitted' in note
                            for note in getattr(primary, '__notes__', ())))

    def test_prepared_media_storage_cannot_be_mutated_through_caller_owned_arrays(self):
        pixels = np.ones((1, 3, 14, 14), dtype=np.float32)
        grids = np.array([[1, 1, 1]], dtype=np.int64)
        media = K3PreparedMedia(pixels, grids, ((),), ('still-v1',))

        pixels[0, 0, 0, 0] = -1
        grids[0, 1] = 2
        self.assertEqual(media.pixels[0, 0, 0, 0], 1.0)
        self.assertEqual(media.grids[0, 1], 1)
        with self.assertRaises(ValueError):
            media.pixels.flags.writeable = True
        with self.assertRaises(ValueError):
            media.grids.flags.writeable = True

    def test_video_input_snapshot_cannot_reenable_frame_writes(self):
        source = np.ones((2, 3, 3), dtype=np.uint8)
        video = K3VideoInput((source,), (0.0,), 'selected-frame-v1')

        source.fill(0)
        self.assertTrue(np.all(video.frames[0] == 1))
        with self.assertRaises(ValueError):
            video.frames[0].flags.writeable = True

    def test_geometry_mismatch_is_rejected_before_detaching_any_frame(self):
        frames = (
            np.zeros((512, 512, 3), dtype=np.uint8),
            np.zeros((256, 512, 3), dtype=np.uint8),
        )
        array = np.array

        with patch('openagent.adaptive_engine.k3_execution.video_input.np.array',
                   wraps=array) as detach:
            with self.assertRaisesRegex(ValueError, 'share one spatial geometry'):
                K3VideoInput(frames, (0.0, 1.0), 'mismatched-geometry')

        detach.assert_not_called()

    def test_cancellation_between_media_items_releases_partial_patch_bank(self):
        references = []

        def preprocess(_frames, **_options):
            pixels = np.ones((1, 3, 14, 14), dtype=np.float32)
            references.append(weakref.ref(pixels))
            return pixels, np.array([[1, 1, 1]], dtype=np.int64)

        def cancelled():
            # The first packed bank has been allocated; the next item boundary
            # must abort and release it even while the exception is retained.
            return bool(references)

        frames = (np.zeros((1, 1, 3), dtype=np.uint8),
                  np.zeros((1, 1, 3), dtype=np.uint8))
        with patch.object(video_input, 'preprocess_images', side_effect=preprocess):
            failure = None
            try:
                video_input.prepare_media(frames, max_bytes=16 * 1024,
                                          cancelled=cancelled)
            except InterruptedError as error:
                failure = error

        self.assertEqual(len(references), 1)
        self.assertIsNotNone(failure)
        self.assertIn('media preparation cancelled', str(failure))
        self.assertIsNotNone(failure.__traceback__)
        self.assertIsNone(references[0]())


if __name__ == '__main__':
    unittest.main()
