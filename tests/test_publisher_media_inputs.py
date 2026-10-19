import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.publisher_media_inputs import PublisherVideoInput
from openagent.adaptive_engine.publisher_media_files import read_selected_video


class PublisherVideoInputTests(unittest.TestCase):
    def test_overflowing_fps_is_a_validation_error_before_frame_inspection(self):
        for fps in (10**1000, float('inf'), float('nan'), 0, -1):
            with self.subTest(fps=type(fps).__name__), self.assertRaisesRegex(
                    ValueError, 'finite positive FPS'):
                PublisherVideoInput((object(),), (0,), 1, fps)

    def test_valid_integer_fps_is_normalized_before_timestamp_projection(self):
        video = PublisherVideoInput(
            (np.zeros((1, 1, 3), dtype=np.uint8),), (2,), 3, 30)

        self.assertEqual(video.fps, 30.0)
        self.assertIs(type(video.fps), float)
        self.assertEqual(video.timestamps, (2 / 30.0,))

    def test_selected_video_manifest_rejects_overflowing_fps_before_frame_open(self):
        with TemporaryDirectory() as directory:
            manifest = Path(directory) / 'selection.json'
            manifest.write_text(json.dumps({
                'schema': 'openagent.publisher-selected-video.v1',
                'frames': ['not-opened.png'],
                'metadata': {'frames_indices': [0], 'total_num_frames': 1,
                             'fps': 10**1000},
            }), encoding='utf-8')

            payload = manifest.read_bytes()
            with patch('openagent.adaptive_engine.publisher_media_files.read_regular',
                       return_value=payload):
                with self.assertRaisesRegex(ValueError, 'frame indices, total frame count and FPS'):
                    read_selected_video(manifest, max_bytes=4096, max_frame_bytes=4096)


if __name__ == '__main__':
    unittest.main()
