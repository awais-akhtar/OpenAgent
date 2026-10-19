import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine import file_cache_transport as transport


class FileCacheTransportStagingCleanupTests(unittest.TestCase):
    def test_close_failure_removes_created_staging_path(self):
        with tempfile.TemporaryDirectory() as directory:
            real_close = os.close

            def close_then_fail(descriptor):
                real_close(descriptor)
                raise OSError('injected cache descriptor close failure')

            with patch.object(transport.os, 'close', side_effect=close_then_fail):
                with self.assertRaisesRegex(OSError, 'injected cache descriptor close failure'):
                    transport._create_staging_path(Path(directory))

            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
