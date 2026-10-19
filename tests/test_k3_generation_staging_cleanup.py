import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine import k3_generation_recovery as recovery


class K3GenerationStagingCleanupTests(unittest.TestCase):
    def test_close_failure_removes_created_staging_path(self):
        with tempfile.TemporaryDirectory() as directory:
            real_close = os.close

            def close_then_fail(descriptor):
                real_close(descriptor)
                raise OSError('injected descriptor close failure')

            with patch.object(recovery.os, 'close', side_effect=close_then_fail):
                with self.assertRaisesRegex(OSError, 'injected descriptor close failure'):
                    recovery._create_staging_path(scratch_directory=directory)

            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
