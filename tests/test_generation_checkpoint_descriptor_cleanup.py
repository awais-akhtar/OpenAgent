import errno
import os
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.generation_checkpoint_files import _fdopen_owned


class GenerationCheckpointDescriptorCleanupTests(unittest.TestCase):
    def test_fdopen_failure_closes_descriptor(self):
        descriptor,name=tempfile.mkstemp()
        try:
            with patch('openagent.adaptive_engine.generation_checkpoint_files.os.fdopen',
                       side_effect=OSError('injected wrapper failure')):
                with self.assertRaisesRegex(OSError,'injected wrapper failure'):
                    _fdopen_owned(descriptor,'wb')
            with self.assertRaises(OSError) as failure:
                os.fstat(descriptor)
            self.assertEqual(failure.exception.errno,errno.EBADF)
        finally:
            os.unlink(name)

    def test_close_failure_is_attached_without_replacing_fdopen_error(self):
        descriptor,name=tempfile.mkstemp()
        real_close=os.close
        try:
            with patch('openagent.adaptive_engine.generation_checkpoint_files.os.fdopen',
                       side_effect=OSError('primary wrapper failure')), \
                 patch('openagent.adaptive_engine.generation_checkpoint_files.os.close',
                       side_effect=OSError('injected close failure')):
                with self.assertRaisesRegex(OSError,'primary wrapper failure') as failure:
                    _fdopen_owned(descriptor,'wb')
            self.assertTrue(any('injected close failure' in note
                                for note in getattr(failure.exception,'__notes__',())))
        finally:
            real_close(descriptor)
            os.unlink(name)


if __name__=='__main__':
    unittest.main()
