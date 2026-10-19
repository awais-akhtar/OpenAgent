import errno
import os
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.owned_file_descriptor import fdopen_owned


class OwnedFileDescriptorTests(unittest.TestCase):
    def test_wrap_failure_closes_descriptor_and_preserves_original_error(self):
        descriptor,name=tempfile.mkstemp()
        try:
            with patch('openagent.adaptive_engine.owned_file_descriptor.os.fdopen',
                       side_effect=OSError('injected fdopen failure')):
                with self.assertRaisesRegex(OSError,'injected fdopen failure'):
                    fdopen_owned(descriptor,'rb',owner='test artifact')
            with self.assertRaises(OSError) as failure:
                os.fstat(descriptor)
            self.assertEqual(failure.exception.errno,errno.EBADF)
        finally:
            os.unlink(name)

    def test_cleanup_failure_is_attached_to_primary_error(self):
        descriptor,name=tempfile.mkstemp()
        real_close=os.close
        try:
            with patch('openagent.adaptive_engine.owned_file_descriptor.os.fdopen',
                       side_effect=OSError('primary fdopen failure')), \
                 patch('openagent.adaptive_engine.owned_file_descriptor.os.close',
                       side_effect=OSError('secondary close failure')):
                with self.assertRaisesRegex(OSError,'primary fdopen failure') as failure:
                    fdopen_owned(descriptor,'rb',owner='test artifact')
            self.assertTrue(any('secondary close failure' in note
                                for note in getattr(failure.exception,'__notes__',())))
        finally:
            real_close(descriptor)
            os.unlink(name)


if __name__=='__main__':
    unittest.main()
