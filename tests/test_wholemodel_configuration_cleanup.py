import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.wholemodel_validation.configuration import write_configuration


class WholeModelConfigurationCleanupTests(unittest.TestCase):
    def test_fdopen_failure_closes_descriptor_and_removes_staging_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'case.json'
            real_mkstemp=tempfile.mkstemp
            opened=[]

            def capture_descriptor(*args,**kwargs):
                descriptor,name=real_mkstemp(*args,**kwargs)
                opened.append((descriptor,name))
                return descriptor,name

            with patch('openagent.adaptive_engine.wholemodel_validation.configuration.tempfile.mkstemp',
                       side_effect=capture_descriptor), \
                 patch('openagent.adaptive_engine.wholemodel_validation.configuration.os.fdopen',
                       side_effect=OSError('injected fdopen failure')):
                with self.assertRaisesRegex(OSError,'injected fdopen failure'):
                    write_configuration({'schema':'fixture'},target)

            self.assertEqual(len(opened),1)
            descriptor,name=opened[0]
            with self.assertRaises(OSError) as failure:
                os.fstat(descriptor)
            self.assertEqual(failure.exception.errno,errno.EBADF)
            self.assertFalse(Path(name).exists())
            self.assertFalse(target.exists())


if __name__=='__main__':
    unittest.main()
