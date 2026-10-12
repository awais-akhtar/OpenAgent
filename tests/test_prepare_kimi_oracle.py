"""Oracle setup rejects unverified downloads and reuses exact cached bytes."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.prepare_kimi_oracle import prepare


class PrepareKimiOracleTests(unittest.TestCase):
    def test_verified_download_then_offline_cache(self):
        data = b'fixture oracle source'
        hashes = {'source.py': hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            with patch('tests.prepare_kimi_oracle.ORACLE_HASHES', hashes), \
                    patch('tests.prepare_kimi_oracle.urllib.request.urlopen', return_value=io.BytesIO(data)) as fetch:
                result = prepare(directory)
                self.assertEqual((result / 'source.py').read_bytes(), data)
                fetch.assert_called_once()
            with patch('tests.prepare_kimi_oracle.ORACLE_HASHES', hashes), \
                    patch('tests.prepare_kimi_oracle.urllib.request.urlopen', side_effect=AssertionError('network')):
                self.assertEqual(prepare(directory), result)

    def test_bad_hash_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.py'
            path.write_bytes(b'previous unverified file')
            with patch('tests.prepare_kimi_oracle.ORACLE_HASHES', {'source.py': '0' * 64}), \
                    patch('tests.prepare_kimi_oracle.urllib.request.urlopen', return_value=io.BytesIO(b'bad source')):
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    prepare(directory)
            self.assertEqual(path.read_bytes(), b'previous unverified file')
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['source.py'])
