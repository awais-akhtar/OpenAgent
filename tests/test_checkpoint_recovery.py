from io import BytesIO
import hashlib
from pathlib import Path
import tempfile
import unittest

from openagent.adaptive_engine.checkpoint_recovery import recover_file, _destination_lock, _HTTPSRedirectHandler
from urllib.request import Request


class Response(BytesIO):
    status = 206


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "weights.bin"
        self.data = b"abcdefghij"
        self.digest = hashlib.sha256(self.data).hexdigest()

    def opener(self, request, timeout):
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        response = Response(self.data[start:end + 1])
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(self.data)}"}
        return response

    def test_resume_and_verified_publication(self):
        self.path.write_bytes(self.data[:3])
        result = recover_file("https://example.test/model", self.path, size=10,
                              sha256=self.digest, block_bytes=3, opener=self.opener)
        self.assertTrue(result["verified"])
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_wrong_range_keeps_original(self):
        self.path.write_bytes(b"abc")
        def wrong(request, timeout):
            response = Response(b"wrong")
            response.status = 200
            response.headers = {}
            return response
        with self.assertRaisesRegex(ValueError, "exact byte range"):
            recover_file("https://example.test/model", self.path, size=10,
                         sha256=self.digest, opener=wrong)
        self.assertEqual(self.path.read_bytes(), b"abc")

    def test_checksum_failure_does_not_replace_original(self):
        self.path.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "checksum"):
            recover_file("https://example.test/model", self.path, size=10,
                         sha256=self.digest, opener=self.opener)
        self.assertEqual(self.path.read_bytes(), b"bad")

    def test_plain_http_rejected(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            recover_file("http://example.test/model", self.path, size=10, sha256=self.digest)

    def test_corrupt_completed_partial_is_preserved_and_retry_starts_fresh(self):
        self.path.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "checksum"):
            recover_file("https://example.test/model", self.path, size=10,
                         sha256=self.digest, opener=self.opener)
        result = recover_file("https://example.test/model", self.path, size=10,
                              sha256=self.digest, opener=self.opener)
        self.assertTrue(result['verified'])
        self.assertEqual(self.path.read_bytes(), self.data)
        failed = list(self.path.parent.glob('*.failed-*'))
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].read_bytes(), b'baddefghij')

    def test_redirect_transport_downgrade_rejected_before_following(self):
        handler = _HTTPSRedirectHandler()
        request = Request('https://example.test/model')
        with self.assertRaisesRegex(ValueError, 'HTTPS'):
            handler.redirect_request(request, None, 302, 'redirect', {}, 'http://example.test/model')
        redirected = handler.redirect_request(request, None, 302, 'redirect', {}, 'https://other.test/model')
        self.assertEqual(redirected.full_url, 'https://other.test/model')

    def test_concurrent_writers_are_rejected_and_lock_is_reusable(self):
        with _destination_lock(self.path):
            with self.assertRaises(OSError):
                recover_file("https://example.test/model", self.path, size=10,
                             sha256=self.digest, opener=self.opener)
        result = recover_file("https://example.test/model", self.path, size=10,
                              sha256=self.digest, opener=self.opener)
        self.assertTrue(result["verified"])
