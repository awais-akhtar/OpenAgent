"""Concurrent route captures must not discard receipt-manifest updates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.device_capture import _receipt_manifest_lock


class ReceiptManifestLockTests(unittest.TestCase):
    def test_unlock_failure_still_closes_file_and_preserves_body_error(self):
        if os.name == 'nt':
            import msvcrt
            module, name, release = msvcrt, 'locking', msvcrt.LK_UNLCK
        else:
            import fcntl
            module, name, release = fcntl, 'flock', fcntl.LOCK_UN
        original_unlock = getattr(module,name)
        original_open = Path.open
        for body_fails in (False,True):
            with self.subTest(body_fails=body_fails),tempfile.TemporaryDirectory() as directory:
                opened=[]
                def track_open(path,*args,**kwargs):
                    stream=original_open(path,*args,**kwargs)
                    opened.append(stream)
                    return stream
                def injected_unlock(fd,operation,*args):
                    if operation == release:
                        raise OSError('injected unlock failure')
                    return original_unlock(fd,operation,*args)
                primary=ValueError('body failure')
                with patch.object(Path,'open',track_open), \
                        patch.object(module,name,side_effect=injected_unlock):
                    with self.assertRaises(ValueError if body_fails else OSError) as caught:
                        with _receipt_manifest_lock(Path(directory)/'receipts.json'):
                            if body_fails:
                                raise primary
                self.assertTrue(opened)
                self.assertTrue(all(stream.closed for stream in opened))
                if body_fails:
                    self.assertIs(caught.exception,primary)

    def test_parallel_capture_processes_preserve_every_route(self):
        with tempfile.TemporaryDirectory(prefix="phase1-receipts-") as temporary:
            manifest = Path(temporary) / "receipts.json"
            worker = (
                "import sys; "
                "from pathlib import Path; "
                "from openagent.adaptive_engine.phase1_capture import _write_receipts; "
                "_write_receipts(Path(sys.argv[1]), {'case_id': sys.argv[2], 'value': sys.argv[2]})"
            )
            children = [
                subprocess.Popen(
                    [sys.executable, "-c", worker, str(manifest), f"route-{index}"],
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
                )
                for index in range(12)
            ]
            failures = []
            for child in children:
                stdout, stderr = child.communicate(timeout=30)
                if child.returncode:
                    failures.append((child.returncode, stdout, stderr))
            self.assertFalse(failures, failures)

            document = json.loads(manifest.read_text(encoding="utf-8"))
            routes = {item["case_id"] for item in document["routes"]}
            self.assertEqual(routes, {f"route-{index}" for index in range(12)})


if __name__ == "__main__":
    unittest.main()
