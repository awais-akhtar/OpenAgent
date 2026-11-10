from __future__ import annotations

import os
import subprocess
import sys
import unittest

from openagent.cpu_budget_job import CPUJobBudget


@unittest.skipUnless(os.name == "nt", "Windows Job Objects only")
class CPUJobBudgetTests(unittest.TestCase):
    def test_child_and_descendant_stay_inside_enforced_job_memory(self) -> None:
        with CPUJobBudget(64 * 1024**2) as job:
            process = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.readline(); x=bytearray(128*1024**2); print(len(x))"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                job.assign(process)
                stdout, stderr = process.communicate("START\n", timeout=10)
                self.assertNotEqual(process.returncode, 0)
                self.assertNotIn("134217728", stdout)
                self.assertIn("MemoryError", stderr)
                self.assertGreater(job.peak_bytes(), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
