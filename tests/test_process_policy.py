from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from openagent.process_policy import DEFAULT_ENV_KEYS, ProcessPolicy, run_process
from openagent.science.execution import ScientificCodeRunner
from openagent.science.tools import CommandSimulationTool


class ProcessPolicyTests(unittest.TestCase):
    def test_cwd_must_remain_inside_an_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as outside:
            with self.assertRaisesRegex(ValueError, "workspace boundary"):
                run_process(
                    [sys.executable, "-c", "print('not reached')"],
                    cwd=outside,
                    policy=ProcessPolicy(
                        allowed_cwd_roots=(Path(allowed),),
                        allowed_executables=(sys.executable,),
                    ),
                )

    def test_environment_is_filtered_and_stdin_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, patch.dict(
            os.environ,
            {"OPENAGENT_ALLOWED_TEST": "visible", "OPENAGENT_SECRET_TEST": "hidden"},
        ):
            result = run_process(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; print(sys.stdin.read()); print(os.getenv('OPENAGENT_ALLOWED_TEST')); print(os.getenv('OPENAGENT_SECRET_TEST'))",
                ],
                cwd=workspace,
                input_data="payload",
                policy=ProcessPolicy(
                    allowed_cwd_roots=(Path(workspace),),
                    allowed_executables=(sys.executable,),
                    env_keys=DEFAULT_ENV_KEYS | frozenset({"OPENAGENT_ALLOWED_TEST"}),
                ),
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.splitlines(), ["payload", "visible", "None"])

    def test_timeout_output_cap_and_cancellation_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            policy = ProcessPolicy(
                timeout_seconds=0.15,
                max_output_bytes=1_000,
                allowed_cwd_roots=(Path(workspace),),
                allowed_executables=(sys.executable,),
            )
            timed_out = run_process(
                [sys.executable, "-c", "import time; print('x' * 5000, flush=True); time.sleep(5)"],
                cwd=workspace,
                policy=policy,
            )
            self.assertTrue(timed_out.timed_out)
            self.assertEqual(timed_out.returncode, 124)
            self.assertTrue(timed_out.output_truncated)
            self.assertLessEqual(len(timed_out.stdout.encode("utf-8")), 1_000)

            cancelled_signal = threading.Event()
            timer = threading.Timer(0.05, cancelled_signal.set)
            timer.start()
            try:
                cancelled = run_process(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    cwd=workspace,
                    policy=ProcessPolicy(
                        timeout_seconds=5,
                        allowed_cwd_roots=(Path(workspace),),
                        allowed_executables=(sys.executable,),
                    ),
                    cancel_event=cancelled_signal,
                )
            finally:
                timer.cancel()
            self.assertTrue(cancelled.cancelled)
            self.assertEqual(cancelled.returncode, 130)


class ScientificProcessBoundaryTests(unittest.TestCase):
    def test_scientific_runner_exposes_cooperative_cancellation(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        record = ScientificCodeRunner().run(
            "import time\ntime.sleep(5)",
            timeout_seconds=5,
            cancel_event=cancel_event,
        )
        self.assertEqual(record.status, "cancelled")
        self.assertEqual(record.returncode, 130)
        self.assertEqual(record.attempts, 1)
        self.assertTrue(any(item.get("status") == "cancelled" for item in record.lifecycle))

    def test_simulator_command_uses_filtered_environment_and_json_stdin(self) -> None:
        script = (
            "import json,os,sys; "
            "payload=json.load(sys.stdin); "
            "print(json.dumps({'value': payload['value'], 'secret_visible': bool(os.getenv('OPENAGENT_PROCESS_SECRET'))}))"
        )
        tool = CommandSimulationTool("test-simulator", sys.executable, ("-c", script))
        with patch.dict(os.environ, {"OPENAGENT_PROCESS_SECRET": "must-not-leak"}):
            result = tool.run(inputs={"value": 42})
        self.assertEqual(result["status"], "completed")
        self.assertIn('"value": 42', result["stdout"])
        self.assertIn('"secret_visible": false', result["stdout"])
        self.assertFalse(result["output_truncated"])


if __name__ == "__main__":
    unittest.main()
