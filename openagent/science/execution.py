"""Policy-checked, bounded isolated subprocess execution and profiling."""

from __future__ import annotations

import ast
import sys
import time
import hashlib
import tempfile
from dataclasses import dataclass, replace
import threading
from pathlib import Path
from typing import Callable

from ..process_policy import ProcessPolicy, ProcessResult, run_process


@dataclass(frozen=True)
class ExecutionRecord:
    status: str
    stdout: str
    stderr: str
    returncode: int
    elapsed_ms: float
    attempts: int
    peak_memory_bytes: int = 0
    attempt_history: tuple[dict[str, object], ...] = ()
    final_code: str = ""
    repairs: int = 0
    lifecycle: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "attempts": self.attempts,
            "peak_memory_bytes": self.peak_memory_bytes,
            "attempt_history": [dict(item) for item in self.attempt_history],
            "final_code": self.final_code,
            "repairs": self.repairs,
            "lifecycle": [dict(item) for item in self.lifecycle],
        }


class ScientificCodeRunner:
    """Runs policy-checked scripts in a bounded isolated subprocess.

    This is defense in depth for local scientific jobs, not a secure sandbox.
    """

    def run(
        self,
        code: str,
        *,
        timeout_seconds: float = 10.0,
        memory_limit_mb: int = 512,
        retries: int = 0,
        repair: Callable[[str, str], str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ExecutionRecord:
        if memory_limit_mb < 32 or memory_limit_mb > 16_384:
            raise ValueError("memory_limit_mb must be between 32 and 16384")
        current = code
        repairer = repair or repair_scientific_code
        started = time.perf_counter()
        last: ProcessResult | None = None
        attempts = 0
        peak_memory = 0
        history: list[dict[str, object]] = []
        lifecycle: list[dict[str, object]] = [{"stage": "code_prepared", "status": "complete", "message": "Python source prepared for bounded isolated subprocess execution."}]
        repairs = 0
        for attempts in range(1, max(1, retries + 1) + 1):
            violations = execution_policy_violations(current)
            if violations:
                message = "Execution policy rejected the source: " + "; ".join(violations)
                last = ProcessResult((sys.executable, "-I", "-c", "<blocked>"), 126, "", message, False, False)
                history.append({
                    "attempt": attempts,
                    "returncode": 126,
                    "elapsed_ms": 0.0,
                    "code_sha256": hashlib.sha256(current.encode("utf-8")).hexdigest(),
                    "stderr_tail": message,
                    "stdout_tail": "",
                    "repair_applied": False,
                    "policy_violations": list(violations),
                })
                lifecycle.append({"stage": "policy", "status": "rejected", "attempt": attempts, "message": message})
                break
            lifecycle.append({"stage": "execute", "status": "running", "attempt": attempts, "message": f"Running attempt {attempts}."})
            attempt_started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="openagent-science-") as workspace:
                last = run_process(
                    [sys.executable, "-I", "-c", current],
                    cwd=workspace,
                    policy=ProcessPolicy(
                        timeout_seconds=max(0.1, timeout_seconds),
                        max_output_bytes=20_000,
                        max_memory_bytes=memory_limit_mb * 1024 * 1024,
                        allowed_cwd_roots=(Path(workspace),),
                        allowed_executables=(sys.executable,),
                    ),
                    cancel_event=cancel_event,
                )
            peak_memory = max(peak_memory, last.peak_memory_bytes)
            stderr = last.stderr
            if last.resource_limited:
                stderr = f"{stderr}\nExecution exceeded the {memory_limit_mb} MB memory limit.".strip()
                last = replace(last, stderr=stderr)
                lifecycle[-1] = {
                    "stage": "resource_limit",
                    "status": "terminated",
                    "attempt": attempts,
                    "message": f"Attempt {attempts} exceeded the {memory_limit_mb} MB memory limit.",
                }
            elif last.cancelled:
                stderr = f"{stderr}\nExecution cancelled.".strip()
                last = replace(last, stderr=stderr)
                lifecycle[-1] = {"stage": "execute", "status": "cancelled", "attempt": attempts, "message": f"Attempt {attempts} was cancelled."}
            elif last.timed_out:
                stderr = f"{stderr}\nExecution timed out.".strip()
                last = replace(last, stderr=stderr)
            entry: dict[str, object] = {
                "attempt": attempts,
                "returncode": last.returncode,
                "elapsed_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                "code_sha256": hashlib.sha256(current.encode("utf-8")).hexdigest(),
                "stderr_tail": last.stderr[-2000:],
                "stdout_tail": last.stdout[-2000:],
                "repair_applied": False,
                "output_truncated": last.output_truncated,
            }
            if last.returncode == 0:
                history.append(entry)
                lifecycle[-1] = {"stage": "execute", "status": "complete", "attempt": attempts, "message": f"Attempt {attempts} completed successfully."}
                break
            if last.cancelled:
                history.append(entry)
                break
            if lifecycle[-1].get("stage") != "resource_limit":
                lifecycle[-1] = {"stage": "execute", "status": "failed", "attempt": attempts, "message": f"Attempt {attempts} returned exit code {last.returncode}."}
            lifecycle.append({"stage": "traceback", "status": "captured", "attempt": attempts, "message": _traceback_summary(last.stderr)})
            if attempts > retries:
                history.append(entry)
                break
            repaired = repairer(current, last.stderr)
            if repaired == current:
                history.append(entry)
                lifecycle.append({"stage": "repair", "status": "not_available", "attempt": attempts, "message": "No safe repair was produced."})
                break
            repairs += 1
            entry["repair_applied"] = True
            history.append(entry)
            lifecycle.append({"stage": "repair", "status": "complete", "attempt": attempts, "message": "A bounded repair was applied from the captured traceback."})
            current = repaired
        assert last is not None
        return ExecutionRecord(
            "cancelled" if last.cancelled else "completed" if last.returncode == 0 else "failed",
            last.stdout[-20000:],
            last.stderr[-20000:],
            last.returncode,
            (time.perf_counter() - started) * 1000,
            attempts,
            peak_memory,
            tuple(history),
            current,
            repairs,
            tuple(lifecycle + [{"stage": "profile", "status": "complete", "message": f"Recorded {round((time.perf_counter() - started) * 1000, 3)} ms elapsed time and {peak_memory} peak bytes."}]),
        )


def repair_scientific_code(code: str, traceback: str) -> str:
    """Repair common missing scientific imports without inventing new logic."""

    repairs = (
        ("name 'math' is not defined", "import math"),
        ("name 'np' is not defined", "import numpy as np"),
        ("name 'pd' is not defined", "import pandas as pd"),
        ("name 'sp' is not defined", "import sympy as sp"),
    )
    for message, statement in repairs:
        if message in traceback and statement not in code:
            return f"{statement}\n{code}"
    return code


_BLOCKED_IMPORTS = frozenset({
    "aiohttp", "builtins", "ctypes", "ftplib", "http", "httpx", "importlib", "multiprocessing",
    "os", "pathlib", "requests", "shutil", "signal", "smtplib", "socket", "subprocess", "urllib", "webbrowser",
})
_BLOCKED_CALLS = frozenset({"breakpoint", "compile", "delattr", "eval", "exec", "getattr", "globals", "input", "locals", "open", "setattr", "vars", "__import__"})
_BLOCKED_ATTRIBUTES = frozenset({
    "call", "check_call", "check_output", "connect", "execv", "execve", "open", "popen", "remove",
    "rename", "replace", "rmdir", "rmtree", "run", "spawn", "system", "unlink", "urlopen", "write_bytes", "write_text",
})


def execution_policy_violations(code: str) -> tuple[str, ...]:
    """Return conservative AST policy violations without executing source."""

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return (f"invalid Python syntax at line {exc.lineno or 0}",)
    violations: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BLOCKED_IMPORTS:
                    violations.add(f"blocked import '{root}'")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _BLOCKED_IMPORTS or node.level:
                violations.add(f"blocked import '{node.module or 'relative'}'")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                violations.add(f"blocked call '{node.func.id}'")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_ATTRIBUTES:
                violations.add(f"blocked call '*.{node.func.attr}'")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.add(f"blocked attribute '{node.attr}'")
    return tuple(sorted(violations))


def _traceback_summary(traceback: str) -> str:
    lines = [line.strip() for line in traceback.splitlines() if line.strip()]
    return (lines[-1] if lines else "Execution failed without a traceback.")[:500]
