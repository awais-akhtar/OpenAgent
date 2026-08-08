"""Local code execution records with bounded retries and profiling."""

from __future__ import annotations

import subprocess
import sys
import time
import hashlib
import os
from dataclasses import dataclass
from typing import Callable


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
    """Runs short local scripts and exposes traceback data for repair loops."""

    def run(self, code: str, *, timeout_seconds: float = 10.0, retries: int = 0, repair: Callable[[str, str], str] | None = None) -> ExecutionRecord:
        current = code
        repairer = repair or repair_scientific_code
        started = time.perf_counter()
        last: subprocess.CompletedProcess[str] | None = None
        attempts = 0
        peak_memory = 0
        history: list[dict[str, object]] = []
        lifecycle: list[dict[str, object]] = [{"stage": "code_prepared", "status": "complete", "message": "Python source prepared for isolated local execution."}]
        repairs = 0
        for attempts in range(1, max(1, retries + 1) + 1):
            lifecycle.append({"stage": "execute", "status": "running", "attempt": attempts, "message": f"Running attempt {attempts}."})
            attempt_started = time.perf_counter()
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", current],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + max(0.1, timeout_seconds)
            while process.poll() is None:
                peak_memory = max(peak_memory, _process_memory_bytes(process.pid))
                if time.monotonic() >= deadline:
                    process.kill()
                    stdout, stderr = process.communicate()
                    last = subprocess.CompletedProcess(process.args, 124, stdout, f"{stderr}\nExecution timed out.".strip())
                    break
                time.sleep(0.01)
            else:
                stdout, stderr = process.communicate()
                last = subprocess.CompletedProcess(process.args, int(process.returncode or 0), stdout, stderr)
            entry: dict[str, object] = {
                "attempt": attempts,
                "returncode": last.returncode,
                "elapsed_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                "code_sha256": hashlib.sha256(current.encode("utf-8")).hexdigest(),
                "stderr_tail": last.stderr[-2000:],
                "stdout_tail": last.stdout[-2000:],
                "repair_applied": False,
            }
            if last.returncode == 0:
                history.append(entry)
                lifecycle[-1] = {"stage": "execute", "status": "complete", "attempt": attempts, "message": f"Attempt {attempts} completed successfully."}
                break
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
            "completed" if last.returncode == 0 else "failed",
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


def _process_memory_bytes(pid: int) -> int:
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            access = 0x0400 | 0x0010
            handle = ctypes.windll.kernel32.OpenProcess(access, False, pid)
            if not handle:
                return 0
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            ctypes.windll.kernel32.CloseHandle(handle)
            return int(counters.PeakWorkingSetSize) if ok else 0
        status = open(f"/proc/{pid}/status", encoding="utf-8").read()
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, AttributeError):
        return 0
    return 0


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


def _traceback_summary(traceback: str) -> str:
    lines = [line.strip() for line in traceback.splitlines() if line.strip()]
    return (lines[-1] if lines else "Execution failed without a traceback.")[:500]
