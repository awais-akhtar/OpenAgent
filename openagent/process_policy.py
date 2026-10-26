"""Shared policy for bounded, non-shell child process execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import threading
import time
from typing import Callable, Iterable, Mapping, Protocol


DEFAULT_ENV_KEYS = frozenset(
    {
        "COMSPEC",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
)


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class ProcessPolicy:
    timeout_seconds: float = 30.0
    max_output_bytes: int = 100_000
    max_input_bytes: int = 1_000_000
    max_memory_bytes: int | None = None
    poll_interval_seconds: float = 0.02
    allowed_cwd_roots: tuple[Path, ...] = ()
    allowed_executables: tuple[str, ...] = ()
    env_keys: frozenset[str] = DEFAULT_ENV_KEYS


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    output_truncated: bool
    cancelled: bool = False
    resource_limited: bool = False
    peak_memory_bytes: int = 0


def parse_argv_template(template: str, **values: str) -> list[str]:
    """Render a command template as argv without invoking a shell."""

    if not template.strip():
        raise ValueError("command template is empty")
    try:
        rendered = template.format(**values)
    except (KeyError, ValueError) as exc:
        raise ValueError("command template may use only declared placeholders") from exc
    parts = shlex.split(rendered, posix=os.name != "nt")
    if os.name == "nt":
        parts = [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in parts]
    if not parts:
        raise ValueError("command template produced no executable")
    return parts


def _contained_cwd(cwd: str | Path | None, roots: Iterable[Path]) -> Path:
    resolved = Path(cwd or Path.cwd()).expanduser().resolve()
    allowed = tuple(Path(root).expanduser().resolve() for root in roots)
    if allowed and not any(resolved == root or root in resolved.parents for root in allowed):
        raise ValueError("process cwd is outside the configured workspace boundary")
    return resolved


def _validate_executable(argv: list[str], allowed: tuple[str, ...]) -> None:
    executable = argv[0].strip()
    resolved = shutil.which(executable) or (str(Path(executable).expanduser().resolve()) if Path(executable).exists() else "")
    if not resolved:
        raise FileNotFoundError(f"executable not found: {executable}")
    if not allowed:
        return
    accepted: set[str] = set()
    for item in allowed:
        configured = str(item).strip()
        if not configured:
            continue
        accepted.add(Path(configured).name.casefold())
        configured_resolved = shutil.which(configured)
        if configured_resolved:
            accepted.add(str(Path(configured_resolved).resolve()).casefold())
            accepted.add(Path(configured_resolved).name.casefold())
        elif Path(configured).is_absolute():
            accepted.add(str(Path(configured).expanduser().resolve()).casefold())
    if str(Path(resolved).resolve()).casefold() not in accepted and Path(resolved).name.casefold() not in accepted:
        raise PermissionError(f"executable is not allowlisted: {Path(resolved).name}")


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=1)
        else:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _process_memory_bytes(pid: int) -> int:
    """Return the process peak working set when the host exposes it."""

    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
            if not handle:
                return 0
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            ctypes.windll.kernel32.CloseHandle(handle)
            return int(counters.PeakWorkingSetSize) if ok else 0
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith(("VmHWM:", "VmRSS:")):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, AttributeError):
        return 0
    return 0


def run_process(
    argv: list[str],
    *,
    cwd: str | Path | None = None,
    policy: ProcessPolicy | None = None,
    env: Mapping[str, str] | None = None,
    input_data: str | bytes | None = None,
    cancel_event: CancellationSignal | None = None,
    on_output: Callable[[str, str], None] | None = None,
) -> ProcessResult:
    """Run argv with cwd, environment, timeout, and output limits enforced."""

    effective = policy or ProcessPolicy()
    clean_argv = [str(part) for part in argv if str(part)]
    if not clean_argv:
        raise ValueError("command is required")
    _validate_executable(clean_argv, effective.allowed_executables)
    resolved_cwd = _contained_cwd(cwd, effective.allowed_cwd_roots)
    source_env = dict(os.environ if env is None else env)
    child_env = {key: value for key, value in source_env.items() if key.upper() in effective.env_keys}
    encoded_input = input_data.encode("utf-8") if isinstance(input_data, str) else input_data
    if encoded_input is not None and len(encoded_input) > max(0, int(effective.max_input_bytes)):
        raise ValueError("process input exceeds the configured policy limit")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        clean_argv,
        cwd=str(resolved_cwd),
        env=child_env,
        stdin=subprocess.PIPE if encoded_input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        creationflags=flags,
        start_new_session=os.name != "nt",
    )
    limit = max(1_000, int(effective.max_output_bytes))
    output: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = threading.Event()

    def drain(name: str, stream: object) -> None:
        pending = bytearray()

        def publish_lines(*, final: bool = False) -> None:
            if on_output is None:
                return
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    if final and pending:
                        raw = bytes(pending)
                        pending.clear()
                    else:
                        return
                else:
                    raw = bytes(pending[: newline + 1])
                    del pending[: newline + 1]
                try:
                    on_output(name, raw.decode("utf-8", errors="replace"))
                except Exception:
                    # Output persistence is best-effort; it must never deadlock
                    # or interrupt the supervised child process.
                    pass

        while True:
            # readline returns as soon as a process flushes a normal line,
            # unlike buffered read(8192), which can withhold small output
            # until EOF and defeats incremental terminal display.
            chunk = stream.readline(8192)  # type: ignore[attr-defined]
            if not chunk:
                break
            remaining = limit - len(output[name])
            if remaining > 0:
                accepted = chunk[:remaining]
                output[name].extend(accepted)
                if on_output is not None:
                    pending.extend(accepted)
                    publish_lines()
            if len(chunk) > remaining:
                truncated.set()
        publish_lines(final=True)

    readers = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    writer: threading.Thread | None = None
    if encoded_input is not None and process.stdin is not None:
        def write_input() -> None:
            try:
                process.stdin.write(encoded_input)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        writer = threading.Thread(target=write_input, daemon=True)
        writer.start()
    timed_out = False
    cancelled = False
    resource_limited = False
    peak_memory = 0
    deadline = time.monotonic() + max(0.1, float(effective.timeout_seconds))
    poll_interval = max(0.005, min(float(effective.poll_interval_seconds), 0.25))
    while process.poll() is None:
        peak_memory = max(peak_memory, _process_memory_bytes(process.pid))
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            _terminate_process_tree(process)
            break
        if effective.max_memory_bytes is not None and peak_memory > max(1, int(effective.max_memory_bytes)):
            resource_limited = True
            _terminate_process_tree(process)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate_process_tree(process)
            break
        try:
            process.wait(timeout=poll_interval)
        except subprocess.TimeoutExpired:
            continue
    if process.poll() is None:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    peak_memory = max(peak_memory, _process_memory_bytes(process.pid))
    if writer is not None:
        writer.join(timeout=1)
    for reader in readers:
        reader.join(timeout=2)
    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        process.stderr.close()
    return ProcessResult(
        argv=tuple(clean_argv),
        returncode=130 if cancelled else 125 if resource_limited else 124 if timed_out else int(process.returncode or 0),
        stdout=bytes(output["stdout"]).decode("utf-8", errors="replace"),
        stderr=bytes(output["stderr"]).decode("utf-8", errors="replace"),
        timed_out=timed_out,
        output_truncated=truncated.is_set(),
        cancelled=cancelled,
        resource_limited=resource_limited,
        peak_memory_bytes=peak_memory,
    )
