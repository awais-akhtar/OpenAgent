"""Opt-in integration commands and existing platform-harness evidence.

No commands run on import. Operator commands are trusted executable code, not
a sandbox. POSIX cleanup contains the process group, not deliberately detached
sessions; Windows uses a kill-on-close Job Object assigned before child startup.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time

from ..process_policy import (
    CancellationSignal, DEFAULT_ENV_KEYS, ProcessPolicy, ProcessResult,
    _contained_cwd, _validate_executable,
)
from .platform_certification.decoder_matrix import BACKENDS
from .platform_certification.__main__ import INTEGRATION_TESTS


@dataclass(frozen=True)
class VerificationCommand:
    name: str
    stage: str  # build, regression, smoke, or platform
    argv: tuple[str, ...]
    cwd: Path
    timeout_seconds: float = 900.0
    required: bool = True
    enabled: bool = True
    skip_reason: str = ''


@dataclass(frozen=True)
class PlatformEvidence:
    backend: str
    command: str
    report: Path
    device_index: int = 0
    min_memory_bytes: int = 0
    device_name: str = ''


@dataclass(frozen=True)
class VerificationPlan:
    workspace: Path
    commands: tuple[VerificationCommand, ...]
    platform_evidence: tuple[PlatformEvidence, ...]
    required_platforms: tuple[str, ...] = BACKENDS
    max_output_bytes: int = 1024 * 1024
    max_report_bytes: int = 4 * 1024 * 1024
    max_total_output_bytes: int = 16 * 1024 * 1024


class _WindowsJob:
    """Contain a suspended child before it can create descendants."""

    def __init__(self):
        from ctypes import wintypes as W

        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', W.DWORD), ('minimum', ctypes.c_size_t),
                        ('maximum', ctypes.c_size_t), ('active_limit', W.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', W.DWORD),
                        ('scheduling', W.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]

        class ThreadEntry(ctypes.Structure):
            _fields_ = [('size', W.DWORD), ('usage', W.DWORD), ('tid', W.DWORD),
                        ('pid', W.DWORD), ('base_priority', W.LONG),
                        ('delta_priority', W.LONG), ('flags', W.DWORD)]

        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, W.LPCWSTR], W.HANDLE),
            'SetInformationJobObject': ([W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD], W.BOOL),
            'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL),
            'TerminateJobObject': ([W.HANDLE, W.UINT], W.BOOL),
            'CloseHandle': ([W.HANDLE], W.BOOL),
            'CreateToolhelp32Snapshot': ([W.DWORD, W.DWORD], W.HANDLE),
            'Thread32First': ([W.HANDLE, ctypes.POINTER(ThreadEntry)], W.BOOL),
            'Thread32Next': ([W.HANDLE, ctypes.POINTER(ThreadEntry)], W.BOOL),
            'OpenThread': ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
            'ResumeThread': ([W.HANDLE], W.DWORD),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = args, result
        self.ThreadEntry = ThreadEntry
        self.handle = self.k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.k.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach_and_resume(self, process):
        if not self.k.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        snapshot = self.k.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            item = self.ThreadEntry()
            item.size = ctypes.sizeof(item)
            present = self.k.Thread32First(snapshot, ctypes.byref(item))
            while present:
                if item.pid == process.pid:
                    thread = self.k.OpenThread(2, False, item.tid)
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if self.k.ResumeThread(thread) == 0xffffffff:
                            raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        self.k.CloseHandle(thread)
                    return
                item.size = ctypes.sizeof(item)
                present = self.k.Thread32Next(snapshot, ctypes.byref(item))
            raise RuntimeError('suspended child primary thread unavailable')
        finally:
            self.k.CloseHandle(snapshot)

    def terminate(self):
        if not self.k.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None


def _execute(command, workspace, limit, cancel_event):
    """Reuse process-policy admission/results; strengthen descendant cleanup."""
    policy = ProcessPolicy(timeout_seconds=command.timeout_seconds,
                           max_output_bytes=limit, allowed_cwd_roots=(workspace,))
    cwd = _contained_cwd(command.cwd, policy.allowed_cwd_roots)
    executable = command.argv[0]
    if Path(executable).is_absolute() or '/' in executable or '\\' in executable:
        executable = str((cwd / executable).resolve())
    else:
        executable = shutil.which(executable) or executable
    argv = [executable, *command.argv[1:]]
    _validate_executable(argv, policy.allowed_executables)
    if os.name == 'nt' and Path(executable).suffix.lower() in ('.bat', '.cmd'):
        raise ValueError('batch commands require an operator-supplied non-shell executable instead')
    env = {key: value for key, value in os.environ.items() if key.upper() in DEFAULT_ENV_KEYS}
    job = None
    process = None
    readers = []
    output = [bytearray(), bytearray()]
    lock = threading.Lock()
    overflow, stopped, read_error = threading.Event(), threading.Event(), threading.Event()
    used = 0
    timed_out = cancelled = False
    cleanup_errors = []

    def drain(index, stream):
        nonlocal used
        try:
            while not stopped.is_set():
                chunk = os.read(stream.fileno(), 8192)
                if not chunk:
                    break
                with lock:
                    remaining = max(0, limit-used)
                    output[index].extend(chunk[:remaining])
                    used += min(len(chunk), remaining)
                    if len(chunk) > remaining:
                        overflow.set()
        except OSError:
            if not stopped.is_set():
                read_error.set()
        finally:
            stream.close()

    try:
        flags = 0
        if os.name == 'nt':
            job = _WindowsJob()
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW | 4  # CREATE_SUSPENDED
        process = subprocess.Popen(argv, cwd=str(cwd), env=env,
                                   shell=False, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=os.name != 'nt', creationflags=flags)
        for index, stream in enumerate((process.stdout, process.stderr)):
            reader = threading.Thread(target=drain, args=(index, stream), daemon=True)
            reader.start()
            readers.append(reader)
        if job is not None:
            job.attach_and_resume(process)
        deadline = time.monotonic()+policy.timeout_seconds
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            if overflow.is_set() or read_error.is_set():
                break
            try:
                process.wait(timeout=policy.poll_interval_seconds)
            except subprocess.TimeoutExpired:
                pass
    except Exception as error:
        if process is None:
            raise
        cleanup_errors.append(f'execution setup/monitor failure: {type(error).__name__}: {error}'[:4096])
    finally:
        if process is not None:
            # Also clean descendants when the leader has already exited.
            try:
                if job is not None:
                    job.terminate()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except OSError as error:
                cleanup_errors.append(str(error))
            if job is not None:
                job.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    cleanup_errors.append('child could not be reaped')
            for reader in readers:
                reader.join(timeout=1)
            stopped.set()
            for reader, stream in zip(readers, (process.stdout, process.stderr)):
                if reader.is_alive():
                    cleanup_errors.append('output pipe remained open after tree cleanup')
                    # The reader owns closure; closing/reusing its fd here could
                    # race a blocked read in a deliberately detached descendant.
            for stream in (process.stdout, process.stderr)[len(readers):]:
                stream.close()
        elif job is not None:
            job.close()
    with lock:
        stdout, stderr = (bytes(value).decode('utf-8', errors='replace') for value in output)
    result = ProcessResult(argv=tuple(argv), returncode=process.returncode,
                           stdout=stdout, stderr=stderr, timed_out=timed_out,
                           output_truncated=overflow.is_set(), cancelled=cancelled)
    return result, cleanup_errors, read_error.is_set()


def _suite_evidence(stdout):
    prefix = 'OPENAGENT_SUITE_EVIDENCE '
    lines = [line[len(prefix):] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError('required regression must emit the existing suite evidence marker exactly once')
    return _validate_suite(json.loads(lines[0]))


def _validate_suite(suite):
    if not isinstance(suite, dict) or suite.get('passed') is not True:
        raise ValueError('suite did not pass')
    if type(suite.get('tests_run')) is not int or suite['tests_run'] <= 0:
        raise ValueError('suite execution count missing')
    for key in ('skipped', 'failures', 'errors', 'expected_failures', 'unexpected_successes'):
        if type(suite.get(key)) is not int or suite[key] != 0:
            raise ValueError('suite has missing or unsuccessful checks: '+key)
    if suite.get('load_errors') != []:
        raise ValueError('suite load-error inventory missing or nonempty')
    successes = suite.get('successful_tests')
    if not isinstance(successes, list) or len(successes) != suite['tests_run'] or any(
            not isinstance(name, str) or not name for name in successes):
        raise ValueError('successful test inventory incomplete')
    return suite


def _platform_evidence(requirement, workspace, since, limit):
    path = _contained_cwd(requirement.report, (workspace,))
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('platform evidence exceeds report admission')
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get('schema_version') != 4:
        raise ValueError('existing platform-harness schema 4 report required')
    stamp = datetime.fromisoformat(report.get('timestamp', ''))
    if stamp.tzinfo is None or not since <= stamp.timestamp() <= time.time()+5:
        raise ValueError('platform evidence is stale or has no aware timestamp')
    if report.get('backend') != requirement.backend or report.get('status') != 'passed' or report.get('scope') != 'engine':
        raise ValueError('requested engine platform evidence did not pass')
    probe = report.get('device_probe_evidence', {})
    runtime = probe.get('runtime', {})
    if probe.get('native_executed') is not True or runtime.get('executed') is not True or runtime.get('backend') != requirement.backend:
        raise ValueError('native/device execution evidence absent')
    if type(runtime.get('device_index')) is not int or runtime['device_index'] != requirement.device_index:
        raise ValueError('requested device was not observed')
    if requirement.min_memory_bytes and (type(runtime.get('memory_bytes')) is not int or runtime['memory_bytes'] < requirement.min_memory_bytes):
        raise ValueError('requested device memory not evidenced')
    if requirement.device_name and requirement.device_name not in runtime.get('name', ''):
        raise ValueError('requested device identity not evidenced')
    engine = report.get('engine_backend_evidence', {})
    if engine.get('executed') is not True or engine.get('backend') != requirement.backend:
        raise ValueError('engine/backend execution evidence absent')
    suite = _validate_suite(engine.get('suite'))
    if INTEGRATION_TESTS[requirement.backend] not in suite['successful_tests']:
        raise ValueError('required platform integration check missing')
    commands = report.get('commands')
    if not isinstance(commands, list) or not commands or any(
            not isinstance(row, dict) or type(row.get('returncode')) is not int or row['returncode'] != 0
            for row in commands):
        raise ValueError('platform command execution inventory incomplete')
    return {'backend': requirement.backend, 'report': str(path),
            'sha256': hashlib.sha256(raw).hexdigest(), 'status': 'evidence_accepted',
            'scope': 'engine', 'tests_run': suite['tests_run'], 'certified': False}


def _evidence_stamp(requirement, workspace):
    path = _contained_cwd(requirement.report, (workspace,))
    try:
        stamp = path.stat()
    except FileNotFoundError:
        return None
    return (stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns)


def verify_integration(plan: VerificationPlan, *, cancel_event: CancellationSignal | None = None):
    """Run only explicitly supplied commands; never certify a decoder or platform.

    Regression commands should invoke platform_certification.suite so skipped
    tests are observable. Platform commands should invoke the existing engine
    platform harness with --report pointing at their evidence requirement.
    Required builds/regressions/smokes/platforms cannot be omitted or skipped.
    """
    workspace = Path(plan.workspace).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError('verification workspace must be a directory')
    if any(type(value) is not int or not 0 < value <= 64*1024*1024
           for value in (plan.max_output_bytes, plan.max_report_bytes, plan.max_total_output_bytes)):
        raise ValueError('verification output/report budgets must be positive and bounded')
    if not plan.commands or len(plan.commands) > 256 or len(plan.platform_evidence) > 32:
        raise ValueError('nonempty bounded verification command plan required')
    names = set()
    for command in plan.commands:
        if (not isinstance(command.name, str) or not 0<len(command.name)<=128 or command.name in names or
                command.stage not in ('build', 'regression', 'smoke', 'platform') or
                type(command.required) is not bool or type(command.enabled) is not bool or
                not isinstance(command.skip_reason,str) or len(command.skip_reason)>4096):
            raise ValueError('unique named verification stages required')
        names.add(command.name)
        if (not isinstance(command.argv, tuple) or not command.argv or len(command.argv) > 1024 or
                any(not isinstance(arg, str) or not arg or '\0' in arg for arg in command.argv) or
                sum(len(arg) for arg in command.argv) > 128*1024):
            raise ValueError('bounded explicit nonempty argv tuple required; no shell strings')
        if isinstance(command.timeout_seconds, bool) or not isinstance(command.timeout_seconds, (float, int)) or not math.isfinite(command.timeout_seconds) or not 0 < command.timeout_seconds <= 86400:
            raise ValueError('finite positive bounded command timeout required')
        _contained_cwd(command.cwd, (workspace,))
    platforms = tuple(plan.required_platforms)
    if not platforms or len(set(platforms)) != len(platforms) or any(name not in BACKENDS for name in platforms):
        raise ValueError('explicit unique required platform inventory required')
    inventory = set()
    by_name = {command.name: command for command in plan.commands}
    for evidence in plan.platform_evidence:
        if evidence.backend not in platforms or evidence.backend in inventory:
            raise ValueError('one evidence requirement per requested platform')
        inventory.add(evidence.backend)
        if evidence.command not in by_name or by_name[evidence.command].stage != 'platform' or not by_name[evidence.command].required:
            raise ValueError('platform evidence must bind a required platform command')
        if type(evidence.device_index) is not int or evidence.device_index < 0 or type(evidence.min_memory_bytes) is not int or evidence.min_memory_bytes < 0 or not isinstance(evidence.device_name, str) or len(evidence.device_name)>1024:
            raise ValueError('invalid required device evidence geometry')
        _contained_cwd(evidence.report, (workspace,))
    report = {'schema_version': 1, 'status': 'failed', 'certified': False,
              'full_model_certified': False, 'commands': [], 'platform_evidence': [],
              'required_platforms': list(platforms), 'gaps': [],
              'trust': 'operator-supplied commands and local reports, not signed hardware attestation',
              'containment': 'Windows Job Object' if os.name == 'nt' else 'POSIX process group; detached sessions are outside containment'}
    for stage in ('build', 'regression', 'smoke'):
        if not any(command.stage == stage and command.required for command in plan.commands):
            report['gaps'].append('missing required '+stage+' command')
    for backend in platforms:
        if backend not in inventory:
            report['gaps'].append('missing required platform evidence plan: '+backend)
    blocked = False
    retained_output = 0
    for command in plan.commands:
        started = time.time()
        row = {'name': command.name, 'stage': command.stage, 'argv': list(command.argv),
               'operator_argv': list(command.argv),
               'cwd': str(Path(command.cwd).resolve()), 'required': command.required,
               'status': 'skipped', 'returncode': None}
        report['commands'].append(row)
        if not command.enabled or blocked or (cancel_event is not None and cancel_event.is_set()):
            row['reason'] = command.skip_reason if not command.enabled else 'prior required failure or cancellation'
            if command.required:
                report['gaps'].append('required command skipped: '+command.name)
                blocked = True
            continue
        clock = time.monotonic()
        try:
            requirements = [evidence for evidence in plan.platform_evidence if evidence.command == command.name]
            prior = {evidence.backend: _evidence_stamp(evidence, workspace) for evidence in requirements}
            # Replacement decoding can expand one captured byte to three UTF-8
            # bytes, so admit aggregate retained text before launching a child.
            allowance = min(plan.max_output_bytes, (plan.max_total_output_bytes-retained_output)//3)
            if allowance <= 0:
                raise MemoryError('aggregate verification output budget exhausted')
            result, cleanup_errors, read_error = _execute(command, workspace, allowance, cancel_event)
            retained_output += len(result.stdout.encode('utf-8'))+len(result.stderr.encode('utf-8'))
            row.update(asdict(result), cleanup_errors=cleanup_errors)
            if result.returncode != 0 or result.timed_out or result.cancelled or result.output_truncated or cleanup_errors or read_error:
                raise RuntimeError('command failed, exceeded bounds, or could not clean up completely')
            if command.stage == 'regression':
                row['suite_evidence'] = _suite_evidence(result.stdout)
            accepted_evidence = []
            for evidence in requirements:
                if _evidence_stamp(evidence, workspace) == prior[evidence.backend]:
                    raise ValueError('platform report was not produced or updated by this command')
                accepted_evidence.append(_platform_evidence(evidence, workspace, started, plan.max_report_bytes))
            report['platform_evidence'].extend(accepted_evidence)
            row['status'] = 'passed'
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}'[:4096])
            if command.required:
                report['gaps'].append('required command failed: '+command.name)
                blocked = True
        finally:
            row['duration_seconds'] = time.monotonic()-clock
    accepted = {evidence['backend'] for evidence in report['platform_evidence']}
    for backend in platforms:
        if backend not in accepted:
            report['gaps'].append('required platform evidence not accepted: '+backend)
    if cancel_event is not None and cancel_event.is_set():
        report['gaps'].append('verification cancelled')
    report['status'] = 'passed' if not report['gaps'] else 'failed'
    return report
