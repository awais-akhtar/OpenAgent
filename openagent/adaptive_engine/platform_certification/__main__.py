"""Trusted argv orchestration. SDK presence alone cannot produce a pass."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from .decoder_matrix import DECODERS, certification_matrix

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
SUITES = {
    "linux": ["tests.test_linux_engine_certification.LinuxEngineCertification"],
    "cuda": ["tests.test_cuda_hip_certification.CudaHipCertificationTests"],
    "hip": ["tests.test_cuda_hip_certification.CudaHipCertificationTests"],
    "metal": ["tests.test_metal_certification.MetalCertificationTests"],
}
INTEGRATION_TESTS = {
    "linux": "tests.test_linux_engine_certification.LinuxEngineCertification.test_real_reads_and_store_integrity",
    "cuda": "tests.test_cuda_hip_certification.CudaHipCertificationTests.test_payloads_scalar_batch_routes_and_dimensions",
    "hip": "tests.test_cuda_hip_certification.CudaHipCertificationTests.test_payloads_scalar_batch_routes_and_dimensions",
    "metal": "tests.test_metal_certification.MetalCertificationTests.test_scalar_and_grouped_batch_all_payloads",
}


class Unavailable(RuntimeError):
    """Required platform prerequisites are absent; never a successful skip."""


def validate_argv(argv):
    if not isinstance(argv, list) or not argv or any(
            not isinstance(arg, str) or not arg or "\0" in arg for arg in argv):
        raise ValueError("commands must be nonempty explicit string argv lists")


def command(argv, env, timeout, records):
    validate_argv(argv)
    # No shell, string splitting, eval, or implicit command-language expansion.
    started = time.monotonic()
    record = {"argv": argv, "returncode": None}
    records.append(record)
    try:
        result = subprocess.run(argv, cwd=ROOT, env=env, shell=False,
                                capture_output=True, text=True, timeout=timeout)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        unavailable = [line[len("OPENAGENT_RUNTIME_UNAVAILABLE "):]
                       for line in result.stdout.splitlines()
                       if line.startswith("OPENAGENT_RUNTIME_UNAVAILABLE ")]
        if result.returncode == 2 and len(unavailable) == 1:
            raise Unavailable(json.loads(unavailable[0])["reason"])
        if result.returncode:
            raise RuntimeError(f"command exited {result.returncode}: {argv[0]}")
        return result.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["error"] = str(exc)
        raise
    finally:
        record["duration_seconds"] = time.monotonic() - started


def tool(name):
    executable = shutil.which(name)
    if not executable:
        raise Unavailable(f"required native build tool unavailable: {name}")
    return executable


def certify(args, report):
    report["stage"] = "preflight"
    report["decoder_backend_matrix"] = certification_matrix(args.backend)
    report["full_model_certified"] = False
    report["routed_expert_certified"] = False
    report["full_model_evidence"] = {"executed": False, "certified": False}
    if args.scope == "full-model":
        report["stage"] = "decoder_prerequisites"
        if not args.decoder:
            raise ValueError("full-model scope requires an explicit --decoder")
        row = next(row for row in report["decoder_backend_matrix"]
                   if row["decoder"] == args.decoder)
        report["full_model_evidence"] = dict(row)
    elif getattr(args, "decoder", None):
        raise ValueError("--decoder requires full-model scope")
    if args.scope != "full-model" and getattr(args, "decoder_manifest", None):
        raise ValueError("--decoder-manifest requires full-model scope")
    required_host = "Darwin" if args.backend == "metal" else "Linux"
    if platform.system() != required_host:
        raise Unavailable(f"{args.backend} requires {required_host}, got {platform.system()}")
    if args.device < 0 or args.timeout <= 0 or args.min_memory_bytes < 0:
        raise ValueError("invalid device, timeout or memory requirement")
    if args.backend in ("metal", "linux") and args.device != 0:
        raise ValueError("this backend supports only the default device index 0")
    if args.scope == "full-model":
        if args.matrix:
            raise ValueError("full-model uses --decoder-manifest, not extension matrices")
        from .full_model import certify_full_model
        certify_full_model(args, report, Unavailable)
        report["status"] = "passed"
        report["stage"] = "complete"
        return
    extra = {"suites": [], "build_commands": [], "commands": []}
    if args.matrix:
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        if not isinstance(matrix, dict) or not isinstance(matrix.get(args.backend), dict):
            raise ValueError("matrix must contain an object for the selected backend")
        entry = matrix[args.backend]
        if set(entry) - set(extra):
            raise ValueError("unknown matrix entry fields")
        extra.update(entry)
    for key in ("suites", "build_commands", "commands"):
        if not isinstance(extra[key], list):
            raise ValueError(f"matrix {key} must be a list")
    if any(not isinstance(name, str) or not name for name in extra["suites"]):
        raise ValueError("suite names must be nonempty strings")
    for argv in extra["build_commands"] + extra["commands"]:
        validate_argv(argv)
    if args.scope == "probe" and extra["suites"]:
        raise ValueError("probe scope cannot silently ignore extension suites")
    env = dict(os.environ, OPENAGENT_CERTIFY_LINUX="1", OPENAGENT_REQUIRE_URING="1",
               OPENAGENT_CUDA_HIP_CERTIFY="1", OPENAGENT_CERTIFY_METAL="1",
               OPENAGENT_CUDA_HIP_DEVICE=f"cuda:{args.device}",
               PYTORCH_ENABLE_MPS_FALLBACK="0")
    records = report["commands"]
    with tempfile.TemporaryDirectory(prefix="openagent-platform-") as directory:
        report["stage"] = "native_build"
        output = Path(directory) / "probe"
        if args.backend in ("cuda", "hip"):
            compiler = tool("nvcc" if args.backend == "cuda" else "hipcc")
            argv = [compiler, "-O2", str(SOURCE / "probe.cu"), "-o", str(output)]
            if args.backend == "hip":
                argv[1:1] = ["-DOA_HIP", "-x", "hip"]
        elif args.backend == "metal":
            argv = [tool("xcrun"), "swiftc", str(SOURCE / "probe.swift"),
                    "-framework", "Metal", "-o", str(output)]
        else:
            compiler = next((shutil.which(x) for x in ("cc", "gcc", "clang") if shutil.which(x)), None)
            if compiler is None:
                raise Unavailable("system C compiler required")
            argv = [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-shared", "-fPIC", str(SOURCE.parent / "native" / "uring_reader.c"),
                    "-o", str(output.with_suffix(".so")), "-luring"]
        command(argv, env, args.timeout, records)
        for argv in extra["build_commands"]:
            command(argv, env, args.timeout, records)
        if args.backend != "linux":
            report["stage"] = "native_execution"
            native = command([str(output), str(args.device)], env, args.timeout, records)
            if "OPENAGENT_NATIVE_EXECUTED" not in native.splitlines():
                raise RuntimeError("native execution evidence absent")
            report["device_probe_evidence"]["native_executed"] = True
        report["stage"] = "runtime_execution"
        runtime_argv = [sys.executable, "-m",
                          "openagent.adaptive_engine.platform_certification.runtime",
                          args.backend, str(args.device)]
        if args.backend == "linux":
            runtime_argv.append(str(output.with_suffix(".so")))
        stdout = command(runtime_argv, env, args.timeout, records)
        lines = [line.removeprefix("OPENAGENT_RUNTIME_EVIDENCE ") for line in stdout.splitlines()
                 if line.startswith("OPENAGENT_RUNTIME_EVIDENCE ")]
        if len(lines) != 1:
            raise RuntimeError("runtime execution evidence absent or ambiguous")
        evidence = json.loads(lines[0])
        if (evidence.get("backend") != args.backend or evidence.get("executed") is not True
                or evidence.get("device_index") != args.device):
            raise RuntimeError("backend execution evidence mismatch")
        if evidence.get("memory_bytes", 0) < args.min_memory_bytes:
            raise RuntimeError("device does not meet memory requirement")
        if args.device_name and args.device_name not in evidence.get("name", ""):
            raise RuntimeError("required device name not matched")
        report["device_probe_evidence"]["runtime"] = evidence
        if args.backend == "linux":
            report["device_probe_evidence"]["native_executed"] = True
        if args.scope == "engine":
            report["stage"] = "engine_suite"
            suite_stdout = command([
                sys.executable, "-m", "openagent.adaptive_engine.platform_certification.suite",
                *SUITES[args.backend], *extra["suites"]], env, args.timeout, records)
            suite_lines = [line.removeprefix("OPENAGENT_SUITE_EVIDENCE ")
                           for line in suite_stdout.splitlines()
                           if line.startswith("OPENAGENT_SUITE_EVIDENCE ")]
            if len(suite_lines) != 1:
                raise RuntimeError("engine integration evidence absent or ambiguous")
            suite_evidence = json.loads(suite_lines[0])
            required_test = INTEGRATION_TESTS[args.backend]
            if (suite_evidence.get("passed") is not True or
                    not isinstance(suite_evidence.get("tests_run"), int) or
                    suite_evidence["tests_run"] <= 0 or
                    any(suite_evidence.get(key) != 0 for key in (
                        "skipped", "failures", "errors", "expected_failures", "unexpected_successes")) or
                    required_test not in suite_evidence.get("successful_tests", [])):
                raise RuntimeError("required backend-specific engine integration did not succeed")
            report["engine_backend_evidence"] = {
                "executed": True, "backend": args.backend,
                "coverage": "io-integrity" if args.backend == "linux" else "routed-expert",
                "actual_decoder_executed": False,
                "implementation": "liburing/O_DIRECT" if args.backend == "linux" else "TorchExpertBackend",
                "required_integration_test": required_test, "suite": suite_evidence,
            }
        for argv in extra["commands"]:
            report["stage"] = "extension_execution"
            command(argv, env, args.timeout, records)
    report["status"] = "passed"
    report["stage"] = "complete"
    report["routed_expert_certified"] = args.scope == "engine" and args.backend != "linux"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(SUITES))
    parser.add_argument("--source-only", action="store_true",
                        help="hash source only; import no decoders and run no commands, builds or device probes")
    parser.add_argument("--engine-root", type=Path, default=SOURCE.parent)
    parser.add_argument("--scope", choices=("engine", "probe", "full-model"), default="engine",
                        help="engine covers routed experts or I/O only; full-model requires decoder certification")
    parser.add_argument("--decoder", choices=tuple(DECODERS))
    parser.add_argument("--decoder-manifest", type=Path,
                        help="local JSON with decoder, backend, commands and wholemodel comparison input")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--device-name")
    parser.add_argument("--min-memory-bytes", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.source_only:
        if (any(value is not None for value in (args.backend,args.decoder,args.decoder_manifest,args.matrix,args.device_name))
                or args.scope!='engine' or args.device!=0 or args.min_memory_bytes!=0):
            parser.error('--source-only accepts source inventory options independently of runtime certification')
        from .source_inventory import main as source_main
        return source_main(['--engine-root',str(args.engine_root),'--output',str(args.report)])
    if args.backend is None:
        parser.error('--backend is required for execution scopes')
    report = {"schema_version": 4, "backend": args.backend, "status": "failed",
              "scope": args.scope, "engine_certified": False,
              "decoder": args.decoder, "full_model_certified": False,
              "routed_expert_certified": False,
              "decoder_backend_matrix": certification_matrix(args.backend),
              "full_model_evidence": {"executed": False, "certified": False},
              "device_probe_evidence": {}, "engine_backend_evidence": {"executed": False},
              "host": platform.platform(), "python": sys.version,
              "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "revision": os.environ.get("GITHUB_SHA"), "commands": [],
              "requirements": {"device": args.device, "device_name": args.device_name,
                               "min_memory_bytes": args.min_memory_bytes}}
    try:
        certify(args, report)
        report["engine_certified"] = args.scope == "engine" and report["status"] == "passed"
    except Exception as exc:
        report["status"] = "unavailable" if isinstance(exc, Unavailable) else "failed"
        report["failure"] = {"stage": report.get("stage", "preflight"),
                             "kind": type(exc).__name__, "reason": str(exc)}
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report)}))
    return 0 if report["status"] == "passed" else 2 if report["status"] == "unavailable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
