"""Capture route-bound Phase 1 runtime observations for one admitted route.

The capture runs the exact canonical route through the production universal
path, records each observation kind from its own measured source, runs an
independent reference implementation, pins every artifact, writes the
receipt into the receipts manifest, and finally verifies the receipt with the
same gate used for Phase 1 closure.  Nothing is asserted: a capture whose
measurements disagree is rejected and not written.

Supported today: native OLMoE manifests executed either by the leased segment
engine (INT8 and Q4/Q5/Q8 block formats) or by an explicit no-fallback native
consumer (FP32, INT2/INT3/INT4, grouped INT4, MXFP4 and block FP8). Every
captured route is compared with the NumPy ``OlmoeModel`` reference; this list
describes capture paths, not a claim that each persisted format has a receipt.

The explicit ``--backend public-range`` option captures the registered native
Segment/Edge public ABI for INT8-row manifests. It is a distinct canonical
route and never uses a native-backend receipt as a substitute.

    python -m openagent.adaptive_engine.phase1_capture olmoe \\
        --checkpoint models/olmoe-1b-7b-instruct \\
        --manifest models/olmoe-1b-7b-instruct/openagent-experts.q8.manifest.json \\
        --build-library build/native --prompt "The capital of France is" --max-tokens 4
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

from .device_observations import (
    OBSERVATION_SCHEMA, RECEIPTS_SCHEMA, REFERENCE_IMPLEMENTATIONS, coverage_applicability, encode_matrix,
    load_receipts, reference_source_pins, sha256_file, verify_route_receipt)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPTS = REPOSITORY_ROOT / "phase1_runtime_receipts.json"
DEFAULT_EVIDENCE = REPOSITORY_ROOT / "phase1_evidence"
OLMOE_CONSUMER = "native_leased_olmoe_segments.NativeLeasedOlmoeSegmentEngine"
OLMOE_ABI_OWNER = "native/segment_executor.c"
OLMOE_DIRECT_CONSUMERS = {
    "float32": "native_expert.NativeExpertBackend",
    "int8-row": "native_expert.NativeExpertBackend",
    "int2-row": "native_lowbit_expert.NativeLowBitExpertBackend",
    "int3-grouped-g64": "native_lowbit_expert.NativeLowBitExpertBackend",
    "int4-row": "native_packed_expert.NativePackedExpertBackend",
    "int4-grouped-g64": "native_grouped_int4_expert.NativeGroupedInt4ExpertBackend",
    "mxfp4-e2m1-g32-ue8m0": "native_mxfp4_expert.NativeMXFP4ExpertBackend",
    "q4_0": "native_q4_0_expert.NativeQ4_0ExpertBackend",
    "q4_1": "native_q4_1_expert.NativeQ4_1ExpertBackend",
    "q5_0": "native_q5_0_expert.NativeQ5_0ExpertBackend",
    "q5_1": "native_q5_1_expert.NativeQ5_1ExpertBackend",
    "q8_0": "native_q8_0_expert.NativeQ8_0ExpertBackend",
    "q8_1": "native_q8_1_expert.NativeQ8_1ExpertBackend",
    "fp8-e4m3-b128-f32": "native_fp8_block_expert.NativeFP8BlockExpertBackend",
}
REFERENCE_IMPLEMENTATION = REFERENCE_IMPLEMENTATIONS["olmoe"]


class CaptureError(RuntimeError):
    """The route could not be captured or its measurements disagree."""


def compile_segment_executor(directory: str | Path) -> Path:
    """Compile the native segment executor with MSVC (Windows) or cc/gcc/clang."""
    source = Path(__file__).resolve().parent / "native" / "segment_executor.c"
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / ("segment_executor.dll" if os.name == "nt" else "segment_executor.so")
    if os.name == "nt":
        args = ["cl", "/nologo", "/W4", "/WX", "/O2", "/fp:precise", "/LD", str(source),
                "/link", "/OUT:" + str(output)]
        compiler = shutil.which("cl")
        if compiler:
            args[0] = compiler
            command: Any = args
        else:
            vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / \
                "Microsoft Visual Studio/Installer/vswhere.exe"
            if not vswhere.is_file():
                raise CaptureError("MSVC build tools are not installed (vswhere.exe not found)")
            location = subprocess.check_output(
                [str(vswhere), "-latest", "-products", "*", "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                text=True).strip()
            if not location:
                raise CaptureError("MSVC C++ tools are not installed")
            vcvars = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
            command = 'cmd.exe /d /s /c ""' + str(vcvars) + '" && ' + subprocess.list2cmdline(args) + '"'
    else:
        compiler = next((shutil.which(name) for name in ("cc", "gcc", "clang") if shutil.which(name)), None)
        if compiler is None:
            raise CaptureError("no C compiler found")
        command = [compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2", "-fPIC", "-shared",
                   str(source), "-o", str(output), "-lm"]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=300)
    if result.returncode or not output.is_file():
        raise CaptureError("segment executor compilation failed:\n" + result.stdout + result.stderr)
    return output


def compile_segment_runtime(directory: str | Path) -> Path:
    """Build the complete Segment/Edge public ABI runtime with its CMake target."""
    source = Path(__file__).resolve().parent / "native"
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cmake = shutil.which("cmake")
    if cmake is None and os.name == "nt":
        roots = [Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")),
                 Path(os.environ.get("ProgramFiles", "C:/Program Files"))]
        vswhere = next((root / "Microsoft Visual Studio/Installer/vswhere.exe"
                        for root in roots
                        if (root / "Microsoft Visual Studio/Installer/vswhere.exe").is_file()), None)
        if vswhere is not None:
            try:
                install = subprocess.check_output(
                    [str(vswhere), "-latest", "-products", "*",
                     "-requires", "Microsoft.VisualStudio.Component.VC.CMake.Project",
                     "-property", "installationPath"], text=True, timeout=30).strip()
            except (OSError, subprocess.SubprocessError):
                install = ""
            if install:
                candidate = Path(install) / (
                    "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")
                if candidate.is_file():
                    cmake = str(candidate)
    if cmake is None:
        raise CaptureError("CMake is required to build the Segment/Edge public runtime")
    configure = [cmake, "-S", str(source), "-B", str(directory),
                 "-DOPENAGENT_BUILD_DECODERS=OFF", "-DOPENAGENT_BUILD_KERNELS=OFF",
                 "-DOPENAGENT_BUILD_URING=OFF", "-DOPENAGENT_BUILD_PLATFORM_PROBES=OFF"]
    try:
        configured = subprocess.run(configure, cwd=directory, capture_output=True,
                                    text=True, timeout=300)
        if configured.returncode:
            raise CaptureError("Segment/Edge CMake configuration failed:\n" +
                               configured.stdout + configured.stderr)
        built = subprocess.run([cmake, "--build", str(directory), "--config", "Release",
                                "--target", "openagent_segment_runtime"],
                               cwd=directory, capture_output=True, text=True, timeout=900)
        if built.returncode:
            raise CaptureError("Segment/Edge public runtime build failed:\n" +
                               built.stdout + built.stderr)
    except subprocess.TimeoutExpired as error:
        raise CaptureError("Segment/Edge public runtime build timed out") from error
    filename = "openagent_segment_runtime.dll" if os.name == "nt" else "libopenagent_segment_runtime.so"
    outputs = (directory / "Release" / filename, directory / filename,
               directory / "lib" / filename)
    result = next((path for path in outputs if path.is_file()), None)
    if result is None:
        raise CaptureError("CMake succeeded but did not produce the Segment/Edge public runtime")
    return result.resolve(strict=True)


def canonical_row(route, *, case_id: str | None = None) -> dict[str, Any]:
    """Resolve an admitted route, requiring an exact pin for publisher variants.

    Multiple pinned publisher artifacts can share the same executable
    six-field route. A route tuple alone must not silently choose one set of
    checkpoint weights as evidence for another.
    """
    from .source_scope import source_scope_matrix
    fields = ("family", "model_type", "backend", "arithmetic", "source_format", "persisted_format")
    wanted = route.as_tuple()
    rows = [case for case in source_scope_matrix()["execution_cases"]
            if tuple(case[name] for name in fields) == wanted]
    admitted = [case for case in rows if case["combination_status"] == "verified-admitted"]
    if case_id is not None:
        if type(case_id) is not str or not case_id:
            raise CaptureError("exact nonempty publisher case ID required")
        admitted = [case for case in admitted if case["case_id"] == case_id]
    if len(admitted) != 1:
        statuses = sorted({case["combination_status"] for case in rows}) or ["absent"]
        reason = ("publisher case ID does not resolve to one admitted row" if case_id is not None else
                  "publisher case ID required for ambiguous admitted route" if len(admitted) > 1 else
                  "route is not verified-admitted")
        raise CaptureError(reason + ": " + "/".join(wanted) +
                           " (" + ", ".join(statuses) + ")")
    return admitted[0]


def _pin(path: Path, role: str) -> dict[str, str]:
    path = Path(path).resolve(strict=True)
    return {"role": role, "path": str(path), "sha256": sha256_file(path)}


def _write_observation(directory: Path, kind: str, payload: dict[str, Any], *, case_id: str,
                       config_sha: str) -> dict[str, str]:
    document = {"schema": OBSERVATION_SCHEMA, "kind": kind, "case_id": case_id,
                "config_sha256": config_sha, **payload}
    path = directory / f"{kind}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return {"kind": kind, "path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_receipts(path: Path, receipt: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _receipt_manifest_lock(path):
        # Read only after acquiring the inter-process lock: capture workers for
        # different routes share this manifest, so an unlocked read/modify/
        # replace sequence can otherwise discard a concurrent worker's row.
        receipts = load_receipts(path)
        receipts[receipt["case_id"]] = receipt
        document = {"schema": RECEIPTS_SCHEMA,
                    "routes": [receipts[key] for key in sorted(receipts)]}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=1, sort_keys=True), encoding="utf-8")
        temporary.replace(path)


@contextmanager
def _receipt_manifest_lock(path: Path, *, timeout_seconds: float = 60.0):
    """Serialize receipt-manifest updates across capture processes/platforms."""
    lock_path = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + timeout_seconds
    lock_file = lock_path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            # msvcrt.locking locks a byte range from the current file position.
            # Ensure byte zero exists; an occasional extra byte from concurrent
            # initializers is harmless because every writer locks byte zero.
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            while True:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if time.monotonic() >= deadline:
                        raise CaptureError("timed out waiting for receipt manifest lock") from error
                    time.sleep(0.05)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if time.monotonic() >= deadline:
                        raise CaptureError("timed out waiting for receipt manifest lock") from error
                    time.sleep(0.05)
        yield
    finally:
        primary = sys.exc_info()[1]
        cleanup_error = None
        try:
            if locked:
                if os.name == "nt":
                    import msvcrt
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except BaseException as error:
            cleanup_error = error
        finally:
            try:
                lock_file.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None:
            if primary is None:
                raise cleanup_error
            if hasattr(primary, 'add_note'):
                primary.add_note('Receipt lock cleanup also failed; original failure preserved.')


def _olmoe_cache_observation(model) -> dict[str, Any]:
    """Digest the direct OLMoE decoder's committed physical KV arrays."""
    import hashlib
    import numpy as np

    position = int(model.position)
    digests = []
    total = 0
    for key, value in zip(model._k, model._v):
        used_key = np.ascontiguousarray(key[:, :position, :])
        used_value = np.ascontiguousarray(value[:, :position, :])
        digest = hashlib.sha256(used_key.tobytes())
        digest.update(used_value.tobytes())
        digests.append(digest.hexdigest())
        total += used_key.nbytes + used_value.nbytes
    return {"position": position, "layers": len(digests),
            "layer_digests": digests, "bytes": int(total)}


def _greedy_olmoe_steps(model, input_ids: Sequence[int], count: int):
    """Measure real decoder logits and tokens while preserving incremental KV."""
    import numpy as np

    pending = list(input_ids)
    generated, logits_rows = [], []
    finish_reason = "length"
    for _ in range(count):
        logits = np.asarray(model.forward(pending), dtype=np.float32)
        logits_rows.append(logits.copy())
        token = int(np.argmax(logits))
        generated.append(token)
        if token == model.config.eos_token_id:
            finish_reason = "eos"
            break
        pending = [token]
    return generated, logits_rows, finish_reason


def capture_olmoe_route(checkpoint: str | Path, *, manifest: str | Path, library: str | Path | None,
                        prompt_ids: Sequence[int], max_tokens: int = 4, recovery_tokens: int = 2,
                        context_tokens: int = 64, cache_mib: int = 256, resident_mib: int = 4096,
                        transfer_mib: int = 4096, staging_mib: int = 1024,
                        conversion_mib: int = 2048, backend: str = "native",
                        case_id: str | None = None,
                        tolerance: float = 1e-3, evidence_root: str | Path = DEFAULT_EVIDENCE,
                        receipts_path: str | Path = DEFAULT_RECEIPTS) -> dict[str, Any]:
    """Capture and gate one exact native or public-range OLMoE route receipt."""
    import numpy as np
    from .expert_runtime import SparseExpertManifest
    from .olmoe import OlmoeModel
    from .universal_adapters import register_production_universal_adapters
    from .universal_engine import UniversalEngine

    checkpoint = Path(checkpoint).resolve(strict=True)
    manifest_path = Path(manifest).resolve(strict=True)
    manifest_record = SparseExpertManifest.read(manifest_path)
    formats = {layer.dtype for layer in manifest_record.layers}
    if len(formats) != 1:
        raise CaptureError("one homogeneous persisted expert format is required")
    persisted_format = next(iter(formats))
    if backend == "public-range":
        if persisted_format != "int8-row":
            raise CaptureError("the OLMoE public-range owner admits only int8-row experts")
        if library is None:
            raise CaptureError("public-range capture requires the compiled Segment/Edge runtime library")
        return _capture_olmoe_public_range_route(
            checkpoint, manifest_path, Path(library).resolve(strict=True), prompt_ids,
            max_tokens=max_tokens, recovery_tokens=recovery_tokens,
            context_tokens=context_tokens, cache_mib=cache_mib,
            resident_mib=resident_mib, transfer_mib=transfer_mib,
            staging_mib=staging_mib, conversion_mib=conversion_mib,
            case_id=case_id, tolerance=tolerance, evidence_root=evidence_root,
            receipts_path=receipts_path)
    if backend != "native":
        raise CaptureError("capture backend must be native or public-range")
    if persisted_format in OLMOE_DIRECT_CONSUMERS:
        return _capture_olmoe_direct_route(
            checkpoint, manifest_path, prompt_ids, max_tokens=max_tokens,
            recovery_tokens=recovery_tokens, context_tokens=context_tokens,
            cache_mib=cache_mib, conversion_mib=conversion_mib,
            persisted_format=persisted_format,
            case_id=case_id, tolerance=tolerance, evidence_root=evidence_root,
            receipts_path=receipts_path)
    if library is None:
        raise CaptureError("this persisted format requires the leased native segment library")
    library = Path(library).resolve(strict=True)
    prompt = [int(token) for token in prompt_ids]
    if not prompt or max_tokens <= 0 or recovery_tokens <= 0:
        raise CaptureError("a nonempty prompt and positive token counts are required")
    if len(prompt) + max_tokens + recovery_tokens > context_tokens:
        raise CaptureError("prompt, generation and recovery replay exceed the context")

    engine = UniversalEngine()
    register_production_universal_adapters(engine)
    session = engine.open(checkpoint, manifest=str(manifest_path), library=str(library),
                          context_tokens=context_tokens, cache_mib=cache_mib,
                          resident_mib=resident_mib, transfer_mib=transfer_mib,
                          staging_mib=staging_mib, conversion_mib=conversion_mib)
    try:
        route = session.manifest.route
        row = canonical_row(route, case_id=case_id)
        owner = session.owner
        runtimes = owner.expert_telemetry_sources()
        if len(runtimes) != 1:
            raise CaptureError("expected exactly one sparse expert runtime for the OLMoE route")
        runtime = runtimes[0]

        # Candidate generation with its own logits and the exact route trace.
        captured: list = []
        runtime.start_route_trace()
        try:
            result = session.generate(prompt, max_tokens=max_tokens, capture=captured)
        finally:
            trace = runtime.stop_route_trace()
        generated = list(result["generated_ids"])
        implementation = owner.numeric_class
        cache = owner.cache_observation()
        routing: dict[str, list[list[int]]] = {}
        for layer_id, rows in trace:
            routing.setdefault(layer_id, []).extend([[int(e) for e in row] for row in rows])

        # Recovery: snapshot, continue, restore, replay.
        receipt = session.snapshot()
        try:
            snapshot_position = owner.position
            continuation = session.generate([generated[-1]], max_tokens=recovery_tokens,
                                            continue_session=True)["generated_ids"]
            session.restore(receipt)
            restored_position = owner.position
            replay = session.generate([generated[-1]], max_tokens=recovery_tokens,
                                      continue_session=True)["generated_ids"]
            session.restore(receipt)
        finally:
            session.release_snapshot(receipt)
    finally:
        engine.close_all()

    # Independent reference: the NumPy OLMoE model over the same packed weights.
    reference_rows, reference_tokens, pending = [], [], prompt
    with OlmoeModel(checkpoint, SparseExpertManifest.read(manifest_path), context_tokens=context_tokens,
                    expert_cache_bytes=cache_mib * 1024 ** 2,
                    max_matrix_decode_bytes=conversion_mib * 1024 ** 2) as model:
        for _ in range(len(generated)):
            logits = np.asarray(model.forward(pending), dtype=np.float32)
            row_logits = logits[-1] if logits.ndim == 2 else logits
            reference_rows.append(row_logits)
            token = int(np.argmax(row_logits))
            reference_tokens.append(token)
            pending = [token]

    case_id = row["case_id"]
    slug = case_id.replace("/", "__")
    directory = Path(evidence_root).resolve() / slug
    directory.mkdir(parents=True, exist_ok=True)
    config_pin = _pin(checkpoint / "config.json", "config")
    config_sha = config_pin["sha256"]
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    finish = result.get("finish_reason")

    write = lambda kind, payload: _write_observation(directory, kind, payload, case_id=case_id,  # noqa: E731
                                                     config_sha=config_sha)
    pins = {
        "tokens": write("tokens", {"prompt_ids": prompt, "generated_ids": generated,
                                   "finish_reason": finish, "max_tokens": max_tokens,
                                   "decoding": "greedy", "implementation": implementation}),
        "logits": write("logits", {"matrix": encode_matrix(np.stack(captured))}),
        "cache": write("cache", cache),
        "routing": write("routing", {"layers": routing}),
        "recovery": write("recovery", {"snapshot_position": snapshot_position,
                                       "restored_position": restored_position,
                                       "continuation_ids": list(continuation),
                                       "replay_ids": list(replay)}),
        "placement": write("placement", {"backend": route.backend, "device": "cpu",
                                         "host_fallback_operations": 0,
                                         "implementation": implementation,
                                         "library_sha256": sha256_file(library)}),
    }
    pins["comparison"] = write("comparison", {
        "candidate_logits_sha256": pins["logits"]["sha256"],
        "reference": {"implementation": REFERENCE_IMPLEMENTATION,
                      "source": reference_source_pins("olmoe")[0],
                      "generated_ids": reference_tokens,
                      "logits": encode_matrix(np.stack(reference_rows))},
        "tolerance": {"max_abs_logit": float(tolerance)},
    })
    checks: dict[str, Any] = {}
    applicability = coverage_applicability(row)
    for requirement, applies in applicability.items():
        if not applies:
            checks[requirement] = {"status": "not-applicable",
                                   "reason": f"canonical route {case_id} does not require {requirement}"}
    format_pin = write("format_device", {"persisted_format": route.persisted_format,
                                         "consumer": OLMOE_CONSUMER,
                                         "placement_sha256": pins["placement"]["sha256"]})
    abi_pin = write("abi_recovery", {"abi_owner": OLMOE_ABI_OWNER,
                                     "recovery_sha256": pins["recovery"]["sha256"]})
    checks["format_device"] = {"status": "verified", "evidence": format_pin}
    checks["abi_recovery"] = {"status": "verified", "evidence": abi_pin}
    for requirement, applies in applicability.items():
        if applies and requirement not in checks:
            raise CaptureError(requirement + " applies to this route but is not captured by this runner")

    artifacts = [config_pin, _pin(manifest_path, "expert_manifest"), _pin(library, "runtime")]
    artifacts += [_pin(path, "weights") for path in sorted(checkpoint.glob("*.safetensors"))]
    payloads = sorted({checkpoint / region.path
                       for layer in SparseExpertManifest.read(manifest_path).layers
                       for region in layer.regions.values()})
    artifacts += [_pin(path, "expert_payload") for path in payloads]
    tokenizer = checkpoint / "tokenizer.json"
    if tokenizer.is_file():
        artifacts.append(_pin(tokenizer, "tokenizer"))

    receipt_record = {
        "case_id": case_id,
        "route": list(route.as_tuple()),
        "artifacts": artifacts,
        "cache_schema": {"format": "openagent.native-olmoe-decoder.v1", "layers": config["num_hidden_layers"]},
        "placement_constraints": {"backend": route.backend, "device": "cpu"},
        "comparison_adapter": REFERENCE_IMPLEMENTATION,
        "observations": [pins[kind] for kind in sorted(pins)],
        "checks": checks,
    }
    measured = verify_route_receipt(row, receipt_record)
    _write_receipts(Path(receipts_path), receipt_record)
    return measured


def _capture_olmoe_public_range_route(checkpoint: Path, manifest_path: Path, library: Path,
                                      prompt_ids: Sequence[int], *, max_tokens: int,
                                      recovery_tokens: int, context_tokens: int, cache_mib: int,
                                      resident_mib: int, transfer_mib: int, staging_mib: int,
                                      conversion_mib: int, tolerance: float,
                                      case_id: str | None,
                                      evidence_root: str | Path,
                                      receipts_path: str | Path) -> dict[str, Any]:
    """Capture the admitted Segment/Edge public-range INT8 OLMoE route."""
    import hashlib
    import numpy as np
    from .expert_runtime import SparseExpertManifest
    from .olmoe import OlmoeConfig, OlmoeModel
    from .segment_edge_adapters import NativeOlmoePublicAdapters
    from .segment_edge_runtime import SegmentEdgeExecution, SegmentRange
    from .universal_engine import ModelRoute

    mib = 1024 ** 2
    prompt = [int(token) for token in prompt_ids]
    if (not prompt or max_tokens <= 0 or recovery_tokens <= 0 or
            len(prompt) + max_tokens + recovery_tokens > context_tokens):
        raise CaptureError("prompt and generation/recovery lengths exceed the admitted context")
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_format = {"float32": "F32", "float16": "F16", "bfloat16": "BF16"}.get(
        config.get("torch_dtype"))
    if source_format is None:
        raise CaptureError("public-range capture requires an F32, F16 or BF16 source checkpoint")
    route = ModelRoute("olmoe", "olmoe", "public-range", "fp32", source_format, "int8-row")
    row = canonical_row(route, case_id=case_id)
    manifest_record = SparseExpertManifest.read(manifest_path)
    if any(layer.dtype != "int8-row" for layer in manifest_record.layers):
        raise CaptureError("public-range capture requires an exact homogeneous INT8-row manifest")
    canonical_manifest = json.dumps(manifest_record.to_dict(), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    manifest_identity = hashlib.sha256(canonical_manifest).hexdigest()
    adapters = None
    execution = None
    primary_error = None
    cleanup_errors = []
    try:
        adapters = NativeOlmoePublicAdapters(
            library, kernel_library=library, model_directory=checkpoint,
            manifest=manifest_record, expert_cache_bytes=cache_mib * mib,
            max_staging_bytes=staging_mib * mib,
            max_snapshot_bytes=max(1, cache_mib) * mib,
            max_batch_rows=min(32, context_tokens),
            max_conversion_bytes=conversion_mib * mib).register()
        execution = SegmentEdgeExecution(
            library, edge_adapter=adapters.edge_id.decode("utf-8"),
            model_directory=str(checkpoint), checkpoint_identity=manifest_identity,
            ranges=(SegmentRange(adapters.segment_id.decode("utf-8"),
                                 str(checkpoint), 0, OlmoeConfig.read(checkpoint).num_hidden_layers),),
            context_tokens=context_tokens, native_memory_bytes=resident_mib * mib,
            max_transfer_bytes=transfer_mib * mib,
            max_snapshot_bytes=max(1, cache_mib) * mib, backend_mask=1 << 8)
        engines = [resource for kind, resource in adapters.objects.values() if kind == "engine"]
        sessions = [resource for kind, resource in adapters.objects.values() if kind == "session"]
        if len(engines) != 1 or len(sessions) != 1:
            raise CaptureError("public runtime did not create exactly one engine and session owner")
        engine = engines[0]
        engine_owner = getattr(engine, "_owner", None)
        if engine_owner is None:
            raise CaptureError("public engine is not held by the registered resource ledger")
        decoder = sessions[0].model
        runtime = engine._loader.experts
        if decoder._engine is not engine_owner:
            raise CaptureError(
                "public session cache owner mismatch: decoder=" +
                type(decoder._engine).__qualname__ + f"@{id(decoder._engine):x}, engine=" +
                type(engine_owner).__qualname__ + f"@{id(engine_owner):x}")

        captured = []
        generated = []
        finish = "length"
        runtime.start_route_trace()
        try:
            pending = prompt
            for _ in range(max_tokens):
                matrix = np.asarray(execution.forward(pending), dtype=np.float32)
                if matrix.ndim != 2 or not matrix.shape[0]:
                    raise CaptureError("public Segment/Edge returned malformed logits")
                logits = matrix[-1].copy()
                if not np.isfinite(logits).all():
                    raise CaptureError("public Segment/Edge returned non-finite logits")
                captured.append(logits)
                token = int(np.argmax(logits))
                generated.append(token)
                if token == OlmoeConfig.read(checkpoint).eos_token_id:
                    finish = "eos"
                    break
                pending = [token]
        finally:
            trace = runtime.stop_route_trace()
        if not generated:
            raise CaptureError("public-range route produced no token observation")
        implementation = "openagent-native-olmoe-public-adapters"
        position = int(execution.position)
        native_cache = decoder._native
        if native_cache.position != position:
            raise CaptureError("public executor and physical cache positions differ")
        cache_digests = []
        cache_bytes = 0
        for key, value in zip(native_cache._keys, native_cache._values, strict=True):
            used_key = np.ascontiguousarray(key[:, :position, :])
            used_value = np.ascontiguousarray(value[:, :position, :])
            digest = hashlib.sha256(used_key.tobytes())
            digest.update(used_value.tobytes())
            cache_digests.append(digest.hexdigest())
            cache_bytes += used_key.nbytes + used_value.nbytes
        cache = {"position": position, "layers": len(cache_digests),
                 "layer_digests": cache_digests, "bytes": int(cache_bytes)}

        snapshot = execution.snapshot()
        try:
            snapshot_position = int(execution.position)
            continuation = _greedy_public_steps(execution, [generated[-1]], recovery_tokens,
                                                OlmoeConfig.read(checkpoint).eos_token_id)
            execution.restore(snapshot)
            restored_position = int(execution.position)
            replay = _greedy_public_steps(execution, [generated[-1]], recovery_tokens,
                                          OlmoeConfig.read(checkpoint).eos_token_id)
            execution.restore(snapshot)
        finally:
            execution.release_snapshot(snapshot)
    except BaseException as error:
        primary_error = error
        if isinstance(error, CaptureError):
            raise
        raise CaptureError("public-range OLMoE capture failed: " + str(error)) from error
    finally:
        if execution is not None:
            try:
                execution.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if adapters is not None:
            try:
                adapters.retry_cleanup()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is not None:
                for cleanup in cleanup_errors:
                    primary_error.add_note("public-range cleanup remains pending: " + str(cleanup))
            else:
                first = cleanup_errors[0]
                for cleanup in cleanup_errors[1:]:
                    first.add_note("additional public-range cleanup failure: " + str(cleanup))
                raise first

    reference_rows, reference_tokens, pending = [], [], prompt
    with OlmoeModel(checkpoint, manifest_record, context_tokens=context_tokens,
                    expert_cache_bytes=cache_mib * mib,
                    max_matrix_decode_bytes=conversion_mib * mib) as model:
        for _ in range(len(generated)):
            matrix = np.asarray(model.forward(pending), dtype=np.float32)
            reference_logits = matrix[-1] if matrix.ndim == 2 else matrix
            reference_rows.append(reference_logits)
            token = int(np.argmax(reference_logits))
            reference_tokens.append(token)
            pending = [token]

    case_id = row["case_id"]
    directory = Path(evidence_root).resolve() / case_id.replace("/", "__")
    directory.mkdir(parents=True, exist_ok=True)
    config_pin = _pin(config_path, "config")
    config_sha = config_pin["sha256"]
    write = lambda kind, payload: _write_observation(  # noqa: E731
        directory, kind, payload, case_id=case_id, config_sha=config_sha)
    routing = {layer_id: [[int(expert) for expert in route_ids] for route_ids in rows]
               for layer_id, rows in trace}
    pins = {
        "tokens": write("tokens", {"prompt_ids": prompt, "generated_ids": generated,
                                   "finish_reason": finish, "max_tokens": max_tokens,
                                   "decoding": "greedy", "implementation": implementation}),
        "logits": write("logits", {"matrix": encode_matrix(np.stack(captured))}),
        "cache": write("cache", cache),
        "routing": write("routing", {"layers": routing}),
        "recovery": write("recovery", {"snapshot_position": snapshot_position,
                                       "restored_position": restored_position,
                                       "continuation_ids": continuation,
                                       "replay_ids": replay}),
        "placement": write("placement", {"backend": route.backend, "device": "cpu",
                                         "host_fallback_operations": 0,
                                         "implementation": implementation,
                                         "runtime_library_sha256": sha256_file(library)}),
    }
    pins["comparison"] = write("comparison", {
        "candidate_logits_sha256": pins["logits"]["sha256"],
        "reference": {"implementation": REFERENCE_IMPLEMENTATION,
                      "source": reference_source_pins("olmoe")[0],
                      "generated_ids": reference_tokens,
                      "logits": encode_matrix(np.stack(reference_rows))},
        "tolerance": {"max_abs_logit": float(tolerance)},
    })
    applicability = coverage_applicability(row)
    checks = {name: {"status": "not-applicable",
                     "reason": f"canonical route {case_id} does not require {name}"}
              for name, applies in applicability.items() if not applies}
    format_pin = write("format_device", {
        "persisted_format": route.persisted_format,
        "consumer": "native_leased_olmoe_segments.NativeLeasedOlmoeSegmentEngine",
        "placement_sha256": pins["placement"]["sha256"]})
    recovery_pin = write("abi_recovery", {
        "abi_owner": "segment_edge_runtime",
        "recovery_sha256": pins["recovery"]["sha256"]})
    checks["format_device"] = {"status": "verified", "evidence": format_pin}
    checks["abi_recovery"] = {"status": "verified", "evidence": recovery_pin}
    if any(applies and name not in checks for name, applies in applicability.items()):
        raise CaptureError("public-range capture lacks applicable route coverage")

    runtime_sources = (
        # Bind receipts to the canonical implementations as well as their
        # public compatibility entrypoints; pinning only a shim would not
        # identify the code that performed capture or validated observations.
        "device_capture.py", "device_observations.py",
        "phase1_capture.py", "phase1_observations.py", "source_scope.py",
        "segment_edge_adapters.py", "segment_edge_runtime.py",
        "native_leased_olmoe_segments.py", "native_olmoe_segments.py",
        "native_segment_executor.py", "expert_runtime.py")
    artifacts = [config_pin, _pin(manifest_path, "expert_manifest"),
                 _pin(library, "runtime")]
    artifacts += [_pin(Path(__file__).with_name(name), "runtime") for name in runtime_sources]
    native_root = Path(__file__).with_name("native")
    native_sources = (
        "CMakeLists.txt", "segment_edge_compat.c", "segment_family_models.c",
        "qwen4_public_model.c", "qwen4_public_session.c", "qwen4_public_adapters.c",
        "original_experts.c", "segment_executor.c", "segment_bridge.c", "segment_math.c",
        "segment_edge_compat.h", "segment_family_models.h", "qwen4_public_internal.h",
        "qwen4_public_session.h", "original_experts.h", "segment_bridge.h", "segment_math.h")
    artifacts += [_pin(native_root / name, "runtime") for name in native_sources]
    artifacts += [_pin(path, "weights") for path in sorted(checkpoint.glob("*.safetensors"))]
    payloads = sorted({checkpoint / region.path
                       for layer in manifest_record.layers for region in layer.regions.values()})
    artifacts += [_pin(path, "expert_payload") for path in payloads]
    tokenizer = checkpoint / "tokenizer.json"
    if tokenizer.is_file():
        artifacts.append(_pin(tokenizer, "tokenizer"))
    receipt_record = {
        "case_id": case_id, "route": list(route.as_tuple()), "artifacts": artifacts,
        "cache_schema": {"format": "openagent.public-segment-edge-olmoe.v1",
                         "layers": config["num_hidden_layers"]},
        "placement_constraints": {"backend": route.backend, "device": "cpu"},
        "comparison_adapter": REFERENCE_IMPLEMENTATION,
        "observations": [pins[kind] for kind in sorted(pins)], "checks": checks,
    }
    measured = verify_route_receipt(row, receipt_record)
    _write_receipts(Path(receipts_path), receipt_record)
    return measured


def _greedy_public_steps(execution, input_ids: Sequence[int], count: int, eos_token: int) -> list[int]:
    import numpy as np

    pending = list(input_ids)
    generated = []
    for _ in range(count):
        matrix = np.asarray(execution.forward(pending), dtype=np.float32)
        if matrix.ndim != 2 or not matrix.shape[0] or not np.isfinite(matrix[-1]).all():
            raise CaptureError("public Segment/Edge returned malformed recovery logits")
        token = int(np.argmax(matrix[-1]))
        generated.append(token)
        if token == eos_token:
            break
        pending = [token]
    return generated


def _olmoe_direct_backend(persisted_format: str):
    """Construct the one explicit no-fallback backend for a direct format."""
    if persisted_format in {"float32", "int8-row"}:
        from .native_expert import NativeExpertBackend
        return NativeExpertBackend()
    if persisted_format in {"int2-row", "int3-grouped-g64"}:
        from .native_lowbit_expert import NativeLowBitExpertBackend
        return NativeLowBitExpertBackend(persisted_format)
    if persisted_format == "int4-row":
        from .native_packed_expert import NativePackedExpertBackend
        return NativePackedExpertBackend()
    if persisted_format == "int4-grouped-g64":
        from .native_grouped_int4_expert import NativeGroupedInt4ExpertBackend
        return NativeGroupedInt4ExpertBackend()
    if persisted_format == "mxfp4-e2m1-g32-ue8m0":
        from .native_mxfp4_expert import NativeMXFP4ExpertBackend
        return NativeMXFP4ExpertBackend()
    if persisted_format == "q4_0":
        from .native_q4_0_expert import NativeQ4_0ExpertBackend
        return NativeQ4_0ExpertBackend()
    if persisted_format == "q4_1":
        from .native_q4_1_expert import NativeQ4_1ExpertBackend
        return NativeQ4_1ExpertBackend()
    if persisted_format == "q5_0":
        from .native_q5_0_expert import NativeQ5_0ExpertBackend
        return NativeQ5_0ExpertBackend()
    if persisted_format == "q5_1":
        from .native_q5_1_expert import NativeQ5_1ExpertBackend
        return NativeQ5_1ExpertBackend()
    if persisted_format == "q8_0":
        from .native_q8_0_expert import NativeQ8_0ExpertBackend
        return NativeQ8_0ExpertBackend()
    if persisted_format == "q8_1":
        from .native_q8_1_expert import NativeQ8_1ExpertBackend
        return NativeQ8_1ExpertBackend()
    if persisted_format == "fp8-e4m3-b128-f32":
        from .native_fp8_block_expert import NativeFP8BlockExpertBackend
        return NativeFP8BlockExpertBackend()
    raise CaptureError("no direct no-fallback backend is registered for " + persisted_format)


def _capture_olmoe_direct_route(checkpoint: Path, manifest_path: Path,
                                prompt_ids: Sequence[int], *, max_tokens: int,
                                recovery_tokens: int, context_tokens: int,
                                cache_mib: int, conversion_mib: int,
                                persisted_format: str, case_id: str | None, tolerance: float,
                                evidence_root: str | Path, receipts_path: str | Path) -> dict[str, Any]:
    """Capture the exact direct OLMoE consumer with kernel fallback disabled."""
    import numpy as np
    from .expert_kernels import ExpertKernelRegistry
    from .expert_runtime import SparseExpertManifest
    from .olmoe import OlmoeModel
    from .source_scope import require_checkpoint_inventory_admission
    from .universal_engine import ModelRoute

    backend = _olmoe_direct_backend(persisted_format)
    if not backend.info().available:
        raise CaptureError(f"the explicit {backend.NAME} backend is unavailable")
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_format = {"float32": "F32", "float16": "F16", "bfloat16": "BF16"}.get(
        config.get("torch_dtype"))
    if source_format is None:
        inventory = require_checkpoint_inventory_admission(
            checkpoint, family="olmoe", model_type="olmoe", backend="native",
            arithmetic="fp32", persisted_format=persisted_format)
        observed = {record.dtype for record in inventory.tensors.values()
                    if record.dtype in {"F32", "F16", "BF16"}}
        if len(observed) != 1:
            raise CaptureError("direct capture requires one homogeneous source dtype")
        source_format = next(iter(observed))
    route = ModelRoute("olmoe", "olmoe", "native", "fp32", source_format, persisted_format)
    row = canonical_row(route, case_id=case_id)
    prompt = [int(token) for token in prompt_ids]
    if (not prompt or max_tokens <= 0 or recovery_tokens <= 0 or
            len(prompt) + max_tokens + recovery_tokens > context_tokens):
        raise CaptureError("prompt and generation/recovery lengths exceed the admitted context")

    manifest_record = SparseExpertManifest.read(manifest_path)
    registry = ExpertKernelRegistry([backend])
    try:
        with OlmoeModel(
                checkpoint, manifest_path, context_tokens=context_tokens,
                expert_cache_bytes=cache_mib * 1024 ** 2,
                max_matrix_decode_bytes=conversion_mib * 1024 ** 2,
                kernels=registry, preferred_kernel=backend.NAME) as model:
            runtime = model.experts
            runtime.start_route_trace()
            try:
                generated, captured, finish = _greedy_olmoe_steps(model, prompt, max_tokens)
                trace = runtime.stop_route_trace()
            except BaseException:
                runtime.stop_route_trace()
                raise
            if not generated:
                raise CaptureError("direct route produced no token observation")
            cache = _olmoe_cache_observation(model)
            snapshot = model.snapshot_state()
            snapshot_position = model.position
            continuation, _, _ = _greedy_olmoe_steps(
                model, [generated[-1]], recovery_tokens)
            model.restore_state(snapshot)
            restored_position = model.position
            replay, _, _ = _greedy_olmoe_steps(
                model, [generated[-1]], recovery_tokens)
            model.restore_state(snapshot)
            implementation = "olmoe-model+" + backend.NAME + "-no-fallback"
    except CaptureError:
        raise
    except Exception as error:
        raise CaptureError("direct OLMoE capture failed: " + str(error)) from error

    # Compare the measured native packed consumer with the reference backend.
    reference_rows, reference_tokens = [], []
    pending = prompt
    with OlmoeModel(
            checkpoint, manifest_path, context_tokens=context_tokens,
            expert_cache_bytes=cache_mib * 1024 ** 2,
            max_matrix_decode_bytes=conversion_mib * 1024 ** 2) as reference:
        for _ in range(len(generated)):
            logits = np.asarray(reference.forward(pending), dtype=np.float32)
            reference_rows.append(logits)
            token = int(np.argmax(logits))
            reference_tokens.append(token)
            pending = [token]

    case_id = row["case_id"]
    directory = Path(evidence_root).resolve() / case_id.replace("/", "__")
    directory.mkdir(parents=True, exist_ok=True)
    config_pin = _pin(checkpoint / "config.json", "config")
    config_sha = config_pin["sha256"]
    write = lambda kind, payload: _write_observation(  # noqa: E731
        directory, kind, payload, case_id=case_id, config_sha=config_sha)
    routing: dict[str, list[list[int]]] = {}
    for layer_id, rows in trace:
        routing.setdefault(layer_id, []).extend([[int(value) for value in item] for item in rows])
    pins = {
        "tokens": write("tokens", {"prompt_ids": prompt, "generated_ids": generated,
                                   "finish_reason": finish, "max_tokens": max_tokens,
                                   "decoding": "greedy", "implementation": implementation}),
        "logits": write("logits", {"matrix": encode_matrix(np.stack(captured))}),
        "cache": write("cache", cache),
        "routing": write("routing", {"layers": routing}),
        "recovery": write("recovery", {"snapshot_position": snapshot_position,
                                       "restored_position": restored_position,
                                       "continuation_ids": continuation,
                                       "replay_ids": replay}),
        "placement": write("placement", {"backend": route.backend, "device": "cpu",
                                         "host_fallback_operations": 0,
                                         "implementation": implementation,
                                         "kernel": backend.NAME,
                                         "kernel_compiled": backend.info().capabilities.compiled}),
    }
    pins["comparison"] = write("comparison", {
        "candidate_logits_sha256": pins["logits"]["sha256"],
        "reference": {"implementation": REFERENCE_IMPLEMENTATION,
                      "source": reference_source_pins("olmoe")[0],
                      "generated_ids": reference_tokens,
                      "logits": encode_matrix(np.stack(reference_rows))},
        "tolerance": {"max_abs_logit": float(tolerance)},
    })
    applicability = coverage_applicability(row)
    checks: dict[str, Any] = {
        name: {"status": "not-applicable",
               "reason": f"canonical route {case_id} does not require {name}"}
        for name, applies in applicability.items() if not applies
    }
    format_pin = write("format_device", {
        "persisted_format": route.persisted_format,
        "consumer": OLMOE_DIRECT_CONSUMERS[persisted_format],
        "placement_sha256": pins["placement"]["sha256"]})
    abi_owner = "model_gateway.OlmoeDirectProvider"
    abi_pin = write("abi_recovery", {"abi_owner": abi_owner,
                                     "recovery_sha256": pins["recovery"]["sha256"]})
    checks["format_device"] = {"status": "verified", "evidence": format_pin}
    checks["abi_recovery"] = {"status": "verified", "evidence": abi_pin}
    if any(applies and name not in checks for name, applies in applicability.items()):
        raise CaptureError("direct capture is missing applicable route coverage")

    import inspect
    backend_source = Path(inspect.getfile(type(backend)))
    artifacts = [config_pin, _pin(manifest_path, "expert_manifest"),
                 _pin(Path(__file__), "runtime"),
                 _pin(Path(__file__).with_name("phase1_observations.py"), "runtime"),
                 _pin(Path(__file__).with_name("source_scope.py"), "runtime"),
                 _pin(backend_source, "runtime"),
                 _pin(Path(__file__).with_name("expert_runtime.py"), "runtime")]
    artifacts += [_pin(path, "weights") for path in sorted(checkpoint.glob("*.safetensors"))]
    payloads = sorted({checkpoint / region.path
                       for layer in manifest_record.layers for region in layer.regions.values()})
    artifacts += [_pin(path, "expert_payload") for path in payloads]
    tokenizer = checkpoint / "tokenizer.json"
    if tokenizer.is_file():
        artifacts.append(_pin(tokenizer, "tokenizer"))
    receipt_record = {
        "case_id": case_id, "route": list(route.as_tuple()), "artifacts": artifacts,
        "cache_schema": {"format": "openagent.olmoe-state.v2", "layers": config["num_hidden_layers"]},
        "placement_constraints": {"backend": route.backend, "device": "cpu"},
        "comparison_adapter": REFERENCE_IMPLEMENTATION,
        "observations": [pins[kind] for kind in sorted(pins)], "checks": checks,
    }
    measured = verify_route_receipt(row, receipt_record)
    _write_receipts(Path(receipts_path), receipt_record)
    return measured


def main(argv: Sequence[str] | None = None, *, capture_fn=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    olmoe = commands.add_parser("olmoe", help="capture one exact OLMoE runtime route")
    olmoe.add_argument("--checkpoint", required=True, type=Path)
    olmoe.add_argument("--manifest", required=True, type=Path)
    olmoe.add_argument("--case-id",
                       help="exact admitted publisher case ID when multiple pins share this route")
    library = olmoe.add_mutually_exclusive_group(required=False)
    library.add_argument("--library", type=Path)
    library.add_argument("--build-library", type=Path, metavar="DIRECTORY",
                         help="compile the leased native segment executor into DIRECTORY first")
    library.add_argument("--build-public-runtime", type=Path, metavar="DIRECTORY",
                         help="build the complete Segment/Edge public ABI runtime into DIRECTORY first")
    prompt = olmoe.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-ids", help="comma-separated prompt token IDs")
    prompt.add_argument("--prompt", help="text encoded with the checkpoint's tokenizer.json")
    olmoe.add_argument("--max-tokens", type=int, default=4)
    olmoe.add_argument("--recovery-tokens", type=int, default=2)
    olmoe.add_argument("--context-tokens", type=int, default=64)
    olmoe.add_argument("--cache-mib", type=int, default=256)
    olmoe.add_argument("--resident-mib", type=int, default=4096)
    olmoe.add_argument("--transfer-mib", type=int, default=4096)
    olmoe.add_argument("--staging-mib", type=int, default=1024)
    olmoe.add_argument("--conversion-mib", type=int, default=2048,
                       help="working memory for decoding one dense matrix (OLMoE embedding/head)")
    olmoe.add_argument("--backend", choices=("native", "public-range"), default="native",
                       help="canonical execution owner to capture; public-range currently admits INT8-row")
    olmoe.add_argument("--tolerance", type=float, default=1e-3)
    olmoe.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    olmoe.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    args = parser.parse_args(argv)
    try:
        if args.prompt is not None:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
            prompt_ids = list(tokenizer.encode(args.prompt).ids)
        else:
            prompt_ids = [int(item) for item in args.prompt_ids.split(",") if item.strip()]
        library_path = (args.library if args.library is not None else
                        compile_segment_executor(args.build_library)
                        if args.build_library is not None else
                        compile_segment_runtime(args.build_public_runtime)
                        if args.build_public_runtime is not None else None)
        print(f"prompt token IDs: {prompt_ids}", file=sys.stderr)
        measured = (capture_fn or capture_olmoe_route)(
            args.checkpoint, manifest=args.manifest, library=library_path,
            prompt_ids=prompt_ids,
            max_tokens=args.max_tokens, recovery_tokens=args.recovery_tokens,
            context_tokens=args.context_tokens, cache_mib=args.cache_mib,
            resident_mib=args.resident_mib, transfer_mib=args.transfer_mib,
            staging_mib=args.staging_mib, conversion_mib=args.conversion_mib,
            backend=args.backend,
            case_id=args.case_id,
            tolerance=args.tolerance,
            evidence_root=args.evidence_root, receipts_path=args.receipts)
    except (CaptureError, ValueError, OSError, ImportError) as error:
        print("capture rejected: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(measured, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CaptureError", "canonical_row", "capture_olmoe_route",
           "compile_segment_executor", "compile_segment_runtime", "main"]
