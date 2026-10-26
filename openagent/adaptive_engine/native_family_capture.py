"""Exact native/device capture for supported text-only Phase 1 family routes.

This runner deliberately does not treat the independent NumPy oracle as the
candidate execution path. It executes a compiled native or specifically
registered device family decoder, captures its cache/routing state, and
compares it with the separately-run reviewed NumPy implementation. Public-range
capture currently supports exact Qwen4 and Inkling text only; other public/media routes
reject until their native observations and consumers are wired to this contract.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Sequence
from uuid import uuid4

from .device_capture import (
    CaptureError, DEFAULT_EVIDENCE, DEFAULT_RECEIPTS, _pin,
    _write_observation, _write_receipts, canonical_row,
)
from .device_observations import (
    reference_implementation, coverage_applicability, encode_matrix,
    reference_source_pins, verify_route_receipt,
    routed_layer_mask,
)


_TEXT_ROUTES = {
    "inkling": {
        "model_type": "inkling_text",
        "persisted_format": "float32",
        "cache_schema": "openagent.inkling-kv-fir.v1",
        "reference": "openagent.adaptive_engine.independent_inkling.IndependentInklingDecoder",
        "context_tokens_for_loader": True,
    },
    "qwen4": {
        "model_type": "qwen4_exp_text",
        "persisted_format": "float32",
        "cache_schema": "openagent.qwen4-kv.v1",
        "reference": "openagent.adaptive_engine.independent_qwen4.IndependentQwen4TextDecoder",
        "context_tokens_for_loader": False,
    },
    "qwen35": {
        "model_type": "qwen3_5_moe_text",
        "persisted_format": "float32",
        "cache_schema": "openagent.qwen35-kv.v1",
        "reference": "openagent.adaptive_engine.independent_qwen35.IndependentQwen35TextDecoder",
        "context_tokens_for_loader": True,
    },
    "glm5-next": {
        "model_type": "glm5_next_text",
        "persisted_format": "float32",
        "cache_schema": "openagent.glm5-kda-dsa-kv.v1",
        "reference": "openagent.adaptive_engine.torch_glm5_decoder.TorchGlm5NextDecoder",
        "context_tokens_for_loader": True,
    },
}

_DEVICE_TEXT_FAMILIES = {"inkling", "qwen4", "glm5-next"}


def compile_native_text_kernel(family: str, directory: str | Path) -> Path:
    """Build a single-family CPU decoder shared library for Linux/Colab or Windows."""
    spec = _TEXT_ROUTES.get(family)
    if spec is None:
        raise CaptureError("native text capture has no compiler contract for " + str(family))
    source_name = {"inkling": "native_inkling.c", "qwen4": "native_qwen4.c",
                   "qwen35": "native_qwen35.c", "glm5-next": "native_glm5_next.c"}[family]
    return compile_native_source(source_name, family, directory)


def compile_native_source(source_name: str, output_name: str, directory: str | Path) -> Path:
    """Compile a repository-owned native kernel for capture/reference execution."""
    admitted = {'native_inkling.c', 'native_qwen4.c', 'native_qwen35.c',
                'native_glm5_next.c', 'native_media_ops.c', 'native_qwen4_hyper.c',
                'native_qwen4_composition.c'}
    if source_name not in admitted or not output_name or Path(output_name).name != output_name:
        raise CaptureError('exact repository kernel and output name required')
    source = Path(__file__).resolve().parent / source_name
    if not source.is_file():
        raise CaptureError("native decoder source is missing: " + source_name)
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        compiler = shutil.which("cl")
        output = destination / (output_name + ".dll")
        arguments = [compiler or "cl", "/nologo", "/W4", "/WX", "/O2", "/fp:strict",
                   "/LD", str(source), "/link", "/OUT:" + str(output)]
        command = arguments
        if compiler is None:
            vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / \
                "Microsoft Visual Studio/Installer/vswhere.exe"
            if not vswhere.is_file():
                raise CaptureError("MSVC C++ build tools are required for the native decoder")
            location = subprocess.check_output([
                str(vswhere), "-latest", "-products", "*", "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"
            ], text=True, timeout=30).strip()
            vcvars = Path(location) / "VC/Auxiliary/Build/vcvars64.bat"
            if not location or not vcvars.is_file():
                raise CaptureError("MSVC C++ build tools are required for the native decoder")
            command = 'cmd.exe /d /s /c ""' + str(vcvars) + '" && ' + subprocess.list2cmdline(arguments) + '"'
    else:
        compiler = next((shutil.which(name) for name in ("cc", "gcc", "clang")
                         if shutil.which(name)), None)
        if compiler is None:
            raise CaptureError("a C compiler is required to build the native decoder")
        output = destination / (output_name + ".so")
        command = [compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
                   "-ffp-contract=off", "-fPIC", "-shared", str(source), "-o", str(output), "-lm"]
    try:
        result = subprocess.run(command, cwd=destination, capture_output=True,
                                text=True, timeout=600)
    except subprocess.TimeoutExpired as error:
        raise CaptureError("native family decoder compilation timed out") from error
    if result.returncode or not output.is_file():
        raise CaptureError("native family decoder compilation failed:\n" +
                           result.stdout + result.stderr)
    return output.resolve(strict=True)


def capture_native_text_route(
        family: str, checkpoint: str | Path, *, library: str | Path | None,
        runtime_library: str | Path | None = None,
        source_format: str, backend: str, prompt_ids: Sequence[int],
        max_tokens: int = 4, recovery_tokens: int = 2, context_tokens: int = 64,
        max_weight_bytes: int = 128 * 1024**3,
        max_runtime_bytes: int = 32 * 1024**3,
        reference_device: str | None = None,
        reference_library: str | Path | None = None,
        execution_device: str | None = None,
        case_id: str | None = None, tolerance: float = 1e-3,
        evidence_root: str | Path = DEFAULT_EVIDENCE,
        receipts_path: str | Path = DEFAULT_RECEIPTS,
) -> dict[str, Any]:
    """Capture an exact native or supported device family text route.

    Source inventory admission is repeated by the production family loader.
    The preflight here additionally requires a homogeneous source dtype so a
    caller cannot select one canonical ``source_format`` for a mixed checkpoint.
    """
    import numpy as np

    spec = _TEXT_ROUTES.get(family)
    if spec is None:
        raise CaptureError("native text capture has no exact family contract for " + str(family))
    model_type = spec["model_type"]
    if backend not in ("native", "device", "public-range"):
        raise CaptureError("unsupported family capture backend")
    if backend == "public-range" and (family, model_type) not in (
            ("qwen4", "qwen4_exp_text"), ("inkling", "inkling_text"),
            ("glm5-next", "glm5_next_text")):
        raise CaptureError("public-range text capture requires an exact supported text identity")
    if backend == "device" and family not in _DEVICE_TEXT_FAMILIES:
        raise CaptureError("device capture has no exact family decoder contract")
    if backend == "device":
        if type(execution_device) is not str or not execution_device:
            raise CaptureError("device capture requires an explicit CUDA/MPS execution device")
        if library is not None:
            raise CaptureError("device capture does not accept a native C candidate library")
        if family == "glm5-next":
            if reference_device is not None:
                raise CaptureError("GLM5 device comparison uses the independent native C reference")
            if reference_library is None:
                raise CaptureError("GLM5 device capture requires a separately compiled native C reference library")
        elif reference_device is not None or reference_library is not None:
            raise CaptureError("reference device/library overrides are specific to GLM5 routes")
        if runtime_library is not None:
            raise CaptureError("--runtime-library is only valid for a public-range family route")
    elif execution_device is not None:
        raise CaptureError("--device is only valid for a supported device route")
    if backend == "native" and runtime_library is not None:
        raise CaptureError("--runtime-library is only valid for a public-range family route")
    if family == "glm5-next" and backend in ("native", "public-range"):
        if type(reference_device) is not str or not reference_device:
            raise CaptureError("GLM5 native capture requires an explicit accelerator for its independent Torch reference")
        if reference_library is not None:
            raise CaptureError("GLM5 native capture compares against Torch, not a second native library")
    elif reference_device is not None:
        raise CaptureError("--reference-device is currently specific to the independent GLM5 Torch reference")
    if backend in ("native", "public-range") and family != "glm5-next" and reference_library is not None:
        raise CaptureError("reference-library is only valid for GLM5 device capture")
    if source_format not in ("F32", "F16", "BF16"):
        raise CaptureError("family text capture supports only source formats F32, F16, and BF16")
    if (type(max_tokens) is not int or max_tokens <= 0 or
            type(recovery_tokens) is not int or recovery_tokens <= 0 or
            type(context_tokens) is not int or context_tokens <= 0 or
            type(max_weight_bytes) is not int or max_weight_bytes <= 0 or
            type(max_runtime_bytes) is not int or max_runtime_bytes <= 0):
        raise CaptureError("positive token, context, and memory budgets are required")
    prompt = list(prompt_ids)
    if (not prompt or any(type(token) is not int or token < 0 for token in prompt) or
            len(prompt) + max_tokens + recovery_tokens > context_tokens):
        raise CaptureError("valid prompt IDs and bounded generation/recovery context required")

    checkpoint = Path(checkpoint).resolve(strict=True)
    if not checkpoint.is_dir():
        raise CaptureError("an exact checkpoint directory is required")
    if backend in ("native", "public-range"):
        if library is None:
            raise CaptureError("compiled native family library is required")
        library = Path(library).resolve(strict=True)
        if not library.is_file():
            raise CaptureError("compiled native family library is not a file")
    if backend == "public-range":
        if runtime_library is None:
            raise CaptureError(
                f"{family} public-range capture requires a compiled Segment/Edge runtime library")
        runtime_library = Path(runtime_library).resolve(strict=True)
        if not runtime_library.is_file():
            raise CaptureError("compiled Segment/Edge runtime library is not a file")
    if reference_library is not None:
        reference_library = Path(reference_library).resolve(strict=True)
        if not reference_library.is_file():
            raise CaptureError("compiled GLM5 native reference library is not a file")

    from .native_checkpoint import load_native_text_decoder, read_native_text_checkpoint
    from .publisher_checkpoint.storage import snapshot
    from .source_scope import require_source_inventory_admission
    from .universal_engine import ModelRoute
    inventory = snapshot(checkpoint, max_metadata_bytes=max_weight_bytes,
                         max_tensors=1_000_000, max_shards=4096)
    source_formats = require_source_inventory_admission(
        family=family, model_type=model_type, backend=backend, arithmetic="fp32",
        persisted_format=spec["persisted_format"], inventory=inventory.tensors)
    if source_formats != (source_format,):
        raise CaptureError("checkpoint source dtypes differ from the selected exact route: " +
                           ", ".join(source_formats))
    route = ModelRoute(family, model_type, backend, "fp32", source_format,
                       spec["persisted_format"])
    row = canonical_row(route, case_id=case_id)
    chain = row.get("callable_chain") or {}

    # Materialize the comparison reference from this exact checkpoint snapshot.
    config, weights = read_native_text_checkpoint(
        checkpoint, model_type=model_type, max_weight_bytes=max_weight_bytes,
        context_tokens=context_tokens if spec["context_tokens_for_loader"] else None,
        source_backend=backend)
    if len(prompt) + max_tokens + recovery_tokens > min(
            context_tokens, int(config.get("max_position_embeddings", context_tokens))):
        raise CaptureError("prompt, generation, and recovery exceed checkpoint context")
    decoder_options: dict[str, Any] = {
        "model_type": model_type,
        "max_weight_bytes": max_weight_bytes, "max_bytes": max_runtime_bytes,
        "source_backend": backend,
    }
    if backend == "native":
        decoder_options["library"] = library
        if spec["context_tokens_for_loader"]:
            decoder_options["context_tokens"] = context_tokens
            decoder_options["max_snapshots"] = 4
        candidate = load_native_text_decoder(checkpoint, **decoder_options)
    elif backend == "public-range":
        decoder_options["library"] = library
        if spec["context_tokens_for_loader"]:
            decoder_options["context_tokens"] = context_tokens
            decoder_options["max_snapshots"] = 4
        base = load_native_text_decoder(checkpoint, **decoder_options)
        try:
            layers = int(config["num_hidden_layers"])
            ranges = tuple((index, index + 1) for index in range(layers))
            if family == "qwen4":
                from .native_qwen4_public_text import NativeQwen4PublicTextDecoder
                public_decoder = NativeQwen4PublicTextDecoder
                options = {"layer_ranges": ranges}
            else:
                from .native_family_public_text import NativeFamilyPublicTextDecoder
                public_decoder = NativeFamilyPublicTextDecoder
                options = {"layer_ranges": ranges}
            candidate = public_decoder(
                runtime_library, base, directory=checkpoint, max_bytes=max_runtime_bytes,
                max_work_bytes=max(4 * 1024**2, min(max_runtime_bytes // 4, 512 * 1024**2)),
                max_snapshots=4, **options)
        except BaseException as error:
            try:
                base.close()
            except BaseException as cleanup:
                from .failure_reporting import add_failure_note, failure_summary
                add_failure_note(error, f"{family} public-range base decoder cleanup retained: " +
                                 failure_summary(cleanup))
            raise CaptureError(f"{family} public-range adapter admission failed: " + str(error)) from error
    else:
        try:
            if family == "qwen4":
                from .torch_qwen4_decoder import TorchQwen4Decoder
                device_decoder = TorchQwen4Decoder
            elif family == "inkling":
                from .torch_inkling_decoder import TorchInklingDecoder
                device_decoder = TorchInklingDecoder
            else:
                from .torch_glm5_decoder import TorchGlm5NextDecoder
                device_decoder = TorchGlm5NextDecoder
            candidate = device_decoder.from_directory(
                checkpoint, device=execution_device, max_weight_bytes=max_weight_bytes,
                max_bytes=max_runtime_bytes,
                **({"max_tokens": context_tokens} if family == "glm5-next" else {}),
                max_snapshots=4)
        except Exception as error:
            raise CaptureError("device family decoder admission failed: " + str(error)) from error
    candidate_runtime_pin = None
    reference_runtime_pin = None
    reference_runtime = None
    reference = None
    try:
        if backend == "device":
            candidate_runtime_pin = _torch_runtime_pin(candidate)
        if family == "glm5-next" and backend in ("native", "public-range"):
            from .torch_glm5_decoder import TorchGlm5NextDecoder
            reference = TorchGlm5NextDecoder.from_directory(
                checkpoint, device=reference_device,
                max_weight_bytes=max_weight_bytes, max_bytes=max_runtime_bytes,
                max_tokens=context_tokens, max_snapshots=4)
            reference_runtime_pin = _torch_runtime_pin(reference)
            reference_runtime = _torch_device_details(reference)
        elif family == "glm5-next" and backend == "device":
            from .native_glm5_next import NativeGlm5NextTextDecoder
            reference = NativeGlm5NextTextDecoder(
                config, weights, library=reference_library, max_tokens=context_tokens,
                max_bytes=max_runtime_bytes, max_snapshots=4)
        else:
            reference_class = _load_reference_class(family)
            reference_options = ({"max_bytes": max_runtime_bytes}
                                 if family in ("inkling", "qwen4") else {})
            reference = reference_class(config, weights, **reference_options)
    except BaseException as primary:
        if reference is not None:
            close = getattr(reference, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    from .failure_reporting import add_failure_note, failure_summary
                    add_failure_note(primary, "independent reference cleanup retained: " +
                                     failure_summary(cleanup))
        try:
            candidate.close()
        except BaseException as cleanup:
            from .failure_reporting import add_failure_note, failure_summary
            add_failure_note(primary, "family capture candidate cleanup retained: " +
                             failure_summary(cleanup))
        raise
    implementation = str(candidate.implementation)
    consumer = f"{type(candidate).__module__.rsplit('.', 1)[-1]}.{type(candidate).__name__}"
    if consumer not in tuple(chain.get("execution", ())):
        primary = CaptureError("loaded decoder is absent from the canonical route execution chain")
        _close_capture_owners(candidate, reference, primary)
        raise primary
    abi_owner = consumer
    if abi_owner not in tuple(chain.get("native_abi", ())):
        primary = CaptureError(
            "canonical route does not bind the executing ABI owner; reconcile P02 before capture")
        _close_capture_owners(candidate, reference, primary)
        raise primary

    candidate_closed = False
    staging: Path | None = None
    try:
        eos_token_id = config.get("eos_token_id")
        candidate_capture_routing = (family in ("glm5-next", "inkling") or
                                     (family == "qwen4" and backend in ("device", "public-range")))
        candidate_routing_keyword = (family == "glm5-next" or
                                     (family == "inkling" and backend == "public-range") or
                                     (family == "qwen4" and backend in ("device", "public-range")))
        reference_capture_routing = family in ("glm5-next", "inkling")
        reference_routing_keyword = family == "glm5-next"
        generated, candidate_rows, finish, route_rows = _greedy(
            candidate, prompt, max_tokens, eos_token_id=eos_token_id,
            capture_routing=candidate_capture_routing,
            routing_keyword=candidate_routing_keyword)
        try:
            ref_generated, reference_rows, _ref_finish, _ = _greedy(
                reference, prompt, max_tokens, eos_token_id=eos_token_id,
                capture_routing=reference_capture_routing,
                routing_keyword=reference_routing_keyword)
        except BaseException as primary:
            close = getattr(reference, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    from .failure_reporting import add_failure_note, failure_summary
                    add_failure_note(primary, "independent reference cleanup retained: " +
                                     failure_summary(cleanup))
            reference = None
            raise
        else:
            close = getattr(reference, "close", None)
            if callable(close):
                close()
            reference = None
        if not generated or generated != ref_generated:
            raise CaptureError("candidate and independent reference token IDs differ")
        if len(route_rows) != config["num_hidden_layers"]:
            raise CaptureError("native decoder did not return routes for every configured layer")

        processed = len(prompt) + len(generated) - 1
        if candidate.position != processed:
            raise CaptureError("native candidate cache position differs from processed tokens")
        cache_bytes, cache_digests = _cache_digests(candidate, config["num_hidden_layers"])
        if cache_bytes <= 0:
            raise CaptureError("native candidate exposed no measurable cache state")

        snapshot_handle = candidate.snapshot()
        snapshot_position = candidate.position
        try:
            continuation, _, _, _ = _greedy(
                candidate, [generated[-1]], recovery_tokens, eos_token_id=eos_token_id,
                capture_routing=candidate_capture_routing,
                routing_keyword=candidate_routing_keyword)
            candidate.restore(snapshot_handle)
            if candidate.position != snapshot_position:
                raise CaptureError("native restore did not return to its saved cache position")
            replay, _, _, _ = _greedy(
                candidate, [generated[-1]], recovery_tokens, eos_token_id=eos_token_id,
                capture_routing=candidate_capture_routing,
                routing_keyword=candidate_routing_keyword)
            if replay != continuation:
                raise CaptureError("native restore replay differs from the original continuation")
        finally:
            candidate.release_snapshot(snapshot_handle)

        case_id_str = row["case_id"]
        root = Path(evidence_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".phase1-capture-", dir=root))
        config_pin = _pin(checkpoint / "config.json", "config")
        config_sha = config_pin["sha256"]
        write = lambda kind, payload: _write_observation(  # noqa: E731
            staging, kind, payload, case_id=case_id_str, config_sha=config_sha)
        candidate_logits = np.stack(candidate_rows)
        reference_logits = np.stack(reference_rows)
        if candidate_logits.shape != reference_logits.shape:
            raise CaptureError("native and reference logit geometry differs")

        if backend == "device":
            measured_device = str(candidate.device)
            placement_details = _torch_device_details(candidate)
        else:
            measured_device = "cpu"
            placement_details = {}
        observations = {
            "tokens": write("tokens", {
                "prompt_ids": prompt, "generated_ids": generated,
                "finish_reason": finish, "max_tokens": len(generated),
                "decoding": "greedy", "implementation": implementation,
            }),
            "logits": write("logits", {"matrix": encode_matrix(candidate_logits)}),
            "cache": write("cache", {
                "position": processed, "layers": config["num_hidden_layers"],
                "layer_digests": cache_digests, "bytes": cache_bytes,
            }),
            "routing": write("routing", {"layers": {
                f"model.layers.{index}": layer_rows
                for index, layer_rows in enumerate(route_rows)
            }}),
            "recovery": write("recovery", {
                "snapshot_position": snapshot_position,
                "restored_position": snapshot_position,
                "continuation_ids": continuation, "replay_ids": replay,
            }),
            "placement": write("placement", {
                "backend": backend, "device": measured_device,
                "host_fallback_operations": 0, "implementation": implementation,
                **placement_details,
            }),
        }
        reference_pin_list = reference_source_pins(family, backend, model_type)
        reference_binding = ({"source": reference_pin_list[0]} if len(reference_pin_list) == 1
                            else {"sources": reference_pin_list})
        reference_name = reference_implementation(family, backend, model_type)
        observations["comparison"] = write("comparison", {
            "candidate_logits_sha256": observations["logits"]["sha256"],
            "reference": {
                "implementation": reference_name,
                **reference_binding,
                "generated_ids": [int(np.argmax(row)) for row in reference_rows],
                "logits": encode_matrix(reference_logits),
                **({"runtime": reference_runtime} if reference_runtime is not None else {}),
            },
            "tolerance": {"max_abs_logit": float(tolerance)},
        })

        applicability = coverage_applicability(row)
        if any(applicability.values()):
            raise CaptureError("native text capture cannot satisfy applicable media/speculative/distributed coverage")
        checks = {
            name: {"status": "not-applicable",
                   "reason": f"canonical route {case_id_str} does not require {name}"}
            for name, applies in applicability.items() if not applies
        }
        checks["format_device"] = {"status": "verified", "evidence": write(
            "format_device", {"persisted_format": route.persisted_format,
                              "consumer": consumer,
                              "placement_sha256": observations["placement"]["sha256"]})}
        checks["abi_recovery"] = {"status": "verified", "evidence": write(
            "abi_recovery", {"abi_owner": abi_owner,
                             "recovery_sha256": observations["recovery"]["sha256"]})}

        inventory.assert_unchanged()
        artifacts = [config_pin]
        artifacts.extend(_pin(path, "weights") for path in sorted(checkpoint.glob("*.safetensors")))
        artifacts.extend(_pin(path, "runtime")
                         for path in _candidate_source_paths(family, backend))
        tokenizer = checkpoint / "tokenizer.json"
        if tokenizer.is_file():
            artifacts.append(_pin(tokenizer, "tokenizer"))
        if backend == "native":
            artifacts.append(_pin(library, "runtime"))
        elif candidate_runtime_pin is not None:
            artifacts.append(candidate_runtime_pin)
        elif backend == "public-range":
            artifacts.extend((_pin(library, "runtime"), _pin(runtime_library, "runtime")))
        if reference_runtime_pin is not None:
            artifacts.append(reference_runtime_pin)
        if backend == "device" and reference_library is not None:
            artifacts.append(_pin(reference_library, "runtime"))
        receipt = {
            "case_id": case_id_str, "route": list(route.as_tuple()),
            "artifacts": artifacts,
            "cache_schema": {"format": spec["cache_schema"],
                             "layers": config["num_hidden_layers"]},
            "placement_constraints": {"backend": backend, "device": measured_device},
            "comparison_adapter": REFERENCE_IMPLEMENTATIONS[family],
            "observations": [observations[kind] for kind in sorted(observations)],
            "checks": checks,
        }
        measured = verify_route_receipt(row, receipt)

        # Promote only gate-accepted evidence from a private staging directory.
        final_directory = root / (case_id_str.replace("/", "__") + "--" + uuid4().hex)
        os.replace(staging, final_directory)
        staging = None
        for pin in receipt["observations"]:
            pin["path"] = str(final_directory / Path(pin["path"]).name)
        for check in checks.values():
            evidence = check.get("evidence")
            if evidence:
                evidence["path"] = str(final_directory / Path(evidence["path"]).name)
        try:
            measured = verify_route_receipt(row, receipt)
            candidate.close()
            candidate_closed = True
            _write_receipts(Path(receipts_path), receipt)
        except BaseException:
            shutil.rmtree(final_directory, ignore_errors=True)
            raise
        result = measured
    except BaseException as primary:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if reference is not None:
            close = getattr(reference, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    from .failure_reporting import add_failure_note, failure_summary
                    add_failure_note(primary, "independent reference cleanup retained: " +
                                     failure_summary(cleanup))
            reference = None
        if not candidate_closed:
            try:
                candidate.close()
            except BaseException as cleanup:
                from .failure_reporting import add_failure_note, failure_summary
                add_failure_note(primary, "family capture decoder cleanup retained: " +
                                 failure_summary(cleanup))
        raise
    else:
        return result
    finally:
        reference = None


def _load_reference_class(family: str):
    if family == "inkling":
        from .independent_inkling import IndependentInklingDecoder
        return IndependentInklingDecoder
    if family == "qwen4":
        from .independent_qwen4 import IndependentQwen4TextDecoder
        return IndependentQwen4TextDecoder
    from .independent_qwen35 import IndependentQwen35TextDecoder
    return IndependentQwen35TextDecoder


def _greedy(decoder, prompt: Sequence[int], max_tokens: int, *, eos_token_id=None,
            capture_routing: bool = False, routing_keyword: bool | None = None):
    import numpy as np
    pending = list(prompt)
    generated: list[int] = []
    rows: list[Any] = []
    finish = "length"
    layers: list[list[list[int]]] | None = None
    for _ in range(max_tokens):
        request_keyword = capture_routing if routing_keyword is None else routing_keyword
        family_config = getattr(decoder, "contract", None) or getattr(decoder, "config", None)
        # Both GLM implementations define an exact batch-one [1, tokens] ABI;
        # the other text decoders accept a flat token sequence.
        forward_ids = (np.asarray(pending, dtype=np.int64)[None, :]
                       if (type(family_config).__name__ == "Glm5NextTextContract" or
                           type(decoder).__name__ in (
                               "NativeQwen4PublicTextDecoder", "NativeFamilyPublicTextDecoder"))
                       else pending)
        result = (decoder.forward(forward_ids, capture_routing=True) if request_keyword
                  else decoder.forward(forward_ids))
        logits = np.asarray(_to_numpy(_result_field(result, "logits"), np), dtype=np.float32)
        if logits.ndim == 3:
            logits = logits[0, -1]
        elif logits.ndim == 2:
            logits = logits[-1]
        elif logits.ndim != 1:
            raise CaptureError("decoder returned malformed logits")
        rows.append(logits.copy())
        routes = _result_field(result, "routes")
        if routes is not None:
            if layers is None:
                layers = [[] for _ in routes]
            if len(routes) != len(layers):
                raise CaptureError("decoder route count changed during generation")
            for index, route_result in enumerate(routes):
                selected = route_result[0] if isinstance(route_result, (tuple, list)) else route_result
                selected = _to_numpy(selected, np)
                if selected.ndim == 3:
                    selected = selected[0]
                if selected.ndim == 1:
                    selected = selected[None, :]
                if selected.ndim != 2 or selected.dtype.kind not in "iu":
                    raise CaptureError("production decoder returned malformed expert IDs")
                layers[index].extend([[int(value) for value in row] for row in selected])
        else:
            routing = _result_field(result, "routing_weights")
            config = getattr(decoder, "contract", None) or getattr(decoder, "config", None)
            if routing is None:
                raise CaptureError("production decoder returned no executed expert-route observation")
            expert_count = _config_value(config, "n_routed_experts")
            if expert_count is None:
                expert_count = _config_value(config, "num_experts")
            top_k = _config_value(config, "num_experts_per_tok")
            values = _to_numpy(routing, np)
            if (type(expert_count) is not int or type(top_k) is not int or
                    values.ndim != 4 or values.shape[0] != 1 or
                    values.shape[-1] < expert_count or top_k > expert_count):
                raise CaptureError("production decoder returned malformed routing weights")
            if layers is None:
                layers = [[] for _ in range(values.shape[2])]
            if values.shape[2] != len(layers):
                raise CaptureError("decoder routing layer count changed during generation")
            for index in range(values.shape[2]):
                scores = values[0, :, index, :expert_count]
                if not routed_layer_mask(config, values.shape[2])[index]:
                    if np.any(scores != 0):
                        raise CaptureError('dense layer returned unexpected routed expert weights')
                    layers[index].extend([[] for _ in range(values.shape[1])])
                    continue
                selected = np.argsort(-scores, axis=-1, kind="stable")[:, :top_k]
                if any(np.count_nonzero(scores[row] > 0) < top_k
                       for row in range(scores.shape[0])):
                    raise CaptureError(
                        "routing weights underflowed; exact selected experts cannot be proved")
                layers[index].extend([[int(value) for value in row] for row in selected])
        token = int(np.argmax(logits))
        generated.append(token)
        config = getattr(decoder, "contract", None) or decoder.config
        eos = eos_token_id
        if eos is None:
            eos = _config_value(config, "eos_token_id")
        if eos is not None and token == eos:
            finish = "eos"
            break
        pending = [token]
    return generated, rows, finish, layers or []


def _cache_digests(decoder, layer_count: int) -> tuple[int, list[str]]:
    import numpy as np
    if type(decoder).__name__ in ("NativeQwen4PublicTextDecoder", "NativeFamilyPublicTextDecoder"):
        return _public_range_cache_digests(decoder, layer_count)
    total = 0
    digests = []
    cache = getattr(decoder, "cache", None)
    if cache is None:
        cache = getattr(decoder, "_cache", None)
    if cache is None:
        raise CaptureError("production decoder exposed no physical cache state")
    extra = getattr(decoder, "ple_cache", None)
    for index in range(layer_count):
        digest = hashlib.sha256()
        banks = [cache[index]]
        if extra is not None:
            banks.append(extra[index])
        for bank in banks:
            _digest_value(digest, bank)
            total += _count_array_bytes(bank, np)
        digests.append(digest.hexdigest())
    return total, digests


def _public_range_cache_digests(decoder, layer_count: int) -> tuple[int, list[str]]:
    """Measure dispatcher-owned physical cache blobs per configured layer."""
    ranges = getattr(decoder, "layer_ranges", None)
    expected = tuple((index, index + 1) for index in range(layer_count))
    if ranges != expected:
        raise CaptureError("public-range cache evidence requires one exact native segment per layer")
    with tempfile.TemporaryDirectory(prefix="oa-family-public-cache-") as directory:
        path = Path(directory) / "cache.bin"
        written = decoder.write_cache(path, max_bytes=int(decoder.max_bytes))
        if type(written) is not int or written <= 0 or path.stat().st_size != written:
            raise CaptureError("public-range cache export size differs from the dispatcher receipt")
        with path.open("rb") as source:
            prefix = source.read(12)
            if len(prefix) != 12 or prefix[:8] != b"OASEKV01":
                raise CaptureError("public-range cache export has an unknown wire format")
            header_size = int.from_bytes(prefix[8:], "little")
            if not 0 < header_size <= 65536:
                raise CaptureError("public-range cache header exceeds its bounded schema")
            try:
                header = json.loads(source.read(header_size))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CaptureError("public-range cache header is malformed") from error
        sizes, digests = header.get("sizes"), header.get("digests")
        if (header.get("version") != 1 or header.get("position") != decoder.position or
                type(sizes) is not list or len(sizes) != layer_count or
                any(type(size) is not int or size < 0 for size in sizes) or
                type(digests) is not list or len(digests) != layer_count or
                any(type(value) is not str or len(value) != 64 or
                    any(char not in "0123456789abcdef" for char in value)
                    for value in digests)):
            raise CaptureError("public-range cache layers/position differ from configured geometry")
        total = sum(sizes)
        if total <= 0:
            raise CaptureError("public dispatcher returned no measurable physical cache bytes")
        return total, list(digests)


def _result_field(result, name):
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


def _config_value(config, name):
    if callable(getattr(config, "get", None)):
        return config.get(name)
    return getattr(config, name, None)


def _to_numpy(value, np):
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            value = numpy()
    return np.asarray(value)


def _torch_runtime_pin(decoder) -> dict[str, str]:
    torch = getattr(decoder, "torch", None)
    extension = getattr(getattr(torch, "_C", None), "__file__", None)
    if type(extension) is not str:
        raise CaptureError("device decoder exposes no concrete Torch runtime binary to pin")
    return _pin(Path(extension), "runtime")


def _close_capture_owners(candidate, reference, primary: BaseException) -> None:
    from .failure_reporting import add_failure_note, failure_summary
    for label, owner in (("independent reference", reference),
                         ("family capture candidate", candidate)):
        close = getattr(owner, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as cleanup:
            add_failure_note(primary, label + " cleanup retained: " + failure_summary(cleanup))


def _candidate_source_paths(family: str, backend: str) -> tuple[Path, ...]:
    """Pin the executed candidate and evidence-gate sources in each receipt."""
    root = Path(__file__).resolve().parent
    common = (
        "native_family_capture.py", "device_capture.py", "device_observations.py",
        "native_checkpoint.py", "source_scope.py",
    )
    if backend == "device":
        if family not in _DEVICE_TEXT_FAMILIES:
            raise CaptureError("no pinned candidate source closure exists for this device route")
        device_sources = {
            "qwen4": (
                "torch_qwen4_decoder.py", "torch_qwen35_decoder.py",
                "device_qwen4_forward.py", "device_qwen4_layers.py",
                "device_qwen4_attention.py", "device_qwen35_layers.py",
                "device_qwen_delta.py", "device_attention.py", "device_logits.py",
                "device_publisher_projections.py", "independent_qwen4.py",
                "independent_qwen35.py", "native_mixtral.py",
            ),
            "glm5-next": (
                "torch_glm5_decoder.py", "device_glm5_forward.py",
                "device_qwen35_layers.py", "device_attention.py",
                "device_publisher_projections.py", "device_logits.py",
                "torch_inkling_decoder.py", "device_inkling_forward.py",
                "independent_glm5_next.py", "independent_inkling.py",
                "native_glm5_next.py", "native_glm5_next.c",
            ),
            "inkling": (
                "torch_inkling_decoder.py", "device_inkling_forward.py",
                "device_qwen35_layers.py", "device_attention.py",
                "device_publisher_projections.py",
                "independent_inkling.py",
            ),
        }[family]
        names = common + device_sources + (
            "device_execution_policy.py",
        )
    elif backend == "public-range":
        if family not in ("qwen4", "inkling", "glm5-next"):
            raise CaptureError("no pinned public-range candidate closure exists for this family")
        names = common + (
            "native_family_public_text.py",
            "native_qwen4.py", "native_qwen4.c", "native_qwen4_public_text.py",
            "native_qwen4_public_adapters.py", "native_qwen4_composition.py",
            "native_qwen35.py", "native_qwen35.c",
            "independent_qwen4.py", "independent_qwen35.py",
            "native_qwen4_hyper.py", "native_library_identity.py",
            "owned_file_descriptor.py", "session_recovery.py",
            "qwen4_segment_wire.py", "segment_edge_runtime.py",
            "segment_edge_cache.py", "segment_snapshot.py", "family_source_binding.py",
            "native_family_public_adapters.py",
            "native_glm5_next.py", "native_glm5_next.c",
            "native_inkling.py", "native_inkling.c", "independent_inkling.py",
            "failure_reporting.py",
            "olmoe_segments.py", "expert_packing.py", "expert_runtime.py",
            "expert_store.py", "expert_kernels.py", "olmoe.py", "olmoe_batch.py",
            "native/segment_edge_compat.h", "native/segment_edge_compat.c",
            "native/segment_family_models.h", "native/segment_family_models.c",
            "native/original_experts.h", "native/original_experts.c",
            "native/segment_executor.h", "native/segment_executor.c",
            "native/segment_bridge.h", "native/segment_bridge.c",
            "native/segment_math.h", "native/segment_math.c",
            "native/sha256.h",
            "native/qwen4_public_model.h", "native/qwen4_public_model.c",
            "native/qwen4_public_session.h", "native/qwen4_public_session.c",
            "native/qwen4_public_internal.h", "native/qwen4_public_adapters.c",
        )
    else:
        wrapper = {
            "inkling": "native_inkling.py",
            "qwen4": "native_qwen4.py",
            "qwen35": "native_qwen35.py",
            "glm5-next": "native_glm5_next.py",
        }.get(family)
        c_source = {
            "inkling": "native_inkling.c",
            "qwen4": "native_qwen4.c",
            "qwen35": "native_qwen35.c",
            "glm5-next": "native_glm5_next.c",
        }.get(family)
        if wrapper is None or c_source is None:
            raise CaptureError("no pinned candidate source closure exists for this native route")
        names = common + (wrapper, c_source)
    paths = tuple((root / name).resolve(strict=True) for name in names)
    if len(paths) != len(set(paths)) or any(not path.is_file() for path in paths):
        raise CaptureError("candidate source closure is incomplete or ambiguous")
    return paths


def _torch_device_details(decoder) -> dict[str, Any]:
    torch = getattr(decoder, "torch", None)
    device = getattr(decoder, "device", None)
    if torch is None or type(device) is not str:
        raise CaptureError("device decoder exposes no measured Torch device")
    details: dict[str, Any] = {"torch_version": str(torch.__version__)}
    if device.startswith("cuda"):
        details["device_name"] = str(torch.cuda.get_device_name(device))
        details["device_capability"] = list(torch.cuda.get_device_capability(device))
    elif device == "mps":
        details["device_name"] = "Apple Metal"
    else:
        raise CaptureError("device capture resolved to an unsupported accelerator identifier")
    return details


def _digest_value(digest, value) -> None:
    import numpy as np
    if value is None:
        digest.update(b"none\0")
    elif isinstance(value, np.ndarray) or callable(getattr(value, "detach", None)):
        contiguous = np.ascontiguousarray(_to_numpy(value, np))
        digest.update(b"array\0" + contiguous.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
        digest.update(contiguous.tobytes())
    elif isinstance(value, (tuple, list)):
        digest.update(b"tuple\0")
        for item in value:
            _digest_value(digest, item)
    else:
        raise CaptureError("native cache contains an unrecognized physical-state value")


def _count_array_bytes(value, np) -> int:
    if value is None:
        return 0
    if isinstance(value, np.ndarray) or callable(getattr(value, "detach", None)):
        return int(_to_numpy(value, np).nbytes)
    if isinstance(value, (tuple, list)):
        return sum(_count_array_bytes(item, np) for item in value)
    raise CaptureError("native cache contains an unrecognized physical-state value")


__all__ = ["capture_native_text_route", "compile_native_text_kernel"]
