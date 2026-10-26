"""Fail-closed CLI adapter for exact Qwen4 text route capture.

Multimodal rows remain outside this text-only capture. Native and public-range
candidates are compared to the independent NumPy decoder; device capture uses
the real Torch decoder on the requested CUDA/MPS device.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .device_capture import (CaptureError, DEFAULT_EVIDENCE, DEFAULT_RECEIPTS,
                             compile_segment_runtime)
from .native_family_capture import capture_native_text_route, compile_native_text_kernel

ADMITTED_MODEL_TYPES = ("qwen4_exp_text",)
ADMITTED_SOURCE_FORMATS = ("F32", "F16", "BF16")
ADMITTED_BACKENDS = ("native", "device", "public-range")


def capture_qwen4_route(checkpoint: str | Path, *, model_type: str = "qwen4_exp_text",
                        source_format: str, backend: str = "native", library: str | Path | None = None,
                        runtime_library: str | Path | None = None,
                        device: str | None = None,
                        prompt_ids: Sequence[int], max_tokens: int = 4,
                        recovery_tokens: int = 2, context_tokens: int = 64,
                        max_weight_bytes: int = 128 * 1024**3,
                        max_runtime_bytes: int = 32 * 1024**3,
                        case_id: str | None = None, tolerance: float = 1e-3,
                        evidence_root: str | Path = DEFAULT_EVIDENCE,
                        receipts_path: str | Path = DEFAULT_RECEIPTS):
    if model_type not in ADMITTED_MODEL_TYPES:
        raise CaptureError("Qwen4 capture currently supports only the exact text identity; "
                           "multimodal media evidence is not implemented")
    if backend == "native" and library is None:
        raise CaptureError("compiled native Qwen4 library is required for native capture")
    if backend == "public-range" and (library is None or runtime_library is None):
        raise CaptureError("Qwen4 public-range capture requires both compiled Qwen4 and Segment/Edge libraries")
    if backend == "device" and (type(device) is not str or not device):
        raise CaptureError("Qwen4 device capture requires an explicit CUDA/MPS device")
    return capture_native_text_route(
        "qwen4", checkpoint, library=library, runtime_library=runtime_library,
        source_format=source_format,
        backend=backend, execution_device=device, prompt_ids=prompt_ids, max_tokens=max_tokens,
        recovery_tokens=recovery_tokens, context_tokens=context_tokens,
        max_weight_bytes=max_weight_bytes, max_runtime_bytes=max_runtime_bytes,
        case_id=case_id, tolerance=tolerance, evidence_root=evidence_root,
        receipts_path=receipts_path)


def add_qwen4_subparser(subparsers) -> None:
    p = subparsers.add_parser("qwen4", help="capture an exact Qwen4 native, device, or public-range text route")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model-type", choices=ADMITTED_MODEL_TYPES, default="qwen4_exp_text")
    p.add_argument("--source-format", required=True, choices=ADMITTED_SOURCE_FORMATS)
    p.add_argument("--backend", choices=ADMITTED_BACKENDS, default="native")
    p.add_argument("--device", help="explicit CUDA/MPS device for --backend device")
    p.add_argument("--library", type=Path)
    p.add_argument("--build-directory", type=Path,
                   help="compile native_qwen4.c into this directory before capture")
    p.add_argument("--runtime-library", type=Path,
                   help="compiled Segment/Edge runtime for --backend public-range")
    p.add_argument("--build-runtime-directory", type=Path,
                   help="build the Segment/Edge runtime into this directory for public-range capture")
    p.add_argument("--case-id")
    p.add_argument("--prompt-ids", required=True, help="comma-separated token IDs")
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--recovery-tokens", type=int, default=2)
    p.add_argument("--context-tokens", type=int, default=64)
    p.add_argument("--max-weight-bytes", type=int, default=128 * 1024**3)
    p.add_argument("--max-runtime-bytes", type=int, default=32 * 1024**3)
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    p.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)


def run_qwen4_subcommand(args) -> int:
    try:
        if args.backend in ("native", "public-range"):
            if args.device is not None:
                raise CaptureError("--device applies only to --backend device")
            if args.library is not None and args.build_directory is not None:
                raise CaptureError("choose --library or --build-directory, not both")
            if args.backend == "native" and (args.runtime_library is not None or
                                               args.build_runtime_directory is not None):
                raise CaptureError("Segment/Edge runtime options apply only to --backend public-range")
            if args.library is None and args.build_directory is None:
                raise CaptureError("compiled native library required; pass --library or --build-directory")
            library = (args.library if args.library is not None else
                       compile_native_text_kernel("qwen4", args.build_directory))
            if args.backend == "public-range":
                if args.runtime_library is not None and args.build_runtime_directory is not None:
                    raise CaptureError("choose --runtime-library or --build-runtime-directory, not both")
                if args.runtime_library is None and args.build_runtime_directory is None:
                    raise CaptureError("public-range requires --runtime-library or --build-runtime-directory")
                runtime_library = (args.runtime_library if args.runtime_library is not None else
                                   compile_segment_runtime(args.build_runtime_directory))
            else:
                runtime_library = None
        else:
            if (args.library is not None or args.build_directory is not None or
                    args.runtime_library is not None or args.build_runtime_directory is not None):
                raise CaptureError("Qwen4 device candidate does not use a native C library")
            if not args.device:
                raise CaptureError("--backend device requires an explicit --device")
            library = None
            runtime_library = None
        prompt = _parse_prompt_ids(args.prompt_ids)
        result = capture_qwen4_route(
            args.checkpoint, model_type=args.model_type, source_format=args.source_format,
            backend=args.backend, library=library, runtime_library=runtime_library,
            device=args.device, prompt_ids=prompt,
            max_tokens=args.max_tokens, recovery_tokens=args.recovery_tokens,
            context_tokens=args.context_tokens, max_weight_bytes=args.max_weight_bytes,
            max_runtime_bytes=args.max_runtime_bytes, case_id=args.case_id,
            tolerance=args.tolerance, evidence_root=args.evidence_root,
            receipts_path=args.receipts)
    except (CaptureError, ValueError, OSError, ImportError) as error:
        print("capture rejected: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


def _parse_prompt_ids(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise CaptureError("--prompt-ids must be comma-separated integers") from error
    if not result or any(value < 0 for value in result):
        raise CaptureError("--prompt-ids must contain nonnegative token IDs")
    return result


__all__ = ["CaptureError", "ADMITTED_BACKENDS", "ADMITTED_MODEL_TYPES",
           "ADMITTED_SOURCE_FORMATS", "capture_qwen4_route", "add_qwen4_subparser",
           "run_qwen4_subcommand"]
