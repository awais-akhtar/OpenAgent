"""Exact native/device Inkling standalone text route capture.

The standalone ``inkling_text`` identity is text-only and its inventory
planner rejects MTP components. The multimodal ``inkling_mm_model`` identity
has separate MTP/media obligations and is not handled by this runner.
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

ADMITTED_SOURCE_FORMATS = ("F32", "F16", "BF16")
ADMITTED_BACKENDS = ("native", "device", "public-range")


def capture_inkling_route(checkpoint: str | Path, *, source_format: str,
                          backend: str = "native", library: str | Path | None = None,
                          runtime_library: str | Path | None = None,
                          device: str | None = None,
                          model_type: str = "inkling_text", prompt_ids: Sequence[int],
                          max_tokens: int = 4, recovery_tokens: int = 2,
                          context_tokens: int = 64, max_weight_bytes: int = 128 * 1024**3,
                          max_runtime_bytes: int = 32 * 1024**3,
                          case_id: str | None = None, tolerance: float = 1e-3,
                          evidence_root: str | Path = DEFAULT_EVIDENCE,
                          receipts_path: str | Path = DEFAULT_RECEIPTS):
    if model_type != "inkling_text":
        raise CaptureError(
            "Inkling text capture rejects inkling_mm_model: actual media plus target/draft "
            "speculative evidence is required for that identity")
    if backend in ("native", "public-range"):
        if library is None:
            raise CaptureError("compiled native_inkling.c library is required")
        if device is not None:
            raise CaptureError("--device is valid only for a device route")
        if backend == "public-range" and runtime_library is None:
            raise CaptureError("Inkling public-range capture requires a compiled Segment/Edge runtime library")
        if backend == "native" and runtime_library is not None:
            raise CaptureError("--runtime-library is valid only for --backend public-range")
    elif backend == "device":
        if library is not None:
            raise CaptureError("Inkling device capture does not accept a native C library")
        if runtime_library is not None:
            raise CaptureError("Inkling device capture does not accept a Segment/Edge library")
        if type(device) is not str or not device:
            raise CaptureError("Inkling device capture requires an explicit CUDA/MPS device")
    else:
        raise CaptureError("Inkling capture backend must be native, device, or public-range")
    return capture_native_text_route(
        "inkling", checkpoint, library=library, runtime_library=runtime_library,
        source_format=source_format,
        backend=backend, execution_device=device, prompt_ids=prompt_ids, max_tokens=max_tokens,
        recovery_tokens=recovery_tokens, context_tokens=context_tokens,
        max_weight_bytes=max_weight_bytes, max_runtime_bytes=max_runtime_bytes,
        case_id=case_id, tolerance=tolerance, evidence_root=evidence_root,
        receipts_path=receipts_path)


def add_inkling_subparser(subparsers) -> None:
    p = subparsers.add_parser("inkling", help="capture exact Inkling native/device/public-range text routes")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--source-format", required=True, choices=ADMITTED_SOURCE_FORMATS)
    p.add_argument("--backend", choices=ADMITTED_BACKENDS, default="native")
    p.add_argument("--library", type=Path)
    p.add_argument("--device", help="explicit CUDA/MPS device for --backend device")
    p.add_argument("--build-directory", type=Path,
                   help="compile native_inkling.c into this directory before capture")
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


def run_inkling_subcommand(args) -> int:
    try:
        if args.backend in ("native", "public-range"):
            if args.device is not None:
                raise CaptureError("--device applies only to --backend device")
            if args.library is not None and args.build_directory is not None:
                raise CaptureError("choose --library or --build-directory, not both")
            if args.library is None and args.build_directory is None:
                raise CaptureError("compiled native library required; pass --library or --build-directory")
            library = (args.library if args.library is not None else
                       compile_native_text_kernel("inkling", args.build_directory))
            if args.backend == "public-range":
                if args.runtime_library is not None and args.build_runtime_directory is not None:
                    raise CaptureError("choose --runtime-library or --build-runtime-directory, not both")
                if args.runtime_library is None and args.build_runtime_directory is None:
                    raise CaptureError("public-range requires --runtime-library or --build-runtime-directory")
                runtime_library = (args.runtime_library if args.runtime_library is not None else
                                   compile_segment_runtime(args.build_runtime_directory))
            else:
                if args.runtime_library is not None or args.build_runtime_directory is not None:
                    raise CaptureError("Segment/Edge options apply only to --backend public-range")
                runtime_library = None
        else:
            if (args.library is not None or args.build_directory is not None or
                    args.runtime_library is not None or args.build_runtime_directory is not None):
                raise CaptureError("Inkling device candidate does not use a native C library")
            if not args.device:
                raise CaptureError("--backend device requires an explicit --device")
            library = None
            runtime_library = None
        result = capture_inkling_route(
            args.checkpoint, source_format=args.source_format, backend=args.backend,
            library=library, runtime_library=runtime_library, device=args.device,
            prompt_ids=_parse_prompt_ids(args.prompt_ids),
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
        tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise CaptureError("--prompt-ids must be comma-separated integers") from error
    if not tokens or any(token < 0 for token in tokens):
        raise CaptureError("--prompt-ids must contain nonnegative token IDs")
    return tokens


__all__ = ["CaptureError", "ADMITTED_BACKENDS", "ADMITTED_SOURCE_FORMATS",
           "capture_inkling_route", "add_inkling_subparser", "run_inkling_subcommand"]
