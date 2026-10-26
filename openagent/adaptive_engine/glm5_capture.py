"""Capture exact GLM5-next text routes with cross-backend comparison.

Native CPU candidates compare against the separate Torch device implementation;
device candidates compare against the separately compiled native C decoder.
Publisher media routes remain a distinct capture contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .device_capture import CaptureError, DEFAULT_EVIDENCE, DEFAULT_RECEIPTS
from .device_capture import compile_segment_runtime
from .native_family_capture import capture_native_text_route, compile_native_text_kernel


def capture_glm5_route(checkpoint: str | Path, *, source_format: str,
                       backend: str = "native", library: str | Path | None = None,
                       runtime_library: str | Path | None = None,
                       reference_device: str | None = None,
                       reference_library: str | Path | None = None,
                       device: str | None = None,
                       prompt_ids: Sequence[int], max_tokens: int = 4,
                       recovery_tokens: int = 2, context_tokens: int = 64,
                       max_weight_bytes: int = 128 * 1024**3,
                       max_runtime_bytes: int = 32 * 1024**3,
                       case_id: str | None = None, tolerance: float = 1e-3,
                       evidence_root: str | Path = DEFAULT_EVIDENCE,
                       receipts_path: str | Path = DEFAULT_RECEIPTS):
    return capture_native_text_route(
        "glm5-next", checkpoint, library=library, source_format=source_format,
        runtime_library=runtime_library, backend=backend, reference_device=reference_device,
        reference_library=reference_library, execution_device=device,
        prompt_ids=prompt_ids,
        max_tokens=max_tokens, recovery_tokens=recovery_tokens,
        context_tokens=context_tokens, max_weight_bytes=max_weight_bytes,
        max_runtime_bytes=max_runtime_bytes, case_id=case_id, tolerance=tolerance,
        evidence_root=evidence_root, receipts_path=receipts_path)


def add_glm5_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "glm5", help="capture exact GLM5-next text routes with cross-backend comparison")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-format", required=True, choices=("F32", "F16", "BF16"))
    parser.add_argument("--backend", choices=("native", "device", "public-range"), default="native")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--build-directory", type=Path,
                         help="compile native_glm5_next.c into this directory")
    parser.add_argument("--runtime-library", type=Path,
                        help="compiled Segment/Edge runtime for public-range candidate execution")
    parser.add_argument("--build-runtime-directory", type=Path,
                        help="build the Segment/Edge runtime for public-range candidate execution")
    parser.add_argument("--reference-library", type=Path,
                         help="compiled native_glm5_next.c reference library for device capture")
    parser.add_argument("--device", help="explicit CUDA/MPS candidate device")
    parser.add_argument("--reference-device",
                         help="explicit CUDA/MPS Torch reference device for native capture")
    parser.add_argument("--case-id")
    parser.add_argument("--prompt-ids", required=True, help="comma-separated token IDs")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--recovery-tokens", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=64)
    parser.add_argument("--max-weight-bytes", type=int, default=128 * 1024**3)
    parser.add_argument("--max-runtime-bytes", type=int, default=32 * 1024**3)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)


def run_glm5_subcommand(args) -> int:
    try:
        runtime_library = None
        if args.backend == "native":
            if (args.device is not None or args.reference_library is not None or
                    args.runtime_library is not None or args.build_runtime_directory is not None):
                raise CaptureError("device/reference/runtime options do not apply to native text capture")
            if args.library is not None and args.build_directory is not None:
                raise CaptureError("choose --library or --build-directory, not both")
            if args.library is None and args.build_directory is None:
                raise CaptureError("compiled native GLM5 library required")
            library = (args.library if args.library is not None else
                       compile_native_text_kernel("glm5-next", args.build_directory))
        elif args.backend == "device":
            if args.library is not None or args.build_directory is not None:
                raise CaptureError("the GLM5 device candidate is Torch, not the native decoder")
            if args.runtime_library is not None or args.build_runtime_directory is not None:
                raise CaptureError("the GLM5 device candidate does not use the Segment/Edge runtime")
            if not args.device or args.reference_library is None:
                raise CaptureError("device capture requires --device and --reference-library")
            if args.reference_device is not None:
                raise CaptureError("device capture compares against native C, not a second Torch device")
            library = None
        else:
            if args.device is not None or args.reference_library is not None:
                raise CaptureError("public-range GLM5 text is CPU-native and compares against the Torch reference")
            if args.library is not None and args.build_directory is not None:
                raise CaptureError("choose --library or --build-directory, not both")
            if args.library is None and args.build_directory is None:
                raise CaptureError("public-range capture requires a compiled native GLM5 text library")
            if args.runtime_library is not None and args.build_runtime_directory is not None:
                raise CaptureError("choose --runtime-library or --build-runtime-directory, not both")
            if args.runtime_library is None and args.build_runtime_directory is None:
                raise CaptureError("public-range capture requires a compiled Segment/Edge runtime")
            library = (args.library if args.library is not None else
                       compile_native_text_kernel("glm5-next", args.build_directory))
            runtime_library = (args.runtime_library if args.runtime_library is not None else
                               compile_segment_runtime(args.build_runtime_directory))
            if not args.reference_device:
                raise CaptureError("public-range capture requires an explicit Torch reference device")
        result = capture_glm5_route(
            args.checkpoint, source_format=args.source_format, backend=args.backend,
            library=library, reference_device=args.reference_device,
            runtime_library=runtime_library,
            reference_library=args.reference_library, device=args.device,
            prompt_ids=_parse_prompt_ids(args.prompt_ids), max_tokens=args.max_tokens,
            recovery_tokens=args.recovery_tokens, context_tokens=args.context_tokens,
            max_weight_bytes=args.max_weight_bytes,
            max_runtime_bytes=args.max_runtime_bytes, case_id=args.case_id,
            tolerance=args.tolerance, evidence_root=args.evidence_root,
            receipts_path=args.receipts)
    except (CaptureError, ValueError, OSError, ImportError, RuntimeError) as error:
        print("capture rejected: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


def _parse_prompt_ids(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise CaptureError("--prompt-ids must be comma-separated integers") from error
    if not result or any(token < 0 for token in result):
        raise CaptureError("--prompt-ids must contain nonnegative token IDs")
    return result


__all__ = ["capture_glm5_route", "add_glm5_subparser", "run_glm5_subcommand"]
