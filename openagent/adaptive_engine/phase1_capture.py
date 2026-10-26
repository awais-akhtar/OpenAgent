"""Phase 1 capture entry point for exact text and publisher media routes.

Inkling text, Qwen4 text, and Qwen3.5 text capture run the compiled production
CPU decoder and compare against separate NumPy implementations. Device and
multimodal paths have separate commands with actual media inputs. Inkling's standalone text identity
has no MTP component; its separate ``inkling_mm_model`` wrapper still requires
real media and speculative target/draft evidence.

GLM5-next host text capture compares the compiled CPU decoder with separate
Torch CUDA/MPS equations; its device capture uses the scalar C comparator.
Publisher media uses the separate ``glm5-publisher-media`` command.

    # OLMoE (existing):
    python -m openagent.adaptive_engine.phase1_capture olmoe \\
        --checkpoint models/olmoe-1b-7b-instruct \\
        --manifest models/olmoe-1b-7b-instruct/openagent-experts.q8.manifest.json \\
        --build-library build/native --prompt "The capital of France is" --max-tokens 4

    # Standalone Inkling text (native backend only):
    python -m openagent.adaptive_engine.phase1_capture inkling \\
        --checkpoint models/inkling-text \\
        --source-format F32 --backend native \\
        --build-directory build/native-inkling --prompt-ids 1,2,3,4 --max-tokens 4

    # Qwen4 text-only native routes (3 formats; compiled C decoder required):
    python -m openagent.adaptive_engine.phase1_capture qwen4 \\
        --checkpoint models/qwen4-flash-next \\
        --model-type qwen4_exp_text --source-format F32 --backend native \\
        --build-directory build/native-qwen4 --prompt-ids 1,2,3,4 --max-tokens 4

    # Qwen3.5 text-only native routes (3 formats; compiled C decoder required):
    python -m openagent.adaptive_engine.phase1_capture qwen35 \\
        --checkpoint models/qwen35-moe \\
        --model-type qwen3_5_moe_text --source-format F32 --backend native \\
        --build-directory build/native-qwen35 --prompt-ids 1,2,3,4 --max-tokens 4

    # GLM5-next text-only native route (Torch reference requires an accelerator):
    python -m openagent.adaptive_engine.phase1_capture glm5 \\
        --checkpoint models/glm5-next-text --source-format F32 --backend native \\
        --reference-device cuda:0 --build-directory build/native-glm5 \\
        --prompt-ids 1,2,3,4 --max-tokens 4
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .device_capture import (
    CaptureError,
    OLMOE_ABI_OWNER,
    OLMOE_CONSUMER,
    OLMOE_DIRECT_CONSUMERS,
    canonical_row,
    capture_olmoe_route,
    compile_segment_executor,
    compile_segment_runtime,
    _write_receipts,
    _olmoe_direct_backend,
)
from .inkling_capture import (
    capture_inkling_route,
    add_inkling_subparser,
    run_inkling_subcommand,
)
from .qwen4_capture import (
    capture_qwen4_route,
    add_qwen4_subparser,
    run_qwen4_subcommand,
)
from .qwen35_capture import (
    capture_qwen35_route,
    add_qwen35_subparser,
    run_qwen35_subcommand,
)
from .glm5_capture import add_glm5_subparser, run_glm5_subcommand
from .glm5_capture import capture_glm5_route
from .inkling_mm_capture import add_inkling_mm_subparser, run_inkling_mm_subcommand
from .qwen4_media_capture import add_qwen4_media_subparser, run_qwen4_media_subcommand
from .glm5_publisher_capture import add_glm5_media_subparser, run_glm5_media_subcommand


def build_parser() -> argparse.ArgumentParser:
    """Build the same command contract used by capture and Colab preflight."""
    parser = argparse.ArgumentParser(
        description="Capture admitted Phase 1 runtime route receipts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- OLMoE subcommand (delegated to device_capture) ---
    from .device_capture import main as _olmoe_main
    # Register OLMoE args by parsing the device_capture subparser once.
    # Reuse one argparse tree for every family capture command.
    olmoe = subparsers.add_parser("olmoe", help="capture one exact OLMoE runtime route")
    from pathlib import Path
    olmoe.add_argument("--checkpoint", required=True, type=Path)
    olmoe.add_argument("--manifest", required=True, type=Path)
    olmoe.add_argument("--case-id", default=None)
    _lib = olmoe.add_mutually_exclusive_group(required=False)
    _lib.add_argument("--library", type=Path)
    _lib.add_argument("--build-library", type=Path, metavar="DIRECTORY")
    _lib.add_argument("--build-public-runtime", type=Path, metavar="DIRECTORY")
    _prompt = olmoe.add_mutually_exclusive_group(required=True)
    _prompt.add_argument("--prompt-ids")
    _prompt.add_argument("--prompt")
    olmoe.add_argument("--max-tokens", type=int, default=4)
    olmoe.add_argument("--recovery-tokens", type=int, default=2)
    olmoe.add_argument("--context-tokens", type=int, default=64)
    olmoe.add_argument("--cache-mib", type=int, default=256)
    olmoe.add_argument("--resident-mib", type=int, default=4096)
    olmoe.add_argument("--transfer-mib", type=int, default=4096)
    olmoe.add_argument("--staging-mib", type=int, default=1024)
    olmoe.add_argument("--conversion-mib", type=int, default=2048)
    olmoe.add_argument("--backend", choices=("native", "public-range"), default="native")
    olmoe.add_argument("--tolerance", type=float, default=1e-3)
    from .device_capture import DEFAULT_EVIDENCE, DEFAULT_RECEIPTS
    olmoe.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    olmoe.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)

    # --- Family text capture subcommands ---
    add_inkling_subparser(subparsers)
    add_qwen4_subparser(subparsers)
    add_qwen35_subparser(subparsers)
    add_glm5_subparser(subparsers)
    add_inkling_mm_subparser(subparsers)
    add_qwen4_media_subparser(subparsers)
    add_glm5_media_subparser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Multi-family Phase 1 capture CLI."""
    args = build_parser().parse_args(argv)

    if args.command == "olmoe":
        return _run_olmoe_subcommand(args)
    if args.command == "inkling":
        return run_inkling_subcommand(args)
    if args.command == "qwen4":
        return run_qwen4_subcommand(args)
    if args.command == "qwen35":
        return run_qwen35_subcommand(args)
    if args.command == "glm5":
        return run_glm5_subcommand(args)
    if args.command == "inkling-mm":
        return run_inkling_mm_subcommand(args)
    if args.command == "qwen4-media":
        return run_qwen4_media_subcommand(args)
    if args.command == "glm5-publisher-media":
        return run_glm5_media_subcommand(args)
    print(f"unknown subcommand: {args.command}", file=sys.stderr)
    return 1


def _run_olmoe_subcommand(args) -> int:
    import json
    from pathlib import Path as _Path
    try:
        if args.prompt is not None:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(_Path(args.checkpoint) / "tokenizer.json"))
            prompt_ids = list(tokenizer.encode(args.prompt).ids)
        else:
            prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
        library_path = (
            args.library if args.library is not None
            else compile_segment_executor(args.build_library)
            if args.build_library is not None
            else compile_segment_runtime(args.build_public_runtime)
            if args.build_public_runtime is not None
            else None
        )
        print(f"prompt token IDs: {prompt_ids}", file=sys.stderr)
        measured = capture_olmoe_route(
            args.checkpoint, manifest=args.manifest, library=library_path,
            prompt_ids=prompt_ids,
            max_tokens=args.max_tokens, recovery_tokens=args.recovery_tokens,
            context_tokens=args.context_tokens, cache_mib=args.cache_mib,
            resident_mib=args.resident_mib, transfer_mib=args.transfer_mib,
            staging_mib=args.staging_mib, conversion_mib=args.conversion_mib,
            backend=args.backend, case_id=args.case_id,
            tolerance=args.tolerance,
            evidence_root=args.evidence_root, receipts_path=args.receipts,
        )
    except (CaptureError, ValueError, OSError, ImportError) as error:
        print("capture rejected: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(measured, indent=1, sort_keys=True))
    return 0


__all__ = [
    "CaptureError",
    "OLMOE_ABI_OWNER", "OLMOE_CONSUMER", "OLMOE_DIRECT_CONSUMERS",
    "canonical_row",
    "capture_olmoe_route", "capture_inkling_route", "capture_qwen4_route", "capture_qwen35_route",
    "capture_glm5_route",
    "compile_segment_executor", "compile_segment_runtime",
    "main", "_write_receipts", "_olmoe_direct_backend",
]


if __name__ == "__main__":
    raise SystemExit(main())
