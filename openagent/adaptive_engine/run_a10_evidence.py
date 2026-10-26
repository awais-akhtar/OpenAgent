"""Retired family-level A10 command; refuses to emit synthetic evidence.

Use :mod:`openagent.adaptive_engine.phase1_capture` for exact-route runtime
evidence. This compatibility command never probes devices or writes files.
"""

import argparse
from pathlib import Path
import sys


def create_test_evidence_binding(family: str, routes: int) -> None:
    """Reject the retired synthetic fixture producer.

    Schema fixtures must never be serialized as runtime evidence. Use
    ``phase1_capture`` with a real checkpoint and an exact admitted route.
    """
    raise RuntimeError(
        "synthetic A10 evidence is prohibited; use "
        "openagent.adaptive_engine.phase1_capture with a real checkpoint")


def run_a10_evidence(
    model_source: str = 'auto',
    output_dir: Path = Path('a10_evidence'),
    routes: int = 414,
    device: str = 'auto',
) -> int:
    """Refuse the retired synthetic workflow (arguments kept for callers)."""

    # The previous implementation mapped unrelated Ollama models to families,
    # synthesized measurements, and set parity_verified=True without running
    # the admitted route. Refuse before creating files or probing devices.
    print(
        "Refusing to generate family-level A10 evidence: this legacy workflow "
        "cannot bind a real model execution to an exact admitted route. Use "
        "python -m openagent.adaptive_engine.phase1_capture with an explicit "
        "checkpoint, manifest, prompt IDs, and route-supported backend.",
        file=sys.stderr)
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 1 A10 Evidence Capture Executor'
    )
    parser.add_argument(
        '--model-source',
        choices=['ollama', 'cpu', 'arc_gpu', 'auto'],
        default='auto',
        help='Model source (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('a10_evidence'),
        help='Output directory for evidence manifests'
    )
    parser.add_argument(
        '--routes',
        type=int,
        default=414,
        help='Number of routes to process (default: all 414)'
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Device to use (xpu:0, cpu, auto)'
    )

    args = parser.parse_args()

    try:
        sys.exit(run_a10_evidence(
            model_source=args.model_source,
            output_dir=args.output_dir,
            routes=args.routes,
            device=args.device,
        ))
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
