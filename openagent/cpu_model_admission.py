"""Bounded, read-only admission for a local CPU OpenVINO model bundle."""

from __future__ import annotations

import os
from pathlib import Path


CPU_MODEL_FILE_LIMIT_BYTES = 4 * 1024**3
REQUIRED_OPENVINO_FILES = (
    "config.json", "openvino_model.xml", "openvino_model.bin", "tokenizer.json",
)


class CPUModelAdmissionError(ValueError):
    """The selected local bundle cannot enter the bounded CPU product path."""


def admit_cpu_openvino_bundle(path: Path, *, max_bytes: int = CPU_MODEL_FILE_LIMIT_BYTES) -> int:
    """Check the complete local bundle before starting or loading a worker.

    The check is deliberately limited to storage and bundle completeness. It
    does not claim that a model fits every host's RAM or has acceptable latency.
    """
    root = path.expanduser().resolve()
    if max_bytes <= 0:
        raise ValueError("CPU model file budget must be positive")
    if not root.is_dir():
        raise CPUModelAdmissionError(f"CPU model directory does not exist: {root}")
    for name in REQUIRED_OPENVINO_FILES:
        file = root / name
        if file.is_symlink() or not file.is_file():
            raise CPUModelAdmissionError(f"CPU model bundle is missing a regular {name}: {root}")
    total = 0
    def fail_walk(error: OSError) -> None:
        raise CPUModelAdmissionError(f"CPU model bundle could not be inspected: {error}") from error

    for current, dirs, files in os.walk(root, followlinks=False, onerror=fail_walk):
        for directory in dirs:
            if (Path(current) / directory).is_symlink():
                raise CPUModelAdmissionError("CPU model bundle cannot contain directory symlinks")
        for name in files:
            file = Path(current) / name
            if file.is_symlink():
                raise CPUModelAdmissionError("CPU model bundle cannot contain file symlinks")
            total += file.stat().st_size
            if total > max_bytes:
                raise CPUModelAdmissionError(
                    f"CPU model bundle uses more than {max_bytes} bytes; choose a smaller local model"
                )
    return total
