"""Numerical summaries for scientific images represented as grayscale samples."""

from __future__ import annotations

import math
from typing import Iterable


def summarise_image(values: Iterable[float], *, width: int = 0, height: int = 0) -> dict[str, float | int]:
    pixels = [float(value) for value in values]
    if not pixels:
        raise ValueError("scientific image needs at least one numeric sample")
    if any(not math.isfinite(value) for value in pixels):
        raise ValueError("scientific image samples must be finite")
    mean = sum(pixels) / len(pixels)
    variance = sum((value - mean) ** 2 for value in pixels) / len(pixels)
    return {
        "samples": len(pixels),
        "width": max(0, int(width)),
        "height": max(0, int(height)),
        "minimum": min(pixels),
        "maximum": max(pixels),
        "mean": mean,
        "stddev": math.sqrt(variance),
    }
