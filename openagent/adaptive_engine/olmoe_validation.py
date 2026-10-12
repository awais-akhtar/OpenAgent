"""Teacher-forced full-decoder comparisons, not public-checkpoint certification.

The injected reference must independently compute a real floating-point array
of shape (sequence_length, vocabulary_size), without a batch dimension. It must
start from empty state on each call. Independence and checkpoint provenance are
caller responsibilities; matching logits alone cannot establish either.
"""

from __future__ import annotations

import platform
import time
from typing import Callable, Iterable

import numpy as np

from .olmoe import OlmoeModel


def _logits(value, shape: tuple[int, ...], source: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{source} logits shape {array.shape}; expected {shape}")
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError(f"{source} logits must be finite real floating-point values")
    result = array.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{source} logits cannot be represented in float64")
    return result


def _metrics(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            difference = np.abs(actual - expected)
            relative = difference / np.maximum(np.abs(expected), np.finfo(np.float64).eps)
            maximum = float(difference.max())
            # Scaling avoids overflow when squaring otherwise finite errors.
            rmse = maximum * float(np.sqrt(np.mean((difference / maximum) ** 2))) if maximum else 0.0
            matches = np.argmax(actual, axis=1) == np.argmax(expected, axis=1)
            return {
                "max_absolute_error": maximum,
                "mean_absolute_error": float(np.mean(difference)),
                "root_mean_square_error": rmse,
                "max_relative_error": float(relative.max()),
                "argmax_matches": int(matches.sum()),
                "argmax_agreement": float(matches.mean()),
                "allclose": bool(np.all(difference <= atol + rtol * np.abs(expected))),
            }
        except FloatingPointError as exc:
            raise ValueError("logit metrics overflowed float64") from exc


def validate_olmoe(
    model: OlmoeModel,
    reference: Callable[[tuple[int, ...]], np.ndarray],
    sequences: Iterable[Iterable[int]],
    *,
    reference_identity: dict[str, str],
    atol: float = 1e-4,
    rtol: float = 1e-4,
    minimum_argmax_agreement: float = 1.0,
    reference_synchronize: Callable[[], None] | None = None,
) -> dict:
    """Compare every teacher-forced position; leave the decoder reset afterward.

    Timing includes forward calls only, excludes conversion/metrics and model
    loading, and has no warmup. OpenAgent's NumPy decoder is synchronous. For an
    asynchronous reference, supply a device synchronization callback. The report
    deliberately makes no assertion about independent-reference provenance.
    """
    for name, value in (("atol", atol), ("rtol", rtol),
                        ("minimum_argmax_agreement", minimum_argmax_agreement)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if minimum_argmax_agreement > 1:
        raise ValueError("minimum_argmax_agreement must not exceed one")
    if not callable(reference):
        raise ValueError("reference must be callable")
    required = ("model_id", "revision", "implementation", "device", "dtype")
    if not isinstance(reference_identity, dict) or any(
        not isinstance(reference_identity.get(key), str) or not reference_identity[key].strip()
        for key in required
    ):
        raise ValueError(f"reference_identity requires nonempty strings: {required}")
    prepared = []
    for sequence in sequences:
        ids = np.asarray(list(sequence))
        if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
            raise ValueError("each sequence must be a nonempty integer vector")
        if np.any(ids < 0) or np.any(ids >= model.config.vocab_size):
            raise ValueError("sequence token outside model vocabulary")
        if len(ids) > model.context_tokens:
            raise ValueError("sequence exceeds model context capacity")
        prepared.append(tuple(int(token) for token in ids))
    if not prepared:
        raise ValueError("at least one sequence is required")
    if getattr(reference, "__self__", None) is model or reference is model:
        raise ValueError("reference must not be the decoder under test")
    records, actual_rows, reference_rows = [], [], []
    try:
        for ids in prepared:
            model.reset()
            if reference_synchronize is not None:
                reference_synchronize()
            start = time.perf_counter()
            raw = reference(ids)
            if reference_synchronize is not None:
                reference_synchronize()
            reference_seconds = time.perf_counter() - start
            expected = _logits(raw, (len(ids), model.config.vocab_size), "reference")
            rows, step_seconds = [], []
            for token in ids:
                start = time.perf_counter()
                raw = model.forward([token])
                step_seconds.append(time.perf_counter() - start)
                rows.append(_logits(raw, (model.config.vocab_size,), "decoder"))
            actual = np.stack(rows)
            metrics = _metrics(actual, expected, atol, rtol)
            records.append({"token_ids": list(ids), "positions": len(ids),
                            "metrics": metrics, "decoder_step_seconds": step_seconds,
                            "decoder_seconds": sum(step_seconds),
                            "reference_seconds": reference_seconds})
            actual_rows.append(actual)
            reference_rows.append(expected)
    finally:
        model.reset()
    metrics = _metrics(np.concatenate(actual_rows), np.concatenate(reference_rows), atol, rtol)
    return {
        "format": "openagent.olmoe-validation.v1",
        "scope": "injected-reference teacher-forced full-decoder logits",
        "public_checkpoint_validated": False,
        "provenance": "Reference independence and checkpoint equivalence are caller-supplied, not verified.",
        "model": {"model_id": model.manifest.model_id,
                  "metadata_fingerprint": model.manifest.fingerprint,
                  "checkpoint_directory": str(model.root),
                  "implementation": "openagent.adaptive_engine.olmoe.OlmoeModel",
                  "dtype": "fp32 dense / int8-row experts", "device": "cpu dense; expert backend selected by runtime",
                  "vocab_size": model.config.vocab_size,
                  "num_hidden_layers": model.config.num_hidden_layers},
        "reference": dict(reference_identity),
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor() or "unknown",
                     "python": platform.python_version(), "numpy": np.__version__},
        "timing": {"clock": "perf_counter", "warmup_runs": 0,
                   "reference_synchronized": reference_synchronize is not None,
                   "scope": "forward only; excludes loading, validation and metrics; cache may be warm",
                   "decoder_seconds": sum(record["decoder_seconds"] for record in records),
                   "reference_seconds": sum(record["reference_seconds"] for record in records)},
        "thresholds": {"atol": atol, "rtol": rtol,
                       "minimum_argmax_agreement": minimum_argmax_agreement},
        "sequence_count": len(records), "positions": sum(map(len, prepared)),
        "metrics": metrics,
        "passed": metrics["allclose"] and metrics["argmax_agreement"] >= minimum_argmax_agreement,
        "sequences": records,
    }
