"""Canonical comparison: no broadcasting, empty or nonfinite success."""
from dataclasses import dataclass
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    atol: float = 1e-6
    rtol: float = 1e-5

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in (self.atol, self.rtol)):
            raise ValueError('finite nonnegative tolerances required')


def canonical(value):
    value = np.asarray(value)
    if value.dtype.kind not in 'biuf':
        raise ValueError('real numeric canonical tensors required')
    return np.array(value, copy=True, order='C')


def identity(value):
    value = canonical(value)
    header = (value.dtype.str + ':' + repr(value.shape) + ':').encode('ascii')
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(header + value.tobytes()).hexdigest())


def compare(actual, expected, tolerance, *, allow_empty=False):
    actual, expected = canonical(actual), canonical(expected)
    result = dict(actual=identity(actual), expected=identity(expected), passed=False)
    if actual.shape != expected.shape:
        return dict(result, reason='shape mismatch')
    if (actual.dtype.kind in 'biu') != (expected.dtype.kind in 'biu'):
        return dict(result, reason='integer/boolean versus floating contract mismatch')
    if actual.dtype.kind == 'b' or expected.dtype.kind == 'b':
        if actual.dtype.kind != expected.dtype.kind:
            return dict(result, reason='boolean versus integer contract mismatch')
    if not actual.size:
        if allow_empty:
            return dict(result, passed=True, exact=True, mismatches=0,
                        reason='explicitly declared structural zero extent')
        return dict(result, reason='empty tensor is not evidence')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return dict(result, reason='nonfinite tensor')
    if actual.dtype.kind in 'biu' or expected.dtype.kind in 'biu':
        # Object integers avoid uint64/int64 promotion to rounded float64.
        mismatch = actual.astype(object) != expected.astype(object)
        metrics = dict(exact=True)
    else:
        a, e = actual.astype(np.float64), expected.astype(np.float64)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            error = np.abs(a - e)
            limit = tolerance.atol + tolerance.rtol * np.abs(e)
            mismatch = (error > limit) | ~np.isfinite(error) | ~np.isfinite(limit)
            relative = error / np.maximum(np.abs(e), np.finfo(np.float64).tiny)
        metrics = dict(exact=False, atol=tolerance.atol, rtol=tolerance.rtol,
                       max_absolute=float(error.max()), max_relative=float(relative.max()))
    return dict(result, **metrics, passed=not bool(mismatch.any()),
                mismatches=int(mismatch.sum()),
                first_mismatch=np.argwhere(mismatch)[0].tolist() if mismatch.any() else None)
