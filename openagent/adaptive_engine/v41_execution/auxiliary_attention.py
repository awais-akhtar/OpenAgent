"""Bounded query-row attention for publisher vision and draft execution.

Retains the FP32 softmax policy, including the zero-value attention sink.
Key banks are borrowed independently; draft windows need no concatenated copy.
This source implementation makes no numerical certification claim.
"""
import math

import numpy as np


def attention_rows(query, key_banks, value_banks, *, max_bytes, sink=None,
                   check_cancelled=None):
    q = np.asarray(query)
    keys, values = tuple(key_banks), tuple(value_banks)
    if (q.dtype != np.float32 or q.ndim != 3 or not all(q.shape) or
            not keys or len(keys) != len(values) or
            type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError('FP32 auxiliary attention geometry and budget required')
    count, heads, width = q.shape
    total = 0
    for key, value in zip(keys, values):
        if (not isinstance(key, np.ndarray) or not isinstance(value, np.ndarray) or
                key.dtype != np.float32 or value.dtype != np.float32 or
                key.shape != value.shape or key.ndim not in (2, 3) or
                key.shape[-1] != width or
                (key.ndim == 3 and key.shape[1] != heads)):
            raise ValueError('auxiliary key/value bank geometry mismatch')
        total += len(key)
    if not total:
        raise ValueError('auxiliary attention requires occupied keys')
    if check_cancelled is not None and not callable(check_cancelled):
        raise TypeError('callable auxiliary cancellation check required')
    # Output, score/probability temporaries, row contraction and finite masks.
    required = 4*(3*count*heads*width + 6*heads*total + 4*heads*width + 8*heads)
    if required > max_bytes:
        raise MemoryError('auxiliary row attention exceeds workspace admission')
    def check():
        if check_cancelled is not None:
            check_cancelled()
    check()
    if sink is not None:
        sink = np.asarray(sink)
        if sink.dtype != np.float32 or sink.shape != (heads,) or not np.isfinite(sink).all():
            raise ValueError('finite FP32 per-head sink required')
    output = np.empty_like(q)
    scores = np.empty((heads, total), dtype=np.float32)
    scale = math.sqrt(width)
    for index, row in enumerate(q):
        check()
        if not np.isfinite(row).all():
            raise ValueError('nonfinite auxiliary queries')
        offset = 0
        for bank in keys:
            check()
            end = offset + len(bank)
            scores[:, offset:end] = (np.einsum('hd,kd->hk', row, bank)
                if bank.ndim == 2 else np.einsum('hd,khd->hk', row, bank)) / scale
            offset = end
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite auxiliary attention scores')
        maximum = scores.max(axis=-1, keepdims=True)
        if sink is not None:
            maximum = np.maximum(maximum, sink[:, None])
        probability = np.exp(scores-maximum)
        denominator = probability.sum(axis=-1, keepdims=True)
        if sink is not None:
            denominator += np.exp(sink[:, None]-maximum)
        probability /= denominator
        result = np.zeros((heads, width), dtype=np.float32)
        offset = 0
        for bank in values:
            check()
            end = offset + len(bank)
            part = probability[:, offset:end]
            result += (np.einsum('hk,kd->hd', part, bank) if bank.ndim == 2
                       else np.einsum('hk,khd->hd', part, bank))
            offset = end
        if not np.isfinite(result).all():
            raise ValueError('nonfinite auxiliary attention output')
        output[index] = result
    check()
    return output
