"""Checks the GGUF quantized kernels against literal scalar ggml reference loops."""

from __future__ import annotations

import random
import struct

import numpy as np

from openagent.adaptive_engine.qwen3moe_gguf import dequantize


def _f16_bytes(value: float) -> bytes:
    return struct.pack("<e", value)


def _f16(data: bytes | list[int]) -> float:
    return struct.unpack("<e", bytes(data))[0]


def _scale_min(j: int, q: list[int]) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def reference_q4_k(block: list[int]) -> list[float]:
    d = _f16(block[0:2])
    dmin = _f16(block[2:4])
    scales = block[4:16]
    qs = block[16:144]
    out: list[float] = []
    q = 0
    is_ = 0
    for _ in range(4):
        sc, m = _scale_min(is_, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = _scale_min(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        for l in range(32):
            out.append(d1 * (qs[q + l] & 0xF) - m1)
        for l in range(32):
            out.append(d2 * (qs[q + l] >> 4) - m2)
        q += 32
        is_ += 2
    return out


def reference_q6_k(block: list[int]) -> list[float]:
    ql_all = block[0:128]
    qh_all = block[128:192]
    sc_all = [v - 256 if v > 127 else v for v in block[192:208]]
    d = _f16(block[208:210])
    y = [0.0] * 256
    for half in range(2):
        ql = ql_all[64 * half: 64 * half + 64]
        qh = qh_all[32 * half: 32 * half + 32]
        sc = sc_all[8 * half: 8 * half + 8]
        base = 128 * half
        for l in range(32):
            is_ = l // 16
            q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
            y[base + l] = d * sc[is_] * q1
            y[base + l + 32] = d * sc[is_ + 2] * q2
            y[base + l + 64] = d * sc[is_ + 4] * q3
            y[base + l + 96] = d * sc[is_ + 6] * q4
    return y


def _random_block(rng: random.Random, size: int, scale_offsets: tuple[int, int]) -> list[int]:
    raw = [rng.randrange(256) for _ in range(size)]
    raw[scale_offsets[0]: scale_offsets[1]] = list(_f16_bytes(rng.uniform(0.001, 0.05)))
    return raw


def test_q4_k_matches_reference_blocks():
    rng = random.Random(7)
    blocks = [_random_block(rng, 144, (0, 2)) for _ in range(3)]
    for block in blocks:
        block[2:4] = list(_f16_bytes(rng.uniform(0.001, 0.05)))
    raw = bytes(v for block in blocks for v in block)
    got = dequantize(raw, 12, 256 * 3)
    expected = np.array([v for block in blocks for v in reference_q4_k(block)], dtype=np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_q6_k_matches_reference_blocks():
    rng = random.Random(11)
    blocks = []
    for _ in range(3):
        block = [rng.randrange(256) for _ in range(210)]
        block[208:210] = list(_f16_bytes(rng.uniform(0.001, 0.05)))
        blocks.append(block)
    raw = bytes(v for block in blocks for v in block)
    got = dequantize(raw, 14, 256 * 3)
    expected = np.array([v for block in blocks for v in reference_q6_k(block)], dtype=np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_q8_0_scales_int8_values():
    block = list(_f16_bytes(0.5)) + [(-3) & 0xFF] + [2] * 31
    got = dequantize(bytes(block), 8, 32)
    expected = np.array([0.5 * -3] + [0.5 * 2] * 31, dtype=np.float32)
    np.testing.assert_allclose(got, expected)


def test_f32_and_f16_round_trip():
    values = np.array([1.5, -2.25, 3.0], dtype="<f4")
    np.testing.assert_array_equal(dequantize(values.tobytes(), 0, 3), values)
    halves = np.array([0.5, -1.0], dtype="<f2")
    np.testing.assert_array_equal(dequantize(halves.tobytes(), 1, 2), halves.astype(np.float32))


def test_native_matvec_matches_numpy_dequant():
    from openagent.adaptive_engine import qwen3moe_native as native
    from openagent.adaptive_engine.qwen3moe_gguf import dequantize as dq

    if not native.native_available():
        import pytest

        pytest.skip("no MSVC build or prebuilt kernel library on this machine")
    rng = random.Random(3)
    rows, cols = 37, 512
    for ggml_type, block_bytes in ((12, 144), (14, 210)):
        raw = bytearray(rng.randrange(256) for _ in range(rows * (cols // 256) * block_bytes))
        for block in range(rows * (cols // 256)):
            start = block * block_bytes + (0 if ggml_type == 14 else 0)
            if ggml_type == 12:
                raw[start + 0: start + 4] = _f16_bytes(rng.uniform(0.001, 0.05)) + _f16_bytes(rng.uniform(0.001, 0.05))
            else:
                raw[start + 208: start + 210] = _f16_bytes(rng.uniform(0.001, 0.05))
        x = np.array([rng.uniform(-1, 1) for _ in range(cols)], dtype=np.float32)
        expected = dq(bytes(raw), ggml_type, rows * cols).reshape(rows, cols) @ x
        got = native.matvec(ggml_type, bytes(raw), rows, cols, x)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_numpy_fallback_matches_native_matvec(monkeypatch):
    from openagent.adaptive_engine import qwen3moe_native as native

    rng = random.Random(9)
    rows, cols = 16, 512
    raw = bytes(rng.randrange(256) for _ in range(rows * (cols // 256) * 144))
    x = np.array([rng.uniform(-1, 1) for _ in range(cols)], dtype=np.float32)
    expected = native.matvec(12, raw, rows, cols, x)
    monkeypatch.setattr(native, "_load_attempted", False)
    monkeypatch.setattr(native, "_library", None)
    monkeypatch.setattr(native, "cpu_has_avx2", lambda: False)
    fallback = native.matvec(12, raw, rows, cols, x)
    np.testing.assert_allclose(fallback, expected, rtol=1e-3, atol=1e-3)
    monkeypatch.setattr(native, "_load_attempted", False)
    monkeypatch.setattr(native, "_library", None)
