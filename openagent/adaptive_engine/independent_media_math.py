"""NumPy media equations for independent capture references; no native kernels."""
import math

import numpy as np


class NumpyMediaMath:
    """Bounded FP32 media arithmetic with FP64 reduction intermediates.

    Attention evaluates one query at a time, keeping scratch linear in frame
    length. Encoders own their model/activation budgets; this class additionally
    bounds attention scratch and validates every public numerical input.
    """

    def __init__(self, *, max_attention_bytes=256 * 1024**2):
        if type(max_attention_bytes) is not int or max_attention_bytes <= 0:
            raise ValueError('positive independent attention memory budget required')
        self.max_attention_bytes = max_attention_bytes

    @staticmethod
    def _value(value):
        value = np.asarray(value)
        if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
            raise ValueError('nonempty finite FP32 reference input required')
        return value

    def linear(self, x, weight, bias=None):
        x, weight = self._value(x), self._value(weight)
        if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
            raise ValueError('reference projection geometry differs')
        result = np.matmul(x.astype(np.float64), weight.astype(np.float64).T)
        if bias is not None:
            bias = self._value(bias)
            if bias.shape != (weight.shape[0],):
                raise ValueError('reference projection bias geometry differs')
            result += bias
        return self._value(result.astype(np.float32))

    def norm(self, x, weight, bias=None, *, rms=False, epsilon=1e-6):
        x, weight = self._value(x), self._value(weight)
        if (x.ndim < 2 or weight.shape != (x.shape[-1],) or type(rms) is not bool
                or not math.isfinite(epsilon) or epsilon <= 0):
            raise ValueError('reference normalization geometry differs')
        values = x.astype(np.float64)
        centered = values if rms else values - values.mean(axis=-1, keepdims=True)
        result = (centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True)
                                     + epsilon)).astype(np.float32) * weight
        if bias is not None:
            bias = self._value(bias)
            if bias.shape != weight.shape:
                raise ValueError('reference normalization bias geometry differs')
            result += bias
        return self._value(result)

    def activation(self, x, *, kind='gelu'):
        x = self._value(x).astype(np.float64)
        if kind == 'gelu':
            erf = np.fromiter((math.erf(float(v)) for v in (x / math.sqrt(2)).flat),
                              dtype=np.float64, count=x.size).reshape(x.shape)
            result = 0.5 * x * (1 + erf)
        elif kind == 'gelu_pytorch_tanh':
            result = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + .044715 * x**3)))
        elif kind == 'silu':
            sigmoid = np.exp(-np.logaddexp(0, -x))
            result = x * sigmoid
        else:
            raise ValueError('unsupported reference activation')
        return self._value(result.astype(np.float32))

    def clamped_swiglu(self, gate, up, *, limit):
        gate, up = self._value(gate), self._value(up)
        if gate.shape != up.shape or not math.isfinite(limit) or limit <= 0:
            raise ValueError('invalid reference SwiGLU inputs')
        return self.activation(np.minimum(gate, np.float32(limit)), kind='silu') * np.clip(up, -limit, limit)

    def vision_attention(self, qkv, positions, offsets, *, heads):
        qkv = self._value(qkv)
        positions, offsets = np.asarray(positions), np.asarray(offsets)
        if (qkv.ndim != 2 or type(heads) is not int or not 1 <= heads <= 256
                or qkv.shape[1] % (3 * heads)):
            raise ValueError('invalid reference vision attention geometry')
        count, dim = qkv.shape[0], qkv.shape[1] // (3 * heads)
        if (not 1 <= count <= 1048576 or not 1 <= dim <= 4096 or dim % 4
                or positions.shape != (count, 2) or positions.dtype.kind not in 'iu'
                or np.any(positions < 0) or np.any(positions > 1048576)
                or offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or offsets.size < 2
                or offsets[0] != 0 or offsets[-1] != count
                or any(int(a) >= int(b) for a, b in zip(offsets[:-1], offsets[1:]))):
            raise ValueError('invalid reference frame boundaries or rotary positions')
        if count * (heads * dim * 24 + 32) > self.max_attention_bytes:
            raise MemoryError('independent vision attention exceeds scratch admission')
        values = qkv.reshape(count, 3, heads, dim)
        frequency = 10000.0 ** (-np.arange(dim // 4, dtype=np.float64) / (dim // 4))
        angle = np.concatenate((positions[:, :1] * frequency,
                                positions[:, 1:] * frequency), axis=1)[:, None, :]
        cosine, sine = np.cos(angle), np.sin(angle)

        def rotate(value):
            first, second = np.split(value.astype(np.float64), 2, axis=-1)
            return np.concatenate((first * cosine - second * sine,
                                   second * cosine + first * sine), axis=-1).astype(np.float32)

        query, key = rotate(values[:, 0]), rotate(values[:, 1])
        output = np.empty((count, heads, dim), np.float32)
        for start, end in zip(offsets[:-1], offsets[1:]):
            start, end = int(start), int(end)
            keys = key[start:end].astype(np.float64)
            frame_values = values[start:end, 2].astype(np.float64)
            for row in range(start, end):
                scores = np.einsum('hd,thd->th', query[row].astype(np.float64), keys)
                scores = (scores / math.sqrt(dim)).astype(np.float32)
                probabilities = np.exp(scores.astype(np.float64) - scores.max(axis=0)).astype(np.float32)
                probabilities = probabilities.astype(np.float64)
                probabilities /= probabilities.sum(axis=0)
                output[row] = np.einsum('th,thd->hd', probabilities, frame_values)
        return self._value(output.reshape(count, heads * dim))

    def close(self):
        pass
