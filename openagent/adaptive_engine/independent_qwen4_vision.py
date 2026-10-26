"""Independent NumPy Qwen4 vision reference for publisher-media comparisons.

This implementation intentionally does not import or call ``Qwen4VisionEncoder``
or its native arithmetic wrapper. It follows the admitted checkpoint schema,
but evaluates each projection, normalization, activation, rotary attention and
merger through the separately implemented NumPy media equations.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from types import MappingProxyType

import numpy as np

from .independent_media_math import NumpyMediaMath


def _shapes(config):
    c = dict(config)
    names = ('hidden_size', 'depth', 'num_heads', 'intermediate_size', 'out_hidden_size',
             'patch_size', 'temporal_patch_size', 'spatial_merge_size', 'in_channels',
             'num_position_embeddings')
    for name in names:
        if type(c.get(name)) is not int or not 1 <= c[name] <= 1_048_576:
            raise ValueError('explicit bounded Qwen4 reference vision geometry required: ' + name)
    h, merge = c['hidden_size'], c['spatial_merge_size']
    side = math.isqrt(c['num_position_embeddings'])
    if (h % c['num_heads'] or (h // c['num_heads']) % 4 or side * side != c['num_position_embeddings']
            or c.get('deepstack_visual_indexes', [])
            or c.get('hidden_act') not in ('gelu', 'gelu_pytorch_tanh')):
        raise ValueError('unsupported Qwen4 reference vision topology')
    merged = h * merge * merge
    result = {
        'patch_embed.proj.weight': (h, c['in_channels'], c['temporal_patch_size'], c['patch_size'], c['patch_size']),
        'patch_embed.proj.bias': (h,), 'pos_embed.weight': (c['num_position_embeddings'], h),
        'merger.norm.weight': (h,), 'merger.norm.bias': (h,),
        'merger.linear_fc1.weight': (merged, merged), 'merger.linear_fc1.bias': (merged,),
        'merger.linear_fc2.weight': (c['out_hidden_size'], merged), 'merger.linear_fc2.bias': (c['out_hidden_size'],),
    }
    for layer in range(c['depth']):
        p = f'blocks.{layer}.'
        for norm in ('norm1', 'norm2'):
            result[p + norm + '.weight'] = (h,)
            result[p + norm + '.bias'] = (h,)
        result.update({
            p + 'attn.qkv.weight': (3*h, h), p + 'attn.qkv.bias': (3*h,),
            p + 'attn.proj.weight': (h, h), p + 'attn.proj.bias': (h,),
            p + 'mlp.linear_fc1.weight': (c['intermediate_size'], h),
            p + 'mlp.linear_fc1.bias': (c['intermediate_size'],),
            p + 'mlp.linear_fc2.weight': (h, c['intermediate_size']),
            p + 'mlp.linear_fc2.bias': (h,),
        })
    return c, result


class IndependentQwen4VisionEncoder:
    """Bounded, independently evaluated Qwen4 patch/vision/merger tower."""

    implementation = 'numpy-independent-qwen4-variable-grid-vision-fp32'

    def __init__(self, config, tensors, *, max_bytes, max_patches=65536,
                 prefix='model.visual.', max_attention_bytes=256*1024**2):
        c, shapes = _shapes(config)
        if (type(max_bytes) is not int or max_bytes <= 0 or type(max_patches) is not int
                or not 1 <= max_patches <= 1_048_576):
            raise ValueError('positive finite Qwen4 reference vision budgets required')
        selected = {name[len(prefix):]: value for name, value in tensors.items()
                    if name.startswith(prefix)}
        if set(selected) != set(shapes):
            raise ValueError('complete exact Qwen4 reference vision tensor inventory required')
        amount = sum(math.prod(shape)*4 for shape in shapes.values())
        if amount * 2 > max_bytes:
            raise MemoryError('independent Qwen4 reference vision weights exceed admission')
        self.config = MappingProxyType(c)
        self.max_bytes, self.max_patches, self.weight_bytes = max_bytes, max_patches, amount
        self._lock = threading.RLock()
        self._busy = self._closed = False
        self.ops = NumpyMediaMath(max_attention_bytes=max_attention_bytes)
        self.weights = {}
        digest = hashlib.sha256(json.dumps(c, sort_keys=True, allow_nan=False).encode())
        try:
            for name, shape in shapes.items():
                value = np.asarray(selected[name])
                if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
                    raise ValueError('invalid independent Qwen4 vision tensor: ' + name)
                owned = np.array(value, dtype=np.float32, order='C', copy=True)
                if not np.isfinite(owned).all():
                    raise ValueError('Qwen4 vision tensor exceeds FP32 reference range')
                owned.flags.writeable = False
                self.weights[name] = owned
                digest.update(name.encode()); digest.update(memoryview(owned).cast('B'))
            self.identity = 'openagent-independent-qwen4-vision-v1:' + digest.hexdigest()
        except BaseException:
            self.weights.clear()
            self.ops.close()
            self._closed = True
            raise

    def _projection(self, x, name):
        weight = self.weights[name + '.weight']
        return self.ops.linear(x, weight.reshape(weight.shape[0], -1), self.weights.get(name + '.bias'))

    def _layout(self, grids):
        c = self.config
        merge = c['spatial_merge_size']
        side = math.isqrt(c['num_position_embeddings'])
        positions, learned, offsets = [], [], [0]
        table = self.weights['pos_embed.weight'].reshape(side, side, -1)
        for temporal, height, width in grids:
            ys = np.linspace(0, side-1, int(height), dtype=np.float32)
            xs = np.linspace(0, side-1, int(width), dtype=np.float32)
            for _ in range(int(temporal)):
                for by in range(0, int(height), merge):
                    for bx in range(0, int(width), merge):
                        for dy in range(merge):
                            for dx in range(merge):
                                y, x = by+dy, bx+dx
                                fy, fx = ys[y], xs[x]
                                y0, x0 = int(fy), int(fx)
                                y1, x1 = min(y0+1, side-1), min(x0+1, side-1)
                                wy, wx = np.float32(fy-y0), np.float32(fx-x0)
                                learned.append((1-wy)*((1-wx)*table[y0,x0]+wx*table[y0,x1])+
                                               wy*((1-wx)*table[y1,x0]+wx*table[y1,x1]))
                                positions.append((y, x))
                offsets.append(len(positions))
        return np.asarray(positions, np.int64), np.asarray(learned, np.float32), np.asarray(offsets, np.uint64)

    def forward(self, pixel_values, grid_thw, *, cancelled=None):
        with self._lock:
            if self._closed or self._busy:
                raise RuntimeError('independent Qwen4 vision owner is closed or executing')
            self._busy = True
            try:
                if cancelled is not None and (not callable(cancelled) or cancelled()):
                    raise InterruptedError('Qwen4 reference vision cancelled before execution')
                c = self.config
                grids, pixels = np.asarray(grid_thw), np.asarray(pixel_values)
                if (grids.ndim != 2 or grids.shape[1] != 3 or not grids.shape[0]
                        or grids.dtype.kind not in 'iu' or np.any(grids <= 0)
                        or np.any(grids > self.max_patches)
                        or np.any(grids[:, 1:] % c['spatial_merge_size'])):
                    raise ValueError('positive merge-aligned Qwen4 reference grids required')
                count = sum(math.prod(int(v) for v in row) for row in grids)
                patch_width = c['in_channels']*c['temporal_patch_size']*c['patch_size']**2
                if (count > self.max_patches or pixels.shape != (count, patch_width)
                        or pixels.dtype != np.float32 or not np.isfinite(pixels).all()):
                    raise ValueError('bounded finite FP32 patches and grid extent must agree')
                scratch = 4*count*(patch_width+16*c['hidden_size']+4*c['intermediate_size']+
                    4*c['hidden_size']*c['spatial_merge_size']**2+2*c['out_hidden_size'])+64*count
                if self.weight_bytes + scratch > self.max_bytes:
                    raise MemoryError('independent Qwen4 vision transaction exceeds admission')
                positions, learned, offsets = self._layout(grids)
                x = self._projection(np.ascontiguousarray(pixels), 'patch_embed.proj') + learned
                for layer in range(c['depth']):
                    if cancelled is not None and cancelled():
                        raise InterruptedError('Qwen4 reference vision cancelled between blocks')
                    p = f'blocks.{layer}.'
                    normalized = self.ops.norm(x, self.weights[p+'norm1.weight'], self.weights[p+'norm1.bias'])
                    qkv = self._projection(normalized, p+'attn.qkv')
                    attention = np.empty_like(x)
                    for start, end in zip(offsets[:-1], offsets[1:]):
                        if cancelled is not None and cancelled():
                            raise InterruptedError('Qwen4 reference vision cancelled between frames')
                        start, end = int(start), int(end)
                        attention[start:end] = self.ops.vision_attention(
                            np.ascontiguousarray(qkv[start:end]),
                            np.ascontiguousarray(positions[start:end]),
                            np.array([0, end-start], np.uint64), heads=c['num_heads'])
                    x = x + self._projection(attention, p+'attn.proj')
                    normalized = self.ops.norm(x, self.weights[p+'norm2.weight'], self.weights[p+'norm2.bias'])
                    value = self._projection(normalized, p+'mlp.linear_fc1')
                    value = self.ops.activation(value, kind=c['hidden_act'])
                    x = x + self._projection(value, p+'mlp.linear_fc2')
                value = self.ops.norm(x, self.weights['merger.norm.weight'], self.weights['merger.norm.bias'])
                value = value.reshape(-1, c['hidden_size']*c['spatial_merge_size']**2)
                value = self.ops.activation(self._projection(value, 'merger.linear_fc1'), kind='gelu')
                projected = self._projection(value, 'merger.linear_fc2')
                if cancelled is not None and cancelled():
                    raise InterruptedError('Qwen4 reference vision cancelled before output commit')
                sizes = [math.prod(int(v) for v in row)//c['spatial_merge_size']**2 for row in grids]
                return tuple(np.split(projected, np.cumsum(sizes)[:-1]))
            finally:
                self._busy = False

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('cannot close executing independent Qwen4 vision owner')
            if not self._closed:
                self.weights.clear()
                self.ops.close()
                self._closed = True


__all__ = ['IndependentQwen4VisionEncoder']
