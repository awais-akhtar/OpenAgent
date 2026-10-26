"""Independent NumPy GLM5 vision reference over the admitted tensor contract.

This intentionally shares only the bounded geometry validator with the
production encoder. Every projection, normalization, attention operation,
spatial downsample, and merger is executed by the separately reviewed NumPy
media-math implementation, never by ``Glm5VisionEncoder.forward`` or the
native media library.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading

import numpy as np

from .glm5_vision import vision_shapes
from .qwen4_vision import _cancel
from .independent_media_math import NumpyMediaMath


class IndependentGlm5VisionEncoder:
    """NumPy-only GLM5 variable-grid vision encoder and publisher merger."""

    implementation = "independent-numpy-glm5-vision-v1"

    def __init__(self, config, tensors, *, max_bytes: int, max_patches: int = 65536,
                 prefix: str = "model.visual."):
        c, shapes = vision_shapes(config)
        if (type(max_bytes) is not int or max_bytes <= 0 or
                type(max_patches) is not int or not 1 <= max_patches <= 1048576):
            raise ValueError("explicit independent GLM vision memory and patch bounds required")
        selected = {name[len(prefix):]: value for name, value in tensors.items()
                    if name.startswith(prefix)}
        if set(selected) != set(shapes):
            raise ValueError("complete original GLM vision tensor inventory required")
        amount = sum(math.prod(shape) * 4 for shape in shapes.values())
        if 2 * amount > max_bytes:
            raise MemoryError("independent GLM vision weight construction exceeds admission")
        self.config = dict(c)
        self.weights = {}
        self.max_bytes, self.max_patches, self.weight_bytes = max_bytes, max_patches, amount
        self._lock = threading.RLock()
        self._busy = False
        self._closed = False
        self.ops = NumpyMediaMath()
        digest = hashlib.sha256(json.dumps(c, sort_keys=True, allow_nan=False).encode())
        try:
            for name, shape in shapes.items():
                value = np.asarray(selected[name])
                if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError("invalid independent GLM vision tensor: " + name)
                owned = np.frombuffer(value.tobytes(), np.float32).reshape(shape)
                self.weights[name] = owned
                digest.update(name.encode())
                digest.update(memoryview(owned).cast("B"))
            digest.update(self.implementation.encode())
            self.identity = "openagent-independent-glm5-vision-v1:" + digest.hexdigest()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                from .failure_reporting import retain_cleanup_failure
                retain_cleanup_failure(primary, self.close, cleanup,
                    "independent GLM vision construction cleanup retained: ")
            raise

    def _projection(self, x, name):
        weight = self.weights[name + ".weight"]
        return self.ops.linear(x, weight.reshape(weight.shape[0], -1),
                               self.weights.get(name + ".bias"))

    def _norm(self, x, name):
        return self.ops.norm(x, self.weights[name + ".weight"], rms=True,
                             epsilon=self.config["rms_norm_eps"])

    def _mlp(self, x, name):
        gate = self._projection(x, name + "gate_proj")
        up = self._projection(x, name + "up_proj")
        hidden = self.ops.clamped_swiglu(gate, up, limit=self.config["swiglu_limit"])
        return self._projection(hidden, name + "down_proj")

    def forward(self, pixel_values, grid_thw, *, cancelled=None):
        with self._lock:
            if self._closed or self._busy:
                raise RuntimeError("independent GLM vision owner is closed or executing")
            c = self.config
            pixels, grids = np.asarray(pixel_values), np.asarray(grid_thw)
            merge = c["spatial_merge_size"]
            if (grids.ndim != 2 or grids.shape[1] != 3 or not grids.shape[0] or
                    grids.dtype.kind not in "iu" or np.any(grids <= 0) or
                    np.any(grids > self.max_patches) or np.any(grids[:, 1:] % merge)):
                raise ValueError("positive merge-aligned original GLM vision grids required")
            count = sum(math.prod(int(value) for value in row) for row in grids)
            patch_width = c["in_channels"] * c["temporal_patch_size"] * c["patch_size"] ** 2
            if count > self.max_patches or pixels.shape != (count, patch_width) or pixels.dtype != np.float32:
                raise ValueError("GLM processor patches differ from the declared grid extent")
            workspace = 4 * count * (2 * patch_width + 24 * c["hidden_size"] +
                8 * c["intermediate_size"] + 4 * c["out_hidden_size"] +
                8 * c["projection_intermediate_size"]) + 128 * count
            if self.weight_bytes + workspace > self.max_bytes:
                raise MemoryError("whole independent GLM vision transaction exceeds admission")
            if not np.isfinite(pixels).all():
                raise ValueError("nonfinite original GLM processor pixels")
            self._busy = True
            try:
                _cancel(cancelled)
                positions = np.empty((count, 2), np.int64)
                offsets = [0]
                cursor = 0
                for temporal, height, width in grids:
                    for _ in range(int(temporal)):
                        for by in range(0, int(height), merge):
                            for bx in range(0, int(width), merge):
                                for dy in range(merge):
                                    for dx in range(merge):
                                        positions[cursor] = (by + dy, bx + dx)
                                        cursor += 1
                        offsets.append(cursor)
                x = self._projection(np.ascontiguousarray(pixels), "patch_embed.proj")
                heads = c["num_heads"]
                dim = c["hidden_size"] // heads
                for index in range(c["depth"]):
                    _cancel(cancelled)
                    p = f"blocks.{index}."
                    qkv = self._projection(self._norm(x, p + "norm1"), p + "attn.qkv")
                    qkv = qkv.reshape(count, 3, heads, dim)
                    qkv[:, 0] = self._norm(np.ascontiguousarray(qkv[:, 0]), p + "attn.q_norm")
                    qkv[:, 1] = self._norm(np.ascontiguousarray(qkv[:, 1]), p + "attn.k_norm")
                    qkv = qkv.reshape(count, 3 * c["hidden_size"])
                    attention = np.empty_like(x)
                    for start, end in zip(offsets[:-1], offsets[1:]):
                        _cancel(cancelled)
                        frame = np.ascontiguousarray(qkv[start:end])
                        frame_positions = np.ascontiguousarray(positions[start:end])
                        frame_offsets = np.array([0, end - start], np.uint64)
                        attention[start:end] = self.ops.vision_attention(
                            frame, frame_positions, frame_offsets, heads=heads)
                    x = x + self._projection(attention, p + "attn.proj")
                    x = x + self._mlp(self._norm(x, p + "norm2"), p + "mlp.")
                grouped = self._norm(x, "post_layernorm").reshape(
                    -1, merge, merge, c["hidden_size"])
                channels = np.ascontiguousarray(grouped.transpose(0, 3, 1, 2)).reshape(
                    -1, c["hidden_size"] * merge * merge)
                downsampled = self._projection(channels, "downsample")
                projected = self._projection(downsampled, "merger.proj")
                projected = self.ops.norm(projected,
                    self.weights["merger.post_projection_norm.weight"],
                    self.weights["merger.post_projection_norm.bias"], epsilon=1e-5)
                projected = self._mlp(self.ops.activation(projected, kind="gelu"), "merger.")
                _cancel(cancelled)
                sizes = [math.prod(int(value) for value in row) // merge ** 2 for row in grids]
                return tuple(np.split(projected, np.cumsum(sizes)[:-1]))
            finally:
                self._busy = False

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError("independent GLM vision inference owns resources")
            if self._closed:
                return
            close = getattr(self.ops, "close", None)
            if callable(close):
                close()
            self.weights.clear()
            self._closed = True


__all__ = ["IndependentGlm5VisionEncoder"]
