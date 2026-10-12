"""Original MoonViT3d/patchmergerv2 equations from the pinned K3 contract.

Input is the publisher processor's patch batch, not an arbitrary RGB image.
No alternate vision architecture, remote model implementation, or Torch import.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class K3VisionConfig:
    hidden: int = 1024
    intermediate: int = 4096
    qkv_hidden: int = 1536
    heads: int = 12
    layers: int = 27
    patch: int = 14
    position_height: int = 64
    position_width: int = 64
    position_time: int = 4
    merge: tuple = (2, 2)
    text_hidden: int = 7168
    projector_epsilon: float = 1e-5
    # nn.RMSNorm(eps=None) uses finfo(input.dtype).eps in publisher source.
    tower_epsilon: float = float(np.finfo(np.float32).eps)

    def __post_init__(self):
        for name in ("hidden", "intermediate", "qkv_hidden", "heads", "layers", "patch",
                     "position_height", "position_width", "position_time", "text_hidden"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("positive vision geometry required: " + name)
        if self.qkv_hidden % self.heads or (self.qkv_hidden // self.heads) % 4:
            raise ValueError("vision head dimension must be divisible by four")
        if self.hidden % 2 or self.heads > 128 or self.qkv_hidden // self.heads > 256:
            raise ValueError("invalid vision temporal/native attention geometry")
        if type(self.merge) is not tuple or len(self.merge) != 2 or any(type(x) is not int or x <= 0 for x in self.merge):
            raise ValueError("explicit spatial merge kernel required")
        for eps in (self.projector_epsilon, self.tower_epsilon):
            if isinstance(eps, bool) or not isinstance(eps, (float, int)) or not np.isfinite(eps) or eps <= 0:
                raise ValueError("positive finite vision epsilon required")


def vision_tensor_shapes(c):
    if type(c) is not K3VisionConfig:
        raise ValueError("K3VisionConfig required")
    h, middle = c.hidden, c.hidden * math.prod(c.merge)
    shapes = {"vision_tower.patch_embed.proj.weight": (h, 3, c.patch, c.patch),
              "vision_tower.patch_embed.pos_emb.weight": (c.position_height, c.position_width, h),
              "vision_tower.encoder.final_layernorm.weight": (h,),
              "mm_projector.proj.0.weight": (middle, middle),
              "mm_projector.proj.2.weight": (c.text_hidden, middle),
              "mm_projector.post_norm.weight": (c.text_hidden,)}
    for i in range(c.layers):
        p = f"vision_tower.encoder.blocks.{i}."
        shapes.update({p + "norm0.weight": (h,), p + "norm1.weight": (h,),
                       p + "wqkv.weight": (3 * c.qkv_hidden, h), p + "wo.weight": (h, c.qkv_hidden),
                       p + "mlp.fc0.weight": (c.intermediate, h), p + "mlp.fc1.weight": (h, c.intermediate)})
    return shapes


def _finite(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite vision arithmetic")
    return a


def _bilinear(weight, height, width):
    """Half-pixel, edge-clamped interpolation (align_corners=False)."""
    sh, sw = weight.shape[:2]
    yy = np.maximum((np.arange(height) + 0.5) * sh / height - 0.5, 0)
    xx = np.maximum((np.arange(width) + 0.5) * sw / width - 0.5, 0)
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, sh - 1), np.minimum(x0 + 1, sw - 1)
    fy, fx = yy - y0, xx - x0
    top = weight[y0[:, None], x0[None, :]] * (1 - fx)[None, :, None] + weight[y0[:, None], x1[None, :]] * fx[None, :, None]
    bottom = weight[y1[:, None], x0[None, :]] * (1 - fx)[None, :, None] + weight[y1[:, None], x1[None, :]] * fx[None, :, None]
    return _finite(top * (1 - fy)[:, None, None] + bottom * fy[:, None, None])


class K3Vision:
    """Bias-free MoonViT3d/projector with explicit FP32 or BF16 arithmetic."""
    published_checkpoint_supported = False

    def __init__(self, config, tensors, *, matrix, attention, operators, vision_ops, max_bytes,
                 arithmetic=None, check_cancelled=None, _borrow_tensors=False,
                 native_projections=None,native_projection_work_bytes=64*1024**2):
        shapes = vision_tensor_shapes(config)
        if set(tensors) != set(shapes):
            raise ValueError("exact MoonViT3d/patchmergerv2 inventory required")
        references=set()
        for name, shape in shapes.items():
            if type(tensors[name]) is not np.ndarray:
                from .executor import _ProjectionReference
                bank=native_projections if arithmetic is None else arithmetic._projection_backend
                if (not _borrow_tensors or type(tensors[name]) is not _ProjectionReference
                        or len(shape)!=2 or tensors[name].shape!=shape or bank is None
                        or not bank.handles(name) or bank.shape(name)!=shape):
                    raise ValueError('unadmitted original vision projection reference: '+name)
                references.add(name);continue
            a = np.asarray(tensors[name])
            if a.shape != shape or a.dtype != np.float32 or not a.flags.c_contiguous or not np.isfinite(a).all():
                raise ValueError("invalid vision tensor: " + name)
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("invalid vision budget")
        self.config, self.max_bytes = config, max_bytes
        self.weight_bytes = sum(np.asarray(a).nbytes for name,a in tensors.items() if name not in references)
        if 3 * self.weight_bytes > max_bytes:
            raise ValueError("vision resident budget exceeded")
        if type(_borrow_tensors) is not bool:raise ValueError('explicit vision weight ownership required')
        if _borrow_tensors and any(np.asarray(value).flags.writeable for name,value in tensors.items() if name not in references):
            raise ValueError('borrowed vision weights must be immutable owner buffers')
        self.weights = (dict(tensors) if _borrow_tensors else
            {name: np.frombuffer(np.asarray(tensors[name]).tobytes(), dtype=np.float32).reshape(shape)
             for name, shape in shapes.items()})
        self.matrix, self.attention, self.operators = matrix, attention, operators
        self.vision_ops = vision_ops
        self.arithmetic=arithmetic
        self.native_projections=native_projections
        self.native_projection_work_bytes=native_projection_work_bytes
        if native_projections is not None:
            from ..native_original_projections import NativeOriginalProjections
            if (type(native_projections) is not NativeOriginalProjections or native_projections.reduction!='fp64'
                    or arithmetic is not None or type(native_projection_work_bytes) is not int or native_projection_work_bytes<=0):
                raise ValueError('admitted native media dense owner and positive scratch required')
        self.check_cancelled=(lambda:None) if check_cancelled is None else check_cancelled
        if not callable(self.check_cancelled):raise ValueError('vision cancellation callback required')
        if arithmetic is not None:
            from .bf16_vision import K3BF16VisionOps
            if config.tower_epsilon!=2**-7:
                raise ValueError('BF16 MoonViT tower requires explicit BF16 RMS epsilon 2**-7')
            self.operators=self.vision_ops=K3BF16VisionOps(arithmetic,check_cancelled=self.check_cancelled)

    def _check(self):
        self.check_cancelled()

    def _add(self,left,right):
        self._check()
        return _finite(left+right) if self.arithmetic is None else self.arithmetic.add(_finite(left),_finite(right))

    def _linear(self, x, name):
        self._check()
        w = self.weights[name]
        if self.native_projections is not None and self.native_projections.handles(name):
            def cancelled():
                self._check();return False
            return self.native_projections.linear(name,_finite(x),
                max_work_bytes=self.native_projection_work_bytes,cancelled=cancelled)
        if self.arithmetic is not None:
            return self.arithmetic.named_matrix(name,w,_finite(x),check_cancelled=self.check_cancelled)
        return _finite(np.stack([self.matrix(w, row) for row in x]))

    def _norm(self, x, name, eps):
        self._check()
        return self.operators.normalize(_finite(x), self.weights[name], epsilon=eps)

    def _gelu(self, x, approximate):
        self._check()
        return self.vision_ops.gelu(x, approximate)

    def projected_lengths(self, grid_thws):
        """Validate media geometry before any tower work or decoder allocation."""
        c = self.config
        grids = np.asarray(grid_thws)
        if grids.ndim != 2 or grids.shape[1] != 3 or not len(grids) or grids.dtype.kind not in "iu":
            raise ValueError("grid_thws must be integer [media,3]")
        if np.any(grids <= 0) or np.any(grids[:, 0] > c.position_time) or np.any(grids[:, 1:] > 512):
            raise ValueError("MoonViT3d time/grid limits exceeded")
        if np.any(grids[:, 1] % c.merge[0]) or np.any(grids[:, 2] % c.merge[1]):
            raise ValueError("grid must divide exactly into merger kernels")
        return tuple((int(h) // c.merge[0]) * (int(w) // c.merge[1]) for _, h, w in grids)

    def encode(self, pixel_values, grid_thws, *, max_bytes=None):
        self._check()
        c = self.config
        pixels, grids = np.asarray(pixel_values), np.asarray(grid_thws)
        self.projected_lengths(grids)
        budget = self.max_bytes if max_bytes is None else max_bytes
        if type(budget) is not int or budget <= 0 or budget > self.max_bytes:
            raise ValueError("invalid vision transaction budget")
        lengths = [int(t) * int(h) * int(w) for t, h, w in grids]
        total = sum(lengths)
        if pixels.shape != (total, 3, c.patch, c.patch) or pixels.dtype != np.float32 or not pixels.flags.c_contiguous:
            raise ValueError("processor patches must be finite contiguous FP32 [sum(t*h*w),3,patch,patch]")
        required = 3 * self.weight_bytes + pixels.nbytes + 4 * total * (12 * c.hidden + 8 * c.qkv_hidden +
                   4 * c.intermediate + 4 * c.text_hidden) + 8 * max(lengths)
        if required > budget:
            raise ValueError("vision transaction budget exceeded")
        flat_pixels=pixels.reshape(-1)
        for offset in range(0,flat_pixels.size,4096):
            self._check()
            if not np.isfinite(flat_pixels[offset:offset+4096]).all():
                raise ValueError('nonfinite processor vision patches')
        patch = self.weights["vision_tower.patch_embed.proj.weight"].reshape(c.hidden, -1)
        x = (self.arithmetic.matrix_batch(patch,pixels.reshape(total,-1),check_cancelled=self.check_cancelled) if self.arithmetic is not None
             else _finite(np.stack([self.matrix(patch, row) for row in pixels.reshape(total, -1)])))
        position_weight = self.weights["vision_tower.patch_embed.pos_emb.weight"]
        positions, angles = [], []
        dim = c.qkv_hidden // c.heads
        omega = 10000.0 ** (-np.arange(0, dim, 4, dtype=np.float32) / dim)
        for t, h, w in grids:
            self._check()
            t, h, w = int(t), int(h), int(w)
            spatial = (self.vision_ops.spatial_position(position_weight,h,w) if self.arithmetic is not None
                       else position_weight if (h, w) == position_weight.shape[:2] else _bilinear(position_weight, h, w))
            spatial = np.tile(spatial.reshape(h * w, c.hidden), (t, 1))
            if t > 1:
                temporal_omega = 10000.0 ** (-np.arange(c.hidden // 2, dtype=np.float32) / (c.hidden / 2))
                phase = np.arange(t, dtype=np.float32)[:, None] * temporal_omega[None, :]
                temporal = np.concatenate((np.sin(phase), np.cos(phase)), axis=-1)
                if temporal.shape[1] != c.hidden:
                    raise ValueError("temporal positional dimension must be even")
                spatial = self._add(spatial,_finite(np.repeat(temporal, h * w, axis=0)))
            positions.append(spatial)
            yy, xx = np.indices((h, w), dtype=np.float32)
            phase = np.stack((xx.ravel()[:, None] * omega, yy.ravel()[:, None] * omega), axis=-1).reshape(h * w, dim // 2)
            angles.append(np.tile(phase, (t, 1)))
        x = self._add(x,_finite(np.concatenate(positions)))
        angle = np.concatenate(angles)
        angle = _finite(angle)
        for i in range(c.layers):
            self._check()
            p = f"vision_tower.encoder.blocks.{i}."
            normalized = self._norm(x, p + "norm0.weight", c.tower_epsilon)
            qkv = self._linear(normalized, p + "wqkv.weight").reshape(total, 3, c.heads, dim)
            q, k, v = (qkv[:, j].copy() for j in range(3))
            q = self.vision_ops.rotate(_finite(q), angle)
            k = self.vision_ops.rotate(_finite(k), angle)
            mixed = np.empty((total, c.qkv_hidden), np.float32)
            offset = 0
            for length in lengths:
                self._check()
                # Bidirectional within each media sequence, never across media.
                keys, values = _finite(k[offset:offset + length]), _finite(v[offset:offset + length])
                if self.arithmetic is not None:
                    mixed[offset:offset+length]=self.vision_ops.attention(
                        _finite(q[offset:offset+length]),keys,values).reshape(length,c.qkv_hidden)
                else:
                    for row in range(offset, offset + length):
                        self._check()
                        mixed[row] = self.attention(_finite(q[row]), keys, values).ravel()
                offset += length
            x = self._add(x,self._linear(mixed, p + "wo.weight"))
            normalized = self._norm(x, p + "norm1.weight", c.tower_epsilon)
            hidden = self._gelu(self._linear(normalized, p + "mlp.fc0.weight"), True)
            x = self._add(x,self._linear(hidden, p + "mlp.fc1.weight"))
        x = self._norm(x, "vision_tower.encoder.final_layernorm.weight", c.tower_epsilon)
        output, offset = [], 0
        kh, kw = c.merge
        for (t, h, w), length in zip(grids, lengths):
            self._check()
            t, h, w = int(t), int(h), int(w)
            seq = x[offset:offset + length].reshape(t, h // kh, kh, w // kw, kw, c.hidden)
            seq = seq.transpose(0, 1, 3, 2, 4, 5)
            seq = (seq.mean(axis=0) if self.arithmetic is None else self.vision_ops.mean(_finite(seq),0))
            merged = _finite(seq.reshape((h // kh) * (w // kw), kh * kw * c.hidden))
            projected = self._linear(merged, "mm_projector.proj.0.weight")
            projected = self._linear(self._gelu(projected, False), "mm_projector.proj.2.weight")
            output.append(self._norm(projected, "mm_projector.post_norm.weight", c.projector_epsilon))
            offset += length
        self._check();return tuple(output)
