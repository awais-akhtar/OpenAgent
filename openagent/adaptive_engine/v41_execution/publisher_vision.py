"""Separate CPU vision equations from pinned V4.1 inference/vision.py.

Revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027. Inputs are already
normalized CHW patches, not images. Numerical publisher parity is unverified.
Provider uses loader target names: tensor(name), linear(name, row), cast(row).
Linear excludes bias; cast owns the activation dtype boundary. No builds occur.
"""
import math
import threading

import numpy as np

from .auxiliary_attention import attention_rows
from .owner_scope import owned_auxiliary


class PublisherV41Vision:
    published_checkpoint_supported = False

    def __init__(self, vision_config, hidden_size, provider, *, max_bytes=64*1024**2):
        self.config = dict(vision_config)
        self.provider = provider
        for method in ('tensor', 'linear', 'cast'):
            if not callable(getattr(provider, method, None)):
                raise ValueError('vision provider missing '+method)
        if self.config.get('model_type') != 'deepseek_v41_vision':
            raise ValueError('loader V4.1 vision configuration required')
        for key in ('hidden_size', 'num_attention_heads', 'intermediate_size',
                    'patch_size', 'downsample_ratio', 'num_hidden_layers'):
            if type(self.config.get(key)) is not int or self.config[key] <= 0:
                raise ValueError('positive vision geometry required: '+key)
        self.width = self.config['hidden_size']
        self.heads = self.config['num_attention_heads']
        if self.width % self.heads or self.width // self.heads % 4:
            raise ValueError('two-axis rotary head geometry required')
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError('positive language hidden size required')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive vision workspace budget required')
        self.hidden, self.max_bytes = hidden_size, max_bytes
        self._lock = threading.RLock()
        self._busy = False
        self._closed = False
        self.theta = self.config.get('rope_theta', 10000.0)
        if not math.isfinite(self.theta) or self.theta <= 0:
            raise ValueError('positive rotary theta required')

    def _check_open(self):
        if self._closed:
            raise RuntimeError('publisher vision owner is closed')
        ensure = getattr(self.provider, '_ensure_open', None)
        if callable(ensure):
            ensure()

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('cannot close an active publisher vision owner')
            self._closed = True

    @staticmethod
    def _array(value, shape):
        # Check the borrowed/source extent before allocating a detached FP32
        # copy.  In particular, malformed patch payloads must not turn a small
        # admitted grid into an unbounded conversion allocation.
        source = np.asarray(value)
        if source.size != math.prod(shape) or source.dtype.kind != 'f':
            raise ValueError('invalid vision tensor extent or arithmetic')
        try:
            finite = np.isfinite(source).all()
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError('invalid vision tensor extent or arithmetic')
        result = np.array(source, dtype=np.float32, order='C', copy=True)
        if not np.isfinite(result).all():
            raise ValueError('vision tensor is not representable as FP32')
        return result.reshape(shape)

    def _cast(self, x):
        return self._array(self.provider.cast(x.reshape(-1)), x.shape)

    def _tensor(self, name, shape):
        return self._array(self.provider.tensor('v41.'+name), shape)

    def _linear(self, name, x, width, bias=False):
        y = np.stack([self._array(self.provider.linear('v41.'+name+'.weight', row),
                                (width,)) for row in x])
        if bias:
            y = y + self._tensor(name+'.bias', (width,))
        return self._cast(y)

    def _norm(self, x, name):
        return self._cast(x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+1e-6)
                          * self._tensor(name+'.weight', (self.width,)))

    @owned_auxiliary
    def encode_image(self, patches, n_h, n_w, *, workspace_bytes=None):
        self._check_open()
        budget = self.max_bytes if workspace_bytes is None else workspace_bytes
        if type(budget) is not int or not 0 < budget <= self.max_bytes:
            raise ValueError('positive admitted vision workspace required')
        if any(type(v) is not int or v <= 0 for v in (n_h, n_w)):
            raise ValueError('positive patch grid required')
        n, d = n_h*n_w, self.width//self.heads
        mid, r = self.config['intermediate_size'], self.config['downsample_ratio']
        retained = 4*n*(16*self.width+4*mid+3*self.config['patch_size']**2)
        attention_budget = budget-retained
        if attention_budget <= 0:
            raise ValueError('vision attention workspace exceeds admission budget')
        patch = self.config['patch_size']
        x = self._array(patches, (n, 3*patch*patch))
        x = self._linear('vision.patch_embed.proj', self._cast(x), self.width, True)
        inv = self.theta ** (-np.arange(0, d//2, 2, dtype=np.float32)/(d//2))
        yy, xx = np.indices((n_h, n_w))
        phase = np.concatenate((yy.reshape(-1, 1)*inv, xx.reshape(-1, 1)*inv), axis=1)
        cos, sin = np.cos(phase)[:, None, :], np.sin(phase)[:, None, :]
        def rotate(t):
            a, b = np.split(t, 2, axis=-1)
            return self._cast(np.concatenate((a*cos-b*sin, b*cos+a*sin), axis=-1))
        for layer in range(self.config['num_hidden_layers']):
            p = f'vision.blocks.{layer}.'
            qkv = self._linear(p+'attn.wqkv', self._norm(x, p+'norm1'), 3*self.width, True)
            q, k, v = [t.reshape(n, self.heads, d) for t in np.split(qkv, 3, axis=-1)]
            q, k = rotate(q), rotate(k)
            out = self._cast(attention_rows(q, (k,), (v,),
                max_bytes=attention_budget,
                check_cancelled=getattr(self.provider, 'check_cancelled', None))).reshape(n, self.width)
            x = self._cast(x+self._linear(p+'attn.wo', out, self.width, True))
            gate, up = np.split(self._linear(p+'mlp.w1', self._norm(x, p+'norm2'), 2*mid), 2, axis=-1)
            silu = self._cast(gate*np.exp(-np.logaddexp(0, -gate)))
            x = self._cast(x+self._linear(p+'mlp.w2', self._cast(silu*up), self.width))
        x = self._norm(x, 'vision.norm').reshape(n_h, n_w, self.width)
        hh, ww = (n_h+r-1)//r, (n_w+r-1)//r
        if 4*(4*hh*ww*self.width*r*r+8*hh*ww*self.hidden+n*self.width) > budget:
            raise ValueError('aligner workspace exceeds admission budget')
        x = np.pad(x, ((0, hh*r-n_h), (0, ww*r-n_w), (0, 0)))
        # F.unfold orders channels before the two kernel coordinates.
        x = x.reshape(hh, r, ww, r, self.width).transpose(0, 2, 4, 1, 3).reshape(hh*ww, -1)
        x = self._linear('aligner.w1', x, self.hidden, True)
        erf = np.fromiter((math.erf(float(v)/math.sqrt(2)) for v in x.flat),
                          dtype=np.float32, count=x.size).reshape(x.shape)
        return self._linear('aligner.w2', self._cast(0.5*x*(1+erf)), self.hidden, True)

    @owned_auxiliary
    def merge_image_embeddings(self, embeddings, images, *, start_pos=0):
        self._check_open()
        if start_pos != 0:
            raise ValueError('image spans require single-chunk initial prefill')
        observed = np.asarray(embeddings)
        if (observed.ndim != 3 or observed.dtype.kind != 'f' or
                not all(observed.shape) or observed.shape[-1] != self.hidden or
                type(images) not in (tuple, list) or len(images) != observed.shape[0] or
                len(images) > 4096):
            raise ValueError('batch language embeddings and image inventory mismatch')
        # Each batch member gets a prompt-sized occupancy bitmap. Account for
        # every bitmap, not just one, before allocating language or mask copies.
        retained = observed.size*12+observed.shape[0]*observed.shape[1]*16
        if retained >= self.max_bytes:
            raise MemoryError('language image insertion exceeds workspace admission')
        h = np.array(observed, dtype=np.float32, copy=True)
        for row in h.reshape(-1, self.hidden):
            if not np.isfinite(row).all():
                raise ValueError('nonfinite language embedding during image insertion')
        for batch, sample in enumerate(images):
            if type(sample) not in (tuple, list) or len(sample) > 4096:
                raise ValueError('bounded image span inventory required')
            occupied = np.zeros(h.shape[1], dtype=bool)
            previous_end = 0
            for img in sample or ():
                try:
                    # Span metadata is borrowed and only read below; avoid a
                    # copy until its length is bounded by the admitted prompt.
                    types = np.asarray(img.types)
                    start = img.start
                    patches, n_vit_h, n_vit_w = img.patches, img.n_vit_h, img.n_vit_w
                except AttributeError as error:
                    raise ValueError('complete V4.1 image span record required') from error
                if (any(type(value) is not int or value <= 0
                        for value in (n_vit_h, n_vit_w)) or
                        types.ndim != 1 or types.dtype.kind not in 'iu' or
                        not types.size or int(types[0]) != 0 or int(types[-1]) != 3 or
                        not np.any(types == 1) or
                        np.any((types < 0) | (types > 3)) or type(start) is not int or
                        start < previous_end or start+types.size > h.shape[1] or
                        occupied[start:start+types.size].any()):
                    raise ValueError('invalid or overlapping image span')
                ratio = self.config['downsample_ratio']
                aligned_h, aligned_w = math.ceil(n_vit_h/ratio), math.ceil(n_vit_w/ratio)
                # Validate geometry without materializing a second array whose
                # size is derived from untrusted checkpoint/request metadata.
                expected_count = aligned_h*(aligned_w+1)+2
                if (types.size != expected_count or int(types[0]) != 0 or
                        int(types[-1]) != 3):
                    raise ValueError('image span types do not match aligned patch-grid geometry')
                grid = types[1:-1].reshape(aligned_h, aligned_w+1)
                if (np.any(grid[:, :-1] != 1) or np.any(grid[:, -1] != 2)):
                    raise ValueError('image span types do not match aligned patch-grid geometry')
                rows = self.encode_image(patches, n_vit_h, n_vit_w,
                                         workspace_bytes=self.max_bytes-retained)
                if np.count_nonzero(types == 1) != len(rows):
                    raise ValueError('image slots differ from aligned patch grid')
                span = h[batch, start:start+types.size]
                span[types == 1] = rows
                for kind, name in ((0, 'image_start'), (2, 'image_newline'), (3, 'image_end')):
                    span[types == kind] = self._cast(self._tensor(name, (self.hidden,)))
                occupied[start:start+types.size] = True
                previous_end = start + types.size
                del rows
        return h
