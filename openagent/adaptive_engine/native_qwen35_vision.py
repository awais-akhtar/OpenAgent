"""Bounded one-image Qwen3.5-MoE vertical subset; all model math is C.

Not support for pretrained Qwen3.5 vision checkpoints. No runtime oracle imports.
"""
import ctypes as C
import _ctypes
import copy
from pathlib import Path
import threading
from types import MappingProxyType
import weakref
from uuid import uuid4

import numpy as np

from .native_qwen35 import NativeQwen35TextDecoder, Tensor, tensor


def _immutable(array):
    return np.frombuffer(array.tobytes(order='C'), dtype=np.float32).reshape(array.shape)


def _own_library(library):
    release = _ctypes.FreeLibrary if hasattr(_ctypes, 'FreeLibrary') else _ctypes.dlclose
    # The finalizer retains only the handle, never the library or function objects.
    return weakref.finalize(library, release, library._handle)


class NativeQwen35Vision:
    """Explicit processed FP32 patches, batch one, grid [1,2,2], no video."""
    implementation = 'c-qwen35-vision-transformer-single-merged-token'
    _geometry = dict(model_type='qwen3_5_moe_vision', depth=0, hidden_size=8,
        num_heads=2, in_channels=3, patch_size=1, temporal_patch_size=1,
        spatial_merge_size=2, out_hidden_size=24, num_position_embeddings=4,
        intermediate_size=16, hidden_act='gelu_pytorch_tanh')
    _shapes = {
        'patch_embed.proj.weight': (8, 3, 1, 1, 1), 'patch_embed.proj.bias': (8,),
        'pos_embed.weight': (4, 8), 'merger.norm.weight': (8,),
        'merger.norm.bias': (8,), 'merger.linear_fc1.weight': (32, 32),
        'merger.linear_fc1.bias': (32,), 'merger.linear_fc2.weight': (24, 32),
        'merger.linear_fc2.bias': (24,)}

    @classmethod
    def validate_config(cls, config, *, max_bytes):
        if type(max_bytes) is not int or not 0 < max_bytes <= np.iinfo(np.intp).max:
            raise ValueError('invalid max_bytes')
        if config.get('model_type') != 'qwen3_5_moe' or config.get('tie_word_embeddings', False):
            raise ValueError('only untied qwen3_5_moe wrapper supported')
        vision = config.get('vision_config', {})
        depth = vision.get('depth')
        if type(depth) is not int or not 0 <= depth <= 4:
            raise ValueError('vision depth must be between zero and four')
        for key, value in cls._geometry.items():
            if key == 'depth':
                continue
            if type(vision.get(key)) is not type(value) or vision[key] != value:
                raise ValueError('unsupported vision geometry: '+key)
        if vision.get('quantization_config') is not None or config.get('quantization_config') is not None:
            raise ValueError('quantization unsupported')
        text = config.get('text_config', {})
        if text.get('hidden_size') != 24 or text.get('tie_word_embeddings', False):
            raise ValueError('untied text width 24 required')
        image_id, video_id = config.get('image_token_id'), config.get('video_token_id')
        if any(type(v) is not int or not 0 <= v < text.get('vocab_size', 0)
               for v in (image_id, video_id)) or image_id == video_id:
            raise ValueError('invalid modality token IDs')
        return vision, text, image_id, video_id

    def __init__(self, config, tensors, *, library, text_library,
                 max_bytes=256*1024*1024):
        vision, text, self.image_id, self.video_id = self.validate_config(config, max_bytes=max_bytes)
        depth = vision['depth']
        size = sum(max(np.asarray(v).nbytes, np.asarray(v).size*4) for v in tensors.values())
        # Base decoder plus transactional candidate plus source/merged buffers.
        if size*6+1024*1024 > max_bytes:
            raise ValueError('wrapper memory budget exceeded')
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._closed=False
        text_tensors = {('model.'+k[len('model.language_model.'):] if k.startswith('model.language_model.') else k): v
                        for k, v in tensors.items() if not k.startswith('model.visual.')}
        self.decoder = NativeQwen35TextDecoder(text, text_tensors, library=text_library,
                                               max_bytes=max_bytes//3)
        original = self.decoder.weights
        frozen = {name: _immutable(a) for name, a in original.items()}
        pointers = {a.ctypes.data: tensor(frozen[name]) for name, a in original.items()}
        for binding in self.decoder._bindings:
            for i in range(len(binding.w)):
                if binding.w[i].data:
                    binding.w[i] = pointers[binding.w[i].data]
        self.decoder.weights = MappingProxyType(frozen)
        self.decoder._weight_owners = tuple(frozen.values())
        self._base_decoder = self.decoder
        self._text_weights = self.decoder.weights
        owners = []
        shapes = dict(self._shapes)
        block_shapes = {
            'norm1.weight': (8,), 'norm1.bias': (8,),
            'norm2.weight': (8,), 'norm2.bias': (8,),
            'attn.qkv.weight': (24, 8), 'attn.qkv.bias': (24,),
            'attn.proj.weight': (8, 8), 'attn.proj.bias': (8,),
            'mlp.linear_fc1.weight': (16, 8), 'mlp.linear_fc1.bias': (16,),
            'mlp.linear_fc2.weight': (8, 16), 'mlp.linear_fc2.bias': (8,)}
        for index in range(depth):
            shapes.update({'blocks.'+str(index)+'.'+k: v for k, v in block_shapes.items()})
        expected = {'model.visual.'+k for k in shapes}
        if {k for k in tensors if k.startswith('model.visual.')} != expected:
            raise ValueError('exact vision tensor names required')
        for name, shape in shapes.items():
            a = np.asarray(tensors['model.visual.'+name])
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('invalid vision tensor '+name)
            owners.append(_immutable(a))
        self._vision_owners = tuple(owners)
        self.library = C.CDLL(str(Path(library).resolve(strict=True)))
        self._vision_release=_own_library(self.library)
        self.library.oa_qwen35_vision_version.restype = C.c_uint64
        if self.library.oa_qwen35_vision_version() != 1:
            raise ValueError('unsupported vision ABI')
        self._encode = self.library.oa_qwen35_vision_encode
        self._encode.argtypes = [Tensor, C.POINTER(Tensor), C.c_uint64, Tensor]
        self._merge = self.library.oa_qwen35_vision_merge
        self._merge.argtypes = [Tensor, C.c_uint64, C.c_uint64, Tensor, Tensor]
        self._positions = self.library.oa_qwen35_vision_positions
        self._positions.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64]
        for f in (self._encode, self._merge, self._positions):
            f.restype = C.c_int
        self._bindings = (Tensor*len(owners))(*(tensor(w) for w in self._vision_owners))

    @property
    def weights(self):
        return self._vision_owners

    def reset(self):
        with self._lock:
            if self._closed:raise RuntimeError('Qwen3.5 vision fixture is closed')
            self.decoder.reset()
            self._base_decoder.reset()
            self.decoder = self._base_decoder

    def forward(self, token_ids, *, pixel_values=None, image_grid_thw=None,
                video_grid_thw=None, pixel_values_videos=None, mm_token_type_ids=None):
        with self._lock:
            if self._closed:raise RuntimeError('Qwen3.5 vision fixture is closed')
            if video_grid_thw is not None or pixel_values_videos is not None:
                raise ValueError('video unsupported')
            ids = np.asarray(token_ids)
            if ids.ndim == 1:
                ids = ids[None, :]
            if ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1] or ids.dtype.kind not in 'iu':
                raise ValueError('batch-one integer tokens required')
            c = self.decoder.config
            if ids.shape[1] > c.max_position_embeddings-self.decoder.position or np.any(ids < 0) or np.any(ids >= c.vocab_size):
                raise ValueError('invalid tokens or context capacity')
            if np.any(ids == self.video_id):
                raise ValueError('video token unsupported')
            has_image = pixel_values is not None
            if has_image:
                if self.decoder.position or np.count_nonzero(ids == self.image_id) != 1:
                    raise ValueError('one image placeholder in initial prefill required')
                grid = np.asarray(image_grid_thw)
                if grid.dtype.kind not in 'iu' or grid.shape != (1, 3) or not np.array_equal(grid, [[1, 2, 2]]):
                    raise ValueError('only image grid [1,2,2] supported')
                pixels = np.asarray(pixel_values)
                if pixels.shape != (4, 3) or pixels.dtype != np.float32 or not np.isfinite(pixels).all():
                    raise ValueError('processed finite FP32 patches [4,3] required')
            elif image_grid_thw is not None or np.any(ids == self.image_id):
                raise ValueError('image placeholder requires pixels and grid')
            if mm_token_type_ids is not None:
                types = np.asarray(mm_token_type_ids)
                if types.shape != ids.shape or types.dtype.kind not in 'iu' or not np.array_equal(types, (ids == self.image_id).astype(np.int64)):
                    raise ValueError('inconsistent modality types')
            positions = np.empty(ids.shape, np.int64)
            self.decoder._check(self._positions(positions.ctypes.data, positions.size,
                self.decoder.position, c.max_position_embeddings))
            candidate = self.decoder
            if has_image:
                image = np.empty(24, np.float32)
                pixels = np.ascontiguousarray(pixels)
                self.decoder._check(self._encode(tensor(pixels), self._bindings, len(self._bindings), tensor(image)))
                merged = np.empty_like(self._text_weights['model.embed_tokens.weight'])
                self.decoder._check(self._merge(tensor(self._text_weights['model.embed_tokens.weight']),
                    c.vocab_size, self.image_id, tensor(image), tensor(merged)))
                weights = dict(self._text_weights, **{'model.embed_tokens.weight': _immutable(merged)})
                # Share immutable layer bindings and DLL ownership, not cache state.
                candidate = copy.copy(self._base_decoder)
                candidate._lock = threading.RLock()
                candidate._release=None
                candidate._borrowed_library_owner=self._base_decoder
                candidate._owner=uuid4().hex
                candidate._snapshots={};candidate._snapshot_bytes=0
                candidate.weights = MappingProxyType(weights)
                candidate._weight_owners = tuple(weights.values())
                import hashlib
                identity=hashlib.sha256(self._base_decoder.identity.encode())
                identity.update(memoryview(weights['model.embed_tokens.weight']).cast('B'))
                candidate.identity='openagent-qwen35-vision-fixture-text-v1:'+identity.hexdigest()
                candidate.reset()
            result = candidate.forward(ids, position_ids=positions)
            self.decoder = candidate
            return result

    def close(self):
        with self._lock:
            if self._closed:return
            self.decoder.close()
            if self._base_decoder is not self.decoder:self._base_decoder.close()
            self._encode=self._merge=self._positions=None
            self._bindings=None;self._vision_owners=();self._text_weights=MappingProxyType({})
            self.library=None
            if self._vision_release.alive:self._vision_release()
            self._closed=True
