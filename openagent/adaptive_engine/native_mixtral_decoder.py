"""Original bounded resident Mixtral C decoder; no model-runtime delegation."""
import ctypes as C
import _ctypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from threading import RLock
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4
import weakref

import numpy as np


class Tensor(C.Structure):
    _fields_ = [('data', C.POINTER(C.c_float)), ('count', C.c_uint64)]


class Config(C.Structure):
    _fields_ = [(key, C.c_uint64) for key in
                ('hidden', 'width', 'layers', 'heads', 'kv_heads', 'dim',
                 'experts', 'topk', 'vocab', 'capacity', 'window')]
    _fields_ += [('epsilon', C.c_float), ('theta', C.c_float)]


class Layer(C.Structure):
    _fields_ = [('weights', Tensor * 9), ('keys', Tensor), ('values', Tensor)]


def tensor(a):
    if a.dtype != np.float32 or not a.flags.c_contiguous:
        raise ValueError('private ABI requires contiguous FP32')
    return Tensor(a.ctypes.data_as(C.POINTER(C.c_float)), a.size)


LAYER_NAMES = ('input_layernorm.weight', 'post_attention_layernorm.weight',
               'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
               'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
               'mlp.gate.weight', 'mlp.experts.gate_up_proj', 'mlp.experts.down_proj')
GLOBAL_NAMES = ('model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight')
ExpertFFN = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_uint64, C.c_uint64,
    C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_uint64, C.c_uint64)


@dataclass(frozen=True)
class MixtralSnapshot:
    identity: str
    owner: str
    snapshot_id: str
    position: int


def configuration(config, max_context):
    raw = dict(config.to_dict() if hasattr(config, 'to_dict') else config)
    if raw.get('model_type') != 'mixtral':
        raise ValueError('expected exact mixtral identity; no aliases')
    metadata = set('model_type architectures transformers_version _name_or_path initializer_range '
                   'id2label label2id problem_type bos_token_id eos_token_id pad_token_id '
                   'router_aux_loss_coef'.split())
    integers = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'num_local_experts',
                'num_experts_per_tok', 'vocab_size', 'max_position_embeddings')
    options = set('head_dim hidden_act rms_norm_eps rope_theta rope_parameters rope_scaling '
                  'sliding_window attention_dropout attention_bias router_jitter_noise '
                  'tie_word_embeddings use_cache output_hidden_states output_attentions '
                  'output_router_logits return_dict dtype torch_dtype is_encoder_decoder '
                  'chunk_size_feed_forward'.split())
    if set(raw) - metadata - set(integers) - options:
        raise ValueError(f'unsupported config fields: {sorted(set(raw)-metadata-set(integers)-options)}')
    if raw.get('architectures') not in (None, ['MixtralForCausalLM'], ('MixtralForCausalLM',)):
        raise ValueError('unsupported architecture declaration')
    bounds = (16384, 65536, 128, 128, 128, 4096, 4096, 1048576, 1048576)
    for key, bound in zip(integers, bounds):
        if type(raw.get(key)) is not int or not 0 < raw[key] <= bound:
            raise ValueError(f'invalid or unbounded {key}')
    for key, default in (('hidden_act', 'silu'), ('attention_dropout', 0),
                         ('attention_bias', False), ('router_jitter_noise', 0),
                         ('is_encoder_decoder', False), ('chunk_size_feed_forward', 0),
                         ('use_cache', True), ('return_dict', True),
                         ('output_hidden_states', False), ('output_attentions', False),
                         ('output_router_logits', False)):
        value = raw.get(key, default)
        if value != default or (isinstance(default, bool) and type(value) is not bool):
            raise ValueError(f'unsupported {key}')
    if any(raw.get(key) not in (None, 'float32', 'float16', 'bfloat16') for key in ('dtype', 'torch_dtype')):
        raise ValueError('explicit supported floating checkpoint dtype required')
    if raw.get('dtype') is not None and raw.get('torch_dtype') is not None and raw['dtype']!=raw['torch_dtype']:
        raise ValueError('conflicting Mixtral source dtype declarations')
    source_dtype=raw.get('dtype') or raw.get('torch_dtype') or 'float32'
    rope = raw.get('rope_parameters')
    scaling = raw.get('rope_scaling')
    theta = raw.get('rope_theta', 1e6)
    factor = 1.0
    rope_type = 'default'
    if rope is not None and scaling is not None:
        raise ValueError('one exact RoPE parameter declaration required')
    if scaling is not None:
        if type(scaling) is not dict or set(scaling)-{'type','rope_type','factor'}:
            raise ValueError('unadmitted Mixtral RoPE scaling fields')
        rope_type = scaling.get('rope_type',scaling.get('type'))
        if 'type' in scaling and 'rope_type' in scaling and scaling['type']!=scaling['rope_type']:
            raise ValueError('conflicting Mixtral RoPE scaling types')
        factor = scaling.get('factor')
    if rope is not None:
        if type(rope) is not dict or set(rope)-{'rope_type','rope_theta','factor'} or 'rope_type' not in rope:
            raise ValueError('unadmitted Mixtral RoPE parameter fields')
        if 'rope_theta' in raw and 'rope_theta' in rope and raw['rope_theta'] != rope['rope_theta']:
            raise ValueError('conflicting RoPE theta')
        theta = rope.get('rope_theta',theta)
        rope_type = rope['rope_type']
        factor = rope.get('factor',1.0)
    if rope_type not in ('default','linear') or (rope_type=='default' and factor!=1):
        raise ValueError('Mixtral requires an admitted default or linear RoPE policy')
    if type(factor) not in (int,float) or not math.isfinite(factor) or not 1<=factor<=1048576:
        raise ValueError('finite bounded Mixtral linear RoPE factor required')
    epsilon = raw.get('rms_norm_eps', 1e-5)
    for key, value, low, high in (('rope_theta', theta, 1, 1e12), ('rms_norm_eps', epsilon, 1e-12, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'invalid bounded {key}')
    h, heads, kv = (raw[key] for key in ('hidden_size', 'num_attention_heads', 'num_key_value_heads'))
    dim = raw.get('head_dim')
    if dim is None:
        if h % heads:
            raise ValueError('hidden_size must divide inferred head count')
        dim = h // heads
    if type(dim) is not int or not 0 < dim <= 256 or dim % 2 or heads % kv or heads * dim > 65536:
        raise ValueError('unsupported grouped-head geometry')
    if raw['num_experts_per_tok'] > raw['num_local_experts']:
        raise ValueError('invalid top-k')
    window = raw.get('sliding_window')
    if window is not None and (type(window) is not int or not 1 <= window <= 1048576):
        raise ValueError('invalid sliding_window')
    capacity = raw['max_position_embeddings'] if max_context is None else max_context
    if type(capacity) is not int or not 0 < capacity <= raw['max_position_embeddings']:
        raise ValueError('max_context outside bounded checkpoint context')
    tied = raw.get('tie_word_embeddings', False)
    if type(tied) is not bool:
        raise ValueError('tie_word_embeddings must be boolean')
    settings = {key: raw[key] for key in integers}
    settings.update(head_dim=dim, sliding_window=window, rope_theta=float(theta),
                    rope_type=rope_type, rope_factor=float(factor),
                    source_dtype=source_dtype,
                    rms_norm_eps=float(epsilon), tie_word_embeddings=tied, model_type='mixtral')
    return MappingProxyType(settings), capacity


def mixtral_metadata_fingerprint(config):
    """Hash the admitted architecture contract, separately from weight identity."""
    settings, _ = configuration(config, None)
    return hashlib.sha256(json.dumps(dict(settings), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


class NativeMixtralDecoder:
    """FP32 token-to-logit C execution, with whole-call staged KV commits.

    Accepts exact root CausalLM names, either complete fused 5.16 banks or
    complete 4.57 per-expert banks. Binary cached padding masks and explicit
    rotary positions and linear scaled RoPE use additive input ABIs. Optional selected packed
    expert execution uses an explicit compiled provider. Imports never compile.
    """
    implementation = 'compiled-c-mixtral-resident-fp32'
    native_range_execution = True
    native_streamed_range_execution = False

    @classmethod
    def from_streamed_directory(cls, directory, *, library, expert_provider,
                                 max_weight_bytes, max_bytes, max_context=None,
                                 max_snapshots=4, read_limits=None, cancel=None):
        """Read dense tensors only after admitting the complete source inventory.

        The source remains an unquantized exact Mixtral checkpoint; experts are
        separately converted into the caller's hashed sparse manifest. The
        provider must describe the same model geometry. No expert banks are
        materialized by this load path.
        """
        from .native_checkpoint import read_native_text_checkpoint, _finish_decoder
        if expert_provider is not None:
            raise ValueError('Mixtral packed experts are disabled until source-content identity is bound')
        def selection(config, inventory):
            # Keep the streamed resident route behind the same family planner
            # used by conversion and other Mixtral inventory consumers. The
            # local selection below determines which already-admitted tensors
            # are materialized; it must not become a second, weaker admission
            # contract as optional metadata rules evolve.
            from .mixtral_inventory import admit_mixtral_inventory
            _, _, admitted_layout, _ = admit_mixtral_inventory(
                config, inventory, context_tokens=max_context)
            settings, _ = configuration(config, max_context)
            h, m, e, d = (settings[key] for key in
                         ('hidden_size', 'intermediate_size', 'num_local_experts', 'head_dim'))
            q, k = settings['num_attention_heads']*d, settings['num_key_value_heads']*d
            global_shapes = dict(zip(GLOBAL_NAMES,
                ((settings['vocab_size'], h), (h,), (settings['vocab_size'], h))))
            if settings['tie_word_embeddings'] and 'lm_head.weight' not in inventory:
                global_shapes.pop('lm_head.weight')
            fused, split = dict(global_shapes), dict(global_shapes)
            keep_fused, keep_split = set(global_shapes), set(global_shapes)
            shapes = ((h,), (h,), (q,h), (k,h), (k,h), (h,q), (e,h), (e,2*m,h), (e,h,m))
            for index in range(settings['num_hidden_layers']):
                prefix = f'model.layers.{index}.'
                for ordinal, (name, shape) in enumerate(zip(LAYER_NAMES, shapes)):
                    fused[prefix+name] = shape
                    if ordinal < 7:
                        keep_fused.add(prefix+name)
                    if ordinal < 6:
                        split[prefix+name] = shape
                        keep_split.add(prefix+name)
                split[prefix+'block_sparse_moe.gate.weight'] = (e,h)
                keep_split.add(prefix+'block_sparse_moe.gate.weight')
                for expert in range(e):
                    for name, shape in (('w1',(m,h)), ('w3',(m,h)), ('w2',(h,m))):
                        split[prefix+f'block_sparse_moe.experts.{expert}.{name}.weight'] = shape
            if set(inventory) == set(fused):
                expected, selected = fused, keep_fused
                layout = 'fused'
            elif set(inventory) == set(split):
                expected, selected = split, keep_split
                layout = 'split'
            else:
                raise ValueError('streamed loading requires one complete exact Mixtral inventory')
            if layout != admitted_layout:
                raise ValueError('Mixtral streamed selection disagrees with family inventory planner')
            for name, shape in expected.items():
                if tuple(inventory[name]['shape']) != shape or inventory[name].get('dtype') not in ('F32','F16','BF16'):
                    raise ValueError('Mixtral checkpoint tensor shape or dtype mismatch: '+name)
            return tuple(sorted(selected))
        config, weights = read_native_text_checkpoint(directory, model_type='mixtral',
            max_weight_bytes=max_weight_bytes, read_limits=read_limits, cancel=cancel,
            tensor_selection=selection,source_backend='native')
        model = cls(config, weights, library=library, expert_provider=expert_provider,
                    max_bytes=max_bytes, max_context=max_context, max_snapshots=max_snapshots)
        return _finish_decoder(model, cancel)

    def __init__(self, config, tensors, *, library, max_bytes=512*1024*1024, max_context=None,
                 max_snapshots=4, expert_provider=None):
        if expert_provider is not None:
            raise ValueError('Mixtral packed experts require verified checkpoint source-content identity')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('max_bytes must be a positive integer')
        self.config, self.max_context = configuration(config, max_context)
        self.max_bytes, self._lock = max_bytes, RLock()
        if type(max_snapshots) is not int or not 1 <= max_snapshots <= 64:
            raise ValueError('snapshot inventory must be between one and 64')
        self.max_snapshots = max_snapshots
        self._closed = False
        self._busy = False
        self._owner = uuid4().hex
        self._snapshots = {}
        self._snapshot_bytes = 0
        self._release = None
        self._expert_provider = expert_provider
        self._expert_identity = None
        self._expert_bytes = 0
        self._run_streamed = self._expert_callback = None
        self._run_inputs = None
        self._run_scaled = None
        self._null_expert = ExpertFFN()
        self._mask = None
        self._expert_error = None
        c = self.config
        h, m, e, d = (c[key] for key in ('hidden_size', 'intermediate_size', 'num_local_experts', 'head_dim'))
        q, k = c['num_attention_heads']*d, c['num_key_value_heads']*d
        if expert_provider is not None:
            from .mixtral_streamed_experts import MixtralStreamedExperts
            if (type(expert_provider) is not MixtralStreamedExperts or
                    expert_provider.runtime.manifest.fingerprint != mixtral_metadata_fingerprint(config) or
                    len(expert_provider.layer_ids) != c['num_hidden_layers'] or
                    expert_provider.hidden_size != h or expert_provider.intermediate_size != m or
                    expert_provider.expert_count != e or
                    expert_provider.selected_experts != c['num_experts_per_tok']):
                raise ValueError('exact admitted streamed Mixtral expert geometry required')
            self._expert_identity = expert_provider.identity
            self._expert_bytes = expert_provider.max_bytes
            self.native_streamed_range_execution = True
            self.implementation = 'compiled-c-mixtral-selected-packed-experts'
        shapes = dict(zip(GLOBAL_NAMES, ((c['vocab_size'], h), (h,), (c['vocab_size'], h))))
        sizes = ((h,), (h,), (q, h), (k, h), (k, h), (h, q), (e, h), (e, 2*m, h), (e, h, m))
        for i in range(c['num_hidden_layers']):
            shapes.update({f'model.layers.{i}.{name}': shape
                for index, (name, shape) in enumerate(zip(LAYER_NAMES, sizes))
                if expert_provider is None or index < 7})
        # Admit before conversion/ownership copies; packed and split sizes agree.
        self._weight_bytes = 4*sum(math.prod(shape) for shape in shapes.values())
        self._scratch_bytes = 4*(5*h+2*q+3*m+2*e)+8*(self.max_context+e)
        self._state_bytes = 8*c['num_hidden_layers']*self.max_context*k
        self._admit(2*self._weight_bytes+3*self._state_bytes+self._scratch_bytes)
        supplied = dict(tensors)
        if c['tie_word_embeddings']:
            embedding = supplied.get(GLOBAL_NAMES[0])
            if GLOBAL_NAMES[2] in supplied and not np.array_equal(supplied[GLOBAL_NAMES[2]], embedding):
                raise ValueError('inconsistent tied weights')
            supplied[GLOBAL_NAMES[2]] = embedding
        split_shapes = {name: shape for name, shape in shapes.items() if '.mlp.' not in name}
        for i in range(c['num_hidden_layers']):
            p = f'model.layers.{i}.block_sparse_moe.'
            split_shapes[p+'gate.weight'] = (e, h)
            for expert in range(e if expert_provider is None else 0):
                for name, shape in (('w1', (m, h)), ('w3', (m, h)), ('w2', (h, m))):
                    split_shapes[p+f'experts.{expert}.{name}.weight'] = shape
        layout = shapes if set(supplied) == set(shapes) else split_shapes
        if set(supplied) != set(layout):
            raise ValueError('tensor names mismatch: require one complete exact fused or split root layout')
        for name, shape in layout.items():
            a = np.asarray(supplied[name])
            if a.shape != shape or a.dtype != np.float32 or not a.flags.c_contiguous or not np.isfinite(a).all():
                raise ValueError(f'{name} requires finite contiguous FP32 shape {shape}')
        owners = {}
        for name, shape in shapes.items():
            if name in supplied:
                a = supplied[name]
            else:
                prefix, leaf = name.split('.mlp.')
                p = prefix+'.block_sparse_moe.'
                if leaf == 'gate.weight':
                    a = supplied[p+'gate.weight']
                elif leaf == 'experts.gate_up_proj':
                    a = np.stack([np.concatenate((supplied[p+f'experts.{j}.w1.weight'],
                                                  supplied[p+f'experts.{j}.w3.weight'])) for j in range(e)])
                else:
                    a = np.stack([supplied[p+f'experts.{j}.w2.weight'] for j in range(e)])
            owners[name] = np.frombuffer(a.tobytes(), dtype=np.float32).reshape(shape)
        self.weights = MappingProxyType(owners)
        digest = hashlib.sha256(json.dumps(dict(self.config), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('ascii'))
        digest.update(str(self.max_context).encode('ascii'))
        if self._expert_identity is not None:
            digest.update(self._expert_identity.encode('ascii'))
        for name, array in sorted(self.weights.items()):
            digest.update(name.encode('utf-8'))
            digest.update(memoryview(array).cast('B'))
        self.identity = 'oa-mixtral-fp32-v1:'+digest.hexdigest()
        borrowed = isinstance(library, C.CDLL)
        self.library = library if borrowed else C.CDLL(str(Path(library).resolve(strict=True)))
        if not borrowed:
            release = _ctypes.FreeLibrary if os.name == 'nt' else _ctypes.dlclose
            self._release = weakref.finalize(self, release, self.library._handle)
        try:
            self._bind_native(h, m, d)
            self.reset()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note('Mixtral library cleanup failed: '+str(cleanup_error))
            raise

    def _bind_native(self, h, m, d):
        c = self.config
        version = self.library.oa_mixtral_decoder_version
        version.argtypes, version.restype = [], C.c_uint64
        if version() != 1:
            raise ValueError('unsupported Mixtral private ABI')
        self._run = self.library.oa_mixtral_decoder_run
        self._run.argtypes = [C.POINTER(Config), C.POINTER(Layer), C.POINTER(Tensor),
                              C.POINTER(C.c_int64), C.c_uint64, C.c_uint64,
                              C.c_uint64, C.c_uint64, Tensor, Tensor, Tensor, Tensor]
        self._run.restype = C.c_int
        input_version = getattr(self.library, 'oa_mixtral_input_version', None)
        if input_version is not None:
            input_version.argtypes, input_version.restype = [], C.c_uint64
            if input_version() != 1:
                raise ValueError('unsupported Mixtral padding/position ABI')
            self._run_inputs = self.library.oa_mixtral_decoder_run_inputs
            self._run_inputs.argtypes = self._run.argtypes+[
                C.POINTER(C.c_uint8), C.c_uint64, C.POINTER(C.c_int64), C.c_uint64,
                ExpertFFN, C.c_void_p]
            self._run_inputs.restype = C.c_int
        elif h > 4096 or self.max_context > 65536 or c['sliding_window'] == 1:
            raise ValueError('expanded Mixtral geometry requires the additive input ABI')
        if c['rope_type']=='linear':
            capability=getattr(self.library,'oa_mixtral_scaled_rope_version',None)
            if capability is None or self._run_inputs is None:
                raise ValueError('linear Mixtral RoPE requires the additive scaled input ABI')
            capability.argtypes,capability.restype=[],C.c_uint64
            if capability()!=1:raise ValueError('unsupported Mixtral scaled RoPE ABI')
            self._run_scaled=self.library.oa_mixtral_decoder_run_scaled_rope
            self._run_scaled.argtypes=self._run_inputs.argtypes+[C.c_double]
            self._run_scaled.restype=C.c_int
        if self._expert_provider is not None:
            capability = self.library.oa_mixtral_streamed_version
            capability.argtypes, capability.restype = [], C.c_uint64
            if capability() != 1:
                raise ValueError('streamed Mixtral arithmetic ABI required')
            self._run_streamed = self.library.oa_mixtral_decoder_run_streamed
            self._run_streamed.argtypes = self._run.argtypes+[ExpertFFN, C.c_void_p]
            self._run_streamed.restype = C.c_int
            def expert(_user, layer, selected, hidden, output, width, intermediate):
                try:
                    if width != h or intermediate != m:
                        raise ValueError('native streamed expert callback geometry mismatch')
                    value = np.ctypeslib.as_array(hidden, shape=(h,)).copy()
                    result = self._expert_provider.execute(int(layer), int(selected), value)
                    C.memmove(output, result.ctypes.data, result.nbytes)
                    return 0
                except BaseException as failure:
                    self._expert_error = failure
                    return 1
            self._expert_callback = ExpertFFN(expert)
        self._c = Config(h, m, c['num_hidden_layers'], c['num_attention_heads'], c['num_key_value_heads'],
                         d, c['num_local_experts'], c['num_experts_per_tok'], c['vocab_size'], self.max_context,
                         c['sliding_window'] or 0, c['rms_norm_eps'], c['rope_theta'])

    def _admit(self, needed):
        needed += self._snapshot_bytes+self._expert_bytes
        needed += 0 if self._mask is None else self._mask.nbytes
        if needed > self.max_bytes:
            raise MemoryError(f'resident transaction needs {needed} bytes; limit {self.max_bytes}')

    def reset(self):
        with self._lock:
            self._check_open()
            self._cache, self._position, self._mask = None, 0, None

    def _check_open(self):
        if self._closed:
            raise RuntimeError('native Mixtral decoder is closed')
        if self._busy:
            raise RuntimeError('native Mixtral decoder has an active operation')

    def close(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('cannot close an active Mixtral decoder')
            if not self._closed:
                self._closed = True
                self._cache = None
                self._mask = None
                self._position = 0
                self._snapshots.clear()
                self._snapshot_bytes = 0
                self.weights = MappingProxyType({})
            if self._release is not None and self._release.alive:
                _, callback, args, kwargs = self._release.detach()
                try:
                    callback(*args, **kwargs)
                except BaseException:
                    self._release = weakref.finalize(self, callback, *args, **kwargs)
                    raise
            self._run = None
            self._run_streamed = self._expert_callback = None
            self._run_inputs = None
            self._run_scaled = None
            self._expert_provider = None
            self.library = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()

    def snapshot(self):
        with self._lock:
            self._check_open()
            if len(self._snapshots) >= self.max_snapshots:
                raise MemoryError('Mixtral snapshot inventory exceeds admission')
            current = 0 if self._cache is None else sum(a.nbytes for layer in self._cache for a in layer)
            size = 0 if self._cache is None else sum(a.shape[0]*self._position*
                math.prod(a.shape[2:])*a.itemsize for layer in self._cache for a in layer)
            size += 0 if self._mask is None else self._mask.nbytes
            self._admit(self._weight_bytes+current+size+self._scratch_bytes)
            cache = None if self._cache is None else tuple(
                tuple(a[:, :self._position].copy() for a in layer) for layer in self._cache)
            if cache is not None:
                for layer in cache:
                    for array in layer:
                        array.flags.writeable = False
            mask = None if self._mask is None else self._mask.copy()
            if mask is not None:
                mask.flags.writeable = False
            handle = MixtralSnapshot(self.identity, self._owner, uuid4().hex, self._position)
            self._snapshots[handle.snapshot_id] = (handle, cache, size, mask)
            self._snapshot_bytes += size
            return handle

    def _owned_snapshot(self, snapshot):
        if (type(snapshot) is not MixtralSnapshot or snapshot.owner != self._owner or
                snapshot.identity != self.identity):
            raise ValueError('foreign Mixtral snapshot')
        owned = self._snapshots.get(snapshot.snapshot_id)
        if owned is None or owned[0] is not snapshot:
            raise ValueError('unknown or released Mixtral snapshot')
        return owned

    def restore(self, snapshot):
        with self._lock:
            self._check_open()
            _, cache, size, mask = self._owned_snapshot(snapshot)
            current = 0 if self._cache is None else sum(a.nbytes for layer in self._cache for a in layer)
            restored_bytes = 0 if cache is None else sum(a.shape[0]*self.max_context*
                math.prod(a.shape[2:])*a.itemsize for layer in cache for a in layer)
            self._admit(self._weight_bytes+current+restored_bytes+self._scratch_bytes+
                        (0 if mask is None else mask.nbytes))
            def expand(array):
                result = np.zeros((array.shape[0], self.max_context, *array.shape[2:]), dtype=array.dtype)
                result[:, :snapshot.position] = array
                return result
            staged = None if cache is None else [tuple(expand(a) for a in layer) for layer in cache]
            staged_mask = None if mask is None else mask.copy()
            self._cache, self._position, self._mask = staged, snapshot.position, staged_mask

    def release_snapshot(self, snapshot):
        with self._lock:
            self._check_open()
            _, _, size, _ = self._owned_snapshot(snapshot)
            del self._snapshots[snapshot.snapshot_id]
            self._snapshot_bytes -= size

    @property
    def position(self):
        with self._lock:
            self._check_open()
            return self._position

    def cache_snapshot(self):
        with self._lock:
            self._check_open()
            size = 0 if self._cache is None else sum(a.nbytes for layer in self._cache for a in layer)
            self._admit(self._weight_bytes+2*size+self._scratch_bytes)
            return () if self._cache is None else tuple(tuple(a.copy() for a in layer) for layer in self._cache)

    def write_cache(self, output, *, max_bytes):
        from .mixtral_cache_recovery import write_mixtral_cache
        return write_mixtral_cache(self, output, max_bytes=max_bytes)

    def restore_cache(self, source, *, max_bytes):
        from .mixtral_cache_recovery import restore_mixtral_cache
        return restore_mixtral_cache(self, source, max_bytes=max_bytes)

    def status(self):
        with self._lock:
            self._check_open()
            return dict(model_type='mixtral', implementation=self.implementation, position=self._position,
                        max_context=self.max_context, native_range_execution=True,
                        native_streamed_range_execution=self.native_streamed_range_execution,
                        padding_mask_supported=self._run_inputs is not None,
                        explicit_positions_supported=self._run_inputs is not None,
                        rope_type=self.config['rope_type'],rope_factor=self.config['rope_factor'],
                        source_dtype=self.config['source_dtype'],arithmetic_dtype='float32',
                        expert_provider_identity=self._expert_identity,
                        published_checkpoint_supported=False)

    def _execute(self, ids, cache, hidden, logits, routers, routes, mask, positions):
        layers = (Layer*self.config['num_hidden_layers'])()
        for i, (keys, values) in enumerate(cache):
            weights = (Tensor*9)(*[tensor(self.weights[f'model.layers.{i}.{name}'])
                if f'model.layers.{i}.{name}' in self.weights else Tensor()
                for name in LAYER_NAMES])
            layers[i] = Layer(weights, tensor(keys), tensor(values))
        globals_ = (Tensor*3)(*[tensor(self.weights[name]) for name in GLOBAL_NAMES])
        arguments = (C.byref(self._c), layers, globals_, ids.ctypes.data_as(C.POINTER(C.c_int64)),
                         ids.size, ids.shape[0], ids.shape[1], self._position,
                         tensor(hidden), tensor(logits), tensor(routers), tensor(routes))
        if self._expert_provider is not None and self._expert_provider.identity != self._expert_identity:
            raise ValueError('streamed Mixtral provider changed after decoder admission')
        if self._run_inputs is not None:
            extra=(self.config['rope_factor'],) if self._run_scaled is not None else ()
            operation=self._run_scaled if self._run_scaled is not None else self._run_inputs
            return operation(*arguments,
                None if mask is None else mask.ctypes.data_as(C.POINTER(C.c_uint8)),
                0 if mask is None else mask.size,
                None if positions is None else positions.ctypes.data_as(C.POINTER(C.c_int64)),
                0 if positions is None else positions.size,
                self._null_expert if self._expert_callback is None else self._expert_callback, None,*extra)
        if mask is not None or positions is not None:
            raise ValueError('padding/position inputs require the additive Mixtral ABI')
        if self._expert_provider is not None:
            return self._run_streamed(*arguments, self._expert_callback, None)
        return self._run(*arguments)

    def forward(self, token_ids, *, attention_mask=None, position_ids=None):
        with self._lock:
            self._check_open()
            ids = np.asarray(token_ids)
            if ids.ndim == 1:
                ids = ids[None, :]
            if ids.ndim != 2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or ids.shape[0] > 16 or ids.shape[1] > 4096:
                raise ValueError('nonempty integer [batch<=16, tokens<=4096] required')
            c = self.config
            b, t = ids.shape
            if np.any(ids < 0) or np.any(ids >= c['vocab_size']) or self._position+t > self.max_context:
                raise ValueError('token or context capacity outside bounds')
            if self._cache is not None and b != self._cache[0][0].shape[0]:
                raise ValueError('cached batch changed; reset first')
            output_bytes = 4*b*t*(c['hidden_size']+c['vocab_size']+2*c['num_hidden_layers']*c['num_local_experts'])
            self._admit(self._weight_bytes+3*b*self._state_bytes+2*output_bytes+
                        self._scratch_bytes+ids.size*32+5*b*(self._position+t))
            ids = np.array(ids, dtype=np.int64, order='C', copy=True)
            end = self._position+t
            if attention_mask is None:
                mask = np.ones((b,end), np.uint8)
                if self._mask is not None:
                    mask[:, :self._position] = self._mask
            else:
                observed = np.asarray(attention_mask)
                if (observed.shape != (b,end) or observed.dtype.kind not in 'biu' or
                        np.any((observed != 0) & (observed != 1))):
                    raise ValueError('binary mask must cover the full physical cache plus input')
                mask = np.array(observed, dtype=np.uint8, order='C', copy=True)
                if self._mask is not None and not np.array_equal(mask[:, :self._position], self._mask):
                    raise ValueError('committed Mixtral padding mask cannot change')
            positions = None
            if position_ids is not None:
                observed = np.asarray(position_ids)
                if observed.ndim == 1 and b == 1:
                    observed = observed[None,:]
                if (observed.shape != ids.shape or observed.dtype.kind not in 'iu' or
                        np.any(observed < 0) or np.any(observed >= c['max_position_embeddings'])):
                    raise ValueError('explicit rotary positions must match incoming tokens')
                positions = np.array(observed, dtype=np.int64, order='C', copy=True)
            native_mask = mask if not np.all(mask) else None
            shape = (b, self.max_context, c['num_key_value_heads'], c['head_dim'])
            cache = ([tuple(a.copy() for a in layer) for layer in self._cache] if self._cache is not None else
                     [(np.zeros(shape, np.float32), np.zeros(shape, np.float32)) for _ in range(c['num_hidden_layers'])])
            hidden = np.empty((b, t, c['hidden_size']), np.float32)
            logits = np.empty((b, t, c['vocab_size']), np.float32)
            routers = np.empty((b, t, c['num_hidden_layers'], c['num_local_experts']), np.float32)
            routes = np.empty_like(routers)
            self._expert_error = None
            self._busy = True
            try:
                code = self._execute(ids, cache, hidden, logits, routers, routes, native_mask, positions)
            finally:
                self._busy = False
            if code:
                if self._expert_error is not None:
                    raise RuntimeError('streamed Mixtral expert failed; state not committed') from self._expert_error
                raise RuntimeError(f'Mixtral C execution status {code}; state not committed')
            if any(not np.isfinite(a).all() for a in (hidden, logits, routers, routes)) or any(
                    not np.isfinite(a).all() for layer in cache for a in layer):
                raise FloatingPointError('nonfinite native result; state not committed')
            result = SimpleNamespace(logits=logits, last_hidden_state=hidden,
                                     router_logits=routers, routing_weights=routes)
            self._cache, self._position, self._mask = cache, self._position+t, mask
            return result

    def padding_mask_snapshot(self):
        with self._lock:
            self._check_open()
            return None if self._mask is None else self._mask.copy()
