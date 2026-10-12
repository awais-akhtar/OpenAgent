"""Explicit Mixtral mapping onto selected packed SwiGLU expert execution."""
import hashlib
import json
import threading
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
import platform
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .expert_runtime import SparseExpertRuntime


def _canonical_info(value):
    if isinstance(value, dict):
        return {key: _canonical_info(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical_info(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_canonical_info(item) for item in value]
    return value


def _engine_artifact_identity():
    """Conservative content identity of independently deployed engine sources.

    Source deployments must remain immutable while their backends are loaded.
    Any engine source change invalidates portability, even if it is unrelated
    to this codec. This deliberately favors rejecting questionable recovery.
    """
    root = Path(__file__).parent
    digest = hashlib.sha256(b'openagent-expert-source-artifact-v1')
    for path in sorted(root.rglob('*.py')):
        name = path.relative_to(root).as_posix().encode('utf-8')
        digest.update(len(name).to_bytes(4, 'little'))
        digest.update(name)
        content = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                content.update(chunk)
        digest.update(content.digest())
    packages = {}
    for package in ('numpy', 'numba', 'llvmlite'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    payload = dict(source_sha256=digest.hexdigest(), packages=packages,
                   python=platform.python_version(), machine=platform.machine(),
                   processor=platform.processor(), system=platform.system())
    numba = sys.modules.get('numba')
    config = getattr(numba, 'config', None)
    payload['compiler_configuration'] = {name: str(getattr(config, name, None)) for name in
        ('CPU_NAME', 'CPU_FEATURES', 'OPT', 'LOOP_VECTORIZE', 'SLP_VECTORIZE',
         'ENABLE_AVX', 'DISABLE_JIT', 'ENABLE_PROFILING', 'BOUNDSCHECK')}
    return payload


def pack_mixtral_experts(directory, destination, *, model_id, dtype,
                          alignment=4096, progress=None, source_layout='split', lattice_contract=None,
                          iq3_grid=None,
                          max_conversion_bytes=1024*1024*1024):
    """Convert exact split or fused local Mixtral experts into a hashed container.

    Uses the existing bounded matrix-at-a-time conversion and explicit codec
    selection. Fused banks use explicit contiguous matrix slices; the full bank
    is never decoded or copied into memory by the converter.
    Source checkpoint quantization is not guessed from names or byte lengths.
    """
    from .native_checkpoint import _open_local, _json, _stamp
    from .native_mixtral_decoder import configuration, mixtral_metadata_fingerprint
    from .expert_packing import (ExpertLayerBinding, ExpertTensorBinding, TensorRangeBinding,
                                 pack_safetensors_directory_int8)
    if source_layout not in ('split', 'fused'):
        raise ValueError('explicit split or fused Mixtral source layout required')
    root = Path(directory).resolve(strict=True)
    with ExitStack() as owned:
        source, stamp = _open_local(root, root/'config.json', owned)
        if stamp[2] > 1024*1024:
            raise MemoryError('Mixtral conversion configuration exceeds admission')
        raw = _json(source, stamp)
        if _stamp(source) != stamp:
            raise ValueError('Mixtral configuration changed during conversion admission')
    config, _ = configuration(raw, None)
    from .native_checkpoint import read_native_text_checkpoint
    from .mixtral_inventory import admit_mixtral_inventory
    def admit_source(candidate, inventory):
        if candidate!=raw:
            raise ValueError('Mixtral configuration changed during source inventory admission')
        _,_,layout,_=admit_mixtral_inventory(candidate,inventory)
        if layout!=source_layout:
            raise ValueError('selected Mixtral conversion layout differs from original checkpoint')
        return ('model.norm.weight',)
    read_native_text_checkpoint(root,model_type='mixtral',max_weight_bytes=max_conversion_bytes,
        tensor_selection=admit_source)
    layers = []
    for index in range(config['num_hidden_layers']):
        prefix = f'model.layers.{index}.block_sparse_moe.experts.'
        if source_layout == 'split':
            experts = tuple(ExpertTensorBinding(str(expert),
                prefix+f'{expert}.w1.weight', prefix+f'{expert}.w3.weight',
                prefix+f'{expert}.w2.weight') for expert in range(config['num_local_experts']))
        else:
            prefix = f'model.layers.{index}.mlp.experts.'
            e, h, m = config['num_local_experts'], config['hidden_size'], config['intermediate_size']
            experts = tuple(ExpertTensorBinding(str(expert),
                TensorRangeBinding(prefix+'gate_up_proj', (e,2*m,h), expert*2*m*h, (m,h)),
                TensorRangeBinding(prefix+'gate_up_proj', (e,2*m,h), (expert*2+1)*m*h, (m,h)),
                TensorRangeBinding(prefix+'down_proj', (e,h,m), expert*h*m, (h,m)))
                for expert in range(e))
        layers.append(ExpertLayerBinding(str(index), config['hidden_size'],
            config['intermediate_size'], config['num_experts_per_tok'], experts))
    result = pack_safetensors_directory_int8(root, destination, model_id=model_id,
        architecture='mixtral', fingerprint=mixtral_metadata_fingerprint(raw),
        layers=tuple(layers), dtype=dtype, alignment=alignment, progress=progress,
        lattice_contract=lattice_contract, iq3_grid=iq3_grid,
        max_conversion_bytes=max_conversion_bytes)
    with ExitStack() as owned:
        source, current = _open_local(root, root/'config.json', owned)
        if current != stamp or _json(source, current) != raw or _stamp(source) != current:
            raise ValueError('Mixtral source configuration changed during expert conversion')
    return result


class MixtralStreamedExperts:
    """Borrow a sparse runtime with an explicit compiled kernel and exact map.

    The decoder still computes router scores, selected IDs and normalized route
    weights. This adapter computes ONE unweighted expert and never substitutes
    an expert, modifies routing, or takes ownership of the runtime/store.
    """

    architecture = 'mixtral'

    def __init__(self, runtime, *, layer_ids, hidden_size, intermediate_size,
                 expert_count, selected_experts, max_work_bytes):
        if type(runtime) is not SparseExpertRuntime or runtime.manifest.architecture != self.architecture:
            raise ValueError('exact Mixtral sparse runtime required')
        if (type(layer_ids) is not tuple or not layer_ids or
                any(type(value) is not str for value in layer_ids) or
                len(set(layer_ids)) != len(layer_ids) or set(layer_ids) != set(runtime.layers)):
            raise ValueError('complete ordered Mixtral layer mapping required')
        for value in (hidden_size, intermediate_size, expert_count, selected_experts, max_work_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError('positive expert geometry and work budget required')
        if selected_experts > expert_count:
            raise ValueError('selected expert count exceeds inventory')
        if runtime.kernels is None or not runtime.preferred_kernel or runtime.allow_kernel_fallback:
            raise ValueError('explicit compiled expert kernel with fallback disabled required')
        matrix_work = 0
        for layer_id in layer_ids:
            layer = runtime.layers[layer_id]
            if (layer.hidden_size != hidden_size or layer.intermediate_size != intermediate_size or
                    layer.experts_used != selected_experts or
                    set(layer.regions) != {str(i) for i in range(expert_count)}):
                raise ValueError('Mixtral expert inventory or geometry differs from decoder')
            if any(not region.sha256 for region in layer.regions.values()):
                raise ValueError('selected Mixtral expert regions require content hashes')
            if any(runtime.store.regions.get(runtime.key(layer_id, expert)) != region
                   for expert, region in layer.regions.items()):
                raise ValueError('expert store regions differ from the admitted manifest')
            backend = runtime.kernels.resolve(dtype=layer.dtype,
                preferred=runtime.preferred_kernel, allow_fallback=False)
            if not backend.info().capabilities.compiled:
                raise ValueError('selected expert backend must execute compiled arithmetic')
            # The existing Numba direct-byte formats avoid full dequantization;
            # its FP16 path promotes one selected expert. Other providers reserve
            # conservative whole-selected-matrix staging, never a full bank.
            if backend.info().provider != 'numba':
                matrix_work = max(matrix_work, 48*hidden_size*intermediate_size)
            elif layer.dtype == 'float16':
                matrix_work = max(matrix_work, 24*hidden_size*intermediate_size)
        if matrix_work+64*(hidden_size+intermediate_size)+4096 > max_work_bytes:
            raise MemoryError('selected expert working buffers exceed admission')
        self.runtime, self.layer_ids = runtime, layer_ids
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.expert_count, self.selected_experts = expert_count, selected_experts
        self.max_bytes = runtime.store.capacity_bytes+max_work_bytes
        self._lock = threading.Lock()
        self._local_bindings = tuple(runtime.kernels.resolve(dtype=runtime.layers[name].dtype,
            preferred=runtime.preferred_kernel, allow_fallback=False) for name in layer_ids)
        self._portable_artifact = (_engine_artifact_identity() if all(
            backend.info().provider == 'numba' and
            type(backend).__module__.startswith('openagent.adaptive_engine.')
            for backend in self._local_bindings) else None)
        self._admitted_identity = self.identity

    @property
    def identity(self):
        kernels = []
        for name in self.layer_ids:
            backend = self.runtime.kernels.resolve(dtype=self.runtime.layers[name].dtype,
                preferred=self.runtime.preferred_kernel, allow_fallback=False)
            if backend is not self._local_bindings[len(kernels)]:
                raise ValueError('Mixtral expert live backend binding changed')
            kernels.append((name, type(backend).__module__, type(backend).__qualname__,
                            _canonical_info(asdict(backend.info())),
                            None if self._portable_artifact is not None else id(backend)))
        payload = dict(manifest=self.runtime.manifest.to_dict(),
            layers=[self.runtime.layers[name].to_dict() for name in self.layer_ids],
            kernel=self.runtime.preferred_kernel,
            fallback=self.runtime.allow_kernel_fallback,
            kernel_bindings=kernels, capacity=self.runtime.store.capacity_bytes,
            store_regions=[(key, region.path, region.offset, region.size, region.sha256, region.storage)
                           for key, region in sorted(self.runtime.store.regions.items())],
            portable_artifact=self._portable_artifact,
            numeric_policy='borrowed-compiled-swiglu-unweighted-v2')
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

    def execute(self, layer, expert, hidden):
        if (type(layer) is not int or not 0 <= layer < len(self.layer_ids) or
                type(expert) is not int or not 0 <= expert < self.expert_count):
            raise ValueError('valid exact Mixtral layer/expert required')
        value = np.asarray(hidden)
        if value.shape != (self.hidden_size,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('finite FP32 expert hidden vector required')
        with self._lock:
            if self.identity != self._admitted_identity:
                raise ValueError('Mixtral expert manifest or kernel policy changed')
            result = self.runtime.execute(self.layer_ids[layer], value, [(str(expert), 1.0)])
            result = np.asarray(result)
            if result.dtype != np.float32 or result.shape != value.shape or not np.isfinite(result).all():
                raise ArithmeticError('invalid streamed Mixtral expert result')
            return np.array(result, copy=True)
