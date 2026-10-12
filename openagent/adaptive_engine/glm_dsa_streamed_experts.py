"""Exact DSA fused publisher slices and borrowed packed expert execution."""
import hashlib
import json
from .independent_glm_dsa import IndependentGlmDsaDecoder
from .mixtral_streamed_experts import MixtralStreamedExperts


def config_identity(config):
    return hashlib.sha256(json.dumps(dict(config),sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


class GlmDsaStreamedExperts(MixtralStreamedExperts):
    architecture = 'glm_moe_dsa'

    def __init__(self, runtime, config, *, max_work_bytes):
        c = IndependentGlmDsaDecoder(config, {}, _inventory_only=True).config
        self.config_digest = config_identity(config)
        if runtime.manifest.fingerprint != self.config_digest:
            raise ValueError('DSA expert manifest differs from exact original configuration')
        self.sparse_layers = tuple(i for i, kind in enumerate(c.mlp_layer_types) if kind == 'sparse')
        super().__init__(runtime,layer_ids=tuple(str(i) for i in self.sparse_layers),
            hidden_size=c.hidden_size,intermediate_size=c.moe_intermediate_size,
            expert_count=c.n_routed_experts,selected_experts=c.num_experts_per_tok,
            max_work_bytes=max_work_bytes)
        self._layer_slots = {layer:slot for slot,layer in enumerate(self.sparse_layers)}

    def execute(self, layer, expert, hidden):
        if type(layer) is not int or layer not in self._layer_slots:
            raise ValueError('exact original sparse DSA layer required')
        return super().execute(self._layer_slots[layer], expert, hidden)


def pack_glm_dsa_experts(directory, destination, *, model_id, dtype,
                         max_conversion_bytes, alignment=4096, progress=None,
                         lattice_contract=None, iq3_grid=None):
    from pathlib import Path
    from contextlib import ExitStack
    from .native_checkpoint import _open_local, _json, _stamp
    from .native_glm_dsa_ranges import admit_inventory
    from .expert_packing import (ExpertLayerBinding, ExpertTensorBinding,
        TensorRangeBinding, pack_safetensors_directory_int8)
    from .native_checkpoint import read_native_text_checkpoint
    root = Path(directory).resolve(strict=True)
    with ExitStack() as owned:
        stream, stamp = _open_local(root,root/'config.json',owned)
        if stamp[2] > 1024*1024:
            raise MemoryError('DSA source configuration exceeds admission')
        raw = _json(stream,stamp)
        if _stamp(stream) != stamp:
            raise ValueError('DSA configuration changed during admission')
    # Admit the complete original inventory before any selected tensor conversion.
    def select(config, records):
        if config != raw:
            raise ValueError('DSA configuration changed during inventory admission')
        admit_inventory(config,records)
        return ('model.norm.weight',)
    read_native_text_checkpoint(root,model_type='glm_moe_dsa',
        max_weight_bytes=max_conversion_bytes,tensor_selection=select)
    c = IndependentGlmDsaDecoder(raw,{},_inventory_only=True).config
    e,h,m = c.n_routed_experts,c.hidden_size,c.moe_intermediate_size
    layers = []
    for layer, kind in enumerate(c.mlp_layer_types):
        if kind != 'sparse':
            continue
        prefix = f'model.layers.{layer}.mlp.experts.'
        experts = tuple(ExpertTensorBinding(str(i),
            TensorRangeBinding(prefix+'gate_up_proj',(e,2*m,h),i*2*m*h,(m,h)),
            TensorRangeBinding(prefix+'gate_up_proj',(e,2*m,h),(2*i+1)*m*h,(m,h)),
            TensorRangeBinding(prefix+'down_proj',(e,h,m),i*h*m,(h,m))) for i in range(e))
        layers.append(ExpertLayerBinding(str(layer),h,m,c.num_experts_per_tok,experts))
    result = pack_safetensors_directory_int8(root,destination,model_id=model_id,
        architecture='glm_moe_dsa',fingerprint=config_identity(raw),layers=tuple(layers),
        dtype=dtype,alignment=alignment,progress=progress,lattice_contract=lattice_contract,
        iq3_grid=iq3_grid,max_conversion_bytes=max_conversion_bytes)
    with ExitStack() as owned:
        stream,current = _open_local(root,root/'config.json',owned)
        if current != stamp or _json(stream,current) != raw or _stamp(stream) != current:
            raise ValueError('DSA configuration changed during conversion')
    return result
