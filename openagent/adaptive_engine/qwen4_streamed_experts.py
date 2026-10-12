"""Qwen4 packed selected experts with exact family fingerprinting."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

from .mixtral_streamed_experts import MixtralStreamedExperts


def qwen4_metadata_fingerprint(config):
    from .native_qwen4_ranges import qwen4_tensor_inventory
    qwen4_tensor_inventory(config,max_bytes=(1<<63)-1)
    return hashlib.sha256(json.dumps(dict(config),sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


class Qwen4StreamedExperts(MixtralStreamedExperts):
    architecture = 'qwen4_exp_text'


def pack_qwen4_experts(directory,destination,*,model_id,dtype,alignment=4096,
                       progress=None,lattice_contract=None,iq3_grid=None,
                       max_conversion_bytes=1024*1024*1024):
    from .native_checkpoint import _open_local,_json,_stamp
    from .expert_packing import ExpertLayerBinding,ExpertTensorBinding,TensorRangeBinding,pack_safetensors_directory_int8
    root = Path(directory).resolve(strict=True)
    with ExitStack() as owned:
        source,stamp = _open_local(root,root/'config.json',owned)
        if stamp[2]>1024*1024:
            raise MemoryError('Qwen4 conversion config exceeds admission')
        config = _json(source,stamp)
        if _stamp(source)!=stamp:
            raise ValueError('Qwen4 config changed during conversion admission')
    fingerprint = qwen4_metadata_fingerprint(config)
    e,h,m = config['num_experts'],config['hidden_size'],config['moe_intermediate_size']
    layers = []
    for index in range(config['num_hidden_layers']):
        prefix = f'model.layers.{index}.mlp.experts.'
        experts = tuple(ExpertTensorBinding(str(expert),
            TensorRangeBinding(prefix+'gate_up_proj',(e,2*m,h),expert*2*m*h,(m,h)),
            TensorRangeBinding(prefix+'gate_up_proj',(e,2*m,h),(expert*2+1)*m*h,(m,h)),
            TensorRangeBinding(prefix+'down_proj',(e,h,m),expert*h*m,(h,m))) for expert in range(e))
        layers.append(ExpertLayerBinding(str(index),h,m,config['num_experts_per_tok'],experts))
    result = pack_safetensors_directory_int8(root,destination,model_id=model_id,
        architecture='qwen4_exp_text',fingerprint=fingerprint,layers=tuple(layers),dtype=dtype,
        alignment=alignment,progress=progress,lattice_contract=lattice_contract,iq3_grid=iq3_grid,
        max_conversion_bytes=max_conversion_bytes)
    with ExitStack() as owned:
        source,current = _open_local(root,root/'config.json',owned)
        if current!=stamp or _json(source,current)!=config or _stamp(source)!=current:
            raise ValueError('Qwen4 config changed during expert conversion')
    return result


__all__ = ['Qwen4StreamedExperts','pack_qwen4_experts','qwen4_metadata_fingerprint']
