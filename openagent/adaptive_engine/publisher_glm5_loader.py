"""Explicit GLM5-next publisher text loading with selected expert streaming."""
from pathlib import Path
import hashlib
import json
import re
from .publisher_checkpoint.storage import snapshot
from .publisher_family_contracts import component_contract,is_mtp_tensor_name
from .independent_glm5_next import Glm5NextTextContract,tensor_layout
from .publisher_tensor_reader import PublisherTensorReader,TensorSlice
from .publisher_weight_mapping import WeightBinding,materialize,expert_bindings
from .glm5_publisher_experts import Glm5PublisherExperts
from .native_glm5_next import NativeGlm5NextTextDecoder
from .native_checkpoint import _pairs,_finish_decoder
from .session_recovery import _path
from .publisher_media_files import read_regular

def _canonical(name):
    for prefix in ('model.language_model.','language_model.','model.'):
        if name.startswith(prefix): name=name[len(prefix):];break
    for source,target in (('hc_attn_fn','attn_hc.fn'),('hc_attn_base','attn_hc.base'),
        ('hc_attn_scale','attn_hc.scale'),('hc_ffn_fn','ffn_hc.fn'),
        ('hc_ffn_base','ffn_hc.base'),('hc_ffn_scale','ffn_hc.scale')):
        name=name.replace(source,target)
    for leaf in ('f_a_proj','f_b_proj','dt_bias','A_log'):
        name=re.sub(r'self_attn\.'+leaf+r'(?=\.|$)','self_attn.forget_gate.'+leaf,name)
    return name


def admit_publisher_glm5_inventory(inventory,components,*,backend='native'):
    """Plan all original wrapper headers before reading any selected payload."""
    from .glm5_vision import vision_shapes
    from .publisher_checkpoint.storage import thaw
    if components.variant.wrapper!='glm5_next':
        raise ValueError('exact GLM5-next publisher inventory required')
    # This planner is callable independently of the directory loader. Bind
    # the component contract to the configuration captured in the same
    # immutable checkpoint snapshot; otherwise a caller could pair a valid
    # GLM5 inventory with a different, shape-compatible wrapper contract.
    if (type(getattr(components,'original_config',None)) is not dict or
            thaw(inventory.config)!=components.original_config):
        raise ValueError('GLM5 component contract differs from checkpoint inventory configuration')
    if components.auxiliary_config:
        raise ValueError('GLM5 publisher rejects auxiliary/MTP configuration: no admitted draft consumer')
    # This wrapper currently owns image/video vision only.  A declared audio
    # component must not be silently accepted merely because no audio tensors
    # happen to be present in the checkpoint inventory.
    audio_names=tuple(name for name in inventory.tensors if name.startswith((
        'model.audio_tower.','audio_tower.','model.language_model.audio_tower.',
        'model.audio.','audio.')))
    if components.audio_config is not None or audio_names:
        raise ValueError('GLM5 publisher loader rejects audio_config/audio tensors: audio execution is not implemented')
    contract=Glm5NextTextContract.parse(components.text_config)
    layout=dict(tensor_layout(contract)[0])
    layout={name:shape for name,shape in layout.items()
        if not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
    layout['lm_head.weight']=(contract.vocab_size,contract.hidden_size)
    visual={'model.visual.'+name:shape for name,shape in vision_shapes(components.vision_config)[1].items()}
    names={};prefixes=set();owned=set();seen_visual=set()
    scales={name[:-7]+'.weight_scale_inv' if name.endswith('.weight') else name+'_scale_inv'
        for name,record in inventory.tensors.items() if record.dtype=='F8_E4M3'}
    from .source_scope import require_source_inventory_admission
    source_tensors={name:record for name,record in inventory.tensors.items()
        if name not in scales and record.dtype in ('F32','F16','BF16')}
    if source_tensors:
        require_source_inventory_admission(
            family='glm5-next',model_type='glm5_next',backend=backend,
            arithmetic='fp32',persisted_format='float32',inventory=source_tensors)
    fp8_tensors={name:record for name,record in inventory.tensors.items()
        if name not in scales and record.dtype=='F8_E4M3'}
    if fp8_tensors:
        require_source_inventory_admission(
            family='glm5-next',model_type='glm5_next',backend=backend,
            arithmetic='fp32',persisted_format='fp8-e4m3-b128-f32',
            inventory=fp8_tensors)
    for name,record in inventory.tensors.items():
        if name in scales:continue
        if record.dtype not in ('F32','F16','BF16','F8_E4M3'):
            raise ValueError('unsupported original GLM component source dtype: '+name)
        if is_mtp_tensor_name(name):
            # GLM MTP tensors used to be counted as owned auxiliary data but
            # were never bound to a draft executor.  Do not admit that
            # incomplete inventory as a runnable GLM5 checkpoint.
            raise ValueError('GLM5 publisher loader rejects MTP tensors: no admitted draft consumer')
        media_name = name
        if media_name.startswith('model.language_model.visual.'):
            media_name = 'model.visual.' + media_name[len('model.language_model.visual.'):]
        if media_name.startswith('model.visual.'):
            if visual.get(media_name)!=record.shape:
                raise ValueError('original GLM media tensor geometry differs: '+name)
            seen_visual.add(media_name);owned.add(name);continue
        canonical=_canonical(name)
        if canonical in names:raise ValueError('ambiguous GLM publisher tensor namespace')
        names[canonical]=name
        if re.search(r'layers\.\d+\.mlp\.experts\.',name):
            prefixes.add(name.split('layers.',1)[0])
    if seen_visual!=set(visual):
        raise ValueError('complete original GLM vision inventory required before payload reads')
    sparse=any(kind=='sparse' for kind in contract.mlp_layer_types)
    if (sparse and len(prefixes)!=1) or (not sparse and prefixes):
        raise ValueError('original GLM expert namespaces disagree with the configured sparse schedule')
    bindings=[]
    for destination,shape in layout.items():
        if destination in names:
            source=names[destination];record=inventory.tensors[source]
            if record.shape!=shape and not (destination.endswith('self_attn.conv1d.weight')
                    and record.shape==(shape[0],1,shape[1])):
                raise ValueError('GLM publisher tensor geometry differs: '+source)
            bindings.append(WeightBinding(destination,(TensorSlice(source,record.shape),),shape))
        elif destination.endswith('self_attn.conv1d.weight'):
            parent=destination[:-len('conv1d.weight')]
            sources=tuple(names.get(parent+axis+'_conv1d.weight') for axis in ('q','k','v'))
            if any(name is None for name in sources):
                raise ValueError('GLM publisher KDA convolution inventory incomplete')
            parts=tuple(TensorSlice(name,inventory.tensors[name].shape) for name in sources)
            if any(part.shape!=(shape[0]//3,1,shape[1]) for part in parts):
                raise ValueError('GLM publisher KDA convolution geometry differs')
            bindings.append(WeightBinding(destination,parts,shape,'concatenate'))
        elif destination=='lm_head.weight' and contract.tie_word_embeddings:
            bindings.append(WeightBinding(destination,(TensorSlice(names['embed_tokens.weight'],shape),),shape))
        else:raise ValueError('missing original GLM publisher tensor: '+destination)
    owned.update(source.source for binding in bindings for source in binding.sources)
    experts=[]
    for layer,kind in enumerate(contract.mlp_layer_types):
        if kind!='sparse':continue
        mapped=expert_bindings(inventory,prefix=f'{next(iter(prefixes))}layers.{layer}.mlp.experts.',
            expert_count=contract.n_routed_experts,hidden_size=contract.hidden_size,
            intermediate_size=contract.moe_intermediate_size)
        owned.update(source.source for binding in mapped for source in binding.sources)
        experts.extend(WeightBinding(_canonical(binding.destination),binding.sources,binding.shape,binding.operation)
            for binding in mapped)
    if owned|scales!=set(inventory.tensors):
        raise ValueError('unowned or incomplete original GLM publisher tensor inventory')
    inventory.assert_unchanged()
    return contract,tuple(bindings),prefixes,tuple(experts)

class OwnedPublisherGlm5Text:
    def __init__(self,decoder,provider,projections=None): self.decoder,self.provider,self.projections=decoder,provider,projections
    def __getattr__(self,name): return getattr(self.decoder,name)
    def close(self):
        # Native callbacks stop borrowing before closing their checkpoint owner.
        failures=[]
        for name in ('decoder','provider','projections'):
            owner=getattr(self,name)
            if owner is None:continue
            try:owner.close()
            except BaseException as error:failures.append((name,error))
            else:setattr(self,name,None)
        if failures:
            primary=failures[0][1]
            for name,error in failures[1:]:
                primary.add_note(f'additional GLM {name} cleanup failure: {error!r}')
            raise primary

def load_publisher_glm5_text_decoder(directory,*,library,expert_library,context_tokens,
        max_weight_bytes,max_bytes,max_expert_bytes,max_conversion_bytes,cancel=None,text_device=None,max_snapshots=4,max_projection_bytes=0):
    root=_path(directory).resolve(strict=True)
    inventory=snapshot(root,max_metadata_bytes=min(max_weight_bytes,256*1024**2),max_tensors=1000000,max_shards=4096)
    raw=read_regular(root/'config.json',max_bytes=1024**2)
    if len(raw)>1024**2 or hashlib.sha256(raw).hexdigest()!=inventory.config_sha256:
        raise ValueError('GLM config changed after checkpoint planning')
    original=json.loads(raw,object_pairs_hook=_pairs)
    if original.get('model_type')!='glm5_next': raise ValueError('exact GLM5-next publisher wrapper required')
    components=component_contract(original,arithmetic='fp32');config=components.text_config
    backend='device' if text_device is not None else 'native'
    contract,bindings,prefixes,expert_mappings=admit_publisher_glm5_inventory(
        inventory,components,backend=backend)
    if type(max_expert_bytes) is not int or max_expert_bytes<0:
        raise ValueError('nonnegative explicit GLM expert streaming admission required')
    if not max_expert_bytes:bindings+=expert_mappings
    quant=original.get('quantization_config')
    if quant is not None and (type(quant) is not dict or quant.get('quant_method')!='fp8'):
        raise ValueError('GLM publisher loading requires BF16/F16 or explicit block-FP8 conversion')
    blocks=None if quant is None else quant.get('weight_block_size')
    sparse=any(kind=='sparse' for kind in contract.mlp_layer_types)
    provider=decoder=projections=None
    reader=PublisherTensorReader(inventory,max_work_bytes=max_conversion_bytes,block_shape=blocks)
    reader.admit_fp8_inventory()
    try:
        if type(max_projection_bytes) is not int or max_projection_bytes<0:
            raise ValueError('nonnegative GLM projection bank admission required')
        if max_projection_bytes:
            if text_device is None: raise ValueError('GLM projection bank requires an explicit accelerator device')
            import importlib
            from .device_publisher_projections import DevicePublisherProjections
            projected={binding.destination:binding.sources[0] for binding in bindings
                if len(binding.shape)==2 and binding.operation=='identity' and len(binding.sources)==1
                and not binding.destination.endswith('index_kpool_compress_ape')
                and not (binding.destination=='lm_head.weight' and contract.tie_word_embeddings)}
            projections=DevicePublisherProjections(inventory,projected,torch=importlib.import_module('torch'),device=text_device,
                max_bytes=max_projection_bytes,max_host_bytes=max_conversion_bytes,block_shape=blocks)
            if contract.tie_word_embeddings:
                duplicate=next((binding.sources[0] for binding in bindings if binding.destination=='lm_head.weight'),None)
                projections.alias('lm_head.weight','embed_tokens.weight',expected_binding=duplicate,cancel=cancel)
            bindings=[binding for binding in bindings if binding.destination not in projections.matrices]
        if sparse and max_expert_bytes:
            if text_device is None:
                provider=Glm5PublisherExperts(inventory,config,library=expert_library,max_bytes=max_expert_bytes,
                    block_shape=blocks,prefix=next(iter(prefixes)))
            else:
                import importlib
                from .device_glm5_publisher_experts import DeviceGlm5PublisherExperts
                provider=DeviceGlm5PublisherExperts(inventory,config,torch=importlib.import_module('torch'),device=text_device,
                    max_bytes=max_expert_bytes,max_host_bytes=max_conversion_bytes,
                    block_shape=blocks,prefix=next(iter(prefixes)))
        weights=materialize(reader,bindings,max_bytes=max_weight_bytes,
            cancelled=None if cancel is None else cancel.is_set)
        inventory.assert_unchanged()
        if text_device is None:
            decoder=NativeGlm5NextTextDecoder(config,weights,library=library,max_tokens=context_tokens,
                max_bytes=max_bytes,expert_provider=provider)
        else:
            from .torch_glm5_decoder import TorchGlm5NextDecoder
            from .torch_glm5_decoder import _SOURCE_ADMISSION as _TEXT_SOURCE_ADMISSION
            decoder=TorchGlm5NextDecoder(config,weights,device=text_device,max_tokens=context_tokens,
                max_bytes=max_bytes,expert_provider=provider,max_snapshots=max_snapshots,projection_provider=projections,
                _source_admission=_TEXT_SOURCE_ADMISSION)
        decoder=_finish_decoder(decoder,cancel)
        if text_device is None:
            from .family_source_binding import bind_decoder_to_snapshot
            bind_decoder_to_snapshot(decoder,inventory,family='glm5-next',
                                     model_type='glm5_next_text')
        return OwnedPublisherGlm5Text(decoder,provider,projections)
    except BaseException as error:
        prior_cleanup=getattr(error,'cleanup',None)
        def cleanup():
            nonlocal decoder,provider,projections,prior_cleanup
            failures=[]
            if prior_cleanup is not None:
                try:prior_cleanup()
                except BaseException as failure:failures.append(('prior',failure))
                else:prior_cleanup=None
            if decoder is not None:
                try:decoder.close()
                except BaseException as failure:failures.append(('decoder',failure))
                else:decoder=None
            if provider is not None:
                try:provider.close()
                except BaseException as failure:failures.append(('expert provider',failure))
                else:provider=None
            if projections is not None:
                try:projections.close()
                except BaseException as failure:failures.append(('projection provider',failure))
                else:projections=None
            if failures:
                primary=failures[0][1]
                for name,failure in failures[1:]:
                    primary.add_note(f'additional GLM {name} cleanup failure: {failure!r}')
                raise primary
        try: cleanup()
        except BaseException as failure:
            from .failure_reporting import retain_cleanup_failure
            retain_cleanup_failure(error,cleanup,failure,
                'GLM checkpoint owner cleanup remains pending: ')
        raise
    finally: reader.close()

__all__=['OwnedPublisherGlm5Text','load_publisher_glm5_text_decoder','admit_publisher_glm5_inventory']
