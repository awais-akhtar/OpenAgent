"""Load original split/FP8 publisher weights into explicit native components."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import numpy as np
from .native_checkpoint import _pairs,_finish_decoder,_cancelled
from .session_recovery import _path
from .publisher_checkpoint.storage import snapshot
from .publisher_family_contracts import component_contract
from .publisher_tensor_reader import PublisherTensorReader,TensorSlice
from .publisher_weight_mapping import expert_bindings,qwen4_ple_bindings,WeightBinding,materialize
from .publisher_media_files import read_regular

def load_publisher_conditional_decoder(directory,*,family,library,text_library,
        max_weight_bytes,max_bytes,context_tokens,max_conversion_bytes,
        max_ple_bytes=64*1024**2,max_snapshots=6,cancel=None,arithmetic='fp32',
        mtp_max_bytes=0,mtp_work_bytes=0,max_expert_bytes=0,text_device=None,max_projection_bytes=0,
        segment_runtime_library=None,segment_work_bytes=0,segment_layer_span=0,native_composition=False,
        native_store_runtime_library=None,native_expert_max_bytes=0,native_ple_max_bytes=0):
    for name, value in (
            ('max_weight_bytes', max_weight_bytes), ('max_bytes', max_bytes),
            ('context_tokens', context_tokens), ('max_conversion_bytes', max_conversion_bytes),
            ('max_ple_bytes', max_ple_bytes), ('max_snapshots', max_snapshots),
            ('max_expert_bytes', max_expert_bytes), ('max_projection_bytes', max_projection_bytes),
            ('segment_work_bytes', segment_work_bytes), ('native_expert_max_bytes', native_expert_max_bytes),
            ('native_ple_max_bytes', native_ple_max_bytes)):
        if type(value) is not int or value < 0:
            raise ValueError('explicit nonnegative publisher admission required: '+name)
    if max_weight_bytes == 0 or max_bytes == 0 or max_conversion_bytes == 0:
        raise ValueError('positive publisher memory and conversion admissions required')
    if max_snapshots == 0 or max_snapshots > 64:
        raise ValueError('positive publisher snapshot admission required')
    if context_tokens == 0 or context_tokens > 2**31:
        raise ValueError('positive bounded publisher context admission required')
    if any(value > 2**63-1 for value in (max_weight_bytes, max_bytes, max_conversion_bytes,
                                         max_ple_bytes, max_expert_bytes, max_projection_bytes,
                                         segment_work_bytes, native_expert_max_bytes,
                                         native_ple_max_bytes)):
        raise ValueError('publisher admission exceeds signed size bounds')
    if type(family) is not str or family not in ('qwen3_5_moe', 'qwen4_exp', 'glm5_next', 'inkling_mm_model'):
        raise ValueError('explicit publisher family identity required')
    if arithmetic not in ('fp32', 'bf16', 'fp8'):
        raise ValueError('unsupported publisher arithmetic policy')
    if arithmetic == 'fp8' and text_device is None:
        raise ValueError('FP8 publisher execution requires an explicit device arithmetic owner')
    if family == 'glm5_next' and (native_expert_max_bytes or native_ple_max_bytes or
                                  native_composition):
        raise ValueError('GLM5-next loader does not accept conditional Qwen native-store options')
    root=_path(directory).resolve(strict=True)
    if any(type(value) is not int or value<0 for value in (mtp_max_bytes,mtp_work_bytes)):
        raise ValueError('explicit nonnegative MTP component admissions required')
    if bool(mtp_max_bytes)!=bool(mtp_work_bytes) or (mtp_max_bytes and mtp_work_bytes>=mtp_max_bytes):
        raise ValueError('MTP component and smaller work admissions must be supplied together')
    if family!='inkling_mm_model' and (mtp_max_bytes or mtp_work_bytes):
        raise ValueError('conditional MTP execution budgets require the implemented Inkling draft owner')
    if type(native_composition) is not bool or (native_composition and (family!='qwen4_exp' or text_device is not None)):
        raise ValueError('native composition requires the Qwen4 CPU publisher path')
    if text_device is not None and segment_runtime_library is not None:
        # A device text decoder cannot be composed with the CPU Segment/Edge
        # range path. Reject the mixed backend tuple before checkpoint
        # inventory planning, provider setup, or tensor materialization.
        raise ValueError('publisher device text and Segment runtime backends are mutually exclusive')
    if (native_store_runtime_library is not None or native_expert_max_bytes or native_ple_max_bytes) and not native_composition:
        raise ValueError('native original store options require explicit composition')
    if type(segment_layer_span) is not int or segment_layer_span<0 or (segment_layer_span and
            (segment_runtime_library is None or family not in ('inkling_mm_model','glm5_next','qwen4_exp'))):
        raise ValueError('native layer partitions require a supported publisher Segment runtime')
    inventory=snapshot(root,max_metadata_bytes=min(max_weight_bytes,256*1024**2),
        max_tensors=1000000,max_shards=4096)
    raw=read_regular(root/'config.json',max_bytes=1024**2)
    if len(raw)>1024**2 or hashlib.sha256(raw).hexdigest()!=inventory.config_sha256:
        raise ValueError('publisher config changed after checkpoint planning')
    config=json.loads(raw,object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if config.get('model_type')!=family: raise ValueError('publisher family differs from selected loader')
    # The component planner validates shapes and namespaces, not the complete
    # P02 route. Bind original publisher headers to the concrete execution
    # backend before any tensor payload is materialized. The persisted format
    # differs for FP8: those tensors retain the block-scale contract rather
    # than being silently treated as ordinary float32 conversion.
    from .source_scope import (require_source_inventory_admission,
                               source_family_for_identity)
    source_family=source_family_for_identity(family)
    backend=('public-range' if segment_runtime_library is not None else
             'device' if text_device is not None else 'native')
    ordinary={name:record for name,record in inventory.tensors.items()
        if record.dtype in ('F32','F16','BF16')}
    fp8={name:record for name,record in inventory.tensors.items()
        if record.dtype=='F8_E4M3'}
    if ordinary:
        require_source_inventory_admission(family=source_family,model_type=family,
            backend=backend,arithmetic=arithmetic,persisted_format='float32',inventory=ordinary)
    if fp8:
        require_source_inventory_admission(family=source_family,model_type=family,
            backend=backend,arithmetic=arithmetic,
            persisted_format='fp8-e4m3-b128-f32',inventory=fp8)
    # Build the parsed component contract before consulting optional/MTP
    # ownership below. Keeping this here ensures direct publisher loader calls
    # cannot reach the auxiliary-bank decision with an uninitialized component
    # inventory.
    components=component_contract(config,arithmetic=arithmetic)
    # Only Inkling has an admitted MTP consumer (InklingMTP, constructed below
    # when both stage budgets are explicit).  The Qwen3.5, Qwen4 and GLM5
    # wrappers previously retained MTP tensors as opaque auxiliary payloads,
    # which made their inventory look supported despite there being no draft
    # execution path.  Reject those headers before any tensor payload read.
    mtp_headers=tuple(name for name in inventory.tensors if name.startswith((
        'mtp.','model.mtp.','model.language_model.mtp.')))
    # The wrapper mapper can preserve an MTP/next-token-prediction config even
    # when the checkpoint has no corresponding tensor namespace.  Such a
    # config is still a claim about an optional execution bank; do not silently
    # admit it for families whose conditional loader has no draft consumer.
    if family!='inkling_mm_model' and components.auxiliary_config:
        raise ValueError(f'{family} publisher loader rejects auxiliary/MTP configuration: no admitted consumer')
    if mtp_headers and family!='inkling_mm_model':
        raise ValueError(f'{family} publisher loader rejects MTP tensors: no admitted draft consumer')
    if family=='qwen3_5_moe':
        # This wrapper implements image/video placement only. Reject audio at
        # metadata admission, before constructing a tensor reader or decoding
        # any checkpoint payload, even when the checkpoint advertises a tower.
        audio_names=tuple(name for name in inventory.tensors if name.startswith((
            'model.audio_tower.','audio_tower.','model.language_model.audio_tower.',
            'model.audio.','audio.')))
        if config.get('audio_config') is not None or audio_names:
            raise ValueError('Qwen3.5 publisher loader rejects audio_config/audio tensors: audio execution is not implemented')
    if family=='qwen4_exp':
        # The admitted Qwen4 wrapper currently composes the vision tower only.
        # Reject an advertised or physically present audio component while
        # checkpoint metadata is still being inspected; constructing the
        # tensor reader and failing later in Qwen4MultimodalDecoder would read
        # payloads for a bank with no execution owner.
        audio_names=tuple(name for name in inventory.tensors if name.startswith((
            'model.audio_tower.','audio_tower.','model.language_model.audio_tower.',
            'model.audio.','audio.')))
        if config.get('audio_config') is not None or audio_names:
            raise ValueError('Qwen4 publisher loader rejects audio_config/audio tensors: audio execution is not implemented')
    if family=='glm5_next':
        if mtp_max_bytes or mtp_work_bytes: raise ValueError('Inkling MTP budgets cannot be applied to GLM')
        from .glm5_multimodal import Glm5MultimodalDecoder
        return _finish_decoder(Glm5MultimodalDecoder(root,config,library=library,text_library=text_library,
            max_bytes=max_bytes,max_weight_bytes=max_weight_bytes,max_conversion_bytes=max_conversion_bytes,
            max_expert_bytes=max_expert_bytes,context_tokens=context_tokens,max_snapshots=max_snapshots,text_device=text_device,
            max_projection_bytes=max_projection_bytes,segment_runtime_library=segment_runtime_library,
            segment_work_bytes=segment_work_bytes,segment_layer_span=segment_layer_span),cancel)
    if segment_runtime_library is not None and family not in ('inkling_mm_model','qwen4_exp'):
        raise ValueError('conditional public Segment execution requires an implemented publisher wrapper')
    c=components.text_config
    if family=='inkling_mm_model':
        from .inkling_mtp import admit_mtp_inventory
        mtp_headers={name:(record.shape,record.dtype) for name,record in inventory.tensors.items()
            if name.startswith(('mtp.','model.mtp.','model.language_model.mtp.'))}
        if mtp_headers and not mtp_max_bytes:
            raise ValueError('Inkling MTP tensors require explicit admitted draft execution budgets')
        admit_mtp_inventory(c,components.auxiliary_config,mtp_headers,
            max_total_bytes=mtp_max_bytes if mtp_max_bytes else None)
    if family=='qwen3_5_moe':
        c=dict(c)
        if type(context_tokens) is not int or not 1<=context_tokens<=c['max_position_embeddings']:
            raise ValueError('explicit bounded Qwen3.5 publisher context required')
        if text_device is None and context_tokens>65536:
            raise ValueError('native Qwen3.5 requires an explicit context of at most 65536 tokens')
    quant=config.get('quantization_config')
    if quant is not None and (type(quant) is not dict or quant.get('quant_method')!='fp8'):
        raise ValueError('this original publisher loader admits explicit FP32 conversion of BF16/F16/FP8 weights')
    blocks=None if quant is None else quant.get('weight_block_size')
    reader=PublisherTensorReader(inventory,max_work_bytes=max_conversion_bytes,block_shape=blocks)
    reader.admit_fp8_inventory()
    ple=expert_provider=projection_provider=None
    try:
        excluded=set();generated={};bindings=[];tables={};metadata_checks=[];tied_alias=None
        if type(max_projection_bytes) is not int or max_projection_bytes<0:
            raise ValueError('nonnegative device projection bank admission required')
        if max_projection_bytes and text_device is None:
            raise ValueError('publisher projection bank requires explicit device text execution')
        if family=='qwen4_exp' and c.get('ple_layer_ids'):
            from .qwen4_ple_store import Qwen4PLEStore
            from .qwen4_ple import metadata
            sources=qwen4_ple_bindings(inventory,c)
            if text_device is None:
                ple=Qwen4PLEStore(inventory,c,sources,max_bytes=max_ple_bytes)
            else:
                for layer,names in sources.items():
                    slices=tuple(TensorSlice(name,inventory.tensors[name].shape) for name in names)
                    if any(len(value.shape)!=2 for value in slices) or len({value.shape[1] for value in slices})!=1:
                        raise ValueError('publisher lexical shard dimensions differ')
                    shape=(sum(value.shape[0] for value in slices),slices[0].shape[1])
                    destination=f'model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.weight'
                    if max_projection_bytes:
                        tables['model.'+destination[len('model.language_model.'):]]=slices
                    else:
                        bindings.append(WeightBinding(destination,slices,shape,'concatenate' if len(slices)>1 else 'identity'))
            excluded.update(name for names in sources.values() for name in names)
            for ordinal,one_based in enumerate(c['ple_layer_ids']):
                values=metadata(SimpleNamespace(**c),ordinal)
                prefix=f'model.language_model.layers.{one_based-1}.ple.ple_embedding.'
                for suffix,value in zip(('layer_multipliers','ngram_heads_vocab_sizes','ngram_heads_offsets'),values):
                    name=prefix+suffix;generated[name]=value
                    if name in inventory.tensors:
                        record=inventory.tensors[name]
                        if record.dtype!='I64' or record.shape!=value.shape:
                            raise ValueError('publisher PLE hash metadata differs')
                        metadata_checks.append((name,value))
                        excluded.add(name)
        if family in ('qwen4_exp','qwen3_5_moe'):
            sparse=range(c['num_hidden_layers']);experts=c['num_experts'];middle=c['moe_intermediate_size']
        elif family=='inkling_mm_model':
            sparse=[i for i,kind in enumerate(c['mlp_layer_types']) if kind=='sparse']
            experts=c['n_routed_experts'] if 'n_routed_experts' in c else c['num_experts'];middle=c['moe_intermediate_size']
        else: raise ValueError('unsupported publisher conditional loader identity')
        if max_expert_bytes:
            from .inkling_publisher_experts import InklingPublisherExperts
            from .qwen4_publisher_experts import Qwen4PublisherExperts
            from .qwen35_publisher_experts import Qwen35PublisherExperts,DeviceQwen35PublisherExperts
            prefixes=('model.language_model.','model.')
            present=[prefix for prefix in prefixes if any(name.startswith(prefix+'layers.') and '.mlp.experts.' in name for name in inventory.tensors)]
            if len(present)!=1: raise ValueError('one original Inkling expert namespace required')
            if text_device is not None:
                import importlib
                from .device_qwen4_publisher_experts import DeviceQwen4PublisherExperts
                from .device_inkling_publisher_experts import DeviceInklingPublisherExperts
                provider_type=(DeviceQwen35PublisherExperts if family=='qwen3_5_moe' else
                    DeviceQwen4PublisherExperts if family=='qwen4_exp' else DeviceInklingPublisherExperts)
                expert_provider=provider_type(inventory,c,torch=importlib.import_module('torch'),device=text_device,
                    max_bytes=max_expert_bytes,max_host_bytes=max_conversion_bytes,block_shape=blocks,prefix=present[0])
            else:
                provider_type=(Qwen35PublisherExperts if family=='qwen3_5_moe' else
                    InklingPublisherExperts if family=='inkling_mm_model' else Qwen4PublisherExperts)
                expert_provider=provider_type(inventory,c,library=library,max_bytes=max_expert_bytes,
                    block_shape=blocks,prefix=present[0])
        for layer in sparse:
            prefixes=(f'model.language_model.layers.{layer}.mlp.experts.',f'model.layers.{layer}.mlp.experts.')
            present=[prefix for prefix in prefixes if any(name.startswith(prefix) for name in inventory.tensors)]
            if len(present)!=1: raise ValueError('one complete publisher expert namespace required')
            mapped=expert_bindings(inventory,prefix=present[0],expert_count=experts,
                hidden_size=c['hidden_size'],intermediate_size=middle)
            if expert_provider is None: bindings.extend(mapped)
            excluded.update(source.source for binding in mapped for source in binding.sources)
        if max_projection_bytes:
            import importlib
            from .device_publisher_projections import DevicePublisherProjections
            from .independent_qwen4 import IndependentQwen4TextDecoder
            from .independent_inkling import IndependentInklingDecoder
            from .independent_qwen35 import IndependentQwen35TextDecoder
            allowed=(IndependentQwen35TextDecoder(c,{},_inventory_only=True).tensor_shapes if family=='qwen3_5_moe' else
                IndependentQwen4TextDecoder(c,{},_inventory_only=True).tensor_shapes if family=='qwen4_exp'
                else IndependentInklingDecoder(c,{},_inventory_only=True).tensor_shapes)
            matrix_bindings={}
            for name,record in inventory.tensors.items():
                if name in excluded or len(record.shape)!=2 or not name.endswith('.weight'):
                    continue
                destination=('model.'+name[len('model.language_model.'):] if name.startswith('model.language_model.') else name)
                if destination not in allowed: continue
                if not (name.startswith(('model.language_model.','model.layers.',
                        'model.embed_tokens.','model.norm.')) or name=='lm_head.weight'):
                    continue
                if destination=='lm_head.weight' and c.get('tie_word_embeddings',False):
                    continue
                matrix_bindings[destination]=TensorSlice(name,record.shape);excluded.add(name)
            projection_provider=DevicePublisherProjections(inventory,matrix_bindings,
                torch=importlib.import_module('torch'),device=text_device,max_bytes=max_projection_bytes,
                max_host_bytes=max_conversion_bytes,block_shape=blocks,table_bindings=tables)
            if family in ('qwen3_5_moe','qwen4_exp') and c.get('tie_word_embeddings',False):
                record=inventory.tensors.get('lm_head.weight')
                tied_alias=(None if record is None else TensorSlice('lm_head.weight',tuple(record.shape)))
                if record is not None:excluded.add('lm_head.weight')
        scales=set()
        for name,record in inventory.tensors.items():
            if record.dtype=='F8_E4M3':
                scale_name=name[:-7]+'.weight_scale_inv' if name.endswith('.weight') else name+'_scale_inv'
                if scale_name not in inventory.tensors: raise ValueError('FP8 publisher source is missing its scale')
                scales.add(scale_name)
        for name,record in inventory.tensors.items():
            if name in excluded or name in scales: continue
            if name.endswith(('.weight_scale_inv','_scale_inv')):
                raise ValueError('unowned publisher quantization scale')
            shape=(1,) if family=='inkling_mm_model' and name.endswith('global_scale') and record.shape==() else record.shape
            bindings.append(WeightBinding(name,(TensorSlice(name,record.shape),),shape))
        if family in ('qwen3_5_moe','qwen4_exp','inkling_mm_model'):
            from .independent_qwen35 import IndependentQwen35TextDecoder
            from .independent_qwen4 import IndependentQwen4TextDecoder
            from .independent_inkling import IndependentInklingDecoder
            from .qwen4_vision import vision_shapes
            planner=(IndependentQwen35TextDecoder if family=='qwen3_5_moe' else
                IndependentQwen4TextDecoder if family=='qwen4_exp' else IndependentInklingDecoder)(c,{},_inventory_only=True)
            text_shapes=dict(planner.tensor_shapes)
            if family=='inkling_mm_model':
                from .inkling_media import media_shapes
                visual_shapes=media_shapes(components.vision_config,components.audio_config,text_hidden=c['hidden_size'])[0]
            else:
                visual_shapes={'model.visual.'+name:shape for name,shape in vision_shapes(components.vision_config).items()}
            seen_text=set();seen_visual=set()
            for binding in bindings:
                name=binding.destination
                if name.startswith(('mtp.','model.mtp.','model.language_model.mtp.')):
                    continue  # Explicitly retained auxiliary namespace, never a text weight.
                if name.startswith(('model.visual.','model.vision_tower.',
                                    'model.video_tower.','model.audio_tower.',
                                    'model.language_model.visual.',
                                    'model.language_model.vision_tower.',
                                    'model.language_model.video_tower.',
                                    'model.language_model.audio_tower.')):
                    short=name
                    if short.startswith('model.language_model.'):
                        short='model.'+short[len('model.language_model.'):]
                    if short in seen_visual or visual_shapes.get(short)!=binding.shape:
                        raise ValueError(f'{family} whole media inventory differs: '+name)
                    seen_visual.add(short)
                else:
                    short='model.'+name[len('model.language_model.'):] if name.startswith('model.language_model.') else name
                    if short in seen_text or text_shapes.get(short)!=binding.shape:
                        raise ValueError(f'{family} whole text inventory differs: '+name)
                    seen_text.add(short)
            required_text=set(text_shapes)
            if projection_provider is not None:
                for name,value in projection_provider.matrices.items():
                    if name in seen_text or text_shapes.get(name)!=value.shape:
                        raise ValueError(f'{family} original projection is outside the whole text inventory')
                    seen_text.add(name)
                if family in ('qwen3_5_moe','qwen4_exp') and c.get('tie_word_embeddings',False):
                    if 'model.embed_tokens.weight' not in seen_text:
                        raise ValueError('tied publisher head requires an admitted original embedding')
                    if tied_alias is not None and tied_alias.shape!=text_shapes['lm_head.weight']:
                        raise ValueError('tied publisher head geometry differs')
                    seen_text.add('lm_head.weight')
            if ple is not None:
                for layer,shape in ple.shapes.items():
                    name=f'model.layers.{layer}.ple.ple_embedding.ngram_embedding.weight'
                    if name in seen_text or text_shapes.get(name)!=shape:
                        raise ValueError('original PLE table differs from complete text geometry')
                    seen_text.add(name)
            expected_metadata=getattr(planner,'integer_metadata',{})
            normalized_metadata={'model.'+name[len('model.language_model.'):]:value for name,value in generated.items()}
            if set(normalized_metadata)!=set(expected_metadata) or any(
                    not np.array_equal(normalized_metadata[name],value) for name,value in expected_metadata.items()):
                raise ValueError('complete publisher integer metadata differs')
            if expert_provider is not None:
                required_text={name for name in required_text if not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
            if c.get('tie_word_embeddings',False) and 'lm_head.weight' not in seen_text:required_text.remove('lm_head.weight')
            if seen_text!=required_text or seen_visual!=set(visual_shapes):
                raise ValueError(f'{family} complete original component inventory required before payload conversion')
        inventory.assert_unchanged();_cancelled(cancel)
        # Admission covers every retained source before metadata/tied payload
        # comparisons too, so an unrelated bad dtype cannot hide behind a
        # selected projection or an early successful comparison.
        for binding in bindings:
            for source in binding.sources:
                if inventory.tensors[source.source].dtype not in ('F32','BF16','F16','F8_E4M3'):
                    raise ValueError('unsupported publisher component source dtype: '+source.source)
        if tied_alias is not None and inventory.tensors[tied_alias.source].dtype not in ('F32','BF16','F16','F8_E4M3'):
            raise ValueError('unsupported original tied publisher head dtype')
        for name,value in metadata_checks:
            record=inventory.tensors[name]
            actual=np.frombuffer(inventory.read_range(name,0,record.nbytes),'<i8').reshape(record.shape)
            if not np.array_equal(actual,value): raise ValueError('publisher PLE hash values differ')
            _cancelled(cancel)
        if projection_provider is not None and family in ('qwen3_5_moe','qwen4_exp') and c.get('tie_word_embeddings',False):
            projection_provider.alias('lm_head.weight','model.embed_tokens.weight',expected_binding=tied_alias,cancel=cancel)
        weights=materialize(reader,bindings,max_bytes=max_weight_bytes,
            cancelled=None if cancel is None else cancel.is_set)
        # Without an accelerator projection provider, a physical tied output
        # tensor is materialized as a distinct NumPy allocation. The wrappers'
        # partitioner intentionally requires one shared owner, not merely
        # equal values, so compare the source-derived head and publish the
        # embedding object under the canonical head name before construction.
        if c.get('tie_word_embeddings',False) and projection_provider is None:
            embedding=next((weights[name] for name in (
                'model.embed_tokens.weight','model.language_model.embed_tokens.weight',
                'language_model.embed_tokens.weight') if name in weights),None)
            if embedding is None:
                raise ValueError('tied publisher head has no admitted embedding owner')
            for name in ('lm_head.weight','model.lm_head.weight','model.language_model.lm_head.weight',
                         'language_model.lm_head.weight','output.weight','shared_head.weight',
                         'model.output.weight','model.shared_head.weight',
                         'model.language_model.output.weight','model.language_model.shared_head.weight'):
                head=weights.pop(name,None)
                if head is not None and not np.array_equal(head,embedding):
                    raise ValueError('physical tied publisher head differs from admitted embedding')
            weights['lm_head.weight']=embedding
        weights.update(generated);inventory.assert_unchanged();_cancelled(cancel)
        if family=='qwen3_5_moe':
            from .qwen35_multimodal import Qwen35MultimodalDecoder, _PUBLISHER_DEVICE_ADMISSION
            decoder=Qwen35MultimodalDecoder(config,weights,library=library,text_library=text_library,
                max_bytes=max_bytes,max_snapshots=max_snapshots,context_tokens=context_tokens,text_device=text_device,
                expert_provider=expert_provider,own_expert_provider=True,
                projection_provider=projection_provider,own_projection_provider=True,
                _publisher_device_admission=_PUBLISHER_DEVICE_ADMISSION)
            expert_provider=projection_provider=None
        elif family=='qwen4_exp':
            from .qwen4_multimodal import Qwen4MultimodalDecoder, _PUBLISHER_DEVICE_ADMISSION
            decoder=Qwen4MultimodalDecoder(config,weights,library=library,text_library=text_library,
                max_bytes=max_bytes,max_snapshots=max_snapshots,ple_provider=ple,own_ple_provider=True,
                expert_provider=expert_provider,own_expert_provider=True,text_device=text_device,
                projection_provider=projection_provider,own_projection_provider=True,
                native_composition=native_composition,native_store_runtime_library=native_store_runtime_library,
                native_expert_max_bytes=native_expert_max_bytes,native_ple_max_bytes=native_ple_max_bytes,
                segment_runtime_library=segment_runtime_library,
                segment_model_directory=root if segment_runtime_library is not None else None,
                segment_work_bytes=segment_work_bytes if segment_runtime_library is not None else None,
                segment_layer_ranges=None if not segment_layer_span else tuple(
                    (begin,min(begin+segment_layer_span,c['num_hidden_layers']))
                    for begin in range(0,c['num_hidden_layers'],segment_layer_span)),
                _checkpoint_inventory=inventory,
                _publisher_device_admission=_PUBLISHER_DEVICE_ADMISSION)
            ple=expert_provider=projection_provider=None
        else:
            from .inkling_multimodal import InklingMultimodalDecoder, _PUBLISHER_DEVICE_ADMISSION
            decoder=InklingMultimodalDecoder(config,weights,library=library,text_library=text_library,
                max_bytes=max_bytes,max_snapshots=max_snapshots,context_tokens=context_tokens,
                mtp_max_bytes=mtp_max_bytes,mtp_work_bytes=mtp_work_bytes,
                expert_provider=expert_provider,own_expert_provider=True,text_device=text_device,
                projection_provider=projection_provider,own_projection_provider=True,
                segment_runtime_library=segment_runtime_library,segment_work_bytes=segment_work_bytes,
                model_directory=root,segment_layer_span=segment_layer_span,
                _checkpoint_inventory=inventory,
                _publisher_device_admission=_PUBLISHER_DEVICE_ADMISSION)
            expert_provider=projection_provider=None
        return _finish_decoder(decoder,cancel)
    except BaseException as primary:
        # A failed wrapper may still borrow a projection or expert provider.
        # Finish its retained destruction before closing those source owners.
        pending=[owner for owner in (ple,expert_provider,projection_provider) if owner is not None]
        prior_cleanup=getattr(primary,'cleanup',None)
        def cleanup():
            nonlocal prior_cleanup
            if prior_cleanup is not None:
                prior_cleanup()
                prior_cleanup=None
            failures=[]
            for owner in tuple(pending):
                try:owner.close()
                except BaseException as failure:failures.append(failure)
                else:pending.remove(owner)
            if failures:
                for failure in failures[1:]:
                    failures[0].add_note('additional publisher provider cleanup failure: '+repr(failure))
                raise failures[0]
        try:cleanup()
        except BaseException as failure:
            primary.cleanup=cleanup
            primary.add_note('conditional publisher ownership cleanup remains pending: '+repr(failure))
        raise
    finally:
        reader.close()

__all__=['load_publisher_conditional_decoder']
