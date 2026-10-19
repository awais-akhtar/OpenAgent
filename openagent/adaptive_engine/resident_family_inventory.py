"""Pure complete-header admission for resident native text checkpoints."""

import math


FAMILIES=frozenset(('mixtral','qwen3_5_moe_text','qwen4_exp_text','glm_moe_dsa','glm5_next_text','inkling_text'))

# The planner is also called by direct construction paths, not only by the
# native checkpoint reader.  Keep the identity mapping here so a caller
# cannot ask (for example) the Qwen4 tensor planner to bless a Qwen3.5 config
# whose names happen to have compatible shapes.
_CONFIG_IDENTITIES = {
    'mixtral': frozenset(('mixtral',)),
    'qwen3_5_moe_text': frozenset(('qwen3_5_moe_text',)),
    'qwen4_exp_text': frozenset(('qwen4_exp_text',)),
    'glm_moe_dsa': frozenset(('glm_moe_dsa',)),
    'glm5_next_text': frozenset(('glm5_next_text',)),
    'inkling_text': frozenset(('inkling_text',)),
}

# These resident text decoders do not own a draft/MTP execution path. Their
# publisher wrappers move these fields into auxiliary_config and reject them;
# keep the direct resident path from accepting the same declaration while
# silently planning only the target-model tensor inventory.
_UNCONSUMED_MTP_FIELDS = {
    'qwen3_5_moe_text': frozenset((
        'mtp_num_hidden_layers', 'mtp_use_dedicated_embeddings')),
    'qwen4_exp_text': frozenset((
        'mtp', 'mtp_num_hidden_layers', 'mtp_use_dedicated_embeddings')),
    'glm5_next_text': frozenset((
        'num_nextn_predict_layers', 'index_share_for_mtp_iteration')),
}


def admit_resident_inventory(config,inventory,*,family,max_bytes,context_tokens=None,
                             backend='native'):
    if type(family) is not str or family not in FAMILIES:
        raise ValueError('explicit resident family inventory contract required')
    if backend not in ('native','device','public-range'):
        raise ValueError('explicit resident family source backend required')
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('positive resident inventory byte admission required')
    identity = config.get('model_type') if type(config) is dict else None
    if (type(identity) is not str or identity not in _CONFIG_IDENTITIES[family]
            or inventory is None or not callable(getattr(inventory, 'items', None))):
        raise ValueError('complete resident configuration and tensor inventory required')
    mtp_config=config.get('mtp_config')
    if mtp_config is not None and type(mtp_config) is not dict:
        raise ValueError('resident auxiliary/MTP configuration must be an object')
    if ((type(mtp_config) is dict and mtp_config) or
            any(name in config for name in _UNCONSUMED_MTP_FIELDS.get(identity, ()))):
        raise ValueError(
            f'{identity} resident inventory rejects MTP configuration: '
            'no admitted draft consumer')
    integers={};names={name:name for name in inventory}
    # Keep this pure admission API safe when used independently of the file
    # reader too: callers may supply an in-memory header map, so validate its
    # record shape and calculate the same FP32 resident/conversion budget
    # before doing family-specific geometry work.
    resident=largest=0
    for name,record in inventory.items():
        if type(name) is not str or not name or type(record) is not dict:
            raise ValueError('resident inventory requires named tensor header records')
        shape,dtype=record.get('shape'),record.get('dtype')
        if (type(shape) not in (tuple,list) or
                any(type(size) is not int or size<0 for size in shape) or
                dtype not in ('F32','F16','BF16','I64')):
            raise ValueError('resident inventory tensor header is malformed: '+name)
        count=math.prod(shape)
        source_width={'F32':4,'F16':2,'BF16':2,'I64':8}[dtype]
        # Every admitted floating source is materialized as FP32. I64 is
        # reserved for a family-specific integer metadata table and remains
        # integer-sized in the loaded inventory.
        loaded=count*(8 if dtype=='I64' else 4)
        source=count*source_width
        resident+=loaded
        largest=max(largest,loaded+source)
    required=resident*2+largest
    if required>max_bytes:
        raise MemoryError('resident family inventory conversion exceeds explicit admission')
    if family!='mixtral':
        _admit_declared_source_dtype(config,inventory)
    if family=='mixtral':
        # Mixtral shares the same header-shape and conversion-budget boundary
        # as the other resident planners. Keep family geometry in its exact
        # planner, but do not let that early family dispatch bypass the
        # caller's resident-memory admission.
        from .mixtral_inventory import admit_mixtral_inventory
        admit_mixtral_inventory(config,inventory,context_tokens=context_tokens)
        from .source_scope import require_source_inventory_admission
        require_source_inventory_admission(
            family='mixtral',model_type='mixtral',backend=backend,
            arithmetic='fp32',persisted_format='float32',inventory=inventory)
        return
    if family=='glm_moe_dsa':
        # Whole-model CPU/native DSA readers call this planner with the full
        # SafeTensors header map before payload materialization. Bind that
        # exact inventory to the native P02 source row here, rather than let
        # a shape-correct direct loader imply tuple admission.
        from .source_scope import require_source_inventory_admission
        require_source_inventory_admission(
            family='glm-dsa',model_type='glm_moe_dsa',backend=backend,
            arithmetic='fp32',persisted_format='float32',inventory=inventory)
    elif family=='qwen4_exp_text':
        from .source_scope import require_source_inventory_admission
        require_source_inventory_admission(
            family='qwen4',model_type='qwen4_exp_text',backend=backend,
            arithmetic='fp32',persisted_format='float32',inventory=inventory)
    elif family=='glm5_next_text':
        from .source_scope import require_source_inventory_admission
        require_source_inventory_admission(
            family='glm5-next',model_type='glm5_next_text',backend=backend,
            arithmetic='fp32',persisted_format='float32',inventory=inventory)
    if family=='qwen3_5_moe_text':
        from .independent_qwen35 import IndependentQwen35TextDecoder
        planner=IndependentQwen35TextDecoder(config,{},_inventory_only=True)
        shapes=dict(planner.tensor_shapes)
        capacity=config['max_position_embeddings'] if context_tokens is None else context_tokens
        if type(capacity) is not int or not 1<=capacity<=min(config['max_position_embeddings'],65536):
            raise ValueError('bounded native Qwen3.5 physical context required before resident payload reads')
    elif family=='qwen4_exp_text':
        from .native_qwen4_ranges import qwen4_tensor_inventory
        shapes,integers=qwen4_tensor_inventory(config,max_bytes=max_bytes)
    elif family=='glm_moe_dsa':
        from .independent_glm_dsa import IndependentGlmDsaDecoder
        shapes=dict(IndependentGlmDsaDecoder(config,{},_inventory_only=True).tensor_shapes)
    elif family=='inkling_text':
        from .independent_inkling import IndependentInklingDecoder
        shapes=dict(IndependentInklingDecoder(config,{},max_bytes=max_bytes,_inventory_only=True).tensor_shapes)
    else:
        from .independent_glm5_next import Glm5NextTextContract,tensor_layout
        contract=Glm5NextTextContract.parse(config);shapes=dict(tensor_layout(contract)[0]);names={}
        for original in inventory:
            canonical=original
            for prefix in ('model.language_model.','language_model.','model.'):
                if canonical.startswith(prefix):canonical=canonical[len(prefix):];break
            if canonical in names:raise ValueError('duplicate canonical GLM5-next source tensor')
            names[canonical]=original
        if 'lm_head.weight' in names or not contract.tie_word_embeddings:
            shapes['lm_head.weight']=(contract.vocab_size,contract.hidden_size)
        capacity=256 if context_tokens is None else context_tokens
        if type(capacity) is not int or not 1<=capacity<=min(contract.max_position_embeddings,65536):
            raise ValueError('bounded native GLM5-next context required before payload reads')
    if config.get('tie_word_embeddings',False) and family!='glm5_next_text' and 'lm_head.weight' not in names:
        shapes.pop('lm_head.weight',None)
    if set(names)!=set(shapes):raise ValueError('complete original resident family tensor inventory differs')
    if config.get('tie_word_embeddings', False):
        embedding_name = next((candidate for candidate in
            ('embed_tokens.weight', 'model.embed_tokens.weight',
             'language_model.embed_tokens.weight') if candidate in names), None)
        if embedding_name is None:
            raise ValueError('tied resident family inventory is missing input embeddings')
        embedding = inventory[names[embedding_name]]
        head_name = next((candidate for candidate in
            ('lm_head.weight', 'output.weight', 'shared_head.weight') if candidate in names), None)
        if head_name is not None:
            head = inventory[names[head_name]]
            if (tuple(head['shape']) != tuple(embedding['shape']) or
                    head['dtype'] != embedding['dtype']):
                raise ValueError('tied resident embedding/head source metadata differs')
    for name,shape in shapes.items():
        record=inventory[names[name]]
        allowed=('I64',) if name in integers else ('F32','F16','BF16')
        if tuple(record['shape'])!=tuple(shape) or record['dtype'] not in allowed:
            raise ValueError('resident family source shape/dtype differs: '+names[name])
    source_family={
        'qwen3_5_moe_text':'qwen35',
        'inkling_text':'inkling',
    }.get(family)
    if source_family is not None:
        from .source_scope import require_source_inventory_admission
        require_source_inventory_admission(
            family=source_family,model_type=family,backend=backend,
            arithmetic='fp32',persisted_format='float32',inventory=inventory)


def _admit_declared_source_dtype(config,inventory):
    """Bind optional config dtype metadata to floating source tensor headers.

    Resident planners normalize loaded weights to FP32, so without this
    preflight a checkpoint declaring (for example) BF16 could silently pass
    with an F16/F32 tensor inventory.  Integer family metadata tables are
    intentionally excluded: their persisted integer encoding is defined by
    the family tensor contract, not the floating model dtype declaration.
    """
    fields=[config[name] for name in ('dtype','torch_dtype')
            if name in config and config[name] is not None]
    if any(type(value) is not str for value in fields):
        raise ValueError('declared resident source dtype must be a string')
    if len(set(fields))>1:
        raise ValueError('resident dtype and torch_dtype declarations disagree')
    if not fields:
        return
    encodings={'float32':'F32','float16':'F16','bfloat16':'BF16'}
    expected=encodings.get(fields[0])
    if expected is None:
        raise ValueError('unsupported declared resident source dtype')
    for name,record in inventory.items():
        if record['dtype']!='I64' and record['dtype']!=expected:
            raise ValueError('resident source tensor disagrees with declared dtype: '+name)


__all__=['FAMILIES','admit_resident_inventory']
