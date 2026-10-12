"""Pure complete-header admission for resident native text checkpoints."""


FAMILIES=frozenset(('mixtral','qwen3_5_moe_text','qwen4_exp_text','glm_moe_dsa','glm5_next_text','inkling_text'))


def admit_resident_inventory(config,inventory,*,family,max_bytes,context_tokens=None):
    if family not in FAMILIES:raise ValueError('explicit resident family inventory contract required')
    if family=='mixtral':
        from .mixtral_inventory import admit_mixtral_inventory
        admit_mixtral_inventory(config,inventory,context_tokens=context_tokens)
        return
    integers={};names={name:name for name in inventory}
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
    if config.get('tie_word_embeddings',False) and family!='glm5_next_text' and 'lm_head.weight' not in inventory:
        shapes.pop('lm_head.weight',None)
    if set(names)!=set(shapes):raise ValueError('complete original resident family tensor inventory differs')
    for name,shape in shapes.items():
        record=inventory[names[name]]
        allowed=('I64',) if name in integers else ('F32','F16','BF16')
        if tuple(record['shape'])!=tuple(shape) or record['dtype'] not in allowed:
            raise ValueError('resident family source shape/dtype differs: '+names[name])


__all__=['FAMILIES','admit_resident_inventory']
