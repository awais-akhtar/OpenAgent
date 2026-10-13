"""Exact whole-inventory admission for selected Mixtral checkpoint loading."""
from types import MappingProxyType


# Original publisher configurations, distinct from the broader bounded fixture
# contract admitted by configuration(). These pins never download artifacts or
# authorize an unobserved checkpoint/platform comparison.
MIXTRAL_PUBLISHER_VARIANTS=MappingProxyType({
    'mistralai/Mixtral-8x7B-v0.1':MappingProxyType(dict(
        revision='41c1fe324a57f555d8df74b85a1353a583746532',
        source_version='4.36.0.dev0',hidden_size=4096,intermediate_size=14336,
        num_hidden_layers=32,num_attention_heads=32,num_key_value_heads=8,
        num_local_experts=8,num_experts_per_tok=2,vocab_size=32000,
        max_position_embeddings=32768,rope_theta=1000000.0,
        sliding_window=None,rope_type='default',rope_factor=1.0,head_dim=128,
        rms_norm_eps=1e-5,tie_word_embeddings=False,source_dtype='bfloat16')),
    'mistralai/Mixtral-8x22B-v0.1':MappingProxyType(dict(
        revision='76fa59362cd95eb633a4e419c24364c90674966c',
        source_version='4.38.0',hidden_size=6144,intermediate_size=16384,
        num_hidden_layers=56,num_attention_heads=48,num_key_value_heads=8,
        num_local_experts=8,num_experts_per_tok=2,vocab_size=32000,
        max_position_embeddings=65536,rope_theta=1000000.0,
        sliding_window=None,rope_type='default',rope_factor=1.0,head_dim=128,
        rms_norm_eps=1e-5,tie_word_embeddings=False,source_dtype='bfloat16')),
})


def publisher_contract(config,*,repository,revision):
    """Admit an explicitly selected original publisher configuration offline."""
    from .native_mixtral_decoder import configuration
    variant=MIXTRAL_PUBLISHER_VARIANTS.get(repository)
    if variant is None or revision!=variant['revision']:
        raise ValueError('explicit pinned original Mixtral publisher variant required')
    settings,_=configuration(config,None)
    for name,value in variant.items():
        if name in ('revision','source_version'):continue
        if settings.get(name)!=value:
            raise ValueError('original Mixtral publisher configuration differs: '+name)
    return dict(variant,repository=repository,
        config_source=f'https://huggingface.co/{repository}/blob/{revision}/config.json',
        numerical_parity_verified=False,platform_certified=False)


def admit_mixtral_inventory(config, inventory, *, context_tokens=None):
    from .native_mixtral_decoder import configuration, GLOBAL_NAMES, LAYER_NAMES
    settings, capacity = configuration(config, context_tokens)
    h,m,e,d = (settings[name] for name in
               ('hidden_size','intermediate_size','num_local_experts','head_dim'))
    q,k = settings['num_attention_heads']*d,settings['num_key_value_heads']*d
    global_shapes = dict(zip(GLOBAL_NAMES,((settings['vocab_size'],h),(h,),(settings['vocab_size'],h))))
    if settings['tie_word_embeddings'] and 'lm_head.weight' not in inventory:
        global_shapes.pop('lm_head.weight')
    fused,split = dict(global_shapes),dict(global_shapes)
    shapes = ((h,),(h,),(q,h),(k,h),(k,h),(h,q),(e,h),(e,2*m,h),(e,h,m))
    for index in range(settings['num_hidden_layers']):
        prefix = f'model.layers.{index}.'
        for ordinal,(name,shape) in enumerate(zip(LAYER_NAMES,shapes)):
            fused[prefix+name] = shape
            if ordinal < 6:
                split[prefix+name] = shape
        split[prefix+'block_sparse_moe.gate.weight'] = (e,h)
        for expert in range(e):
            for leaf,shape in (('w1',(m,h)),('w3',(m,h)),('w2',(h,m))):
                split[prefix+f'block_sparse_moe.experts.{expert}.{leaf}.weight'] = shape
    if set(inventory) == set(fused):
        expected,layout = fused,'fused'
    elif set(inventory) == set(split):
        expected,layout = split,'split'
    else:
        raise ValueError('exact complete fused or split Mixtral source inventory required')
    for name,shape in expected.items():
        if tuple(inventory[name]['shape']) != shape or inventory[name].get('dtype') not in ('F32','F16','BF16'):
            raise ValueError('Mixtral checkpoint tensor shape or dtype mismatch: '+name)
    return settings,capacity,layout,global_shapes
