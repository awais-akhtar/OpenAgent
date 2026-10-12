"""GLM5-next configuration defaults and schedules pinned to Transformers 5.16.1.

Architecture source: huggingface/transformers v5.16.1,
src/transformers/models/glm5_next/configuration_glm5_next.py (Apache-2.0).
"""
from copy import deepcopy

_DEFAULTS=dict(vocab_size=154880,hidden_size=4096,intermediate_size=12288,
    moe_intermediate_size=2048,num_hidden_layers=45,num_attention_heads=64,
    num_key_value_heads=64,n_shared_experts=1,n_routed_experts=288,
    routed_scaling_factor=2.5,kv_lora_rank=512,q_lora_rank=1536,qk_rope_head_dim=0,
    v_head_dim=256,qk_nope_head_dim=256,n_group=1,topk_group=1,num_experts_per_tok=8,
    norm_topk_prob=True,hidden_act='silu',max_position_embeddings=1048576,
    rms_norm_eps=1e-5,use_cache=True,tie_word_embeddings=False,attention_bias=False,
    attention_dropout=0.0,index_topk=2048,index_head_dim=128,index_n_heads=32,
    head_dim=0,swiglu_limit=10.0,linear_head_dim=128,linear_num_heads=64,
    linear_conv_kernel_dim=4,linear_lower_bound=-5.0,hc_mult=4,hc_eps=1e-6,
    hc_sinkhorn_iters=20,index_kpool=16,index_kpool_always_select_tail=True)

VISION_DEFAULTS=dict(model_type='glm5_next_vision',depth=24,hidden_size=1024,
    hidden_act='silu',attention_bias=True,attention_dropout=0.0,num_heads=16,
    in_channels=3,image_size=336,patch_size=14,rms_norm_eps=1e-5,
    spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=1536,
    intermediate_size=4096,projection_intermediate_size=10240,swiglu_limit=10.0)

def text_configuration(config):
    c=deepcopy(dict(config));c['model_type']='glm5_next_text'
    if 'num_local_experts' in c:
        value=c.pop('num_local_experts')
        if 'n_routed_experts' in c and c['n_routed_experts']!=value: raise ValueError('conflicting GLM expert count')
        c['n_routed_experts']=value
    for name,value in _DEFAULTS.items(): c.setdefault(name,value)
    count=c['num_hidden_layers']
    if type(count) is not int or not 1<=count<=128: raise ValueError('bounded GLM publisher layer count required')
    if c.get('num_key_value_heads') is None: c['num_key_value_heads']=c['num_attention_heads']
    linear=c.pop('linear_attn_config',None)
    if linear is not None:
        if type(linear) is not dict: raise ValueError('GLM linear attention geometry must be an object')
        for old,new in (('num_heads','linear_num_heads'),('head_dim','linear_head_dim'),
                        ('short_conv_kernel_size','linear_conv_kernel_dim'),('gate_lower_bound','linear_lower_bound')):
            if old in linear: c[new]=linear[old]
        safe=linear.get('safe_gate',True)
        if type(safe) is not bool: raise ValueError('explicit GLM safe gate boolean required')
        if safe and c['linear_lower_bound'] is None: c['linear_lower_bound']=-5.0
    if c.get('layer_types') is None:
        if linear is not None and 'kda_layers' in linear:
            layers=linear['kda_layers']
            if type(layers) is not list or len(set(layers))!=len(layers) or any(type(i) is not int or not 0<=i<count for i in layers):
                raise ValueError('valid GLM KDA layer inventory required')
        else: layers=[i for i in range(count) if i%4!=3]
        c['layer_types']=['linear_attention' if i in layers else 'deepseek_sparse_attention' for i in range(count)]
    if type(c['layer_types']) is not list or len(c['layer_types'])!=count: raise ValueError('complete GLM attention schedule required')
    c['layer_types']=['deepseek_sparse_attention' if v=='full_attention' else v for v in c['layer_types']]
    if linear is not None:
        for field,kind in (('kda_layers','linear_attention'),('full_attn_layers','deepseek_sparse_attention')):
            if field in linear and linear[field]!=[i for i,v in enumerate(c['layer_types']) if v==kind]:
                raise ValueError('GLM linear geometry layer list disagrees with schedule')
    dense=c.pop('first_k_dense_replace',min(3,count))
    if type(dense) is not int or not 0<=dense<=count: raise ValueError('valid GLM dense prefix required')
    if c.get('mlp_layer_types') is None: c['mlp_layer_types']=['dense' if i<dense else 'sparse' for i in range(count)]
    elif 'first_k_dense_replace' in config and c['mlp_layer_types']!=['dense' if i<dense else 'sparse' for i in range(count)]:
        raise ValueError('GLM publisher dense prefix disagrees with its MLP schedule')
    pattern=c.pop('index_topk_pattern',None);freq=c.pop('index_topk_freq',1);offset=c.pop('index_skip_topk_offset',2)
    if c.get('indexer_types') is None:
        if pattern is not None:
            if type(pattern) is str:
                if len(pattern)!=count or set(pattern)-{'F','S'}: raise ValueError('complete GLM F/S index schedule required')
                c['indexer_types']=[{'F':'full','S':'shared'}[v] for v in pattern]
            elif type(pattern) is list: c['indexer_types']=list(pattern)
            else: raise ValueError('explicit GLM index schedule required')
        else:
            if type(freq) is not int or type(offset) is not int or not -128<=offset<=128:
                raise ValueError('bounded GLM index schedule frequency and offset required')
            freq=max(freq,1)
            c['indexer_types']=['full' if max(i-offset+1,0)%freq==0 else 'shared' for i in range(count)]
    c['head_dim']=c['qk_rope_head_dim'];c['qk_head_dim']=c['qk_rope_head_dim']+c['qk_nope_head_dim']
    return c

__all__=['text_configuration','VISION_DEFAULTS']
