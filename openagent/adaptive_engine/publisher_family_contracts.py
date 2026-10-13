"""Explicit publisher wrapper-to-component contracts; never model aliases.

These releases identify the authoritative variant discovered for phase one.
Runtime arithmetic is separately requested as FP32. Source dtype, auxiliary
heads and media contracts remain in the wrapper identity and are not erased.
"""
from dataclasses import dataclass
from copy import deepcopy
from types import MappingProxyType
import hashlib
import json

@dataclass(frozen=True)
class PublisherVariant:
    repository: str
    revision: str
    wrapper: str
    text: str
    source_version: str

PUBLISHER_VARIANTS=MappingProxyType({
    'qwen3_5_moe':PublisherVariant('Qwen/Qwen3.5-35B-A3B','59d61f3ce65a6d9863b86d2e96597125219dc754',
        'qwen3_5_moe','qwen3_5_moe_text','5.16.1'),
    'glm5_next':PublisherVariant('zai-org/GLM-5.3-Flash','eb9eb208eb0d988989d07a6a12d0fdeb5f52574a',
        'glm5_next','glm5_next_text','5.16.1'),
    'inkling_mm_model':PublisherVariant('thinkingmachines/Inkling','91b051f1ec836e6d56596c624c3775b495d797b1',
        'inkling_mm_model','inkling_text','5.16.1'),
    'qwen4_exp':PublisherVariant('Qwen/Qwen3.8-Flash-Next','34567a4712bc9766c4449e2e98e4468bfa24d915',
        'qwen4_exp','qwen4_exp_text','5.16.1'),
})

@dataclass(frozen=True)
class PublisherComponents:
    variant: PublisherVariant
    original_config: dict
    text_config: dict
    source_text_dtype: str
    execution_arithmetic: str
    vision_config: dict
    audio_config: dict | None
    auxiliary_config: dict
    identity: str

def component_contract(config,*,arithmetic):
    if arithmetic!='fp32':
        raise ValueError('this component mapper requires explicit FP32 arithmetic admission')
    original=deepcopy(dict(config));family=original.get('model_type')
    if family not in PUBLISHER_VARIANTS:
        raise ValueError('exact supported publisher multimodal wrapper required')
    if family=='glm5_next' and original.get('text_config') is None:
        outer={'vision_config','quantization_config','mtp_config','image_token_id','video_token_id',
               'image_start_token_id','image_end_token_id','video_start_token_id','video_end_token_id'}
        text={name:deepcopy(value) for name,value in original.items() if name not in outer}
        text['model_type']='glm5_next_text'
    else:
        if type(original.get('text_config')) is not dict: raise ValueError('publisher text component object required')
        text=deepcopy(original['text_config'])
    variant=PUBLISHER_VARIANTS[family]
    if text.get('model_type',variant.text)!=variant.text:
        raise ValueError('publisher text component identity differs')
    text['model_type']=variant.text
    source_dtype=text.get('dtype',text.get('torch_dtype'))
    if source_dtype not in (None,'float32','bfloat16','float16'):
        raise ValueError('unadmitted publisher component source dtype')
    text.pop('torch_dtype',None);text['dtype']='float32'
    auxiliary=deepcopy(original.get('mtp_config') or {})
    if family=='qwen3_5_moe':
        text['architectures']=['Qwen3_5MoeForCausalLM']
        auxiliary.update({name:text.pop(name) for name in ('mtp_num_hidden_layers',
            'mtp_use_dedicated_embeddings') if name in text})
        if text.get('rope_parameters',{}).get('mrope_interleaved',True) is not True:
            raise ValueError('Qwen3.5 publisher requires interleaved MRoPE')
        for name,value in (('attn_output_gate',True),('mamba_ssm_dtype','float32')):
            if text.get(name,value)!=value:raise ValueError('unadmitted Qwen3.5 publisher switch: '+name)
        if text.get('mlp_only_layers',[]):raise ValueError('Qwen3.5 dense-only layer overrides are not admitted')
    elif family=='qwen4_exp':
        text['architectures']=['Qwen4ExpForCausalLM']
        # Defaults pinned to the publisher's v5.16.1 configuration class.
        for name,value in (('seed',1234),('norm_topk_prob',True),
            ('ple_conv_kernel_size',4),('ngram_size',3),('heads_per_ngram',8),
            ('ngram_vocab_size_base',20000000),('make_ngram_vocab_size_divisible_by',128),
            ('split_ngram_parts',512)):
            text.setdefault(name,value)
        if text.get('ple_embed_dim') is None:
            text['ple_embed_dim']=text['hidden_size']
        auxiliary.update({name:text.pop(name) for name in ('mtp','mtp_num_hidden_layers',
            'mtp_use_dedicated_embeddings') if name in text})
        schedule=text.get('layer_types')
        if type(schedule) is not list:
            raise ValueError('publisher Qwen4 must declare its layer schedule')
        # Publisher calls these full_attention although the block uses QSA.
        text['layer_types']=['qwen_sparse_attention' if value=='full_attention' else value for value in schedule]
    elif family=='inkling_mm_model':
        text['architectures']=['InklingForCausalLM']
        auxiliary.update({name:text.pop(name) for name in ('num_mtp_layers','mtp_local_layer_ids',
            'chain_hidden_post_norm','mtp_hidden_states_first') if name in text})
        for old,new in (('model_max_length','max_position_embeddings'),('sconv_kernel_size','conv_kernel_size')):
            if old in text:
                if new in text and text[new]!=text[old]:
                    raise ValueError('conflicting Inkling publisher geometry: '+old)
                text[new]=text.pop(old)
        count=text.get('num_hidden_layers');local=text.pop('local_layer_ids',None)
        if type(count) is not int or count<=0 or type(local) is not list or len(set(local))!=len(local) or any(type(v) is not int or not 0<=v<count for v in local):
            raise ValueError('explicit valid Inkling local layer ownership required')
        text.setdefault('layer_types',['hybrid_sliding' if i in local else 'hybrid' for i in range(count)])
        dense=text.pop('dense_mlp_idx',None)
        if type(dense) is not int or not 0<=dense<=count:
            raise ValueError('explicit Inkling dense prefix required')
        text.setdefault('mlp_layer_types',['dense' if i<dense else 'sparse' for i in range(count)])
        dense_width=text.pop('dense_intermediate_size',None)
        if dense_width is not None:
            text['moe_intermediate_size']=text['intermediate_size'];text['intermediate_size']=dense_width
        for key,value in (('q_bias',False),('o_bias',False),('use_sconv',True),('use_embed_norm',True),
                          ('use_gate_bias',True),('gate_activation','sigmoid'),('norm_after_topk',True),('use_global_scale',True)):
            if text.pop(key,value)!=value:
                raise ValueError('unsupported Inkling publisher switch: '+key)
        if text.pop('final_logit_softcapping',None) is not None:
            raise ValueError('Inkling logit softcap is not admitted')
        text.setdefault('rms_norm_eps_moe_gate',1e-6)
        text.setdefault('log_scaling_alpha',0.1)
    else:
        from .publisher_glm5_config import text_configuration
        text=text_configuration(text)
        for key,value in (('mhc',True),('mla_use_nope',True),('index_kpool_compress',True),
            ('indexer_rope_interleave',True),('moe_router_dtype','float32')):
            if text.pop(key,value)!=value:
                raise ValueError('unadmitted GLM publisher switch: '+key)
        for name in ('num_nextn_predict_layers','index_share_for_mtp_iteration'):
            if name in text: auxiliary[name]=text.pop(name)
        dense=text.pop('first_k_dense_replace',None)
        if dense is not None and text.get('mlp_layer_types')!=['dense' if i<dense else 'sparse' for i in range(text['num_hidden_layers'])]:
            raise ValueError('GLM dense prefix disagrees with the decoder schedule')
    vision=deepcopy(original.get('vision_config'))
    if vision is None and family=='glm5_next':
        from .publisher_glm5_config import VISION_DEFAULTS
        vision=deepcopy(VISION_DEFAULTS)
    if type(vision) is not dict:
        raise ValueError('publisher wrapper requires its own vision component')
    audio=deepcopy(original.get('audio_config'))
    if audio is not None and type(audio) is not dict:
        raise ValueError('publisher audio component must be an explicit configuration object')
    identity=hashlib.sha256(json.dumps(dict(original=original,arithmetic=arithmetic,
        repository=variant.repository,revision=variant.revision),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return PublisherComponents(variant,original,text,source_dtype or 'unspecified',arithmetic,
        vision,audio,auxiliary,'openagent-publisher-components-v1:'+identity)

def partition_weights(tensors,components):
    """Partition every tensor exactly once and retain optional head namespaces."""
    text,vision,audio,auxiliary={},{},{},{}
    def retain(owner,name,value):
        if name in owner:
            raise ValueError('duplicate mapped publisher checkpoint tensor: '+name)
        owner[name]=value
    for name,value in tensors.items():
        if type(name) is not str:
            raise ValueError('checkpoint tensor names must be strings')
        if name=='lm_head.weight':
            retain(text,name,value)
        elif name in ('model.language_model.lm_head.weight',
                      'language_model.lm_head.weight', 'model.lm_head.weight',
                      'output.weight', 'shared_head.weight',
                      'model.output.weight', 'model.shared_head.weight',
                      'model.language_model.output.weight',
                      'model.language_model.shared_head.weight'):
            retain(text,'lm_head.weight',value)
        elif name.startswith(('mtp.','model.mtp.','model.language_model.mtp.')):
            retain(auxiliary,name,value)
        elif name.startswith(('model.language_model.visual.',
                              'model.language_model.vision_tower.',
                              'model.language_model.video_tower.')):
            retain(vision,name,value)
        elif name.startswith('model.language_model.audio_tower.'):
            retain(audio,name,value)
        elif name.startswith(('model.language_model.', 'language_model.')):
            prefix = 'model.language_model.' if name.startswith('model.language_model.') else 'language_model.'
            destination='model.'+name[len(prefix):]
            retain(text,destination,value)
        elif name.startswith(('model.visual.', 'model.vision_tower.',
                              'model.video_tower.', 'visual.', 'vision_tower.',
                              'video_tower.')):
            retain(vision,name,value)
        elif name.startswith(('model.audio_tower.', 'audio_tower.')):
            retain(audio,name,value)
        elif name.startswith(('model.layers.','model.embed_tokens.','model.norm.')):
            retain(text,name,value)
        else:
            raise ValueError('unowned publisher checkpoint tensor: '+name)
    if not text or not vision or (components.audio_config is not None and not audio):
        raise ValueError('publisher checkpoint is missing a required component')
    if components.audio_config is None and audio:
        raise ValueError('publisher checkpoint contains an unadmitted audio component')
    # Tied publishers may omit a physical lm_head tensor.  Keep the public
    # runtime namespace complete by binding the head to the admitted embedding
    # owner; this is an ownership alias, not a decoded or copied fallback.
    text_config = components.text_config
    tied = text_config.get('tie_word_embeddings', False)
    if type(tied) is not bool:
        raise ValueError('publisher tied-embedding policy must be explicit boolean')
    if tied and 'lm_head.weight' not in text:
        embedding = text.get('model.embed_tokens.weight')
        if embedding is None:
            raise ValueError('tied publisher head requires an admitted embedding tensor')
        text['lm_head.weight'] = embedding
    elif tied:
        embedding = text.get('model.embed_tokens.weight')
        if embedding is None or text['lm_head.weight'] is not embedding:
            raise ValueError('tied publisher head must share the admitted embedding owner')
    if 'lm_head.weight' not in text:
        raise ValueError('publisher checkpoint is missing its output head')
    return text,vision,audio,auxiliary

__all__=['PublisherVariant','PublisherComponents','PUBLISHER_VARIANTS','component_contract','partition_weights']
