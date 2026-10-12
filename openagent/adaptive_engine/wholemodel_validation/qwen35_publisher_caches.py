"""Complete Qwen3.5 unmasked text cache observations for opt-in comparisons.

Publisher fields follow Transformers v5.16.1 cache_utils.py. No oracle is
imported here; observations must come from the caller's pinned executed owner.
"""
from collections.abc import Mapping
import numpy as np
from ..independent_qwen35 import IndependentQwen35TextDecoder
from .cache_readers import detached,UnsupportedCacheContract


def contract(config,max_bytes):
    if type(max_bytes) is not int or max_bytes<=0:raise ValueError('positive cache export admission required')
    return IndependentQwen35TextDecoder(config,{},_inventory_only=True).config


def schema_for(c,*,media=False):
    result={'position':'global'}
    if media:result['rope_delta']='global'
    for layer,kind in enumerate(c.layer_types):
        names=('key','value') if kind=='full_attention' else ('convolution','recurrent')
        result.update((f'layer.{layer}.{name}',layer) for name in names)
    return result


class Qwen35PublisherCacheReader:
    def __init__(self,*,family,config,role,max_export_bytes,recurrent_layout='key-value',routing_max_export_bytes=None,
                 media=False,rope_delta_reader=None):
        if family!='qwen3_5_moe_text' or role not in ('candidate','oracle'):
            raise ValueError('explicit Qwen3.5 publisher cache role required')
        if type(max_export_bytes) is not int or max_export_bytes<=0:
            raise ValueError('bounded Qwen3.5 physical cache export required')
        if recurrent_layout not in ('key-value','value-key'):
            raise ValueError('explicit recurrent orientation required')
        if type(media) is not bool or (rope_delta_reader is not None and not callable(rope_delta_reader)):
            raise ValueError('explicit media observation policy and callable rotary observation required')
        if media and role=='oracle' and rope_delta_reader is None:
            raise ValueError('publisher media comparison requires a pinned observed rotary-delta accessor')
        if not media and rope_delta_reader is not None:
            raise ValueError('rotary-delta accessor requires the complete media cache schema')
        self.config=contract(config,max_export_bytes);self.geometry=None
        self.media=media;self.rope_delta_reader=rope_delta_reader
        self.schema=schema_for(self.config,media=media);self.role=role;self.max_bytes=max_export_bytes
        self.recurrent_layout=recurrent_layout
        if routing_max_export_bytes is not None:self.enable_routing_observations(max_export_bytes=routing_max_export_bytes)

    def enable_routing_observations(self,*,max_export_bytes):
        if type(max_export_bytes) is not int or max_export_bytes<=0:raise ValueError('bounded Qwen3.5 routing export required')
        self.routing_max_bytes=max_export_bytes
        self.evidence_schema={f'layer.{layer}.{name}':layer for layer in range(self.config.num_hidden_layers)
            for name in ('router_logits','routing_weights')}
        return dict(self.evidence_schema)

    def evidence(self,context):
        # This export only normalizes observed logits and selected route weights;
        # it neither computes a router nor borrows Qwen4 neural equations.
        from .qwen4_publisher_caches import Qwen4PublisherCacheReader
        if self.role=='candidate' and getattr(context.model,'last_output',None) is None and context.output is not None:
            from types import SimpleNamespace
            from .cache_readers import CacheContext
            context=CacheContext(context.cache,SimpleNamespace(last_output=context.output),
                context.output,context.position,context.appended_tokens)
        return Qwen4PublisherCacheReader.evidence(self,context)

    def __call__(self,context):
        position=context.position;c=self.config
        if type(position) is not int or not 0<position<=c.max_position_embeddings:
            raise ValueError('occupied bounded Qwen3.5 observation position required')
        result={'position':np.array(position,np.int64)};batch=None
        if self.media:
            if self.role=='candidate':
                from ..qwen35_multimodal import Qwen35MultimodalDecoder
                wrapper=context.model
                if not isinstance(wrapper,Qwen35MultimodalDecoder):
                    wrapper=getattr(wrapper,'decoder',None)
                if not isinstance(wrapper,Qwen35MultimodalDecoder):
                    raise ValueError('complete Qwen3.5 wrapper owner required for media continuation')
                delta=wrapper.rope_delta
            else:
                # The accessor reads the pinned executed publisher owner/output;
                # no positions are reconstructed from the candidate or fixture.
                delta=self.rope_delta_reader(context)
                if hasattr(delta,'detach'):delta=detached(delta)
                if isinstance(delta,np.ndarray):
                    if delta.size!=1 or delta.dtype.kind not in 'iu':
                        raise ValueError('single exact integer publisher rotary delta required')
                    delta=int(delta.reshape(-1)[0])
            if type(delta) is not int or abs(delta)>c.max_position_embeddings:
                raise ValueError('bounded observed Qwen3.5 rotary continuation required')
            result['rope_delta']=np.array(delta,np.int64)
        if self.role=='candidate':
            model=context.model;seen=set()
            for _ in range(8):
                if id(model) in seen:raise ValueError('cyclic Qwen3.5 decoder ownership')
                seen.add(id(model))
                if not self.media and getattr(model,'rope_delta',0)!=0:
                    raise ValueError('text cache comparison cannot omit media rotary continuation')
                if hasattr(model,'cache') or hasattr(model,'_cache'):break
                model=next((getattr(model,name) for name in ('decoder','text','model')
                            if getattr(model,name,None) is not None),None)
                if model is None:raise ValueError('complete Qwen3.5 physical owner required')
            else:raise ValueError('Qwen3.5 ownership depth exceeds admission')
            if model.position!=position:raise ValueError('Qwen3.5 live cache position differs')
            banks=getattr(model,'cache',getattr(model,'_cache',None))
            mask=getattr(model,'mask',getattr(model,'_mask',None))
            if mask is None or mask.ndim!=2 or mask.shape[1]!=position or not np.all(mask):
                raise ValueError('unmasked Qwen3.5 text comparison requires complete occupied mask')
            batch=mask.shape[0]
        else:
            banks=getattr(context.cache,'layers',None)
        if banks is None or len(banks)!=c.num_hidden_layers:
            raise UnsupportedCacheContract('complete Qwen3.5 cache layers required',fields=['layers'])
        width=2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
        for layer,(kind,bank) in enumerate(zip(c.layer_types,banks)):
            if self.role=='oracle':
                if not type(bank).__module__.startswith('transformers.'):
                    raise UnsupportedCacheContract('pinned publisher cache owner required',fields=[f'layer.{layer}.type'])
                if kind=='full_attention':values=(getattr(bank,'keys',None),getattr(bank,'values',None))
                else:
                    histories=getattr(bank,'conv_states',None);states=getattr(bank,'recurrent_states',None)
                    if not isinstance(histories,Mapping) or set(histories)!={0} or not isinstance(states,Mapping) or set(states)!={0}:
                        raise ValueError('complete single-state publisher delta cache required')
                    values=(histories[0],states[0])
            else:
                if not isinstance(bank,(tuple,list)) or len(bank)!=2:
                    raise ValueError('complete two-leaf Qwen3.5 physical bank required')
                values=bank
            names=('key','value') if kind=='full_attention' else ('convolution','recurrent')
            tails=((c.num_key_value_heads,position,c.head_dim),)*2 if kind=='full_attention' else (
                (width,c.linear_conv_kernel_dim),(c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim))
            for name,value,tail in zip(names,values,tails):
                if value is None:raise ValueError('missing Qwen3.5 cache leaf: '+name)
                if kind=='full_attention':
                    if value.ndim!=4 or value.shape[-2]<position:raise ValueError('complete occupied KV extent required')
                    value=value[...,:position,:]
                elif name=='recurrent' and self.role=='oracle' and self.recurrent_layout=='value-key':
                    value=value.transpose(-1,-2) if hasattr(value,'permute') else value.swapaxes(-1,-2)
                if value.ndim!=len(tail)+1 or tuple(value.shape[1:])!=tail or not 1<=value.shape[0]<=128:
                    raise ValueError('Qwen3.5 physical cache geometry differs: '+name)
                if batch is None:batch=value.shape[0]
                if value.shape[0]!=batch:raise ValueError('Qwen3.5 physical cache batch differs')
                result[f'layer.{layer}.{name}']=value
        total=0
        for value in result.values():
            dtype=str(value.dtype)
            if dtype not in ('float32','float16','int64','torch.float32','torch.float16','torch.bfloat16'):
                raise ValueError('unsupported Qwen3.5 observation dtype')
            size=4 if dtype=='torch.bfloat16' else value.element_size() if hasattr(value,'element_size') else value.dtype.itemsize
            total+=int(np.prod(value.shape,dtype=object))*size
        if 3*total>self.max_bytes:raise MemoryError('complete Qwen3.5 physical cache export exceeds admission')
        exported={name:detached(value) for name,value in result.items()}
        if any(not np.isfinite(value).all() for value in exported.values()):
            raise ValueError('nonfinite Qwen3.5 physical observation')
        return exported


def make_reader(*,schema,options):
    options=dict(options)
    if type(options.get('rope_delta_reader')) is str:
        from .adapters import resolve_callable
        options['rope_delta_reader']=resolve_callable(options['rope_delta_reader'])
    reader=Qwen35PublisherCacheReader(**options)
    if dict(schema)!=reader.schema:raise ValueError('complete Qwen3.5 declared cache schema differs')
    return reader


__all__=['contract','schema_for','Qwen35PublisherCacheReader','make_reader']
