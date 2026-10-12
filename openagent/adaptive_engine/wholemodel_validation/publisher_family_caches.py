"""Explicit GLM/Inkling logical cache exports for later pinned comparisons."""
import numpy as np
from ..independent_inkling import configuration
from ..independent_glm5_next import Glm5NextTextContract
from .cache_readers import detached,UnsupportedCacheContract

def schema(family,config):
    if family=='qwen3_5_moe_text':
        from .qwen35_publisher_caches import contract,schema_for
        return schema_for(contract(config,1024**3))
    if family=='qwen4_exp_text':
        from .qwen4_publisher_caches import contract,schema_for
        return schema_for(contract(config,1024**3))
    result={'position':'global'}
    if family=='inkling_text':
        c,geometry=configuration(config)
        names=('key','value','key_fir','value_fir','attention_fir','mlp_fir')
        for layer in range(len(geometry)):
            result.update((f'layer.{layer}.{name}',layer) for name in names)
    elif family=='glm5_next_text':
        c=Glm5NextTextContract.parse(config)
        for layer,kind in enumerate(c.layer_types):
            names=('convolution','recurrent') if kind=='linear_attention' else ('key','value')
            if kind!='linear_attention' and c.indexer_types[layer]=='full':names+=('indexer',)
            result.update((f'layer.{layer}.{name}',layer) for name in names)
    else:raise ValueError('explicit GLM/Inkling cache family required')
    return result

def empty_shapes(family,config):
    """Declare structural zero token extents without admitting empty histories."""
    result={}
    if family=='qwen3_5_moe_text':
        from .qwen35_publisher_caches import contract
        c=contract(config,1024**3)
        for layer,kind in enumerate(c.layer_types):
            if kind=='full_attention':
                for name in ('key','value'):
                    result[f'layer.{layer}.{name}']=[None,c.num_key_value_heads,0,c.head_dim]
    elif family=='inkling_text':
        _,geometry=configuration(config)
        for layer,g in enumerate(geometry):
            for name in ('key','value'):
                result[f'layer.{layer}.{name}']=[None,g.kv_heads,0,g.dim]
    elif family=='glm5_next_text':
        c=Glm5NextTextContract.parse(config)
        for layer,kind in enumerate(c.layer_types):
            if kind=='linear_attention':continue
            for name,width in (('key',c.qk_nope_head_dim),('value',c.v_head_dim)):
                result[f'layer.{layer}.{name}']=[None,c.num_attention_heads,0,width]
            if c.indexer_types[layer]=='full':
                result[f'layer.{layer}.indexer']=[None,0,2*c.index_head_dim+1]
    elif family=='qwen4_exp_text':
        from .qwen4_publisher_caches import contract
        c=contract(config,1024**3)
        for layer,kind in enumerate(c.layer_types):
            if kind!='linear_attention':
                result[f'layer.{layer}.key']=[None,c.num_key_value_heads,0,c.head_dim]
                result[f'layer.{layer}.value']=[None,c.num_key_value_heads,0,c.head_dim]
                result[f'layer.{layer}.indexer']=[None,0,c.indexer_head_dim]
            if layer+1 in c.ple_layer_ids:
                if c.ngram_size==1:result[f'layer.{layer}.ple_tokens']=[None,0]
                if c.ple_conv_kernel_size==1:result[f'layer.{layer}.ple_convolution']=[None,c.hc_count*c.hidden_size,0]
    else:raise ValueError('explicit GLM/Inkling/Qwen4 cache family required')
    return result

def _transpose(value,axes):
    return value.permute(*axes) if hasattr(value,'permute') else value.transpose(axes)

class _IndexerExport:
    """Describe the implicit validity channel without allocating it yet."""
    def __init__(self,value):
        self.value=value
        self.shape=(*value.shape[:-1],value.shape[-1]+1)

    def export(self):
        source=detached(self.value)
        result=np.empty(self.shape,dtype=source.dtype)
        result[...,:-1]=source
        result[...,-1]=1
        return result

def _export_bytes(value):
    source=value.value if isinstance(value,_IndexerExport) else value
    # bfloat16 is widened losslessly by detached(), so budget its exported size.
    width=4 if str(source.dtype)=='torch.bfloat16' else (
        source.element_size() if hasattr(source,'element_size') else source.dtype.itemsize)
    shape=value.shape
    return int(np.prod(shape,dtype=object))*width

class PublisherFamilyCacheReader:
    def __init__(self,*,family,config,role,max_export_bytes,recurrent_layout='key-value',routing_max_export_bytes=None):
        if role not in ('candidate','oracle') or type(max_export_bytes) is not int or max_export_bytes<=0:
            raise ValueError('explicit cache role and bounded export admission required')
        if recurrent_layout not in ('key-value','value-key'):
            raise ValueError('explicit publisher recurrent matrix orientation required')
        self.family,self.role,self.max_bytes,self.recurrent_layout=family,role,max_export_bytes,recurrent_layout
        self.schema=schema(family,config)
        if family=='inkling_text':self.config,self.geometry=configuration(config)
        else:self.config=Glm5NextTextContract.parse(config);self.geometry=None
        if routing_max_export_bytes is not None:
            self.enable_routing_observations(max_export_bytes=routing_max_export_bytes)

    def enable_routing_observations(self,*,max_export_bytes):
        if self.family!='glm5_next_text':
            raise ValueError('observed GLM publisher routing required')
        if type(max_export_bytes) is not int or max_export_bytes<=0:
            raise ValueError('bounded routing observation export required')
        self.routing_max_bytes=max_export_bytes
        self.evidence_schema={}
        for layer,kind in enumerate(self.config.mlp_layer_types):
            if kind=='sparse':
                for name in ('router_logits','routing_weights'):
                    self.evidence_schema[f'layer.{layer}.{name}']=layer
        return self.evidence_schema

    def evidence(self,context):
        from collections.abc import Mapping
        if not hasattr(self,'routing_max_bytes'):
            raise ValueError('routing observations were not explicitly admitted')
        output=(getattr(context.output,'_openagent_routing_observations',None) if self.role=='oracle'
                else getattr(context.model,'last_output',None))
        if output is None:raise ValueError('complete decoder routing observation missing')
        result={};values={}
        for name in ('router_logits','routing_weights'):
            value=output.get(name) if isinstance(output,Mapping) else getattr(output,name,None)
            if value is None or value.ndim!=4 or tuple(value.shape[1:])!=(
                    context.appended_tokens,self.config.num_hidden_layers,self.config.n_routed_experts):
                raise ValueError('complete token/layer/expert routing observation required')
            values[name]=value
        if values['router_logits'].shape!=values['routing_weights'].shape:
            raise ValueError('routing observation batch geometry differs')
        # Admit the complete retained outputs and detached export copies, even
        # though dense layers contribute no routed-expert evidence leaves.
        if 3*sum(_export_bytes(value) for value in values.values())>self.routing_max_bytes:
            raise MemoryError('routing evidence exceeds export admission')
        for key in self.evidence_schema:
            _,layer,name=key.split('.')
            result[key]=detached(values[name][:,:,int(layer),:])
        return result

    def __call__(self,context):
        position=context.position
        if type(position) is not int or position<0:
            raise ValueError('nonnegative integer cache observation position required')
        result={'position':np.array(position,dtype=np.int64)}
        if self.role=='oracle':self._oracle(context,result)
        else:self._candidate(context,result)
        if set(result)!=set(self.schema):raise ValueError('complete publisher family cache inventory required')
        total=sum(_export_bytes(value) for value in result.values())
        if 3*total>self.max_bytes:raise MemoryError('complete logical cache export exceeds admission')
        return {name:value.export() if isinstance(value,_IndexerExport) else detached(value)
                for name,value in result.items()}

    def _candidate(self,context,result):
        model=context.model
        seen=set()
        from ..native_family_public_text import NativeFamilyPublicTextDecoder
        for _ in range(8):
            if id(model) in seen:raise ValueError('cyclic candidate decoder ownership')
            seen.add(id(model))
            if type(model) is NativeFamilyPublicTextDecoder:
                from .native_family_caches import read_public_cache
                result.update(read_public_cache(model,family=self.family,config=self.config,
                    geometry=self.geometry,position=context.position,max_bytes=self.max_bytes))
                return
            if hasattr(model,'_cache'):break
            child=next((getattr(model,name) for name in ('model','text','decoder')
                        if getattr(model,name,None) is not None),None)
            if child is None:raise ValueError('explicit candidate physical cache owner required')
            model=child
        else:raise ValueError('candidate decoder ownership exceeds admitted depth')
        if model.position!=context.position:raise ValueError('candidate cache position differs from observation')
        banks=model._cache
        if banks is None or len(banks)!=len(self.geometry or self.config.layer_types):
            raise ValueError('complete candidate physical cache required')
        if self.family=='inkling_text':
            c=self.config;capacity=model.config['max_position_embeddings'];kernel=c['conv_kernel_size'];h=c['hidden_size']
            if context.position>capacity:raise ValueError('candidate position exceeds physical cache capacity')
            for layer,(bank,g) in enumerate(zip(banks,self.geometry)):
                batch=bank.shape[0];v=g.kv_width;offset=0;values=[]
                shapes=((batch,capacity,g.kv_heads,g.dim),(batch,capacity,g.kv_heads,g.dim),
                    (batch,kernel,v),(batch,kernel,v),(batch,kernel,h),(batch,kernel,h))
                for shape in shapes:
                    size=int(np.prod(shape[1:],dtype=object));values.append(bank[:,offset:offset+size].reshape(shape));offset+=size
                if bank.ndim!=2 or bank.shape[1]!=offset:
                    raise ValueError('candidate Inkling physical cache geometry differs')
                first=max(0,context.position-g.window+1) if g.window else 0
                for name,value in zip(('key','value'),values[:2]):
                    result[f'layer.{layer}.{name}']=_transpose(value[:,first:context.position],(0,2,1,3))
                for name,value in zip(('key_fir','value_fir','attention_fir','mlp_fir'),values[2:]):
                    result[f'layer.{layer}.{name}']=_transpose(value,(0,2,1))
        else:
            c=self.config;capacity=getattr(model,'max_tokens',c.max_position_embeddings)
            if context.position>capacity:raise ValueError('candidate position exceeds physical cache capacity')
            for layer,(bank,kind) in enumerate(zip(banks,c.layer_types)):
                if hasattr(model,'torch'):
                    batch=bank.shape[0]
                    if kind=='linear_attention':
                        shape=(batch,3*c.linear_num_heads*c.linear_head_dim,c.linear_conv_kernel_dim)
                        size=int(np.prod(shape[1:],dtype=object));first=bank[:,:size].reshape(shape)
                        second=bank[:,size:].reshape(batch,c.linear_num_heads,c.linear_head_dim,c.linear_head_dim)
                    else:
                        shape=(batch,capacity,c.num_attention_heads,c.qk_nope_head_dim+c.v_head_dim)
                        size=int(np.prod(shape[1:],dtype=object));first=bank[:,:size].reshape(shape)
                        second=bank[:,size:].reshape(batch,capacity,2*c.index_head_dim)
                else:
                    first,second=bank;first=first[None];second=second[None]
                if kind=='linear_attention':
                    result[f'layer.{layer}.convolution']=first
                    result[f'layer.{layer}.recurrent']=second
                else:
                    occupied=first[:,:context.position]
                    result[f'layer.{layer}.key']=_transpose(occupied[...,:c.qk_nope_head_dim],(0,2,1,3))
                    result[f'layer.{layer}.value']=_transpose(occupied[...,c.qk_nope_head_dim:],(0,2,1,3))
                    if c.indexer_types[layer]=='full':
                        result[f'layer.{layer}.indexer']=_IndexerExport(second[:,:context.position])

    def _oracle(self,context,result):
        cache=context.cache
        layers=getattr(cache,'layers',None)
        if layers is None or len(layers)!=len(self.geometry or self.config.layer_types):
            raise UnsupportedCacheContract('publisher cache must expose every layer',fields=['layers'])
        for layer,bank in enumerate(layers):
            if not type(bank).__module__.startswith('transformers.'):
                raise UnsupportedCacheContract('pinned Transformers cache layer required',fields=[f'layer.{layer}.type'])
            if self.family=='inkling_text':
                g=self.geometry[layer];span=min(context.position,g.window-1) if g.window else context.position
                for name,attribute in (('key','keys'),('value','values')):
                    value=getattr(bank,attribute,None)
                    if value is None or value.ndim!=4 or value.shape[-2]<span:
                        raise UnsupportedCacheContract('publisher KV state missing',fields=[f'layer.{layer}.{name}'])
                    if value.shape[1]!=g.kv_heads or value.shape[-1]!=g.dim:
                        raise UnsupportedCacheContract('publisher KV geometry differs',fields=[f'layer.{layer}.{name}'])
                    result[f'layer.{layer}.{name}']=value[...,-span:,:] if span else value[...,:0,:]
                histories=getattr(bank,'conv_states',None)
                if histories is None or len(histories)!=4:
                    raise UnsupportedCacheContract('all four publisher FIR histories required',fields=[f'layer.{layer}.conv_states'])
                for name,value in zip(('key_fir','value_fir','attention_fir','mlp_fir'),histories):
                    if value is None:raise UnsupportedCacheContract('publisher FIR history missing',fields=[name])
                    width=g.kv_width if name in ('key_fir','value_fir') else self.config['hidden_size']
                    if value.ndim!=3 or tuple(value.shape[1:])!=(width,self.config['conv_kernel_size']):
                        raise UnsupportedCacheContract('publisher FIR geometry differs',fields=[f'layer.{layer}.{name}'])
                    result[f'layer.{layer}.{name}']=value
            elif self.config.layer_types[layer]=='linear_attention':
                conv=getattr(bank,'conv_states',None);recurrent=getattr(bank,'recurrent_states',None)
                if conv is None or recurrent is None or len(conv)!=1 or len(recurrent)!=1 or conv[0] is None or recurrent[0] is None:
                    raise UnsupportedCacheContract('complete GLM KDA state required',fields=[f'layer.{layer}.kda'])
                result[f'layer.{layer}.convolution']=conv[0]
                value=recurrent[0]
                c=self.config
                if conv[0].ndim!=3 or tuple(conv[0].shape[1:])!=(3*c.linear_num_heads*c.linear_head_dim,c.linear_conv_kernel_dim):
                    raise UnsupportedCacheContract('publisher KDA convolution geometry differs',fields=[f'layer.{layer}.convolution'])
                if value.ndim!=4 or tuple(value.shape[1:])!=(c.linear_num_heads,c.linear_head_dim,c.linear_head_dim):
                    raise UnsupportedCacheContract('publisher KDA recurrent geometry differs',fields=[f'layer.{layer}.recurrent'])
                result[f'layer.{layer}.recurrent']=_transpose(value,(0,1,3,2)) if self.recurrent_layout=='value-key' else value
            else:
                for name,attribute in (('key','keys'),('value','values')):
                    value=getattr(bank,attribute,None)
                    if value is None:raise UnsupportedCacheContract('GLM KV state missing',fields=[f'layer.{layer}.{name}'])
                    width=self.config.qk_nope_head_dim if name=='key' else self.config.v_head_dim
                    if value.ndim!=4 or value.shape[1]!=self.config.num_attention_heads or value.shape[-1]!=width or value.shape[-2]<context.position:
                        raise UnsupportedCacheContract('GLM KV geometry differs',fields=[f'layer.{layer}.{name}'])
                    result[f'layer.{layer}.{name}']=value[...,:context.position,:]
                if self.config.indexer_types[layer]=='full':
                    value=getattr(bank,'indexer_keys',None)
                    if value is None:raise UnsupportedCacheContract('GLM packed indexer state missing',fields=[f'layer.{layer}.indexer'])
                    if value.ndim!=3 or value.shape[-1]!=2*self.config.index_head_dim+1 or value.shape[1]<context.position:
                        raise UnsupportedCacheContract('GLM packed indexer geometry differs',fields=[f'layer.{layer}.indexer'])
                    result[f'layer.{layer}.indexer']=value[:,:context.position]

def make_reader(*,schema,options):
    if options.get('family')=='qwen3_5_moe_text':
        from .qwen35_publisher_caches import make_reader as make_qwen35_reader
        return make_qwen35_reader(schema=schema,options=options)
    if options.get('family')=='qwen4_exp_text':
        from .qwen4_publisher_caches import make_reader as make_qwen4_reader
        return make_qwen4_reader(schema=schema,options=options)
    reader=PublisherFamilyCacheReader(**options)
    if dict(schema)!=reader.schema:raise ValueError('declared publisher cache schema differs')
    return reader

__all__=['schema','empty_shapes','PublisherFamilyCacheReader','make_reader']
