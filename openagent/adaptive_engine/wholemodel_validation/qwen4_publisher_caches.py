"""Complete unmasked-text Qwen4 cache observations, including lexical state."""
import numpy as np
from collections.abc import Mapping

from ..independent_qwen4 import IndependentQwen4TextDecoder
from .cache_readers import detached,UnsupportedCacheContract


def contract(config,max_bytes):
    return IndependentQwen4TextDecoder(config,{},_inventory_only=True,max_bytes=max_bytes).config


def schema_for(c,*,media=False):
    result={'position':'global'}
    if media:result.update((name,'global') for name in ('rope_delta','mask','positions'))
    for layer,kind in enumerate(c.layer_types):
        names=('convolution','recurrent') if kind=='linear_attention' else ('key','value','indexer')
        if layer+1 in c.ple_layer_ids:names+=('ple_tokens','ple_convolution')
        result.update((f'layer.{layer}.{name}',layer) for name in names)
    return result


class Qwen4PublisherCacheReader:
    def __init__(self,*,family,config,role,max_export_bytes,recurrent_layout='key-value',routing_max_export_bytes=None,
                 media=False,auxiliary_reader=None):
        if family!='qwen4_exp_text' or role not in ('candidate','oracle'):
            raise ValueError('explicit Qwen4 publisher cache role required')
        if type(max_export_bytes) is not int or max_export_bytes<=0:
            raise ValueError('bounded Qwen4 cache observation admission required')
        if recurrent_layout not in ('key-value','value-key'):
            raise ValueError('explicit Qwen4 recurrent matrix orientation required')
        if type(media) is not bool or (auxiliary_reader is not None and not callable(auxiliary_reader)):
            raise ValueError('explicit media policy and pinned observed auxiliary accessor required')
        if media and role=='oracle' and auxiliary_reader is None:
            raise ValueError('publisher media caches require observed mask/positions/rotary-delta access')
        if not media and auxiliary_reader is not None:raise ValueError('auxiliary access requires a complete media schema')
        self.config=contract(config,max_export_bytes);self.geometry=None
        self.media=media;self.auxiliary_reader=auxiliary_reader
        self.schema=schema_for(self.config,media=media);self.role=role;self.max_bytes=max_export_bytes
        self.recurrent_layout=recurrent_layout
        if routing_max_export_bytes is not None:
            self.enable_routing_observations(max_export_bytes=routing_max_export_bytes)

    def enable_routing_observations(self, *, max_export_bytes):
        if type(max_export_bytes) is not int or max_export_bytes <= 0:
            raise ValueError('bounded Qwen4 routing export admission required')
        self.routing_max_bytes = max_export_bytes
        self.evidence_schema = {f'layer.{layer}.{name}':layer
            for layer in range(self.config.num_hidden_layers)
            for name in ('router_logits','routing_weights')}
        return dict(self.evidence_schema)

    def evidence(self, context):
        if not hasattr(self, 'routing_max_bytes'):
            raise ValueError('Qwen4 routing observations were not explicitly admitted')
        output = (getattr(context.output, '_openagent_routing_observations', None)
                  if self.role == 'oracle' else getattr(context.model, 'last_output', None))
        if self.role=='candidate' and output is None:output=context.output
        if output is None:
            raise ValueError('complete executed Qwen4 routing observation required')
        def field(name):
            return output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
        raw, weights = field('router_logits'), field('routing_weights')
        c = self.config
        tokens = context.appended_tokens
        if type(tokens) is not int or tokens <= 0:
            raise ValueError('positive Qwen4 routing observation token count required')
        result = {}
        if isinstance(raw, (tuple, list)):
            routes = field('routes')
            if self.role != 'candidate' or len(raw) != c.num_hidden_layers or not isinstance(routes,(tuple,list)) or len(routes)!=len(raw):
                raise ValueError('complete Qwen4 executed routing trace inventory required')
            batch = None
            for layer, (logits, route) in enumerate(zip(raw,routes)):
                if not isinstance(route,(tuple,list)) or len(route)!=2:
                    raise ValueError('Qwen4 selected route trace required')
                indices, scales = route
                rows = logits.shape[0] if getattr(logits,'ndim',None)==2 else 0
                if (not rows or rows%tokens or tuple(logits.shape)!=(rows,c.num_experts) or
                        tuple(indices.shape)!=(rows,c.num_experts_per_tok) or scales.shape!=indices.shape):
                    raise ValueError('Qwen4 executed routing trace geometry differs')
                observed_batch = rows//tokens
                if not 0 < observed_batch <= 128 or (batch is not None and batch!=observed_batch):
                    raise ValueError('Qwen4 routing trace batch inventory differs')
                batch = observed_batch
                if batch*tokens*c.num_hidden_layers*c.num_experts*32 > self.routing_max_bytes:
                    raise MemoryError('complete Qwen4 routing evidence exceeds export admission')
                logits, indices, scales = detached(logits), detached(indices), detached(scales)
                if (indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=c.num_experts) or
                        not np.isfinite(logits).all() or not np.isfinite(scales).all() or
                        any(np.unique(row).size!=c.num_experts_per_tok for row in indices)):
                    raise ValueError('invalid Qwen4 executed route selection')
                dense = np.zeros((rows,c.num_experts),np.float32)
                np.put_along_axis(dense,indices,scales,axis=-1)
                result[f'layer.{layer}.router_logits'] = logits.reshape(batch,tokens,c.num_experts)
                result[f'layer.{layer}.routing_weights'] = dense.reshape(batch,tokens,c.num_experts)
        else:
            expected = (tokens,c.num_hidden_layers,c.num_experts)
            if (raw is None or weights is None or getattr(raw,'ndim',None)!=4 or
                    tuple(raw.shape[1:])!=expected or tuple(weights.shape)!=tuple(raw.shape) or
                    not 0 < raw.shape[0] <= 128):
                raise ValueError('complete Qwen4 token/layer/expert evidence required')
            if int(np.prod(raw.shape,dtype=object))*32 > self.routing_max_bytes:
                raise MemoryError('complete Qwen4 routing evidence exceeds export admission')
            for layer in range(c.num_hidden_layers):
                for name, value in (('router_logits',raw),('routing_weights',weights)):
                    leaf = detached(value[:,:,layer,:])
                    if not np.isfinite(leaf).all():
                        raise ValueError('nonfinite Qwen4 routing evidence')
                    result[f'layer.{layer}.{name}'] = leaf
        if set(result)!=set(self.evidence_schema):
            raise ValueError('complete Qwen4 routing evidence schema required')
        return result

    def __call__(self,context):
        c=self.config;position=context.position
        if type(position) is not int or not 0<position<=c.max_position_embeddings:
            raise ValueError('occupied bounded Qwen4 observation position required')
        result={'position':np.asarray(position,dtype=np.int64)};batch=None
        if self.media:
            if self.role=='candidate':
                from ..qwen4_multimodal import Qwen4MultimodalDecoder
                wrapper=context.model
                if type(wrapper) is not Qwen4MultimodalDecoder:wrapper=getattr(wrapper,'decoder',None)
                if type(wrapper) is not Qwen4MultimodalDecoder:
                    raise ValueError('complete Qwen4 publisher wrapper required for rotary continuation')
                delta=wrapper.rope_delta
            else:
                auxiliary=self.auxiliary_reader(context)
                if not isinstance(auxiliary,Mapping) or set(auxiliary)!={'rope_delta','mask','positions'}:
                    raise ValueError('complete actually observed publisher auxiliary continuation required')
                delta=auxiliary['rope_delta']
                if hasattr(delta,'detach'):delta=detached(delta)
                if isinstance(delta,np.ndarray):
                    if delta.size!=1 or delta.dtype.kind not in 'iu':raise ValueError('exact integer rotary delta required')
                    delta=int(delta.reshape(-1)[0])
                result['mask']=auxiliary['mask'];result['positions']=auxiliary['positions']
            if type(delta) is not int or abs(delta)>c.max_position_embeddings:
                raise ValueError('bounded actual Qwen4 rotary continuation required')
            result['rope_delta']=np.asarray(delta,dtype=np.int64)
        def leaf(layer,name,value,tail):
            nonlocal batch
            if value is None or value.ndim!=len(tail)+1 or tuple(value.shape[1:])!=tuple(tail):
                raise UnsupportedCacheContract('Qwen4 cache leaf geometry differs',fields=[f'layer.{layer}.{name}'])
            if batch is None:batch=value.shape[0]
            if not 0<batch<=128 or value.shape[0]!=batch:
                raise ValueError('Qwen4 observed cache batch differs')
            result[f'layer.{layer}.{name}']=value
        if self.role=='candidate':
            model=context.model;seen=set()
            for _ in range(8):
                if id(model) in seen:raise ValueError('cyclic Qwen4 decoder owner')
                seen.add(id(model))
                from ..native_qwen4_public_text import NativeQwen4PublicTextDecoder
                if type(model) is NativeQwen4PublicTextDecoder:
                    from .native_qwen4_caches import read_public_qwen4_cache
                    exported=read_public_qwen4_cache(model,position=position,max_bytes=self.max_bytes,include_auxiliary=self.media)
                    if self.media:
                        if not np.all(exported['mask']==1) or exported['positions'].shape!=(3,1,position):
                            raise ValueError('complete unpadded public media rotary continuation required')
                        exported['rope_delta']=result['rope_delta']
                    if set(exported)!=set(self.schema):raise ValueError('complete public Qwen4 cache schema required')
                    return {name:detached(value) for name,value in exported.items()}
                if hasattr(model,'_ple') or hasattr(model,'ple_cache'):break
                model=next((getattr(model,name) for name in ('decoder','text','model')
                            if getattr(model,name,None) is not None),None)
                if model is None:raise ValueError('explicit Qwen4 physical cache owner required')
            else:raise ValueError('Qwen4 ownership depth exceeds admission')
            if model.position!=position:raise ValueError('Qwen4 live observation position differs')
            banks=model._cache if hasattr(model,'_ple') else model.cache
            lexical=model._ple if hasattr(model,'_ple') else model.ple_cache
            mask=model._mask if hasattr(model,'_ple') else model.mask
            positions=model._positions if hasattr(model,'_ple') else model.positions
            if len(banks)!=c.num_hidden_layers or len(lexical)!=c.num_hidden_layers:
                raise ValueError('complete Qwen4 physical layer inventory required')
            # This case declares unmasked contiguous text. Check the auxiliary
            # continuation inputs instead of silently dropping media positions.
            if mask is None or positions is None or tuple(mask.shape)[-1]!=position:
                raise ValueError('complete Qwen4 mask/position continuation required')
            if mask.ndim!=2 or not 0<mask.shape[0]<=128 or positions.ndim not in (2,3) or tuple(positions.shape[-2:])!=tuple(mask.shape):
                raise ValueError('Qwen4 auxiliary continuation geometry differs')
            if positions.ndim==3 and positions.shape[0]!=3:
                raise ValueError('Qwen4 media rotary position axis inventory differs')
            if 3*int(np.prod(positions.shape,dtype=object))+position*8>self.max_bytes:
                raise MemoryError('Qwen4 position observation exceeds export admission')
            if hasattr(mask,'detach'):
                if not bool(mask.all()):raise ValueError('unmasked Qwen4 text observation required')
                if not self.media:
                    torch=model.torch;expected=torch.arange(position,device=positions.device)
                    if not bool((positions==expected).all()):raise ValueError('contiguous Qwen4 text positions required')
            else:
                if not np.all(mask) or (not self.media and not np.all(positions==np.arange(position))):
                    raise ValueError('unmasked contiguous Qwen4 text observation required')
            if self.media:result.update(mask=mask,positions=positions)
        else:
            banks=getattr(context.cache,'layers',None);lexical=None
            if banks is None or len(banks)!=c.num_hidden_layers:
                raise UnsupportedCacheContract('complete publisher Qwen4 cache layers required',fields=['layers'])
        width=2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
        for layer,(kind,bank) in enumerate(zip(c.layer_types,banks)):
            if self.role=='oracle':
                if not type(bank).__module__.startswith('transformers.'):
                    raise UnsupportedCacheContract('pinned publisher cache layer required',fields=[f'layer.{layer}.type'])
                histories=getattr(bank,'conv_states',None)
                required_conv=({0} if kind=='linear_attention' else set())|({1,2} if layer+1 in c.ple_layer_ids else set())
                recurrent=getattr(bank,'recurrent_states',None)
                required_recurrent={0} if kind=='linear_attention' else set()
                # The pinned cache preallocates None slots. Only occupied
                # states are physical continuation; unknown occupied slots
                # cannot be dropped from this complete observation schema.
                for states,required in ((histories,required_conv),(recurrent,required_recurrent)):
                    if states is None and not required:continue
                    if (not isinstance(states,Mapping) or len(states)>1024 or
                            any(type(index) is not int or index<0 for index in states) or
                            {index for index,value in states.items() if value is not None}!=required):
                        raise ValueError('complete pinned Qwen4 occupied state-slot inventory required')
                if kind=='linear_attention':
                    if histories is None or not histories or recurrent is None or not recurrent:
                        raise ValueError('complete publisher Qwen4 recurrent state required')
                    values=(histories[0],recurrent[0])
                else:values=(getattr(bank,'keys',None),getattr(bank,'values',None),getattr(bank,'indexer_keys',None))
            else:
                if bank is None or len(bank)!=(2 if kind=='linear_attention' else 3):
                    raise ValueError('complete candidate Qwen4 state bank required')
                values=bank
            if kind=='linear_attention':
                leaf(layer,'convolution',values[0],(width,c.linear_conv_kernel_dim))
                recurrent=values[1]
                if self.role=='oracle' and self.recurrent_layout=='value-key' and recurrent is not None:
                    recurrent=recurrent.transpose(-1,-2) if hasattr(recurrent,'permute') else recurrent.swapaxes(-1,-2)
                leaf(layer,'recurrent',recurrent,(c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim))
            else:
                for name,value in zip(('key','value'),values[:2]):
                    if value is None or value.ndim!=4 or value.shape[-2]<position:
                        raise ValueError('occupied complete Qwen4 KV state required')
                    leaf(layer,name,value[...,:position,:],(c.num_key_value_heads,position,c.head_dim))
                indexer=values[2]
                if indexer is None or indexer.ndim!=3 or indexer.shape[1]<position:
                    raise ValueError('complete Qwen4 QSA indexer keys required')
                leaf(layer,'indexer',indexer[:,:position],(position,c.indexer_head_dim))
            if layer+1 in c.ple_layer_ids:
                if self.role=='oracle':
                    if histories is None or histories.get(1) is None or histories.get(2) is None:
                        raise ValueError('publisher PLE histories missing')
                    tokens,convolution=histories[2],histories[1]
                else:
                    if lexical[layer] is None or len(lexical[layer])!=2:raise ValueError('candidate PLE histories missing')
                    tokens,convolution=lexical[layer]
                if tokens is None or str(tokens.dtype) not in ('torch.int64','int64'):
                    raise ValueError('Qwen4 PLE history must preserve exact int64 token IDs')
                leaf(layer,'ple_tokens',tokens,(c.ngram_size-1,))
                leaf(layer,'ple_convolution',convolution,(c.hc_count*c.hidden_size,(c.ple_conv_kernel_size-1)*c.ngram_size))
            elif self.role=='candidate' and lexical[layer] is not None:
                raise ValueError('unexpected Qwen4 lexical cache owner')
        if self.media:
            mask,positions=result['mask'],result['positions']
            if (tuple(mask.shape)!=(batch,position) or tuple(positions.shape)!=(3,batch,position) or
                    str(positions.dtype) not in ('int64','torch.int64') or
                    str(mask.dtype) not in ('uint8','bool','torch.uint8','torch.bool')):
                raise ValueError('complete observed Qwen4 mask and three-axis rotary geometry required')
            if (mask.numel() if hasattr(mask,'numel') else mask.size)*4+int(np.prod(positions.shape,dtype=object))*24>self.max_bytes:
                raise MemoryError('Qwen4 media auxiliary export exceeds admission')
            dm,dp=detached(mask),detached(positions)
            if np.any(dm!=1) or np.any(dp<0) or np.any(dp>=c.max_position_embeddings):
                raise ValueError('occupied unpadded media mask and bounded actual rotary positions required')
            result['mask']=dm.astype(np.uint8);result['positions']=dp
        total=0
        for value in result.values():
            width=4 if str(value.dtype)=='torch.bfloat16' else (value.element_size() if hasattr(value,'element_size') else value.dtype.itemsize)
            total+=int(np.prod(value.shape,dtype=object))*width
        if 3*total>self.max_bytes:raise MemoryError('complete Qwen4 cache export exceeds admission')
        if set(result)!=set(self.schema):raise ValueError('complete Qwen4 cache schema required')
        if self.role=='candidate' and batch!=mask.shape[0]:
            raise ValueError('Qwen4 state and auxiliary continuation batches differ')
        exported={name:detached(value) for name,value in result.items()}
        if any(not np.isfinite(value).all() for value in exported.values()):
            raise ValueError('nonfinite complete Qwen4 physical observation')
        return exported


def make_reader(*,schema,options):
    options=dict(options)
    if type(options.get('auxiliary_reader')) is str:
        from .adapters import resolve_callable
        options['auxiliary_reader']=resolve_callable(options['auxiliary_reader'])
    reader=Qwen4PublisherCacheReader(**options)
    if dict(schema)!=reader.schema:raise ValueError('complete declared Qwen4 cache schema differs')
    return reader


__all__=['contract','schema_for','Qwen4PublisherCacheReader','make_reader']
