"""Exact additive Qwen4 native composition descriptor and ABI binding.

The borrowed decoder library owns the function pointers. Callers stage all
cache/history/trace buffers and publish them only after a successful return.
No Python model arithmetic callback is admitted by this binding.
"""
import ctypes as C
from contextlib import ExitStack
import numpy as np
from types import SimpleNamespace
from .native_qwen35 import Config,Tensor


class ComposeLayer(C.Structure):
    _fields_=[('kind',C.c_uint64),('ple_enabled',C.c_uint64),
        ('attention_hyper',Tensor*4),('mlp_hyper',Tensor*4),('attention',Tensor*9),('mlp',Tensor*7),
        ('cache',Tensor*3),('ple_weights',Tensor*7),('ple_history',Tensor),
        ('multipliers',C.POINTER(C.c_int64)),('sizes',C.POINTER(C.c_int64)),('offsets',C.POINTER(C.c_int64)),
        ('previous',C.POINTER(C.c_int64)),('multiplier_count',C.c_uint64),('lane_count',C.c_uint64),
        ('previous_count',C.c_uint64),('ple_table_rows',C.c_uint64),('ple_user',C.c_void_p)]


class ComposeConfig(C.Structure):
    _fields_=[('size',C.c_uint32),('version',C.c_uint32),('text',Config),
        ('layers',C.c_uint64),('streams',C.c_uint64),('rank',C.c_uint64),('index_heads',C.c_uint64),
        ('index_dim',C.c_uint64),('index_budget',C.c_uint64),('index_ratio',C.c_uint64),
        ('ple_embedding',C.c_uint64),('ple_kernel',C.c_uint64),('ple_ngram',C.c_uint64),
        ('ple_lanes',C.c_uint64),('eos',C.c_uint64),('vocabulary',C.c_uint64),('max_work_bytes',C.c_uint64),
        ('rope_scale',C.c_double),('mrope_sections',C.c_int32*3),('mrope_enabled',C.c_uint32),
        ('sigmoid_gate',C.c_uint32),('normalize_topk',C.c_uint32),('finalize',C.c_uint32),
        ('final_hyper',Tensor*3),('head',Tensor)]


class ComposeTrace(C.Structure):
    _fields_=[('router',Tensor),('scores',Tensor),('attention',Tensor),('incoming_hidden',Tensor),
        ('selected',C.POINTER(C.c_int64)),('chosen',C.POINTER(C.c_uint8)),
        ('selected_count',C.c_uint64),('chosen_count',C.c_uint64)]


class NativeQwen4Composition:
    """Borrowed native whole/range operation with exact structure probes."""
    def __init__(self,library):
        self.library=library
        version=library.oa_qwen4_compose_version
        version.argtypes=[];version.restype=C.c_uint64
        if version()!=1:raise ValueError('unsupported complete Qwen4 native composition ABI')
        probe=library.oa_qwen4_compose_layout
        probe.argtypes=[C.c_uint32,C.POINTER(C.c_uint64),C.c_size_t];probe.restype=C.c_size_t
        for number,record in enumerate((ComposeConfig,ComposeLayer,ComposeTrace),1):
            expected=[C.sizeof(record),len(record._fields_)]+[getattr(record,name).offset for name,_ in record._fields_]
            actual=(C.c_uint64*len(expected))()
            if probe(number,actual,len(actual))!=len(expected) or list(actual)!=expected:
                raise ValueError('Qwen4 composition descriptor layout differs: '+str(number))
        operation=library.oa_qwen4_compose
        operation.argtypes=[C.POINTER(ComposeConfig),C.POINTER(ComposeLayer),C.POINTER(ComposeTrace)]+[
            C.c_uint64]*6+[C.POINTER(C.c_int64),C.POINTER(C.c_uint8),C.POINTER(C.c_int64)]+[
            Tensor]*3+[C.c_void_p]*6
        operation.restype=C.c_int
        self.operation=operation

    def configure(self,decoder,*,max_work_bytes):
        """Capture admitted geometry; live pending buffers are supplied separately."""
        from .native_qwen4 import NativeQwen4TextDecoder
        from .native_qwen4_ranges import NativeQwen4Range
        from .native_qwen35 import tensor
        if type(decoder) not in (NativeQwen4TextDecoder,NativeQwen4Range) or decoder.library is not self.library:
            raise ValueError('matching exact native Qwen4 decoder owner required')
        decoder._check_open()
        if type(max_work_bytes) is not int or not 0<max_work_bytes<2**64:
            raise ValueError('positive bounded native composition workspace required')
        c=decoder.config;result=ComposeConfig()
        result.size=C.sizeof(result);result.version=1;result.text=decoder._c
        for destination,source in (('layers','num_hidden_layers'),('streams','hc_count'),('rank','hc_lowrank'),
                ('index_heads','indexer_n_heads'),('index_dim','indexer_head_dim'),('index_budget','indexer_budget'),
                ('index_ratio','indexer_compress_ratio'),('vocabulary','vocab_size')):
            setattr(result,destination,getattr(c,source) or 0)
        if c.ple_layer_ids:
            for destination,source in (('ple_embedding','ple_embed_dim'),('ple_kernel','ple_conv_kernel_size'),
                    ('ple_ngram','ngram_size'),('ple_lanes','heads_per_ngram')):
                setattr(result,destination,getattr(c,source))
            result.eos=c.eos_token_id[0] if isinstance(c.eos_token_id,list) else c.eos_token_id
        result.max_work_bytes=max_work_bytes;result.rope_scale=decoder.rope_scale
        result.mrope_enabled=int(decoder.mrope_section is not None)
        if result.mrope_enabled:result.mrope_sections=(C.c_int32*3)(*decoder.mrope_section)
        result.sigmoid_gate=int(getattr(c,'output_gate_type',None)=='sigmoid')
        result.normalize_topk=int(getattr(c,'norm_topk_prob',True))
        result.finalize=int(not decoder.range_mode)
        if result.finalize:
            prefix='model.hyper_connection_mixer.'
            result.final_hyper=(Tensor*3)(*[tensor(decoder.weights[prefix+name]) for name in
                ('hc_norm.weight','input_mix_weight_down.weight','input_mix_weight_up.weight')])
            result.head=tensor(decoder.weights['lm_head.weight'])
        return result

    def forward_staged(self,decoder,ids,*,attention_mask=None,position_ids=None,
                       incoming_hidden=None,inputs_embeds=None):
        """Execute resident weights in C, committing all continuation together.

        The decoder's request lock and outer rollback transaction must be held.
        Original-streaming stores attach their native callbacks separately;
        Python expert/PLE arithmetic callbacks are not installed here.
        """
        from .native_qwen35 import tensor
        native_bank=getattr(decoder,'_composition_experts',None)
        native_ple=getattr(decoder,'_composition_ple',None)
        if ((decoder._expert_provider is not None and native_bank is None)
                or (decoder._ple_provider is not None and native_ple is None)):
            raise ValueError('native composition requires native store callbacks for streamed banks')
        decoder._check_open();c=decoder.config
        ids=np.asarray(ids)
        if (ids.ndim!=2 or not all(ids.shape) or ids.dtype.kind not in 'iu'
                or np.any(ids<0) or np.any(ids>=c.vocab_size)):
            raise ValueError('occupied original Qwen4 token inventory required')
        batch,tokens=ids.shape;start=decoder.position;end=start+tokens
        decoder._admit_native(batch,tokens,end)
        ids=np.array(ids,dtype=np.int64,order='C',copy=True)
        mask=np.ones((batch,end),np.uint8) if attention_mask is None else np.asarray(attention_mask)
        if mask.shape!=(batch,end) or not np.isin(mask,[0,1]).all():raise ValueError('complete occupied mask required')
        mask=np.array(mask,dtype=np.uint8,order='C',copy=True)
        if decoder.mask is not None and not np.array_equal(decoder.mask,mask[:,:start]):
            raise ValueError('cached batch or attention mask changed')
        positions=np.broadcast_to(np.arange(start,end),ids.shape) if position_ids is None else np.asarray(position_ids)
        if decoder.mrope_section is not None and positions.shape==ids.shape:
            positions=np.broadcast_to(positions,(3,*ids.shape))
        shape=(3,*ids.shape) if decoder.mrope_section is not None else ids.shape
        if (positions.shape!=shape or positions.dtype.kind not in 'iu'
                or np.any(positions<0) or np.any(positions>=c.max_position_embeddings)):
            raise ValueError('bounded original text/MRoPE positions required')
        positions=np.array(positions,dtype=np.int64,order='C',copy=True)
        full_positions=positions if decoder.positions is None else np.concatenate((decoder.positions,positions),axis=-1)
        width=c.hc_count*c.hidden_size
        if decoder.range_mode:
            if inputs_embeds is not None:raise ValueError('media embeddings belong to the input edge')
            incoming=np.asarray(incoming_hidden)
            if incoming.dtype!=np.float32 or incoming.shape!=(batch,tokens,width) or not np.isfinite(incoming).all():
                raise ValueError('occupied complete hyper-channel input required')
            incoming=np.array(incoming,order='C',copy=True)
        else:
            if incoming_hidden is not None:raise ValueError('selected hidden range contract required')
            if inputs_embeds is None:incoming=decoder._embed(ids)
            else:
                embedding=np.asarray(inputs_embeds)
                if embedding.dtype!=np.float32 or embedding.shape!=(batch,tokens,c.hidden_size) or not np.isfinite(embedding).all():
                    raise ValueError('original finite projected media embeddings required')
                incoming=np.tile(embedding,(1,1,c.hc_count))
        layers=(ComposeLayer*c.num_hidden_layers)();traces=(ComposeTrace*c.num_hidden_layers)()
        pending=[None]*c.num_hidden_layers;pending_ple=[None]*c.num_hidden_layers
        arrays=[ids,mask,positions,full_positions,incoming]
        hidden_states=[];routers=[];routes=[];attentions=[];selections=[]
        def allocate(shape,dtype=np.float32,zeros=False):
            value=(np.zeros if zeros else np.empty)(shape,dtype=dtype);arrays.append(value);return value
        def hyper(prefix):
            return (Tensor*4)(*[tensor(decoder.weights[prefix+name]) for name in
                ('hc_norm.weight','input_mix_weight_down.weight','input_mix_weight_up.weight','block_inject_weight.weight')])
        for index in range(decoder.layer_begin,decoder.layer_end):
            layer=layers[index];trace=traces[index];prefix=f'model.layers.{index}.'
            layer.attention_hyper=hyper(prefix+'attn_hyper_connection.')
            layer.mlp_hyper=hyper(prefix+'mlp_hyper_connection.')
            old=decoder.cache[index]
            if c.layer_types[index]=='linear_attention':
                channels=2*c.linear_num_key_heads*c.linear_key_head_dim+c.linear_num_value_heads*c.linear_value_head_dim
                history=allocate((batch,channels,c.linear_conv_kernel_dim),zeros=True)
                recurrent=allocate((batch,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim),zeros=True)
                if old is not None:history[:]=old[0];recurrent[:]=old[1]
                state=(history,recurrent)
                names=('in_proj_qkv.weight','in_proj_z.weight','in_proj_b.weight','in_proj_a.weight',
                    'conv1d.weight','norm.weight','dt_bias','A_log','out_proj.weight')
                bank='linear_attn.'
            else:
                layer.kind=1
                keys=allocate((batch,c.num_key_value_heads,end,c.head_dim),zeros=True)
                values=allocate(keys.shape,zeros=True);raw=allocate((batch,end,c.indexer_head_dim),zeros=True)
                if old is not None:keys[:,:,:start]=old[0];values[:,:,:start]=old[1];raw[:,:start]=old[2]
                state=(keys,values,raw)
                names=('q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight','q_norm.weight',
                    'k_norm.weight','indexer.index_qk_proj.weight','indexer.q_layernorm.weight','indexer.k_layernorm.weight')
                bank='self_attn.'
                attention=allocate((batch,c.num_attention_heads,tokens,end))
                chosen=allocate((batch,1,tokens,end),np.uint8)
                trace.attention=tensor(attention);trace.chosen=chosen.ctypes.data_as(C.POINTER(C.c_uint8))
                trace.chosen_count=chosen.size;attentions.append(attention);selections.append(chosen)
            pending[index]=state
            for slot,value in enumerate(state):layer.cache[slot]=tensor(value)
            layer.attention=(Tensor*9)(*[tensor(decoder.weights[prefix+bank+name]) for name in names])
            layer.mlp=(Tensor*7)(*[Tensor() if native_bank is not None and slot in (1,2)
                else tensor(decoder.weights[prefix+'mlp.'+name]) for slot,name in enumerate(
                ('gate.weight','experts.gate_up_proj','experts.down_proj','shared_expert.gate_proj.weight',
                 'shared_expert.up_proj.weight','shared_expert.down_proj.weight','shared_expert_gate.weight'))])
            router=allocate((batch*tokens,c.num_experts));scores=allocate((batch*tokens,c.num_experts_per_tok))
            selected=allocate(scores.shape,np.int64);prior_hidden=allocate(incoming.shape)
            trace.router=tensor(router);trace.scores=tensor(scores);trace.incoming_hidden=tensor(prior_hidden)
            trace.selected=selected.ctypes.data_as(C.POINTER(C.c_int64));trace.selected_count=selected.size
            routers.append(router);routes.append((selected,scores));hidden_states.append(prior_hidden)
            if index+1 in c.ple_layer_ids:
                layer.ple_enabled=1;old=decoder.ple_cache[index]
                eos=c.eos_token_id[0] if isinstance(c.eos_token_id,list) else c.eos_token_id
                previous=allocate((batch,c.ngram_size-1),np.int64);previous.fill(eos)
                history=allocate((batch,width,(c.ple_conv_kernel_size-1)*c.ngram_size),zeros=True)
                if old is not None:previous[:]=old[0];history[:]=old[1]
                pending_ple[index]=(previous,history)
                layer.previous=previous.ctypes.data_as(C.POINTER(C.c_int64));layer.previous_count=previous.size
                layer.ple_history=tensor(history)
                p=prefix+'ple.'
                layer.ple_weights=(Tensor*7)(*[Tensor() if slot==0 and native_ple is not None
                    else tensor(decoder.weights[p+name]) for slot,name in enumerate(
                    ('ple_embedding.ngram_embedding.weight','key_proj.weight','value_proj.weight','norm_key.weight',
                     'norm_query.weight','norm_conv.weight','conv1d.weight'))])
                if native_ple is not None:layer.ple_table_rows=native_ple.shapes[index][0]
                for field,name in (('multipliers','layer_multipliers'),('sizes','ngram_heads_vocab_sizes'),
                                   ('offsets','ngram_heads_offsets')):
                    value=decoder.weights[p+'ple_embedding.'+name]
                    setattr(layer,field,value.ctypes.data_as(C.POINTER(C.c_int64)))
                layer.multiplier_count=c.ngram_size;layer.lane_count=(c.ngram_size-1)*c.heads_per_ngram
        final=allocate((batch,tokens,width if decoder.range_mode else c.hidden_size))
        logits=None if decoder.range_mode else allocate((batch,tokens,c.vocab_size))
        occupied=sum(value.nbytes for value in {id(value):value for value in arrays}.values())+ids.nbytes
        workspace=decoder.max_bytes-decoder.weight_bytes-decoder._snapshot_bytes-decoder._state_bytes()-occupied
        workspace-=decoder._expert_bytes+decoder._ple_bytes
        if workspace<=0:raise MemoryError('native composition transaction exceeds aggregate admission')
        config=self.configure(decoder,max_work_bytes=workspace)
        cancel_error=[]
        def check(user):
            try:
                callback=getattr(decoder,'_cancelled',None)
                return int(callback is not None and callback())
            except BaseException as error:cancel_error.append(error);return 1
        cancel=C.CFUNCTYPE(C.c_int,C.c_void_p)(check)
        decoder._executing=True
        try:
            with ExitStack() as borrowed:
                expert_function=expert_handle=None
                ple_function=None
                if native_ple is not None:
                    borrowed.enter_context(native_ple._lock)
                    if native_ple.provider_identity!=decoder._ple_identity:
                        raise ValueError('native Qwen4 logical PLE owner changed')
                    ple_function=native_ple.callback
                    for layer in range(decoder.layer_begin,decoder.layer_end):
                        if layer+1 in c.ple_layer_ids:layers[layer].ple_user=native_ple.context(layer)
                if native_bank is not None:
                    borrowed.enter_context(native_bank._lock)
                    if (native_bank.provider_identity!=decoder._expert_identity
                            or native_bank.hidden_size!=c.hidden_size
                            or native_bank.intermediate_size!=c.moe_intermediate_size):
                        raise ValueError('native Qwen4 original expert owner changed')
                    if any((layer,expert) not in native_bank.bindings
                           for layer in range(decoder.layer_begin,decoder.layer_end) for expert in range(c.num_experts)):
                        raise ValueError('complete selected-range native expert inventory required')
                    expert_handle=native_bank.handle
                    expert_function=C.cast(native_bank._runtime.library.oa_original_expert_evaluate_counted64,C.c_void_p)
                status=self.operation(C.byref(config),layers,traces,batch,tokens,start,decoder.layer_begin,decoder.layer_end,
                    ids.ctypes.data_as(C.POINTER(C.c_int64)),mask.ctypes.data_as(C.POINTER(C.c_uint8)),
                    full_positions.ctypes.data_as(C.POINTER(C.c_int64)),tensor(incoming),tensor(final),
                    Tensor() if logits is None else tensor(logits),expert_function,expert_handle,
                    ple_function,None,C.cast(cancel,C.c_void_p),None)
        finally:decoder._executing=False
        if cancel_error:raise InterruptedError('native Qwen4 composition cancellation failed') from cancel_error[0]
        if status==4:raise InterruptedError('native Qwen4 composition cancelled; continuation unchanged')
        decoder._result(status)
        if any(value.dtype.kind=='f' and not np.isfinite(value).all() for value in arrays):
            raise ValueError('nonfinite native Qwen4 complete observation')
        decoder._cancellation_boundary()
        decoder.cache,decoder.ple_cache=pending,pending_ple
        decoder.mask=mask.astype(bool);decoder.positions=full_positions;decoder.position=end
        common=dict(hidden_states=tuple(hidden_states) if decoder.range_mode else tuple([*hidden_states,final]),
            router_logits=tuple(routers),routes=tuple(routes),attentions=tuple(attentions),
            selections=tuple(value.view(np.bool_) for value in selections))
        return SimpleNamespace(hidden=final,**common) if decoder.range_mode else SimpleNamespace(logits=logits,**common)


__all__=['ComposeConfig','ComposeLayer','ComposeTrace','NativeQwen4Composition']
