"""Original KDA equations. This module is not a checkpoint decoder.

Inputs are projected Q/K/V, log(alpha), beta and sigmoid output gates.
No reference implementation is imported, and no model identity is aliased.
"""
import numpy as np
import copy
import threading
from types import MappingProxyType, SimpleNamespace


def kda_equations(q, k, v, log_decay, beta, state):
    """Paper equation (1), evaluated as a dense transition in float64.

    Q/K are already normalized and Q already includes the caller's scale.
    State orientation is [heads, key, value]. Returns output and new state.
    This deliberately forms the transition matrix, unlike the C rank-one path.
    """
    q, k, v, log_decay, beta, state = (
        np.asarray(a, dtype=np.float64) for a in (q, k, v, log_decay, beta, state)
    )
    outputs = []
    state = state.copy()
    for t in range(q.shape[0]):
        for h in range(q.shape[1]):
            transition = (np.eye(k.shape[-1]) - beta[t, h] *
                          np.outer(k[t, h], k[t, h])) @ np.diag(np.exp(log_decay[t, h]))
            state[h] = transition @ state[h] + beta[t, h] * np.outer(k[t, h], v[t, h])
        outputs.append(np.einsum('hk,hkv->hv', q[t], state))
    return np.asarray(outputs), state


def projected_kda_equations(q, k, v, taps, log_decay, beta, gate, norm,
                            state, history, *, epsilon, q_scale, l2_epsilon=None):
    """Causal depthwise convolution, SiLU, L2, KDA, head RMS and sigmoid gate.

    Taps are oldest-to-newest. History stores the preceding kernel-1 raw
    projections, not convolved values. Gate and beta are probabilities.
    """
    l2_epsilon = epsilon if l2_epsilon is None else l2_epsilon
    history = [a.astype(np.float64).copy() for a in history]
    filtered = []
    for stream, weights, past in zip((q, k, v), taps, history):
        rows = []
        for row in stream:
            window = np.concatenate((past, row[None]), axis=0)
            z = np.sum(window * weights.transpose(2, 0, 1), axis=0)
            rows.append(z * np.exp(-np.logaddexp(0., -z)))
            past = window[1:].copy()
        filtered.append(np.asarray(rows))
        history[len(filtered)-1] = past
    for i in (0, 1):
        filtered[i] /= np.sqrt(np.sum(filtered[i]**2, axis=-1, keepdims=True) + l2_epsilon)
    filtered[0] *= q_scale
    out, state = kda_equations(*filtered, log_decay, beta, state)
    out = out / np.sqrt(np.mean(out**2, axis=-1, keepdims=True) + epsilon)
    return out * norm * gate, state, tuple(history)


class IndependentKimiLinearDecoder:
    """Original batch-one resident official Linear subset, token IDs to logits.

    FP32, SiLU, low-rank KDA gates, NoPE MLA, direct Q, single router group.
    K3 extensions, quantized tensors and nontrivial padding are rejected.
    Numerical equation checks are not a vendor-runtime certification.
    """
    implementation = 'original-numpy-kimi-linear'
    full_decoder = True
    vendor_verified = False

    @staticmethod
    def tensor_shapes(config):
        c = SimpleNamespace(**config)
        h, la = c.hidden_size, c.linear_attn_config
        width = la['num_heads']*la['head_dim']
        shapes = {'model.embed_tokens.weight': (c.vocab_size,h),
                  'model.norm.weight': (h,), 'lm_head.weight': (c.vocab_size,h)}
        for i in range(c.num_hidden_layers):
            p = f'model.layers.{i}.'
            shapes[p+'input_layernorm.weight'] = (h,)
            shapes[p+'post_attention_layernorm.weight'] = (h,)
            a = p+'self_attn.'
            if i+1 in la['kda_layers']:
                for name in ('q_proj','k_proj','v_proj'):
                    shapes[a+name+'.weight'] = (width,h)
                for name in ('f_a_proj','g_a_proj'):
                    shapes[a+name+'.weight'] = (la['head_dim'],h)
                for name in ('f_b_proj','g_b_proj'):
                    shapes[a+name+'.weight'] = (width,la['head_dim'])
                for name in ('q_conv1d','k_conv1d','v_conv1d'):
                    shapes[a+name+'.weight'] = (width,1,la['short_conv_kernel_size'])
                shapes.update({a+'b_proj.weight': (la['num_heads'],h),
                    a+'A_log': (1,1,la['num_heads'],1), a+'dt_bias': (width,),
                    a+'o_norm.weight': (la['head_dim'],), a+'o_proj.weight': (h,width)})
            else:
                shapes.update({a+'q_proj.weight': (c.num_attention_heads*(c.qk_nope_head_dim+c.qk_rope_head_dim),h),
                    a+'kv_a_proj_with_mqa.weight': (c.kv_lora_rank+c.qk_rope_head_dim,h),
                    a+'kv_a_layernorm.weight': (c.kv_lora_rank,),
                    a+'kv_b_proj.weight': (c.num_attention_heads*(c.qk_nope_head_dim+c.v_head_dim),c.kv_lora_rank),
                    a+'o_proj.weight': (h,c.num_attention_heads*c.v_head_dim)})
            if i >= c.first_k_dense_replace and i % c.moe_layer_freq == 0:
                b = p+'block_sparse_moe.'
                shapes[b+'gate.weight'] = (c.num_experts,h)
                shapes[b+'gate.e_score_correction_bias'] = (c.num_experts,)
                for e in range(c.num_experts):
                    for name, shape in {'w1': (c.moe_intermediate_size,h),
                        'w3': (c.moe_intermediate_size,h), 'w2': (h,c.moe_intermediate_size)}.items():
                        shapes[b+f'experts.{e}.'+name+'.weight'] = shape
                m = c.moe_intermediate_size*c.num_shared_experts
                for name,shape in {'gate_proj': (m,h),'up_proj': (m,h),'down_proj': (h,m)}.items():
                    shapes[b+'shared_experts.'+name+'.weight'] = shape
            else:
                for name,shape in {'gate_proj': (c.intermediate_size,h),
                    'up_proj': (c.intermediate_size,h),'down_proj': (h,c.intermediate_size)}.items():
                    shapes[p+'mlp.'+name+'.weight'] = shape
        return shapes

    @staticmethod
    def validate_config(config, *, capacity, max_bytes):
        raw = copy.deepcopy(dict(config))
        if raw.get('model_type') != 'kimi_linear' or raw.get('architectures') != ['KimiLinearForCausalLM']:
            raise ValueError('exact official Kimi Linear identity required; K3 is not an alias')
        known = {'model_type','architectures','auto_map','bos_token_id','eos_token_id','pad_token_id',
            'dtype','torch_dtype','initializer_range','head_dim','hidden_act','hidden_size',
            'vocab_size','num_hidden_layers','num_attention_heads','num_key_value_heads',
            'first_k_dense_replace','intermediate_size','kv_lora_rank','q_lora_rank',
            'linear_attn_config','mla_use_nope','model_max_length','max_position_embeddings',
            'moe_intermediate_size','moe_layer_freq','moe_renormalize','moe_router_activation_func',
            'num_expert_group','num_experts','num_experts_per_token','num_nextn_predict_layers',
            'num_shared_experts','qk_nope_head_dim','qk_rope_head_dim','v_head_dim','rms_norm_eps','rope_scaling',
            'rope_theta','routed_scaling_factor','tie_word_embeddings','topk_group',
            'transformers_version','use_cache','use_grouped_topk','attention_dropout',
            '_name_or_path','_attn_implementation'}
        if set(raw)-known:
            raise ValueError(f'unsupported config extensions (including K3): {sorted(set(raw)-known)}')
        for name in ('hidden_size','vocab_size','num_hidden_layers','num_attention_heads','num_key_value_heads',
                     'intermediate_size','kv_lora_rank','moe_intermediate_size','moe_layer_freq',
                     'num_experts','num_experts_per_token','num_shared_experts','qk_nope_head_dim',
                     'qk_rope_head_dim','v_head_dim'):
            if type(raw.get(name)) is not int or raw[name]<=0:
                raise ValueError(f'invalid {name}')
        # v_head_dim is architectural, never inferred from head_dim.
        if type(raw.get('first_k_dense_replace')) is not int or not 0 <= raw['first_k_dense_replace'] <= raw['num_hidden_layers']:
            raise ValueError('invalid first_k_dense_replace')
        if (raw.get('hidden_act') != 'silu' or raw.get('q_lora_rank') is not None or
            raw.get('mla_use_nope') is not True or raw.get('moe_router_activation_func') != 'sigmoid' or
            raw.get('num_expert_group') != 1 or raw.get('topk_group') != 1 or
            raw.get('rope_scaling') is not None or raw.get('num_nextn_predict_layers',0) != 0 or
            raw.get('use_grouped_topk') is not True or type(raw.get('moe_renormalize')) is not bool or
            type(raw.get('tie_word_embeddings')) is not bool or
            raw['num_attention_heads'] != raw['num_key_value_heads'] or
            raw['num_experts']<2 or raw['num_experts_per_token']>raw['num_experts']):
            raise ValueError('unsupported exact Linear variant')
        for name in ('rms_norm_eps','routed_scaling_factor'):
            x = raw.get(name)
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or x<=0:
                raise ValueError(f'invalid {name}')
        la = raw.get('linear_attn_config')
        if not isinstance(la,dict) or set(la) != {'num_heads','head_dim','short_conv_kernel_size','kda_layers','full_attn_layers'}:
            raise ValueError('explicit official low-rank KDA config required')
        for name,bound in (('num_heads',128),('head_dim',256),('short_conv_kernel_size',64)):
            if type(la[name]) is not int or not 0 < la[name] <= bound:
                raise ValueError('KDA geometry exceeds native bounds')
        lists = (la['kda_layers'],la['full_attn_layers'])
        if any(not isinstance(a,list) or any(type(x) is not int for x in a) for a in lists):
            raise ValueError('explicit one-based layer lists required')
        combined = lists[0]+lists[1]
        if sorted(combined) != list(range(1,raw['num_hidden_layers']+1)):
            raise ValueError('layer lists must partition the decoder exactly')
        if (raw['hidden_size']>4096 or raw['num_hidden_layers']>128 or raw['vocab_size']>1048576 or
            raw['num_attention_heads']>128 or raw['qk_nope_head_dim']+raw['qk_rope_head_dim']>512 or
            raw['v_head_dim']>256 or raw['num_experts']>4096 or raw['kv_lora_rank']>65536 or
            raw['intermediate_size']>65536 or raw['moe_intermediate_size']*raw['num_shared_experts']>65536):
            raise ValueError('decoder geometry exceeds bounded CPU subset')
        if type(capacity) is not int or not 0 < capacity <= 65536 or type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('invalid capacity or memory budget')
        limit = raw.get('max_position_embeddings',raw.get('model_max_length'))
        if type(limit) is not int or capacity>limit:
            raise ValueError('capacity exceeds explicit checkpoint context limit')
        return raw

    def __init__(self, config, tensors, *, capacity, max_bytes=256*1024*1024):
        raw = self.validate_config(config,capacity=capacity,max_bytes=max_bytes)
        la = raw['linear_attn_config']
        self.config = c = SimpleNamespace(**raw)
        self.capacity, self.max_bytes = capacity,max_bytes
        shapes = self.tensor_shapes(raw)
        tensors = dict(tensors)
        if c.tie_word_embeddings:
            if 'lm_head.weight' in tensors and not np.array_equal(tensors['lm_head.weight'],tensors.get('model.embed_tokens.weight')):
                raise ValueError('inconsistent tied head')
            tensors['lm_head.weight'] = tensors.get('model.embed_tokens.weight')
        if set(tensors) != set(shapes):
            raise ValueError(f'exact tensor names mismatch: missing={set(shapes)-set(tensors)}, extra={set(tensors)-set(shapes)}')
        weight_bytes = 4*sum(np.prod(shape,dtype=object) for shape in shapes.values())
        self._cache_bytes = 0
        for i in range(c.num_hidden_layers):
            if i+1 in la['kda_layers']:
                self._cache_bytes += 4*la['num_heads']*(la['head_dim']**2+3*(la['short_conv_kernel_size']-1)*la['head_dim'])
            else:
                self._cache_bytes += 4*capacity*c.num_attention_heads*(c.qk_nope_head_dim+c.qk_rope_head_dim+c.v_head_dim)
        self._weight_bytes = int(weight_bytes)
        self._workspace_bytes = 4*64*max(c.hidden_size,c.intermediate_size,
            c.moe_intermediate_size*c.num_shared_experts,c.num_experts,c.num_attention_heads*512,
            la['num_heads']*la['head_dim'],c.kv_lora_rank)+8*capacity
        self._admit(weight_bytes*2+self._cache_bytes+self._workspace_bytes)
        owners = {}
        for name,shape in shapes.items():
            a = np.asarray(tensors[name])
            if a.shape != shape or a.dtype != np.float32 or not a.flags.c_contiguous or not np.isfinite(a).all() or a.size>67108864:
                raise ValueError(f'invalid resident FP32 tensor {name}')
            owners[name] = a.copy()
            owners[name].flags.writeable = False
        self.weights = MappingProxyType(owners)
        self._lock = threading.RLock()
        self.reset()

    def _admit(self, size):
        if size>self.max_bytes:
            raise ValueError('decoder resident plus transaction memory budget exceeded')

    def reset(self):
        with self._lock:
            c,la = self.config,self.config.linear_attn_config
            cache = []
            for i in range(c.num_hidden_layers):
                if i+1 in la['kda_layers']:
                    h,d,w = la['num_heads'],la['head_dim'],la['short_conv_kernel_size']
                    cache.append((np.zeros((h,d,d),np.float32),
                                  *(np.zeros((w-1,h,d),np.float32) for _ in range(3))))
                else:
                    cache.append((np.zeros((self.capacity,c.num_attention_heads,c.qk_nope_head_dim+c.qk_rope_head_dim),np.float32),
                                  np.zeros((self.capacity,c.num_attention_heads,c.v_head_dim),np.float32)))
            self._cache,self._position = cache,0

    @property
    def position(self):
        return self._position

    def snapshot(self):
        with self._lock:
            return self._position,tuple(tuple(a.copy() for a in layer) for layer in self._cache)

    def _mat(self,w,x):
        return (w.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)

    def _vec(self,op,x,w=None,param=1.):
        x = x.astype(np.float64)
        if op==0: y=x/np.sqrt(np.mean(x*x)+param)*w
        elif op==1: y=x+w
        elif op==2: y=x*w
        elif op==3: y=np.exp(-np.logaddexp(0.,-x))
        elif op==4: y=x*np.exp(-np.logaddexp(0.,-x))
        elif op==5: y=np.logaddexp(0.,x)
        elif op==6: y=np.exp(x)
        else: y=x*param
        return np.asarray(y,dtype=np.float32)

    def _attention(self,q,k,v):
        score = np.einsum('hd,thd->ht',q.astype(np.float64),k.astype(np.float64))/np.sqrt(q.shape[-1])
        score = np.exp(score-score.max(axis=-1,keepdims=True))
        score /= score.sum(axis=-1,keepdims=True)
        return np.einsum('ht,thv->hv',score,v).astype(np.float32)

    def _route(self,scores,bias):
        c = self.config
        scores = self._vec(1,scores,bias)
        ids = np.argsort(-scores,kind='stable')[:c.num_experts_per_token]
        selected = scores[ids].astype(np.float64)
        if len(ids)>1 and c.moe_renormalize:
            selected /= selected.sum()+1e-20
        return ids,(selected*c.routed_scaling_factor).astype(np.float32)

    def _kda(self,streams,taps,decay,beta,gate,norm,cache):
        la = self.config.linear_attn_config
        out,state,hist = projected_kda_equations(*streams,taps,decay,beta,gate,norm,
            cache[0],cache[1:],epsilon=self.config.rms_norm_eps,
            l2_epsilon=1e-6,q_scale=la['head_dim']**-.5)
        return out.astype(np.float32),(state.astype(np.float32),*(a.astype(np.float32) for a in hist))

    def _mlp(self,x,p,sparse=False):
        names = ('w1','w3','w2') if sparse else ('gate_proj','up_proj','down_proj')
        gate = self._vec(4,self._mat(self.weights[p+names[0]+'.weight'],x))
        return self._mat(self.weights[p+names[2]+'.weight'],
                         self._vec(2,gate,self._mat(self.weights[p+names[1]+'.weight'],x)))

    def _token(self,token,cache,position):
        c,la,w = self.config,self.config.linear_attn_config,self.weights
        x = w['model.embed_tokens.weight'][token].copy()
        for i in range(c.num_hidden_layers):
            p = f'model.layers.{i}.'; a = p+'self_attn.'
            z = self._vec(0,x,w[p+'input_layernorm.weight'],c.rms_norm_eps)
            if i+1 in la['kda_layers']:
                h,d = la['num_heads'],la['head_dim']
                streams = tuple(self._mat(w[a+name+'_proj.weight'],z).reshape(1,h,d) for name in ('q','k','v'))
                taps = tuple(w[a+name+'_conv1d.weight'].reshape(h,d,-1) for name in ('q','k','v'))
                f = self._mat(w[a+'f_b_proj.weight'],self._mat(w[a+'f_a_proj.weight'],z))
                dt = self._vec(5,self._vec(1,f,w[a+'dt_bias'])).reshape(h,d)
                factors = self._vec(6,w[a+'A_log'].reshape(h))
                decay = np.stack([self._vec(7,dt[j],param=-float(factors[j])) for j in range(h)])[None]
                beta = self._vec(3,self._mat(w[a+'b_proj.weight'],z))[None]
                gate = self._vec(3,self._mat(w[a+'g_b_proj.weight'],self._mat(w[a+'g_a_proj.weight'],z))).reshape(1,h,d)
                mixed,cache[i] = self._kda(streams,taps,decay,beta,gate,w[a+'o_norm.weight'],cache[i])
                att = self._mat(w[a+'o_proj.weight'],mixed.ravel())
            else:
                h,kd,vd = c.num_attention_heads,c.qk_nope_head_dim,c.v_head_dim
                q = self._mat(w[a+'q_proj.weight'],z).reshape(h,-1)
                compressed = self._mat(w[a+'kv_a_proj_with_mqa.weight'],z)
                latent = self._vec(0,compressed[:c.kv_lora_rank].copy(),w[a+'kv_a_layernorm.weight'],1e-6)
                kv = self._mat(w[a+'kv_b_proj.weight'],latent).reshape(h,kd+vd)
                keys,values = cache[i]
                keys[position] = np.concatenate((kv[:,:kd],np.broadcast_to(compressed[c.kv_lora_rank:],(h,c.qk_rope_head_dim))),axis=-1)
                values[position] = kv[:,kd:]
                att = self._mat(w[a+'o_proj.weight'],self._attention(q,keys[:position+1],values[:position+1]).ravel())
            x = self._vec(1,x,att)
            z = self._vec(0,x,w[p+'post_attention_layernorm.weight'],c.rms_norm_eps)
            if i >= c.first_k_dense_replace and i % c.moe_layer_freq == 0:
                b = p+'block_sparse_moe.'
                ids,scores = self._route(self._vec(3,self._mat(w[b+'gate.weight'],z)),w[b+'gate.e_score_correction_bias'])
                ffn = np.zeros(c.hidden_size,np.float32)
                for expert,score in zip(ids,scores):
                    result = self._mlp(z,b+f'experts.{int(expert)}.',True)
                    ffn = self._vec(1,ffn,self._vec(7,result,param=float(score)))
                ffn = self._vec(1,ffn,self._mlp(z,b+'shared_experts.'))
            else:
                ffn = self._mlp(z,p+'mlp.')
            x = self._vec(1,x,ffn)
        return self._mat(w['lm_head.weight'],self._vec(0,x,w['model.norm.weight'],c.rms_norm_eps))

    def forward(self, token_ids):
        with self._lock:
            ids = np.asarray(token_ids)
            if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not 0 < ids.size <= 4096:
                raise ValueError('batch-one nonempty integer token IDs required')
            if (ids>=self.config.vocab_size).any() or (ids<0).any() or self.position+ids.size>self.capacity:
                raise ValueError('token IDs or cache capacity out of range')
            self._admit(self._weight_bytes+2*self._cache_bytes+self._workspace_bytes+
                        ids.size*self.config.vocab_size*8)
            cache = [tuple(a.copy() for a in layer) for layer in self._cache]
            rows = [self._token(int(t),cache,self.position+j) for j,t in enumerate(ids.copy())]
            logits = np.stack(rows)
            if not np.isfinite(logits).all() or any(not np.isfinite(a).all() for layer in cache for a in layer):
                raise ValueError('nonfinite decoder transaction; cache unchanged')
            self._cache,self._position = cache,self.position+len(ids)
            return logits
