"""Explicit GLM-MoE-DSA accelerator owner with physical latent recovery.

Resident FP32 projections run on the requested device. The flattened cache
includes every latent, rotary and full-indexer bank plus the exact prefix mask.
No reference package, CPU arithmetic fallback or automatic compilation is used.
"""
import hashlib
import importlib
import json
import math
import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from .independent_glm_dsa import IndependentGlmDsaDecoder, yarn_magnitude
from .torch_inkling_decoder import TorchInklingDecoder
from .device_logits import device_logits
from .device_execution_policy import admit_no_fallback,admit_fp32

_SOURCE_ADMISSION = object()


class TorchGlmDsaDecoder(TorchInklingDecoder):
    implementation = 'torch-glm-moe-dsa-device-fp32'
    cache_magic = b'OATDSA01'

    def __init__(self, config, tensors, *, device, max_bytes, max_snapshots=4,
                 max_work_bytes=64*1024**2, max_head_bytes=64*1024**2,expert_provider=None,
                 _source_admission=None):
        if _source_admission is not _SOURCE_ADMISSION:
            raise ValueError('GLM-DSA device construction requires metadata-admitted source tensors')
        if expert_provider is not None:
            raise ValueError('GLM-DSA external experts are disabled until packed/source tensor content identity is bound')
        admit_no_fallback(device)
        inventory = IndependentGlmDsaDecoder(config, {}, _inventory_only=True)
        c = inventory.config
        if any(type(v) is not int or v <= 0 for v in (max_bytes, max_work_bytes, max_head_bytes)):
            raise ValueError('explicit GLM DSA device allocation budgets required')
        if type(max_snapshots) is not int or not 1 <= max_snapshots <= 64:
            raise ValueError('bounded GLM DSA snapshot inventory required')
        if type(device) is not str or (device not in ('cuda', 'mps') and not
                (device.startswith('cuda:') and device[5:].isdigit())):
            raise ValueError('explicit GLM DSA CUDA/HIP or Metal device required')
        supplied = dict(tensors)
        shapes=dict(inventory.tensor_shapes)
        self.expert_provider=expert_provider
        self._expert_identity=None
        provider_bytes=0
        if expert_provider is not None:
            from .device_glm_dsa_experts import DeviceGlmDsaExperts
            from .torch_glm_dsa_experts import TorchGlmDsaExperts
            if type(expert_provider) not in (DeviceGlmDsaExperts,TorchGlmDsaExperts) or vars(expert_provider.contract)!=vars(c):
                raise ValueError('exact original DSA device expert owner required')
            shapes={name:shape for name,shape in shapes.items()
                    if not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
            provider_bytes=expert_provider.max_bytes
            self._expert_identity=expert_provider.identity
        if c.tie_word_embeddings:
            embedding = supplied.get('model.embed_tokens.weight')
            if 'lm_head.weight' in supplied and not np.array_equal(supplied['lm_head.weight'], embedding):
                raise ValueError('GLM DSA tied head differs from embedding')
            supplied['lm_head.weight'] = embedding
        if set(supplied) != set(shapes):
            raise ValueError('complete exact resident GLM DSA inventory required')
        resident_bytes=sum(math.prod(shape)*4 for shape in shapes.values())
        self.weight_bytes=resident_bytes+provider_bytes
        if 3*resident_bytes+provider_bytes+max_work_bytes+max_head_bytes > max_bytes:
            raise MemoryError('GLM DSA source/device construction exceeds admission')
        admitted = {}
        digest = hashlib.sha256(json.dumps(vars(c), sort_keys=True, allow_nan=False).encode())
        if expert_provider is not None:digest.update(expert_provider.identity.encode())
        for name, shape in sorted(shapes.items()):
            value = np.asarray(supplied[name])
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError('invalid GLM DSA device tensor: '+name)
            admitted[name] = np.array(value, copy=True, order='C')
            digest.update(name.encode()); digest.update(memoryview(admitted[name]).cast('B'))
        torch = importlib.import_module('torch')
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('requested GLM DSA CUDA/HIP device unavailable')
            index = torch.cuda.current_device() if device == 'cuda' else int(device.split(':')[1])
            if index >= torch.cuda.device_count():
                raise ValueError('GLM DSA device index unavailable')
            device = 'cuda:'+str(index)
        elif not torch.backends.mps.is_available():
            raise RuntimeError('requested GLM DSA Metal device unavailable')
        self.torch, self.device, self.contract = torch, device, c
        if expert_provider is not None and hasattr(expert_provider,'bank') and (expert_provider.bank.torch is not torch
                or expert_provider.bank.device!=torch.device(device)):
            raise ValueError('DSA device expert owner is on a different runtime/device')
        self.config = dict(vars(c))
        self._configuration_identity=hashlib.sha256(json.dumps(vars(c),sort_keys=True,allow_nan=False).encode()).digest()
        self.weights = {name: torch.tensor(value, device=device, dtype=torch.float32)
                        for name, value in admitted.items()}
        # PyTorch reports CUDA and ROCm devices through the same ``cuda:N``
        # namespace. Bind cache/recovery identity to the linked runtime too;
        # otherwise snapshots from CUDA and HIP could be mistaken for the
        # same selected-device execution owner.
        version = getattr(torch, 'version', None)
        self.device_runtime = dict(
            cuda=getattr(version, 'cuda', None),
            hip=getattr(version, 'hip', None),
            mps=getattr(version, 'mps', None),
        )
        digest.update(repr((device, torch.__version__, self.device_runtime,
                            self.implementation)).encode())
        self.identity = 'openagent-torch-glm-dsa-v1:'+digest.hexdigest()
        self.geometry = tuple(c.indexer_types)+('mask',)
        self.max_bytes, self.max_snapshots = max_bytes, max_snapshots
        self.work_bytes, self.head_bytes = max_work_bytes, max_head_bytes
        self._lock, self._local = threading.RLock(), threading.local()
        self._owner, self._snapshots = uuid4().hex, {}
        self._closed = self._busy = False
        self._snapshot_bytes, self.position, self._cache = 0, 0, ()

    def _check(self):
        super()._check()
        if self.expert_provider is not None and (self.expert_provider.identity!=self._expert_identity
                or getattr(self.expert_provider,'_closed',False) or vars(self.expert_provider.contract)!=vars(self.contract)):
            raise ValueError('DSA selected expert owner changed after admission')
        if (self.config!=vars(self.contract) or hashlib.sha256(json.dumps(vars(self.contract),
                sort_keys=True,allow_nan=False).encode()).digest()!=self._configuration_identity):
            raise ValueError('GLM DSA configuration changed after device admission')

    def _state_size(self, kind):
        c = self.contract
        if kind == 'mask':
            return c.max_position_embeddings
        return c.max_position_embeddings*(c.kv_lora_rank+c.qk_rope_head_dim+
                                           (c.index_head_dim if kind == 'full' else 0))

    def _validate_recovered_state(self, banks, position):
        if not banks:
            if position:raise ValueError('GLM DSA nonempty position without physical state')
            return
        if len(banks)!=len(self.geometry):
            raise ValueError('GLM DSA recovery requires every physical bank')
        mask=banks[-1]
        for row in mask:
            for begin in range(0,row.size,262144):
                values=row[begin:begin+262144]
                if np.any((values!=0)&(values!=1)):
                    raise ValueError('GLM DSA recovered mask is not binary')
            for begin in range(position,row.size,262144):
                if np.any(row[begin:begin+262144]!=0):
                    raise ValueError('GLM DSA recovered mask extends beyond its active prefix')
        for bank,kind in zip(banks[:-1],self.geometry[:-1]):
            width=self._state_size(kind)//self.contract.max_position_embeddings
            for row in bank:
                for begin in range(position*width,row.size,262144):
                    if np.any(row[begin:begin+262144]!=0):
                        raise ValueError('GLM DSA recovered bank extends beyond its active prefix')

    def export_cache(self, *, max_bytes):
        """Detach every active physical leaf for caller-pinned comparisons.

        This method observes stored latent/rotary/indexer values directly.
        It performs no expansion, rounding, model replay or certification.
        """
        with self._lock:
            self._check();self._cancel()
            if type(max_bytes) is not int or max_bytes<=0:
                raise ValueError('explicit GLM DSA cache observation budget required')
            c=self.contract;batch=self._cache[0].shape[0] if self._cache else 0
            payload=8+batch*self.position
            for kind in self.geometry[:-1]:
                payload+=batch*self.position*(self._state_size(kind)//c.max_position_embeddings)*4
            if 3*payload>max_bytes:
                raise MemoryError('GLM DSA detached cache observation exceeds admission')
            self._admit(3*payload);self._busy=True
            try:
                result={'position':np.asarray([self.position],dtype=np.int64)}
                result['mask']=(self._cache[-1][:,:self.position].detach().cpu().numpy().astype(np.uint8)
                    if self._cache else np.empty((0,0),dtype=np.uint8))
                for layer,kind in enumerate(self.geometry[:-1]):
                    self._cancel();width=self._state_size(kind)//c.max_position_embeddings
                    values=(self._cache[layer].reshape(batch,c.max_position_embeddings,width)[:,:self.position]
                        .detach().cpu().numpy().copy() if self._cache else
                        np.empty((0,0,width),dtype=np.float32))
                    first=c.kv_lora_rank;last=first+c.qk_rope_head_dim
                    result[f'layer.{layer}.latent']=values[...,:first].copy()
                    result[f'layer.{layer}.rotary']=values[...,first:last].copy()
                    if kind=='full':result[f'layer.{layer}.indexer']=values[...,last:].copy()
                self._cancel()
                return result
            finally:
                self._busy=False

    @classmethod
    def from_directory(cls, directory, *, device, max_weight_bytes, max_bytes,
                       read_limits=None, cancel=None, **options):
        if options.get('expert_provider') is not None:
            raise ValueError(
                'GLM-DSA packed experts lack checkpoint source-content identity; '
                'device directory loading is restricted')
        from .native_checkpoint import read_native_text_checkpoint, _finish_decoder
        def select(config, records):
            inventory = IndependentGlmDsaDecoder(config, {}, _inventory_only=True)
            shapes = dict(inventory.tensor_shapes)
            if inventory.config.tie_word_embeddings and 'lm_head.weight' not in records:
                shapes.pop('lm_head.weight')
            if set(shapes) != set(records) or any(tuple(records[name]['shape']) != shape or
                                                  records[name].get('dtype') not in ('F32','F16','BF16')
                                                  for name, shape in shapes.items()):
                raise ValueError('exact complete GLM DSA device checkpoint required')
            from .source_scope import require_source_inventory_admission
            require_source_inventory_admission(
                family='glm-dsa',model_type='glm_moe_dsa',backend='device',
                arithmetic='fp32',persisted_format='float32',inventory=records)
            expert=options.get('expert_provider')
            if expert is not None:
                raise ValueError('GLM-DSA external experts are disabled until packed/source tensor content identity is bound')
            if 4*sum(math.prod(shape)*4 for shape in shapes.values()) > max_bytes:
                raise MemoryError('GLM DSA checkpoint conversion exceeds admission')
            return tuple(sorted(shapes))
        config, weights = read_native_text_checkpoint(directory, model_type='glm_moe_dsa',
            max_weight_bytes=max_weight_bytes, read_limits=read_limits, cancel=cancel,
            tensor_selection=select,source_backend='device')
        return _finish_decoder(cls(config, weights, device=device, max_bytes=max_bytes,
            _source_admission=_SOURCE_ADMISSION, **options), cancel)

    def _rotary(self, value, positions):
        torch, rope = self.torch, self.contract.rope_parameters
        width = value.shape[-1]
        lane = torch.arange(width//2, dtype=torch.float32, device=self.device)
        base = float(rope['rope_theta'])**(2*lane/width)
        frequency, amplitude = 1/base, 1.
        if rope['rope_type'] == 'yarn':
            bounds = [width*math.log(rope['original_max_position_embeddings']/(2*math.pi*rope[name]))/
                      (2*math.log(rope['rope_theta'])) for name in ('beta_fast', 'beta_slow')]
            if rope['truncate']:
                bounds = [math.floor(bounds[0]), math.ceil(bounds[1])]
            low, high = max(bounds[0], 0), min(bounds[1], width-1)
            if low == high:
                high += .001
            ramp = ((lane-low)/(high-low)).clamp(0, 1)
            frequency = (ramp/rope['factor']+1-ramp)/base
            amplitude = rope.get('attention_factor')
            if amplitude is None:
                amplitude = (yarn_magnitude(rope['factor'], rope['mscale'])/
                    yarn_magnitude(rope['factor'], rope['mscale_all_dim'])
                    if rope['mscale'] and rope['mscale_all_dim'] else yarn_magnitude(rope['factor']))
        angles = positions.to(dtype=torch.float32)[..., None, None]*frequency
        cosine, sine = angles.cos()*amplitude, angles.sin()*amplitude
        pairs = value.reshape(*value.shape[:-1], width//2, 2)
        return torch.cat((pairs[..., 0]*cosine-pairs[..., 1]*sine,
                          pairs[..., 1]*cosine+pairs[..., 0]*sine), dim=-1)

    def forward(self, token_ids, *, attention_mask=None, position_ids=None, use_cache=True):
        admit_fp32(self.torch,self.device)
        with self._lock:
            self._check(); self._cancel()
            torch, c, w = self.torch, self.contract, self.weights
            ids = np.asarray(token_ids)
            if ids.ndim == 1:
                ids = ids[None, :]
            if (ids.ndim != 2 or not all(ids.shape) or ids.dtype.kind not in 'iu' or
                    ids.shape[0] > 128 or ids.shape[1] > 4096 or
                    np.any(ids < 0) or np.any(ids >= c.vocab_size) or type(use_cache) is not bool):
                raise ValueError('bounded GLM DSA integer token batch required')
            batch, count = ids.shape
            start = self.position if use_cache else 0
            end = start+count
            if end > c.max_position_embeddings or (use_cache and self._cache and self._cache[0].shape[0] != batch):
                raise ValueError('GLM DSA continuation capacity or batch differs')
            self._admit(batch*end*8+ids.size*24)
            observed = np.ones((batch, end), dtype=np.uint8) if attention_mask is None else np.asarray(attention_mask)
            if (observed.shape != (batch, end) or observed.dtype.kind not in 'biu' or
                    np.any((observed != 0) & (observed != 1))):
                raise ValueError('exact full-prefix GLM DSA binary mask required')
            mask = np.array(observed, dtype=np.uint8, copy=True)
            positions = (np.broadcast_to(np.arange(start, end), ids.shape) if position_ids is None
                         else np.asarray(position_ids))
            if (positions.shape != ids.shape or positions.dtype.kind not in 'iu' or
                    np.any(positions < 0) or np.any(positions >= c.max_position_embeddings)):
                raise ValueError('bounded GLM DSA rotary positions required')
            # Query/key scores are formed one query and one head at a time.
            scratch = (batch*count*(8*c.hidden_size+2*c.q_lora_rank+
                c.num_attention_heads*(c.qk_nope_head_dim+c.qk_rope_head_dim+c.v_head_dim)+
                2*c.index_n_heads*c.index_head_dim+c.kv_lora_rank+2*c.qk_rope_head_dim+
                4*max(c.intermediate_size,c.moe_intermediate_size*c.n_shared_experts))+
                end*(4*c.kv_lora_rank+4*c.index_head_dim+16)+
                8*max(c.intermediate_size,c.moe_intermediate_size*c.n_shared_experts)+
                8*c.n_routed_experts)*4+batch*count*min(c.index_topk,end)*8
            if scratch > self.work_bytes:
                raise MemoryError('GLM DSA device query/layer scratch exceeds admission')
            state = batch*sum(self._state_size(kind) for kind in self.geometry)*4
            output = batch*count*(c.hidden_size+c.vocab_size)*4
            self._admit(2*state+2*output+self.work_bytes+self.head_bytes+ids.size*24+mask.size*8)
            self._busy = True
            try:
                with torch.inference_mode():
                    incoming = torch.tensor(ids.astype(np.int64), device=self.device)
                    physical = torch.tensor(positions.astype(np.int64), device=self.device)
                    occupied = torch.tensor(mask, device=self.device, dtype=torch.bool)
                    if use_cache and self._cache and not bool(torch.equal(self._cache[-1][:, :start], occupied[:, :start].float())):
                        raise ValueError('GLM DSA committed mask changed')
                    pending = (tuple(bank.clone() for bank in self._cache) if use_cache and self._cache else
                        tuple(torch.zeros((batch, self._state_size(kind)), device=self.device,
                                          dtype=torch.float32) for kind in self.geometry))
                    pending[-1][:, :end] = occupied.float()
                    x = w['model.embed_tokens.weight'][incoming]
                    def rms(value, weight, eps):
                        return value*torch.rsqrt(value.square().mean(dim=-1,keepdim=True)+eps)*weight
                    def mlp(value, prefix):
                        return (torch.nn.functional.silu(value@w[prefix+'gate_proj.weight'].T)*
                                (value@w[prefix+'up_proj.weight'].T))@w[prefix+'down_proj.weight'].T
                    selected = None
                    multiplier = (yarn_magnitude(c.rope_parameters['factor'],c.rope_parameters['mscale_all_dim'])**2
                                  if c.rope_parameters['rope_type'] == 'yarn' else 1.)
                    for layer in range(c.num_hidden_layers):
                        self._cancel(); p = f'model.layers.{layer}.'; a = p+'self_attn.'
                        z = rms(x,w[p+'input_layernorm.weight'],c.rms_norm_eps)
                        residual = rms(z@w[a+'q_a_proj.weight'].T,w[a+'q_a_layernorm.weight'],1e-6)
                        query = (residual@w[a+'q_b_proj.weight'].T).reshape(batch,count,c.num_attention_heads,-1)
                        qn, qr = query[..., :c.qk_nope_head_dim], self._rotary(query[..., c.qk_nope_head_dim:], physical)
                        raw = z@w[a+'kv_a_proj_with_mqa.weight'].T
                        latent = rms(raw[..., :c.kv_lora_rank],w[a+'kv_a_layernorm.weight'],1e-6)
                        rotary = self._rotary(raw[..., c.kv_lora_rank:].unsqueeze(2),physical).squeeze(2)
                        bank = pending[layer].reshape(batch,c.max_position_embeddings,-1)
                        bank[:,start:end,:c.kv_lora_rank] = latent
                        bank[:,start:end,c.kv_lora_rank:c.kv_lora_rank+c.qk_rope_head_dim] = rotary
                        latent = bank[:,:end,:c.kv_lora_rank]
                        rotary = bank[:,:end,c.kv_lora_rank:c.kv_lora_rank+c.qk_rope_head_dim]
                        if c.indexer_types[layer] == 'full':
                            ip = a+'indexer.'
                            iq = (residual@w[ip+'wq_b.weight'].T).reshape(batch,count,c.index_n_heads,c.index_head_dim)
                            iq = torch.cat((self._rotary(iq[...,:c.qk_rope_head_dim],physical),iq[...,c.qk_rope_head_dim:]),dim=-1)
                            ik = z@w[ip+'wk.weight'].T; centered = ik-ik.mean(dim=-1,keepdim=True)
                            ik = centered*torch.rsqrt(centered.square().mean(dim=-1,keepdim=True)+1e-6)*w[ip+'k_norm.weight']+w[ip+'k_norm.bias']
                            ik = torch.cat((self._rotary(ik[...,:c.qk_rope_head_dim].unsqueeze(2),physical).squeeze(2),ik[...,c.qk_rope_head_dim:]),dim=-1)
                            bank[:,start:end,c.kv_lora_rank+c.qk_rope_head_dim:] = ik
                            keys = bank[:,:end,c.kv_lora_rank+c.qk_rope_head_dim:]
                            head_weights = (z@w[ip+'weights_proj.weight'].T)/math.sqrt(c.index_n_heads)
                            selected = torch.empty((batch,count,min(c.index_topk,end)),device=self.device,dtype=torch.int64)
                            for row in range(batch):
                                for step in range(count):
                                    self._cancel(); score = torch.zeros(end,device=self.device,dtype=torch.float32)
                                    for head in range(c.index_n_heads):
                                        score += ((keys[row]@iq[row,step,head])/math.sqrt(c.index_head_dim)).clamp_min(0)*head_weights[row,step,head]
                                    if not bool(torch.isfinite(score).all()):
                                        raise ArithmeticError('nonfinite GLM DSA index scores')
                                    allowed = occupied[row] & (torch.arange(end,device=self.device) <= start+step)
                                    score = score.masked_fill(~allowed,-torch.finfo(torch.float32).max)
                                    selected[row,step] = torch.argsort(score,descending=True,stable=True)[:selected.shape[-1]]
                        if selected is None:
                            raise ValueError('GLM DSA shared indexer lacks preceding selection')
                        projection = w[a+'kv_b_proj.weight'].reshape(c.num_attention_heads,c.qk_nope_head_dim+c.v_head_dim,c.kv_lora_rank)
                        core = torch.empty((batch,count,c.num_attention_heads,c.v_head_dim),device=self.device,dtype=torch.float32)
                        for row in range(batch):
                            for step in range(count):
                                sparse = torch.zeros(end,device=self.device,dtype=torch.bool); sparse[selected[row,step]] = True
                                allowed = sparse & occupied[row] & (torch.arange(end,device=self.device) <= start+step)
                                for head in range(c.num_attention_heads):
                                    self._cancel(); lq = qn[row,step,head]@projection[head,:c.qk_nope_head_dim]
                                    score = (latent[row]@lq+rotary[row]@qr[row,step,head])*multiplier/math.sqrt(c.qk_nope_head_dim+c.qk_rope_head_dim)
                                    if not bool(torch.isfinite(score).all()):
                                        raise ArithmeticError('nonfinite GLM DSA attention scores')
                                    prob = score.masked_fill(~allowed,-torch.finfo(torch.float32).max).softmax(dim=0)
                                    core[row,step,head] = (prob@latent[row])@projection[head,c.qk_nope_head_dim:].T
                        x = x+core.reshape(batch,count,-1)@w[a+'o_proj.weight'].T
                        z = rms(x,w[p+'post_attention_layernorm.weight'],c.rms_norm_eps)
                        if c.mlp_layer_types[layer] == 'dense':
                            x = x+mlp(z,p+'mlp.')
                        else:
                            result = mlp(z,p+'mlp.shared_experts.')
                            for row in range(batch):
                                for step in range(count):
                                    self._cancel(); value=z[row,step]
                                    router=value@w[p+'mlp.gate.weight'].T
                                    if not bool(torch.isfinite(router).all()):
                                        raise ArithmeticError('nonfinite GLM DSA router logits')
                                    scores=router.sigmoid()
                                    choice=scores+w[p+'mlp.gate.e_score_correction_bias']
                                    group=choice.reshape(c.n_group,-1).sort(dim=-1).values[:,-2:].sum(dim=-1)
                                    groups=torch.argsort(group,descending=True,stable=True)[:c.topk_group]
                                    permitted=torch.zeros(c.n_group,device=self.device,dtype=torch.bool);permitted[groups]=True
                                    choice=choice.masked_fill(~permitted.repeat_interleave(c.n_routed_experts//c.n_group),-torch.inf)
                                    experts=torch.argsort(choice,descending=True,stable=True)[:c.num_experts_per_tok]
                                    factors=scores[experts]
                                    if c.norm_topk_prob:factors=factors/(factors.sum()+1e-20)
                                    factors=factors*c.routed_scaling_factor
                                    for rank in torch.argsort(experts):
                                        self._cancel();expert=int(experts[rank])
                                        if self.expert_provider is None:
                                            gate,up=(value@w[p+'mlp.experts.gate_up_proj'][expert].T).chunk(2,dim=-1)
                                            routed=(torch.nn.functional.silu(gate)*up)@w[p+'mlp.experts.down_proj'][expert].T
                                        else:
                                            routed=self.expert_provider.execute(layer,expert,value[None],torch=torch,
                                                device=self.device,cancel=getattr(self._local,'cancel',None))[0]
                                        result[row,step] += routed*factors[rank]
                            x = x+result
                    hidden=rms(x,w['model.norm.weight'],c.rms_norm_eps)
                    logits=device_logits(torch,hidden,w['lm_head.weight'],max_device_bytes=self.head_bytes,
                        max_host_bytes=batch*count*c.vocab_size*4,cancel=getattr(self._local,'cancel',None))
                    if any(not bool(torch.isfinite(bank).all()) for bank in (hidden,*pending)):
                        raise ArithmeticError('nonfinite GLM DSA device continuation')
                    result=SimpleNamespace(logits=logits)
                    self._cancel()
                    if use_cache:self._cache,self.position=pending,end
                    return result
            finally:
                self._busy=False

    def status(self):
        return dict(model_type='glm_moe_dsa',implementation=self.implementation,
                    device=self.device,device_runtime=dict(self.device_runtime),
                    position=self.position,closed=self._closed)

    def close(self):
        with self._lock:
            if self._closed:return
            if self._busy:raise RuntimeError('cannot close active GLM DSA device owner')
            self.weights.clear();self._cache=();self._snapshots.clear()
            self._snapshot_bytes=self.position=0;self._closed=True


__all__ = ['TorchGlmDsaDecoder']
