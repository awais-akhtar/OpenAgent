"""Isolated vision and DSpark wiring to source-name native publisher arithmetic.

The caller owns the arithmetic libraries and payload providers. Construction
does not build libraries or alter the backbone. Publisher parity is unverified.
"""
from dataclasses import replace
import math

import numpy as np

from .publisher import PublisherV41Execution
from .publisher_arithmetic import NativePublisherV41Arithmetic
from .publisher_mtp import PublisherV41MTP
from .publisher_vision import PublisherV41Vision


class PublisherV41AuxiliaryProvider:
    """Resolve exact loader targets, with only explicit draft norm aliases."""

    def __init__(self, arithmetic):
        if not isinstance(arithmetic, NativePublisherV41Arithmetic):
            raise ValueError('NativePublisherV41Arithmetic required')
        arithmetic._ensure_open()
        self.arithmetic = arithmetic
        self.attention_rows = getattr(arithmetic,'attention_rows',None)
        self.auxiliary_math = getattr(arithmetic,'auxiliary_math',None)
        self.payload = arithmetic.payload
        if self.payload.unsupported:
            raise ValueError('supported publisher payload required')
        self.rules = {}
        for rule in self.payload.contracts:
            if rule.target in self.rules:
                raise ValueError('duplicate auxiliary loader target: '+rule.target)
            self.rules[rule.target] = rule
        self.aliases = {}
        text = self.payload.publisher_config['text_config']
        self.stages = text['num_nextn_predict_layers']
        for stage in range(self.stages):
            prefix = f'v41.mtp.{stage}.'
            for suffix in ('attention.q_norm', 'attention.kv_norm', 'attn_norm', 'ffn_norm'):
                alias = prefix+suffix
                target = alias+'.weight'
                if target in self.rules and alias not in self.rules:
                    self.aliases[alias] = target
        self.identity = arithmetic.identity+':auxiliary-target-adapter-v1'

    def contract(self, name):
        target = self.aliases.get(name, name)
        try:
            return self.rules[target]
        except KeyError:
            raise ValueError('missing auxiliary loader target: '+target) from None

    def tensor(self, name):
        self.arithmetic._ensure_open()
        rule = self.contract(name)
        source = rule.sources[0]
        if source not in self.payload.tensors:
            raise ValueError('auxiliary tensor must be resident: '+source)
        value = np.asarray(self.payload.tensors[source], dtype=np.float32)
        if value.shape != rule.shape or not np.isfinite(value).all():
            raise ValueError('invalid resident auxiliary tensor: '+source)
        # Publisher DSpark contracts wo_a directly after converting to BF16.
        if source.endswith('.wo_a.weight'):
            return self.cast(value.reshape(-1)).reshape(value.shape)
        return value

    def linear(self, name, row):
        return self.arithmetic.linear(self.contract(name).sources[0], row)

    def grouped_linear(self, name, rows, *, groups):
        return self.arithmetic.grouped_linear(self.contract(name).sources[0], rows,
                                              groups=groups)

    def tensor_row(self, name, index):
        self.arithmetic._ensure_open()
        rule = self.contract(name)
        if len(rule.shape) != 2 or type(index) is not int or not 0 <= index < rule.shape[0]:
            raise ValueError('valid indexed auxiliary matrix row required')
        provider=getattr(self.payload,'embedding_provider',None)
        if provider is not None and rule.sources[0] in provider.tensor_shapes:
            return provider.read_row(rule.sources[0],index)
        value = np.asarray(self.payload.tensors[rule.sources[0]])
        if value.shape != rule.shape:
            raise ValueError('auxiliary matrix extent mismatch')
        row = np.array(value[index], dtype=np.float32, copy=True)
        if not np.isfinite(row).all():
            raise ValueError('nonfinite auxiliary matrix row')
        return row

    def cast(self, row):
        return self.arithmetic.cast(row)

    def check_cancelled(self):
        self.arithmetic.check_cancelled()

    def roundtrip(self, row, encoding):
        return self.arithmetic.roundtrip(row, encoding)

    def rotate(self, stage, rows, positions, inverse=False):
        self.arithmetic._ensure_open()
        if type(stage) is not int or not 0 <= stage < self.stages:
            raise ValueError('valid draft stage required')
        p = self.payload.profile
        values = np.array(rows, dtype=np.float32, copy=True)
        positions = np.asarray(positions)
        if (values.ndim not in (2, 3) or values.shape[-1] != p.head_dim or
                positions.shape != (values.shape[0],) or positions.dtype.kind not in 'iu' or
                np.any(positions < 0) or np.any(positions >= p.context) or
                not np.isfinite(values).all()):
            raise ValueError('valid draft rotary rows and admitted positions required')
        if values.nbytes > self.arithmetic.max_bytes:
            raise MemoryError('draft rotary workspace exceeds arithmetic admission')
        rd = p.rotary_dim
        if not rd:
            return values
        theta = self.payload.publisher_config['text_config']['rope_theta']
        # Window-only draft layers use plain RoPE, not compressed-layer YaRN.
        policy = (theta, 0, p.rotary_factor, p.rotary_beta_fast, p.rotary_beta_slow)
        for i, position in enumerate(positions):
            tail = np.ascontiguousarray(values[i, ..., -rd:])
            rotated = self.arithmetic._kernel(17, tail, policy, a=int(position),
                                              b=rd, c=-1 if inverse else 1)
            values[i, ..., -rd:] = rotated.reshape(tail.shape)
        return self.cast(values.reshape(-1)).reshape(values.shape)


class PublisherV41DraftBlock:
    """Publisher HC/FFN callback with draft-only 128-expert, top-3 routing."""

    _row = PublisherV41Execution._row
    _tensor = PublisherV41Execution._tensor
    _linear = PublisherV41Execution._linear
    _cast = PublisherV41Execution._cast
    _norm = PublisherV41Execution._norm
    _mix = PublisherV41Execution._mix
    _coefficients = PublisherV41Execution._coefficients
    _post = PublisherV41Execution._post
    _expert = PublisherV41Execution._expert

    def __init__(self, provider):
        if not isinstance(provider, PublisherV41AuxiliaryProvider):
            raise ValueError('auxiliary publisher provider required')
        self.provider = provider
        text = provider.payload.publisher_config['text_config']
        if (text.get('dspark_n_routed_experts') != 128 or
                text.get('dspark_n_activated_experts') != 3):
            raise ValueError('explicit DSpark 128 routed experts / top-3 configuration required')
        p = provider.payload.profile
        if (p.streams > 8 or p.sinkhorn_iterations > 64 or p.shared_experts != 1 or
                p.scoring_function != 'sqrtsoftplus' or not p.normalize_routed_weights):
            raise ValueError('unsupported native draft HC or routing policy')
        self.profile = replace(p, expert_count=128, selected_experts=3)
        self.max_bytes = provider.arithmetic.max_bytes
        self.temperature = text.get('gate_temp', 1.0)
        if (type(self.temperature) not in (int, float) or
                not math.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError('positive finite draft gate temperature required')
        for stage in range(provider.stages):
            prefix = f'v41.mtp.{stage}.'
            for sublayer in ('attn', 'ffn'):
                for suffix in ('fn', 'base', 'scale'):
                    provider.contract(prefix+f'hc_{sublayer}_{suffix}')
                provider.contract(prefix+sublayer+'_norm')
            rule = provider.contract(prefix+'moe.router')
            if rule.shape != (128, p.hidden_size):
                raise ValueError('draft router must use its 128-expert inventory')
            provider.contract(prefix+'moe.router_bias')
            for expert in range(128):
                for projection in ('gate', 'up', 'down'):
                    provider.contract(prefix+f'moe.experts.{expert}.'+projection)
            for projection in ('w1', 'w3', 'w2'):
                provider.contract(prefix+'moe.shared.'+projection)

    def _kernel(self, *args, **kwargs):
        self.provider.arithmetic._ensure_open()
        return self.provider.arithmetic._kernel(*args, **kwargs)

    def _moe(self, x, stage):
        p = self.profile
        prefix = f'v41.mtp.{stage}.moe.'
        logits = self._linear(prefix+'router', x, p.expert_count)
        route = self._kernel(21, logits, self._tensor(prefix+'router_bias', p.expert_count),
                             (self.temperature, p.routed_scale, 1.0),
                             a=p.expert_count, b=p.selected_experts, c=0,
                             n=2*p.selected_experts)
        # Drafts are text-only: never select router_bias_vl.
        out = self._expert(x, prefix+'shared.', 1.0, shared=True)
        for rank in range(p.selected_experts):
            expert = int(route[rank])
            branch = self._expert(x, prefix+f'experts.{expert}.',
                                  float(route[p.selected_experts+rank]))
            out = self._kernel(2, out, branch)
        return self._cast(out)

    def __call__(self, stage, streams, position, pre_mix, main_x, attention):
        if type(stage) is not int or not 0 <= stage < self.provider.stages:
            raise ValueError('valid draft stage required')
        if type(position) is not int or position <= 0:
            raise ValueError('draft blocks require incremental positive position')
        p = self.profile
        streams = np.asarray(streams, dtype=np.float32)
        if streams.ndim != 3 or streams.shape[1:] != (p.streams, p.hidden_size) or not len(streams):
            raise ValueError('draft token/HC/hidden stream geometry required')
        pre_mix = np.asarray(pre_mix, dtype=np.float32)
        if pre_mix.shape != streams.shape[:2] or not np.isfinite(pre_mix).all() or not np.isfinite(streams).all():
            raise ValueError('finite incoming draft streams and pre coefficients required')
        if 8*streams.nbytes > self.max_bytes:
            raise MemoryError('draft HC workspace exceeds arithmetic admission')
        prefix = f'v41.mtp.{stage}.'
        residuals = [self._row(row, p.streams*p.hidden_size) for row in streams]
        attn_coeff = [self._coefficients(row, prefix+'hc_attn') for row in residuals]
        inputs = np.stack([self._norm(self._mix(row, mix), prefix+'attn_norm')
                           for row, mix in zip(residuals, pre_mix)])
        # Exactly one attention call per stage preserves DSpark's window update.
        branches = np.asarray(attention(stage, inputs, position, main_x), dtype=np.float32)
        if branches.shape != (len(streams), p.hidden_size) or not np.isfinite(branches).all():
            raise ValueError('invalid draft attention branch')
        output, next_pre = [], []
        for residual, coefficients, branch in zip(residuals, attn_coeff, branches):
            updated = self._post(residual, coefficients, branch)
            ffn_coeff = self._coefficients(updated, prefix+'hc_ffn')
            # Attention uses incoming pre; FFN uses this attention's pre.
            ffn_input = self._norm(self._mix(updated, coefficients[:p.streams]), prefix+'ffn_norm')
            output.append(self._post(updated, ffn_coeff, self._moe(ffn_input, stage)))
            next_pre.append(ffn_coeff[:p.streams])
        return (np.stack(output).reshape(streams.shape), np.stack(next_pre))


class PublisherV41Auxiliary:
    """Construct isolated vision/MTP executors without taking provider ownership."""

    published_checkpoint_supported = False

    def __init__(self, arithmetic):
        self.provider = PublisherV41AuxiliaryProvider(arithmetic)

    def vision(self, *, max_bytes=64*1024**2):
        config = self.provider.payload.publisher_config
        if config.get('vision_config') is None:
            raise ValueError('payload has no publisher vision configuration')
        vision_type=PublisherV41Vision
        if callable(self.provider.auxiliary_math):
            from .device_vision import DevicePublisherV41Vision
            vision_type=DevicePublisherV41Vision
        return vision_type(config['vision_config'],
                                  self.provider.payload.profile.hidden_size,
                                  self.provider, max_bytes=max_bytes)

    def mtp(self, *, rng=None, max_bytes=64*1024**2):
        return PublisherV41MTP(self.provider.payload.publisher_config['text_config'],
                               self.provider, PublisherV41DraftBlock(self.provider),
                               rng=rng, max_bytes=max_bytes)
