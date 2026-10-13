"""Separate DSpark/MTP equations for pinned V4.1 model.py.

Revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027. This is not a speculative
accept/reject decoder. Provider uses loader target names and owns packed GEMMs,
cast, roundtrip and rotate(stage, rows, positions, inverse=False). Draft blocks
are supplied explicitly: block(stage, streams, position, pre_mix, main_x,
attention) -> (streams, next_pre_mix), with publisher HC and draft MoE math.
`publisher_auxiliary` supplies the executed draft blocks and `dspark` binds
observed target evidence to `SpeculativeDecoder` verification and recovery.
These source paths do not establish numerical parity.
"""
import copy
import math
import threading
from functools import wraps
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
from .auxiliary_attention import attention_rows
from .owner_scope import arithmetic_scope


def _exclusive(operation):
    @wraps(operation)
    def owned(self, *args, **kwargs):
        with self._lock:
            if getattr(self, '_busy', False) or getattr(self, '_cache_transport', False):
                raise RuntimeError('MTP transaction owns continuation state')
            self._busy = True
            try:
                scope=(arithmetic_scope(self.provider) if operation.__name__ in
                       ('seed_window','attention','forward_spec') else nullcontext())
                with scope:
                    return operation(self, *args, **kwargs)
            finally:
                self._busy = False
    return owned


@dataclass(frozen=True)
class PublisherV41MTPSnapshot:
    identity: str
    position: object
    windows: tuple
    rng_state: dict


class PublisherV41MTP:
    published_checkpoint_supported = False

    def __init__(self, text_config, provider, block, *, rng=None, max_bytes=64*1024**2):
        self.config, self.provider, self.block = dict(text_config), provider, block
        for name in ('tensor', 'linear', 'cast', 'roundtrip', 'rotate'):
            if not callable(getattr(provider, name, None)):
                raise ValueError('MTP provider missing '+name)
        if not callable(block):
            raise ValueError('explicit publisher draft HC/MoE block required')
        for key in ('hidden_size', 'vocab_size', 'hc_mult', 'num_nextn_predict_layers',
                    'dspark_block_size', 'dspark_markov_rank', 'num_attention_heads',
                    'head_dim', 'o_groups', 'o_lora_rank', 'q_lora_rank', 'sliding_window'):
            if type(self.config.get(key)) is not int or self.config[key] <= 0:
                raise ValueError('positive MTP geometry required: '+key)
        c = self.config
        self.h, self.vocab, self.hc = c['hidden_size'], c['vocab_size'], c['hc_mult']
        self.context = c.get('max_position_embeddings')
        if type(self.context) is not int or self.context <= 0:
            raise ValueError('positive MTP publisher context required')
        profile = getattr(getattr(provider, 'payload', None), 'profile', None)
        if profile is not None:
            self.context = min(self.context, profile.context)
        self.stages, self.size = c['num_nextn_predict_layers'], c['dspark_block_size']
        self.targets = c.get('dspark_target_layer_ids')
        if not isinstance(self.targets, tuple) or not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError('explicit unique DSpark target inventory required')
        n = c.get('num_hidden_layers')
        ratios = c.get('compress_ratios')
        if (type(n) is not int or n <= 0 or
                any(type(i) is not int or not 0 <= i < n for i in self.targets) or
                not isinstance(ratios, tuple) or len(ratios) != n+self.stages or any(ratios[n:])):
            raise ValueError('backbone targets and window-only draft compression required')
        self.noise = c.get('dspark_noise_token_id')
        if type(self.noise) is not int or not 0 <= self.noise < self.vocab:
            raise ValueError('explicit DSpark noise token required')
        self.eps = c.get('rms_norm_eps', c.get('norm_eps'))
        if type(self.eps) not in (int, float) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('explicit positive publisher norm epsilon required')
        if c['num_attention_heads']*c['head_dim'] % c['o_groups']:
            raise ValueError('draft grouped output extent mismatch')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive MTP workspace budget required')
        if 4*(3*self.stages*c['sliding_window']*c['head_dim']+self.size*self.vocab*6
              + self.size*(4*self.h*self.hc+4*c['dspark_markov_rank'])) > max_bytes:
            raise ValueError('MTP workspace exceeds admission budget')
        self.max_bytes = max_bytes
        self.rng = np.random.default_rng() if rng is None else rng
        self.windows = [np.zeros((c['sliding_window'], c['head_dim']), np.float32) for _ in range(self.stages)]
        self.position = None
        self._lock = threading.RLock()
        self._busy = False
        self._cache_transport = False
        self._quarantined = False
        self._closed = False

    def _check_transport(self):
        if self._closed:
            raise RuntimeError('MTP owner is closed')
        ensure = getattr(self.provider, '_ensure_open', None)
        if callable(ensure):
            ensure()
        if self._cache_transport or self._quarantined:
            raise RuntimeError('MTP owner is active or requires recovery')

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._busy or self._cache_transport:
                raise RuntimeError('cannot close an active MTP owner')
            self._closed = True
            self.windows = []
            self.position = None

    def write_cache(self, output, *, max_bytes):
        from .mtp_cache import transport_mtp_cache
        with self._lock:
            self._check_transport()
        return transport_mtp_cache(self, output, max_bytes=max_bytes)

    def restore_cache(self, source, *, max_bytes):
        from .mtp_cache import transport_mtp_cache
        with self._lock:
            self._check_transport()
        return transport_mtp_cache(self, source, max_bytes=max_bytes, restoring=True)

    @_exclusive
    def snapshot(self):
        with self._lock:
            self._check_transport()
            identity = getattr(self.provider, 'identity', None)
            if not isinstance(identity, str) or not identity:
                raise ValueError('stable MTP provider identity required for snapshots')
            return PublisherV41MTPSnapshot(identity, self.position,
                tuple(w.astype('<f4', copy=False).tobytes() for w in self.windows),
                copy.deepcopy(self.rng.bit_generator.state))

    @_exclusive
    def restore(self, snapshot):
        with self._lock:
            self._check_transport()
            identity = getattr(self.provider, 'identity', None)
            if (type(snapshot) is not PublisherV41MTPSnapshot or
                    snapshot.identity != identity or type(snapshot.windows) is not tuple or
                    len(snapshot.windows) != self.stages):
                raise ValueError('MTP snapshot identity or stage inventory mismatch')
            if (snapshot.position is not None and
                    (type(snapshot.position) is not int or not 0 < snapshot.position <= self.context)):
                raise ValueError('invalid MTP snapshot position')
            shape = (self.config['sliding_window'], self.config['head_dim'])
            if 12*self.stages*math.prod(shape)+262144 > self.max_bytes:
                raise MemoryError('MTP snapshot restore exceeds workspace admission')
            restored = []
            for raw in snapshot.windows:
                if type(raw) is not bytes or len(raw) != math.prod(shape)*4:
                    raise ValueError('invalid MTP snapshot window extent')
                bank = self._array(np.frombuffer(raw, dtype='<f4'), shape)
                occupied = min(snapshot.position or 0, shape[0])
                if occupied < shape[0] and np.any(bank[occupied:] != 0):
                    raise ValueError('nonzero unoccupied MTP window state')
                restored.append(bank)
            from .mtp_cache import _rng
            _rng(snapshot.rng_state, self.rng.bit_generator.state['bit_generator'])
            old_rng = copy.deepcopy(self.rng.bit_generator.state)
            try:
                self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)
            except BaseException as primary:
                try:
                    self.rng.bit_generator.state = old_rng
                except BaseException as rollback:
                    self._quarantined = True
                    primary.add_note('MTP restore RNG rollback failed: '+repr(rollback))
                raise
            self.windows, self.position = restored, snapshot.position

    @_exclusive
    def reset(self):
        with self._lock:
            self._check_transport()
            self._quarantined = True
            for window in self.windows:
                window.fill(0)
            self.position = None
            self._quarantined = False

    @staticmethod
    def _array(value, shape):
        x = np.array(value, dtype=np.float32, copy=True)
        if x.size != math.prod(shape) or not np.isfinite(x).all():
            raise ValueError('invalid MTP extent or arithmetic')
        return x.reshape(shape)

    def _cast(self, x):
        return self._array(self.provider.cast(x.reshape(-1)), x.shape)

    def _tensor(self, name, shape):
        return self._array(self.provider.tensor(name), shape)

    def _tensor_row(self, name, index, width):
        read_row = getattr(self.provider, 'tensor_row', None)
        if callable(read_row):
            return self._array(read_row(name, index), (width,))
        table = np.asarray(self.provider.tensor(name))
        if table.shape != (self.vocab, width) or not 0 <= index < self.vocab:
            raise ValueError('invalid indexed MTP tensor geometry')
        return self._array(table[index], (width,))

    def _linear(self, name, x, width):
        return np.stack([self._array(self.provider.linear(name, row), (width,)) for row in x])

    def _norm(self, x, name):
        operation = getattr(self.provider,'auxiliary_math',None)
        if callable(operation):
            return operation('norm',x,self._tensor(name,(x.shape[-1],)),epsilon=self.eps)
        return self._cast(x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+self.eps)
                          * self._tensor(name, (x.shape[-1],)))

    def _cancelled(self):
        check = getattr(self.provider, 'check_cancelled', None)
        if callable(check):
            check()

    def _kv(self, stage, rows, positions):
        hd = self.config['head_dim']
        prefix = f'v41.mtp.{stage}.attention.'
        out = self._norm(self._linear(prefix+'wkv', rows, hd), prefix+'kv_norm.weight')
        out = self._array(self.provider.rotate(stage, out, positions), out.shape)
        result = []
        for row in out:
            self._cancelled()
            result.append(self._array(self.provider.roundtrip(row,
                'fp8-e4m3-g32-ue8m0'), (hd,)))
        return np.stack(result)

    @_exclusive
    def seed_window(self, main_hidden, *, start_pos):
        """Seed occupied draft windows from an absolute-position prefix tail.

        Only the final sliding-window rows are required. Projection and rotary
        positions match full prefill; untouched physical slots remain zero.
        Commit all stages together so interrupted reseeding preserves state.
        """
        with self._lock:
            self._check_transport()
            rows = np.asarray(main_hidden)
            window, hd = self.config['sliding_window'], self.config['head_dim']
            if (type(start_pos) is not int or start_pos < 0 or
                    rows.dtype != np.float32 or rows.ndim != 2 or
                    rows.shape[1] != self.h*len(self.targets) or
                    not 0 < len(rows) <= window or
                    (start_pos > 0 and len(rows) != window) or
                    start_pos+len(rows) > self.context):
                raise ValueError('complete occupied prefix tail and absolute position required')
            required = (rows.nbytes*3 + 4*len(rows)*(8*self.h+8*hd) +
                        12*self.stages*window*hd)
            if required > self.max_bytes:
                raise MemoryError('MTP window reseeding exceeds workspace admission')
            self._cancelled()
            if not np.isfinite(rows).all():
                raise ValueError('nonfinite draft reseeding evidence')
            main = self._norm(self._linear('v41.mtp.0.main_proj', rows, self.h),
                              'v41.mtp.0.main_norm')
            positions = np.arange(start_pos, start_pos+len(rows), dtype=np.int64)
            staged = []
            for stage in range(self.stages):
                self._cancelled()
                keys = self._kv(stage, main, positions)
                bank = np.zeros((window, hd), dtype=np.float32)
                for position, row in zip(positions, keys):
                    bank[int(position) % window] = row
                staged.append(bank)
            self._cancelled()
            self.windows, self.position = staged, start_pos+len(rows)

    @_exclusive
    def attention(self, stage, x, start_pos, main_x):
        self._check_transport()
        if type(stage) is not int or not 0<=stage<self.stages:
            raise ValueError('valid MTP attention stage required')
        retained=sum(window.nbytes for window in self.windows)
        if retained+self.windows[stage].nbytes>self.max_bytes:
            raise MemoryError('MTP attention rollback exceeds workspace admission')
        saved=self.windows[stage].copy()
        try:
            return self._attention_locked(stage,x,start_pos,main_x,rollback_bytes=saved.nbytes)
        except BaseException:
            self.windows[stage]=saved
            raise

    def _attention_locked(self, stage, x, start_pos, main_x, *, rollback_bytes=0):
        """One-session DSpark attention; drafts attend bidirectionally to all draft KV.

        The block callback calls this exactly once per stage, including prefill.
        Window mutations are rolled back if forward_spec fails.
        """
        self._check_transport()
        if type(stage) is not int or not 0 <= stage < self.stages:
            raise ValueError('valid MTP attention stage required')
        c, win = self.config, self.config['sliding_window']
        heads, hd, groups, rank = c['num_attention_heads'], c['head_dim'], c['o_groups'], c['o_lora_rank']
        p = f'v41.mtp.{stage}.attention.'
        def kv(rows, positions):
            return self._kv(stage, rows, positions)
        main_kv = kv(main_x, np.arange(start_pos, start_pos+len(main_x)))
        if start_pos == 0:
            for pos, row in enumerate(main_kv):
                self._cancelled()
                self.windows[stage][pos % win] = row
            return x.copy()
        if len(main_x) != 1:
            raise ValueError('publisher incremental DSpark main input must contain one token')
        positions = np.arange(start_pos+1, start_pos+1+len(x))
        qr = self._norm(self._linear(p+'wq_a', x, c['q_lora_rank']), p+'q_norm.weight')
        q = self._linear(p+'wq_b', qr, heads*hd).reshape(len(x), heads, hd)
        q = self._array(self.provider.rotate(stage, q, positions), q.shape)
        draft_kv = kv(x, positions)
        self.windows[stage][start_pos % win] = main_kv[0]
        banks = (self.windows[stage][:min(win, start_pos+1)], draft_kv)
        retained = sum(window.nbytes for window in self.windows)+q.nbytes+draft_kv.nbytes+main_kv.nbytes+rollback_bytes
        attention = getattr(self.provider,'attention_rows',None) or attention_rows
        out = self._cast(attention(q, banks, banks,
            sink=self._tensor(p+'attn_sink', (heads,)),
            max_bytes=self.max_bytes-retained,
            check_cancelled=getattr(self.provider, 'check_cancelled', None)))
        out = self._array(self.provider.rotate(stage, out, positions, inverse=True), out.shape)
        out = out.reshape(len(x), groups, -1)
        # Publisher bypasses quantized Linear for wo_a and contracts logical weights.
        grouped = getattr(self.provider, 'grouped_linear', None)
        if callable(grouped):
            out = self._array(grouped(p+'wo_a', out, groups=groups),
                              (len(x), groups*rank))
            out = self._cast(out)
        else:
            weight = self._tensor(p+'wo_a', (groups, rank, out.shape[-1]))
            out = self._cast(np.einsum('sgd,grd->sgr', out, weight)).reshape(len(x), groups*rank)
        return self._linear(p+'wo_b', out, self.h)

    @_exclusive
    def forward_spec(self, input_id, main_hidden, start_pos=0, *, temperature=1.0):
        with self._lock:
            self._check_transport()
            return self._forward_spec(input_id, main_hidden, start_pos, temperature=temperature)

    def _forward_spec(self, input_id, main_hidden, start_pos=0, *, temperature=1.0):
        """Single-batch forward only; returns (ids, biased_logits, raw_confidence).

        main_hidden concatenates attention-input target stream means in backbone
        traversal order. Prefill returns None after seeding every stage cache.
        """
        if type(input_id) is not int or not 0 <= input_id < self.vocab:
            raise ValueError('valid sampled backbone token required')
        if type(start_pos) is not int or start_pos < 0:
            raise ValueError('nonnegative DSpark position required')
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('nonnegative finite temperature required')
        main_hidden = np.asarray(main_hidden, dtype=np.float32)
        if (main_hidden.ndim != 2 or main_hidden.shape[1] != self.h*len(self.targets)
                or not len(main_hidden) or not np.isfinite(main_hidden).all()):
            raise ValueError('concatenated target attention inputs required')
        if (start_pos+len(main_hidden) > self.context or
                (start_pos and start_pos+1+self.size > self.context)):
            raise ValueError('MTP input or draft rotary positions exceed admitted context')
        if start_pos and (self.position != start_pos or len(main_hidden) != 1):
            raise ValueError('draft calls must follow seeded contiguous backbone positions')
        if 4*(len(main_hidden)*(2*self.h+3*self.config['head_dim'])+
                3*self.stages*self.config['sliding_window']*self.config['head_dim']+
                6*self.size*self.vocab) > self.max_bytes:
            raise MemoryError('MTP prefill and draft workspace exceeds admission budget')
        saved, old_pos = [w.copy() for w in self.windows], self.position
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            if start_pos == 0:
                for window in self.windows:
                    window.fill(0)
            main_x = self._norm(self._linear('v41.mtp.0.main_proj', main_hidden, self.h), 'v41.mtp.0.main_norm')
            ids = np.full(self.size, self.noise, dtype=np.int64)
            ids[0] = input_id
            embedded = np.stack([self._tensor_row('v41.text.embedding', int(token), self.h) for token in ids])
            x = np.repeat(embedded[:, None, :], self.hc, axis=1)
            pre = np.zeros((self.size, self.hc), np.float32)
            pre[:, 0] = 1
            for stage in range(self.stages):
                self._cancelled()
                if start_pos == 0:
                    self._attention_locked(stage, x, 0, main_x)
                else:
                    x, pre = self.block(stage, x, start_pos, pre, main_x, self._attention_locked)
                    x = self._array(x, (self.size, self.hc, self.h))
                    pre = self._array(pre, (self.size, self.hc))
            self._cancelled()
            self.position = len(main_x) if start_pos == 0 else start_pos+1
            if start_pos == 0:
                return None
            operation = getattr(self.provider,'auxiliary_math',None)
            hidden = (operation('streams',x,pre,axis=1) if callable(operation)
                      else self._cast(np.sum(pre[..., None]*x, axis=1)))
            last = f'v41.mtp.{self.stages-1}.'
            normed = self._norm(hidden, last+'norm')
            logits = self._linear('v41.text.lm_head', normed, self.vocab)
            rank = self.config['dspark_markov_rank']
            output = np.empty(self.size+1, np.int64)
            output[0] = input_id
            markov = []
            for i in range(self.size):
                self._cancelled()
                row = self._tensor_row(last+'markov_head.embed', int(output[i]), rank)
                markov.append(row)
                bias = self._linear(last+'markov_head.head', row[None, :], self.vocab)[0]
                if callable(operation):
                    logits[i] = operation('add_fp32',logits[i],bias)
                else:
                    logits[i] += bias
                if not np.isfinite(logits[i]).all():
                    raise ValueError('nonfinite biased MTP vocabulary logits')
                if temperature == 0:
                    output[i+1] = logits[i].argmax()
                else:
                    scaled = logits[i]/max(temperature, 1e-5)
                    probs = np.exp(scaled-scaled.max())
                    probs /= probs.sum()
                    output[i+1] = (probs/np.maximum(self.rng.exponential(size=self.vocab), np.finfo(np.float32).tiny)).argmax()
            features = np.concatenate((hidden, np.stack(markov)), axis=-1)
            confidence = self._linear(last+'confidence_head.proj', features, 1)
            self._cancelled()
            return output, logits, confidence[:, 0]
        except BaseException as primary:
            self.windows, self.position = saved, old_pos
            try:
                self.rng.bit_generator.state = rng_state
            except BaseException as rollback:
                self._quarantined = True
                primary.add_note('MTP proposal RNG rollback failed: '+repr(rollback))
            raise
