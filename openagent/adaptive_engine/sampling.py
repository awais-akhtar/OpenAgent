"""Seeded CPU sampling with explicit vocabulary-buffer admission."""
from dataclasses import dataclass,asdict
import copy
import hashlib
import hmac
import json
import math

import numpy as np


@dataclass(frozen=True)
class SamplingPolicy:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int = 0

    def __post_init__(self):
        for name in ('temperature', 'top_p', 'min_p', 'repetition_penalty',
                     'frequency_penalty', 'presence_penalty'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError('finite sampling scalar required: '+name)
        if (self.temperature < 0 or not 0 < self.top_p <= 1 or
                not 0 <= self.min_p <= 1 or self.repetition_penalty <= 0):
            raise ValueError('invalid sampling probability or penalty')
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError('nonnegative top-k required')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('nonnegative integer sampling seed required')


class TokenSampler:
    """One request's RNG; callers serialize access to a sampler instance.

    max_buffer_bytes bounds numerical buffers, not interpreter or model RSS.
    Greedy sampling retains lowest-token-ID tie breaking. Stochastic policies
    are opt-in and do not assert publisher-specific sampler equivalence.
    """
    def __init__(self, policy, *, max_buffer_bytes, max_history_tokens=131072):
        if type(policy) is not SamplingPolicy:
            raise TypeError('SamplingPolicy required')
        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise ValueError('positive sampling buffer budget required')
        if type(max_history_tokens) is not int or max_history_tokens <= 0:
            raise ValueError('positive sampling history limit required')
        self.policy, self.max_buffer_bytes = policy, max_buffer_bytes
        self.max_history_tokens = max_history_tokens
        self.rng = np.random.default_rng(policy.seed)

    @property
    def needs_history(self):
        p = self.policy
        return bool(p.repetition_penalty != 1 or p.frequency_penalty or p.presence_penalty)

    def snapshot(self,*,max_bytes=65536):
        """Detach the complete policy/RNG boundary for exact request recovery."""
        if type(max_bytes) is not int or not 64<=max_bytes<=1024**2:
            raise ValueError('bounded sampler transport admission required')
        metadata=dict(version=1,policy=asdict(self.policy),rng=copy.deepcopy(self.rng.bit_generator.state),
            max_history_tokens=self.max_history_tokens)
        encoded=json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
        if len(encoded)+40>max_bytes:raise MemoryError('sampler boundary exceeds admission')
        prefix=b'OASAMP01'+encoded
        return prefix+hashlib.sha256(prefix).digest()

    def restore(self,frame,*,max_bytes=65536):
        """Validate in a temporary generator, then replace live RNG atomically."""
        if (type(max_bytes) is not int or not 64<=max_bytes<=1024**2 or type(frame) is not bytes
                or not 40<len(frame)<=max_bytes or frame[:8]!=b'OASAMP01'):
            raise ValueError('bounded immutable sampler boundary required')
        if not hmac.compare_digest(frame[-32:],hashlib.sha256(frame[:-32]).digest()):
            raise ValueError('sampler boundary checksum differs')
        def pairs(items):
            result={}
            for key,value in items:
                if key in result:raise ValueError('duplicate sampler boundary field')
                result[key]=value
            return result
        metadata=json.loads(frame[8:-32],object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if (type(metadata) is not dict or set(metadata)!={'version','policy','rng','max_history_tokens'}
                or type(metadata['version']) is not int or metadata['version']!=1
                or metadata['policy']!=asdict(self.policy)
                or type(metadata['max_history_tokens']) is not int
                or metadata['max_history_tokens']!=self.max_history_tokens
                or type(metadata['rng']) is not dict or metadata['rng'].get('bit_generator')!='PCG64'):
            raise ValueError('sampler boundary policy, history or generator differs')
        if type(metadata['policy']) is not dict or SamplingPolicy(**metadata['policy'])!=self.policy:
            raise ValueError('sampler boundary contains an invalid policy')
        generator=np.random.PCG64();generator.state=copy.deepcopy(metadata['rng'])
        self.rng=np.random.Generator(generator)

    def sample(self, logits, *, history=()):
        scores = np.asarray(logits)
        if scores.ndim != 1 or not scores.size or scores.dtype.kind != 'f':
            raise ValueError('floating vocabulary row required')
        if scores.size*64 > self.max_buffer_bytes:
            raise MemoryError('sampling buffers exceed vocabulary admission')
        if not np.isfinite(scores).all():
            raise ValueError('finite vocabulary logits required')
        p = self.policy
        if p.top_k > scores.size:
            raise ValueError('top-k exceeds vocabulary')
        row = np.array(scores, dtype=np.float64, copy=True)
        if self.needs_history:
            if type(history) not in (list, tuple) or len(history) > self.max_history_tokens:
                raise ValueError('sampling history exceeds admission')
            if any(
                    type(t) is not int or not 0 <= t < row.size for t in history):
                raise ValueError('bounded vocabulary history required for penalties')
            counts = np.zeros(row.size, dtype=np.int64)
            for token in history:
                counts[token] += 1
            with np.errstate(over='ignore', invalid='ignore'):
                for token in np.flatnonzero(counts):
                    count = int(counts[token])
                    row[token] = row[token]/p.repetition_penalty if row[token] > 0 else row[token]*p.repetition_penalty
                    row[token] -= count*p.frequency_penalty+p.presence_penalty
            if not np.isfinite(row).all():
                raise ValueError('sampling penalties produced nonfinite logits')
        if p.temperature == 0:
            return int(row.argmax())
        if p.top_k:
            selected = np.argsort(-row, kind='stable')[:p.top_k]
            keep = np.zeros(row.size, dtype=bool)
            keep[selected] = True
            row[~keep] = -np.inf
        with np.errstate(over='ignore', under='ignore', invalid='raise', divide='raise'):
            row -= row.max()
            row /= p.temperature
            np.exp(row, out=row)
        total = row.sum()
        if not math.isfinite(total) or total <= 0:
            raise ValueError('invalid sampling normalization')
        row /= total
        if p.min_p:
            row[row < row.max()*p.min_p] = 0
        if p.top_p < 1:
            order = np.argsort(-row, kind='stable')
            normalized = row[order]
            normalized /= normalized.sum()
            stop = int(np.searchsorted(np.cumsum(normalized), p.top_p, side='left'))
            row[order[stop+1:]] = 0
        total = row.sum()
        if not math.isfinite(total) or total <= 0:
            raise ValueError('sampling filters removed all candidates')
        row /= total
        return int(self.rng.choice(row.size, p=row))
