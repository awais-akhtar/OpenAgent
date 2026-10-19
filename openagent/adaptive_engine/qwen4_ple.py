"""Qwen4 lexical feature metadata and resident PLE equations."""
import numpy as np

from .independent_qwen35 import sigmoid, silu
from .native_mixtral import rmsnorm


_INT64_MAX = (1 << 63) - 1


def _is_prime_int64(value):
    """Deterministic primality test for the signed-int64 PLE metadata domain."""
    if value < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small:
        if value % prime == 0:
            return value == prime
    odd, shifts = value - 1, 0
    while odd % 2 == 0:
        odd //= 2
        shifts += 1
    # This base set is deterministic for every integer below 2**64.  PLE
    # metadata is stored as signed int64, so this covers its complete domain.
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        base %= value
        if base == 0:
            continue
        witness = pow(base, odd, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def metadata(config, ordinal):
    """Prime-sized hash lanes and deterministic unsigned SplitMix64 seeds."""
    c = config
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError('nonnegative Qwen4 PLE layer ordinal required')
    lanes = (c.ngram_size-1)*c.heads_per_ngram
    if type(lanes) is not int or lanes <= 0:
        raise ValueError('positive Qwen4 PLE hash-lane count required')
    primes, candidate = [], c.ngram_vocab_size_base
    if type(candidate) is not int or not 2 <= candidate <= _INT64_MAX:
        raise ValueError('Qwen4 PLE prime base is outside signed-int64 metadata range')
    while len(primes) < (ordinal+1)*lanes:
        if candidate > _INT64_MAX:
            raise ValueError('Qwen4 PLE prime metadata exceeds signed-int64 range')
        if _is_prime_int64(candidate):
            primes.append(candidate)
        candidate += 1
    sizes = np.array(primes[ordinal*lanes:], dtype=np.int64)
    if sum(int(value) for value in sizes) > _INT64_MAX:
        raise ValueError('Qwen4 PLE hash offsets exceed signed-int64 range')
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(sizes[:-1])))
    modulus = (1 << 64)-1
    bound = max(1, (((1 << 63)-1)//c.vocab_size)//2)
    values = []
    for index in range(c.ngram_size):
        word = (c.seed+10007*ordinal+0x9E3779B97F4A7C15*(index+2)) & modulus
        for shift, multiplier in ((30, 0xBF58476D1CE4E5B9), (27, 0x94D049BB133111EB)):
            word = ((word ^ (word >> shift))*multiplier) & modulus
        word ^= word >> 31
        values.append(2*(word % bound)+1)
    return np.array(values, dtype=np.int64), sizes, offsets


def execute(decoder, x, ids, prefix, old, mask):
    c, w = decoder.config, decoder.weights
    b, t, _ = x.shape
    p = prefix+'ple.'
    eos = c.eos_token_id[0] if isinstance(c.eos_token_id, list) else c.eos_token_id
    ids = np.where(mask, ids, eos)
    previous = np.full((b, c.ngram_size-1), eos, np.int64) if old is None else old[0]
    tokens = np.concatenate((previous, ids), axis=1)
    lane = p+'ple_embedding.'
    multipliers = w[lane+'layer_multipliers']
    sizes, offsets = w[lane+'ngram_heads_vocab_sizes'], w[lane+'ngram_heads_offsets']
    hashed = np.empty((b, t, sizes.size), np.int64)
    for row in range(b):
        for step in range(t):
            here, combined = c.ngram_size-1+step, 0
            for lag in range(c.ngram_size):
                source = here-lag
                valid = source >= 0 and not np.any(tokens[row, source:here] == eos)
                token = int(tokens[row, source]) if valid else eos
                combined ^= token*int(multipliers[lag])
                if lag:
                    begin = (lag-1)*c.heads_per_ngram
                    hashed[row, step, begin:begin+c.heads_per_ngram] = combined % sizes[begin:begin+c.heads_per_ngram]+offsets[begin:begin+c.heads_per_ngram]
    embed = w[lane+'ngram_embedding.weight'][hashed].reshape(b, t, c.ple_embed_dim)
    group_shape = (b, t, c.hc_count, c.hidden_size)
    def norm(value, name):
        return rmsnorm(value.reshape(group_shape), 1+w[p+name+'.weight'].reshape(c.hc_count, c.hidden_size), c.rms_norm_eps)
    key = norm(embed @ w[p+'key_proj.weight'].T, 'norm_key')
    query = norm(x, 'norm_query')
    score = (key*query).sum(axis=-1, keepdims=True)/np.float32(np.sqrt(c.hidden_size))
    gate = np.sign(score)*np.sqrt(np.maximum(np.abs(score), np.float32(1e-6)))
    value = embed @ w[p+'value_proj.weight'].T
    gated = (sigmoid(gate)*value[..., None, :]).reshape(x.shape)
    normalized = norm(gated, 'norm_conv').reshape(x.shape)*mask[..., None]
    gated = gated*mask[..., None]
    dilation, kernel = c.ngram_size, c.ple_conv_kernel_size
    span = (kernel-1)*dilation
    history = np.zeros((b, x.shape[-1], span), np.float32) if old is None else old[1].copy()
    output = np.empty_like(x)
    weights = w[p+'conv1d.weight'][:, 0]
    for step in range(t):
        window = np.concatenate((history, normalized[:, step, :, None]), axis=-1)
        output[:, step] = gated[:, step]+silu((window[..., ::dilation]*weights).sum(axis=-1))
        history = window[..., 1:] if span else window[..., :0]
    return output, (tokens[:, -(c.ngram_size-1):].copy(), history.copy())
