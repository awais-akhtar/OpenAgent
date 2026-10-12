"""Original NumPy equations for the explicit dense-MLA K3 fixture only."""
import numpy as np

from .native_kimi_k3_decoder import admit_weights


def evaluate_dense_mla(config, weights, tokens):
    w = admit_weights(config, weights)
    c = config
    ids = np.asarray(tokens)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not 1 <= ids.size <= c.capacity:
        raise ValueError('invalid fixture tokens')
    if np.any(ids < 0) or np.any(ids >= c.vocab):
        raise ValueError('token outside fixture vocabulary')

    def norm(x, gain, eps=c.epsilon):
        x = x.astype(np.float64)
        return (x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)*gain).astype(np.float32)

    def project(x, weight):
        return (x.astype(np.float64) @ weight.astype(np.float64).T).astype(np.float32)

    def combine(rows, gain, query):
        x = np.stack(rows, axis=1).astype(np.float64)
        score = np.sum(x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+c.epsilon)*gain*query, axis=-1)
        p = np.exp(score-score.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        return np.sum(x*p[..., None], axis=1).astype(np.float32)

    def sigmoid(x):
        return np.exp(-np.logaddexp(0, -x.astype(np.float64)))

    prefix = w['model.embed_tokens.weight'][ids].copy()
    blocks, keys, values = [], [], []
    for i in range(c.layers):
        p = f'model.layers.{i}.'
        def get(name):
            return w[p+name+'.weight']
        mixed = combine(blocks+[prefix], get('self_attention_res_norm'),
                        get('self_attention_res_proj')) if blocks else prefix.copy()
        if i % c.block == 0:
            blocks.append(prefix.copy())
            prefix = np.zeros_like(prefix)
        x = norm(mixed, get('input_layernorm'))
        qlatent = norm(project(x, get('self_attn.q_a_proj')), get('self_attn.q_a_layernorm'), 1e-6)
        q = project(qlatent, get('self_attn.q_b_proj')).reshape(-1, c.heads, c.nope+c.rope)
        latent = project(x, get('self_attn.kv_a_proj_with_mqa'))
        expanded = project(norm(latent[:, :c.kvrank], get('self_attn.kv_a_layernorm'), 1e-6),
                           get('self_attn.kv_b_proj')).reshape(-1, c.heads, c.nope+c.value)
        k = np.concatenate([expanded[..., :c.nope],
            np.broadcast_to(latent[:, None, c.kvrank:], (len(ids), c.heads, c.rope))], axis=-1)
        v = expanded[..., c.nope:]
        keys.append(k.copy())
        values.append(v.copy())
        scores = np.einsum('thd,shd->hts', q.astype(np.float64), k.astype(np.float64))/np.sqrt(c.nope+c.rope)
        scores[:, np.triu_indices(len(ids), 1)[0], np.triu_indices(len(ids), 1)[1]] = -np.inf
        probs = np.exp(scores-scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        attention = np.einsum('hts,shv->thv', probs, v.astype(np.float64)).reshape(len(ids), -1)
        gated = (attention*sigmoid(project(x, get('self_attn.g_proj')))).astype(np.float32)
        prefix += project(gated, get('self_attn.o_proj'))
        x = norm(combine(blocks+[prefix], get('mlp_res_norm'), get('mlp_res_proj')), get('post_attention_layernorm'))
        g, u = project(x, get('mlp.gate_proj')), project(x, get('mlp.up_proj'))
        activated = (c.beta*np.tanh(g.astype(np.float64)/c.beta)*sigmoid(g)*
                     c.linear_beta*np.tanh(u.astype(np.float64)/c.linear_beta)).astype(np.float32)
        prefix += project(activated, get('mlp.down_proj'))
    x = norm(combine(blocks+[prefix], w['model.output_attn_res_norm.weight'],
                     w['model.output_attn_res_proj.weight']), w['model.norm.weight'])
    logits = project(x, w['lm_head.weight'])
    if not np.isfinite(logits).all():
        raise ValueError('nonfinite fixture result')
    return logits, np.stack(keys), np.stack(values)
