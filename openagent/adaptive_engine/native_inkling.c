/* Independently designed Inkling token-stream kernel. Private ABI v2.
 * FIR histories and token-major KV are caller-owned staging buffers.
 * Architecture facts: Apache-2.0 Transformers 5.16.1 (not translated code).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct { float *data; uint64_t count; } Tensor;
typedef int (*ExpertCall)(void *,uint64_t,uint64_t,const float *,float *);
typedef struct {
    uint64_t hidden, vocab, output_vocab, layers, width, kernel, relative, capacity, floor;
    uint64_t moe_width, experts, shared, topk;
    float epsilon, multiplier, alpha, route_scale;
} Config;
typedef struct {
    uint64_t heads, kv_heads, dim, extent, window, sparse;
    Tensor weights[26];
    Tensor state;
} Layer;

API uint64_t oa_inkling_version(void) { return 2; }

static int tensor_ok(Tensor t, uint64_t size) {
    return t.data != NULL && t.count == size;
}

static int finite_tensor(Tensor t) {
    uint64_t i;
    for (i = 0; i < t.count; ++i) if (!isfinite(t.data[i])) return 0;
    return 1;
}

static uint64_t state_size(const Config *c, const Layer *l) {
    uint64_t v = l->kv_heads * l->dim;
    return 2 * c->capacity * v + 2 * c->kernel * (v + c->hidden);
}

static int geometry_ok(const Config *c, const Layer *l) {
    if (!c || !l || !c->hidden || c->hidden > 1048576 || !c->vocab || c->vocab > 1048576 ||
        !c->output_vocab || c->output_vocab > c->vocab || !c->layers || c->layers > 128 ||
        !c->width || c->width > 1048576 || !c->kernel || c->kernel > 64 ||
        !c->relative || c->relative > 1048576 || !c->capacity || c->capacity > 65536 ||
        c->floor > 1048576 || !isfinite(c->epsilon) || c->epsilon <= 0 ||
        !isfinite(c->multiplier) || c->multiplier <= 0 || !isfinite(c->alpha) ||
        c->alpha < 0 || c->alpha > 100 || !c->moe_width || c->moe_width > 1048576 ||
        !c->experts || c->experts > 65536 || !c->shared || c->shared > 65536 ||
        c->experts + c->shared > 65536 || !c->topk || c->topk > c->experts ||
        !isfinite(c->route_scale) || c->route_scale <= 0) return 0;
    if (!l->heads || l->heads > 65536 || !l->kv_heads || l->kv_heads > 65536 ||
        !l->dim || l->dim > 65536 || l->heads % l->kv_heads ||
        l->heads * l->dim > 65536 || l->kv_heads * l->dim > 65536 ||
        l->heads * c->relative > 65536 || !l->extent || l->extent > 1048576 ||
        l->window > 1048576 || l->window == 1 || l->sparse > 1 ||
        (l->window && l->extent != l->window)) return 0;
    return 1;
}

static void matvec(const float *matrix, const float *x, float *y, uint64_t rows, uint64_t cols) {
    uint64_t r, j;
    for (r = 0; r < rows; ++r) {
        float sum = 0;
        for (j = 0; j < cols; ++j) sum += matrix[r * cols + j] * x[j];
        y[r] = sum;
    }
}

static void normalize(const float *x, const float *w, float *y, uint64_t n, float eps) {
    uint64_t j;
    float sum = 0, scale;
    for (j = 0; j < n; ++j) sum += x[j] * x[j];
    if (!isfinite(sum)) {
        for (j = 0; j < n; ++j) y[j] = NAN;
        return;
    }
    scale = 1.0f / sqrtf(sum / (float)n + eps);
    for (j = 0; j < n; ++j) y[j] = (x[j] * scale) * w[j];
}

/* Histories are time-major, oldest first. The newest sample occupies tap K-1. */
static void filter(float *x, const float *bank, float *history, uint64_t channels, uint64_t taps) {
    uint64_t d, k;
    memmove(history, history + channels, (size_t)((taps - 1) * channels) * sizeof(float));
    memcpy(history + (taps - 1) * channels, x, (size_t)channels * sizeof(float));
    for (d = 0; d < channels; ++d) {
        float sum = 0;
        for (k = 0; k < taps; ++k) sum += history[k * channels + d] * bank[d * taps + k];
        x[d] += sum;
    }
}

static float sigmoid(float x) {
    if (x >= 0) return 1.0f / (1.0f + expf(-x));
    { float e = expf(x); return e / (1.0f + e); }
}

static float log_sigmoid(float x) {
    return x >= 0 ? -log1pf(expf(-x)) : x - log1pf(expf(x));
}

/* Selection is a bounded scan, separate from mixture normalization. Equal
 * cutoff scores are rejected rather than assuming a backend-specific top-k tie. */
static int mixture(const Config *c, const Layer *l, const float *n, float *out,
                   float *temporary, float *shared_out, float *gate, float *up,
                   float *router, float *choice, float *selected, float *gamma,
                   uint64_t layer,ExpertCall callback,void *user) {
    uint64_t e, j, step, best;
    uint64_t count = c->experts + c->shared, h = c->hidden, z = c->moe_width;
    float cutoff = 0, maximum = -FLT_MAX, denominator = 0;
    matvec(l->weights[18].data, n, router, count, h);
    for (e = 0; e < count; ++e) if (!isfinite(router[e])) return 3;
    for (e = 0; e < c->experts; ++e) {
        choice[e] = sigmoid(router[e]) + l->weights[20].data[e];
        if (!isfinite(choice[e])) return 3;
        selected[e] = 0;
    }
    for (step = 0; step < c->topk; ++step) {
        float largest = -FLT_MAX;
        best = c->experts;
        for (e = 0; e < c->experts; ++e) if (!selected[e] && (best == c->experts || choice[e] > largest)) {
            largest = choice[e]; best = e;
        }
        selected[best] = 1;
        cutoff = largest;
    }
    if (c->topk < c->experts) for (e = 0; e < c->experts; ++e)
        if (!selected[e] && choice[e] == cutoff) return 5;
    for (e = 0; e < count; ++e) {
        gamma[e] = 0;
        if (e >= c->experts || selected[e]) {
            router[e] = log_sigmoid(router[e]);
            if (router[e] > maximum) maximum = router[e];
        }
    }
    for (e = 0; e < count; ++e) if (e >= c->experts || selected[e]) {
        gamma[e] = expf(router[e] - maximum); denominator += gamma[e];
    }
    for (e = 0; e < count; ++e) gamma[e] = (gamma[e] / denominator) * c->route_scale * l->weights[19].data[0];
    memset(out, 0, (size_t)h * sizeof(float));
    memset(shared_out, 0, (size_t)h * sizeof(float));
    for (e = 0; e < c->experts; ++e) if (selected[e]) {
        if(callback) {
            if(callback(user,layer,e,n,temporary)) return 6;
            for(j=0;j<h;++j) if(!isfinite(temporary[j])) return 3;
        } else {
        const float *bank = l->weights[21].data + e * 2 * z * h;
        matvec(bank, n, gate, z, h);
        matvec(bank + z * h, n, up, z, h);
        for (j = 0; j < z; ++j) gate[j] = (gate[j] * sigmoid(gate[j])) * up[j];
        matvec(l->weights[22].data + e * h * z, gate, temporary, h, z);
        }
        for (j = 0; j < h; ++j) out[j] += temporary[j] * gamma[e];
    }
    for (e = 0; e < c->shared; ++e) {
        matvec(l->weights[23].data + e * z * h, n, gate, z, h);
        matvec(l->weights[24].data + e * z * h, n, up, z, h);
        for (j = 0; j < z; ++j) gate[j] = ((gate[j] * sigmoid(gate[j])) * up[j]) * gamma[c->experts + e];
        matvec(l->weights[25].data + e * h * z, gate, temporary, h, z);
        for (j = 0; j < h; ++j) shared_out[j] += temporary[j];
    }
    for (j = 0; j < h; ++j) out[j] += shared_out[j];
    return 0;
}

static int inkling_run(const Config *,Layer *,const Tensor *,const int64_t *,uint64_t,
    uint64_t,uint64_t,uint64_t,Tensor,Tensor,Tensor,Tensor,int,uint64_t,ExpertCall,void *);
API int oa_inkling_run(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor hidden,Tensor logits,Tensor routing) {
    Tensor empty={NULL,0};
    return inkling_run(c,layers,global,ids,id_count,batch,tokens,start,hidden,logits,routing,empty,0,0,NULL,NULL);
}
API uint64_t oa_inkling_embeddings_version(void) {return 1;}
API int oa_inkling_run_embeddings(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor hidden,Tensor logits,Tensor routing,Tensor embeddings) {
    if(!embeddings.data) return 1;
    return inkling_run(c,layers,global,ids,id_count,batch,tokens,start,hidden,logits,routing,embeddings,0,0,NULL,NULL);
}
API uint64_t oa_inkling_block_version(void) {return 1;}
API uint64_t oa_inkling_range_version(void) {return 1;}
typedef struct {ExpertCall callback;void *user;uint64_t begin;} RangeExpert;
static int range_expert(void *owner,uint64_t layer,uint64_t expert,const float *input,float *output) {
    RangeExpert *range=(RangeExpert *)owner;
    return range->callback(range->user,range->begin+layer,expert,input,output);
}
API int oa_inkling_run_range_streamed(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    uint64_t begin,uint64_t end,Tensor hidden,Tensor routing,Tensor incoming,
    ExpertCall expert,void *user) {
    Config local;Tensor empty={NULL,0};uint64_t row;int status;RangeExpert range;
    if(!c||!layers||!global||!incoming.data||!c->layers||c->layers>128||begin>=end||end>c->layers||!expert)return 1;
    local=*c;local.layers=end-begin;range.callback=expert;range.user=user;range.begin=begin;
    status=inkling_run(&local,layers+begin,global,ids,id_count,batch,tokens,start,
        hidden,empty,routing,incoming,1,0,range_expert,&range);
    if(status)return status;
    if(end==c->layers)for(row=0;row<batch*tokens;row++)
        normalize(hidden.data+row*c->hidden,global[2].data,hidden.data+row*c->hidden,c->hidden,c->epsilon);
    return finite_tensor(hidden)?0:3;
}
API int oa_inkling_run_range(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    uint64_t begin,uint64_t end,Tensor hidden,Tensor routing,Tensor incoming) {
    Config local;Tensor empty={NULL,0};uint64_t row;int status;
    if(!c||!layers||!global||!incoming.data||!c->layers||c->layers>128||begin>=end||end>c->layers) return 1;
    local=*c;local.layers=end-begin;
    status=inkling_run(&local,layers+begin,global,ids,id_count,batch,tokens,start,
        hidden,empty,routing,incoming,1,0,NULL,NULL);
    if(status) return status;
    /* Intermediate ranges retain the residual. The last range alone owns the
     * final normalization; the public Edge owns the vocabulary projection. */
    if(end==c->layers) for(row=0;row<batch*tokens;row++)
        normalize(hidden.data+row*c->hidden,global[2].data,hidden.data+row*c->hidden,
            c->hidden,c->epsilon);
    return finite_tensor(hidden)?0:3;
}
API int oa_inkling_run_block(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor hidden,Tensor logits,Tensor routing,Tensor embeddings,uint64_t position_origin) {
    if(!embeddings.data||position_origin>65536) return 1;
    return inkling_run(c,layers,global,ids,id_count,batch,tokens,start,hidden,logits,routing,embeddings,1,position_origin,NULL,NULL);
}
API uint64_t oa_inkling_streamed_version(void) {return 1;}
API int oa_inkling_run_streamed(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor hidden,Tensor logits,Tensor routing,Tensor embeddings,ExpertCall callback,void *user) {
    if(!callback) return 1;
    return inkling_run(c,layers,global,ids,id_count,batch,tokens,start,hidden,logits,routing,embeddings,0,0,callback,user);
}

/* No committed storage is accepted by this API: the wrapper stages the whole
 * transaction. A nonzero return invalidates every staged output and state. */
static int inkling_run(const Config *c, Layer *layers, const Tensor *global,
                      const int64_t *ids, uint64_t id_count, uint64_t batch,
                      uint64_t tokens, uint64_t start, Tensor hidden, Tensor logits, Tensor routing,Tensor embeddings,
                      int raw_hidden,uint64_t position_origin,ExpertCall callback,void *user) {
    uint64_t i, b, t, j, head, d, r, p, maxq = 0, maxv = 0, maxr = 0, scratch_count;
    uint64_t mlp_width, expert_count;
    float *scratch, *x, *n, *q, *k, *v, *rel, *context, *out, *gate, *up, *scores;
    float *temporary, *shared_out, *router, *choice, *selected;
    if (!c || !layers || !global || !ids || !batch || batch > 128 || !tokens ||
        tokens > 65536 || start > 65536 || !geometry_ok(c, layers) ||
        start > c->capacity || tokens > c->capacity - start || id_count != batch * tokens)
        return 1;
    if (!tensor_ok(global[0], c->vocab * c->hidden) || !tensor_ok(global[1], c->hidden) ||
        !tensor_ok(global[2], c->hidden) || !tensor_ok(global[3], c->vocab * c->hidden) ||
        !tensor_ok(hidden, batch * tokens * c->hidden) ||
        (!raw_hidden&&!tensor_ok(logits, batch * tokens * c->output_vocab)) ||
        (raw_hidden&&(logits.data||logits.count)) ||
        !tensor_ok(routing, batch * tokens * c->layers * (c->experts + c->shared))) return 1;
    for (i = 0; i < 4; ++i) if (!finite_tensor(global[i])) return 3;
    for (i = 0; i < id_count; ++i) if (ids[i] < 0 || (uint64_t)ids[i] >= c->vocab) return 2;
    if(embeddings.data) {
        if(!tensor_ok(embeddings,batch*tokens*c->hidden)||!finite_tensor(embeddings)) return 3;
    } else if(embeddings.count) return 1;
    for (i = 0; i < c->layers; ++i) {
        Layer *l = layers + i;
        uint64_t qw, vw, rw, h = c->hidden, m = c->width, kt = c->kernel;
        uint64_t expected[26];
        if (!geometry_ok(c, l)) return 1;
        qw = l->heads * l->dim; vw = l->kv_heads * l->dim; rw = l->heads * c->relative;
        expected[0] = h; expected[1] = h; expected[2] = qw * h;
        expected[3] = vw * h; expected[4] = vw * h; expected[5] = rw * h;
        expected[6] = h * qw; expected[7] = vw * kt; expected[8] = vw * kt;
        expected[9] = l->dim; expected[10] = l->dim;
        expected[11] = c->relative * l->extent; expected[12] = h * kt; expected[13] = h * kt;
        expected[14] = m * h; expected[15] = m * h; expected[16] = h * m; expected[17] = 1;
        expected[18] = (c->experts + c->shared) * h; expected[19] = 1; expected[20] = c->experts;
        expected[21] = c->experts * 2 * c->moe_width * h;
        expected[22] = c->experts * h * c->moe_width;
        expected[23] = c->shared * c->moe_width * h; expected[24] = expected[23];
        expected[25] = c->shared * h * c->moe_width;
        for (j = 0; j < 26; ++j) {
            int used = j < 14 || (l->sparse ? j >= 18 : j < 18);
            if(callback&&l->sparse&&(j==21||j==22)) used=0;
            if (used) {
                if (!tensor_ok(l->weights[j], expected[j])) return 1;
                if (!finite_tensor(l->weights[j])) return 3;
            } else if (l->weights[j].count || l->weights[j].data) return 1;
        }
        if (!tensor_ok(l->state, batch * state_size(c, l))) return 1;
        if (!finite_tensor(l->state)) return 3;
        if (qw > maxq) maxq = qw;
        if (vw > maxv) maxv = vw;
        if (rw > maxr) maxr = rw;
    }
    mlp_width = c->width > c->moe_width ? c->width : c->moe_width;
    expert_count = c->experts + c->shared;
    scratch_count = 5 * c->hidden + 2 * maxq + 2 * maxv + maxr + 2 * mlp_width + c->capacity + 3 * expert_count;
    if (scratch_count > SIZE_MAX / sizeof(float)) return 4;
    scratch = (float *)malloc((size_t)scratch_count * sizeof(float));
    if (!scratch) return 4;
    x = scratch; n = x + c->hidden; out = n + c->hidden;
    temporary = out + c->hidden; shared_out = temporary + c->hidden;
    q = shared_out + c->hidden; context = q + maxq; k = context + maxq; v = k + maxv;
    rel = v + maxv; gate = rel + maxr; up = gate + mlp_width; scores = up + mlp_width;
    router = scores + c->capacity; choice = router + expert_count; selected = choice + expert_count;
    memset(routing.data, 0, (size_t)routing.count * sizeof(float));
    for (b = 0; b < batch; ++b) for (t = 0; t < tokens; ++t) {
        uint64_t row = b * tokens + t, position = start + t;
        if(embeddings.data) memcpy(x,embeddings.data+row*c->hidden,(size_t)c->hidden*sizeof(float));
        else normalize(global[0].data + (uint64_t)ids[row] * c->hidden, global[1].data, x, c->hidden, c->epsilon);
        for (i = 0; i < c->layers; ++i) {
            Layer *l = layers + i;
            uint64_t qw = l->heads * l->dim, vw = l->kv_heads * l->dim;
            uint64_t first = l->window && position >= l->window ? position - l->window + 1 : 0;
            float *keys = l->state.data + b * state_size(c, l);
            float *values = keys + c->capacity * vw;
            float *kc = values + c->capacity * vw, *vc = kc + c->kernel * vw;
            float *ac = vc + c->kernel * vw, *mc = ac + c->kernel * c->hidden;
            float tau = 1;
            normalize(x, l->weights[0].data, n, c->hidden, c->epsilon);
            matvec(l->weights[2].data, n, q, qw, c->hidden);
            matvec(l->weights[3].data, n, k, vw, c->hidden);
            matvec(l->weights[4].data, n, v, vw, c->hidden);
            matvec(l->weights[5].data, n, rel, l->heads * c->relative, c->hidden);
            filter(k, l->weights[7].data, kc, vw, c->kernel);
            filter(v, l->weights[8].data, vc, vw, c->kernel);
            for (head = 0; head < l->heads; ++head)
                normalize(q + head * l->dim, l->weights[9].data, q + head * l->dim, l->dim, c->epsilon);
            for (head = 0; head < l->kv_heads; ++head)
                normalize(k + head * l->dim, l->weights[10].data, k + head * l->dim, l->dim, c->epsilon);
            memcpy(keys + position * vw, k, (size_t)vw * sizeof(float));
            memcpy(values + position * vw, v, (size_t)vw * sizeof(float));
            if (!l->window && c->floor && position + position_origin + 1 > c->floor)
                tau += c->alpha * logf((float)(position + position_origin + 1) / (float)c->floor);
            for (j = 0; j < qw; ++j) q[j] *= tau;
            for (head = 0; head < l->heads; ++head) {
                uint64_t kh = head / (l->heads / l->kv_heads);
                float maximum = -FLT_MAX, denominator = 0;
                for (p = first; p <= position; ++p) {
                    uint64_t distance = position - p;
                    float dot = 0, bias = 0;
                    for (d = 0; d < l->dim; ++d)
                        dot += q[head * l->dim + d] * keys[p * vw + kh * l->dim + d];
                    if (distance < l->extent) for (r = 0; r < c->relative; ++r)
                        bias += rel[head * c->relative + r] * l->weights[11].data[r * l->extent + distance];
                    scores[p] = dot / (float)l->dim + bias * tau;
                    if (scores[p] > maximum) maximum = scores[p];
                }
                for (p = first; p <= position; ++p) {
                    scores[p] = expf(scores[p] - maximum); denominator += scores[p];
                }
                for (d = 0; d < l->dim; ++d) {
                    float sum = 0;
                    for (p = first; p <= position; ++p)
                        sum += (scores[p] / denominator) * values[p * vw + kh * l->dim + d];
                    context[head * l->dim + d] = sum;
                }
            }
            matvec(l->weights[6].data, context, out, c->hidden, qw);
            filter(out, l->weights[12].data, ac, c->hidden, c->kernel);
            for (j = 0; j < c->hidden; ++j) x[j] += out[j];
            normalize(x, l->weights[1].data, n, c->hidden, c->epsilon);
            if (l->sparse) {
                int code = mixture(c, l, n, out, temporary, shared_out, gate, up, router, choice, selected,
                                   routing.data + (row * c->layers + i) * expert_count,i,callback,user);
                if (code) { free(scratch); return code; }
            } else {
                matvec(l->weights[14].data, n, gate, c->width, c->hidden);
                matvec(l->weights[15].data, n, up, c->width, c->hidden);
                for (j = 0; j < c->width; ++j) gate[j] = (gate[j] * sigmoid(gate[j])) * up[j];
                matvec(l->weights[16].data, gate, out, c->hidden, c->width);
                for (j = 0; j < c->hidden; ++j) out[j] *= l->weights[17].data[0];
            }
            filter(out, l->weights[13].data, mc, c->hidden, c->kernel);
            for (j = 0; j < c->hidden; ++j) x[j] += out[j];
        }
        if(raw_hidden) memcpy(hidden.data+row*c->hidden,x,(size_t)c->hidden*sizeof(float));
        else {
            normalize(x, global[2].data, hidden.data + row * c->hidden, c->hidden, c->epsilon);
            for (j = 0; j < c->hidden; ++j) n[j] = hidden.data[row * c->hidden + j] / c->multiplier;
            matvec(global[3].data, n, logits.data + row * c->output_vocab, c->output_vocab, c->hidden);
        }
    }
    free(scratch);
    if (!finite_tensor(hidden) || (!raw_hidden&&!finite_tensor(logits)) || !finite_tensor(routing)) return 3;
    for (i = 0; i < c->layers; ++i) if (!finite_tensor(layers[i].state)) return 3;
    return 0;
}
