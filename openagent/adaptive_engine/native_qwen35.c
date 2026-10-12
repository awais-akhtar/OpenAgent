/* Independent Qwen3.5 text inference math. Private counted-buffer ABI. */
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct { float *data; uint64_t count; } Tensor;
typedef struct {
    uint64_t hidden, heads, kv, dim, key_heads, value_heads, key_dim, value_dim;
    uint64_t kernel, experts, topk, intermediate, shared, capacity, rotary;
    double epsilon, theta;
} Config;
/* Common: norms/router/banks/shared weights; mixer fields are kind-specific. */
typedef struct { uint64_t kind; Tensor w[19]; Tensor left, right; } Layer;
typedef int (*ExpertFFN)(void *, uint64_t, uint64_t, const float *, uint64_t, float *, uint64_t);

static int geometry(const Config *c) {
    uint64_t fields[15];
    size_t i;
    if (!c) return 0;
    memcpy(fields, c, sizeof(fields));
    for (i = 0; i < 15; ++i) if (!fields[i] || fields[i] > 65536) return 0;
    return c->hidden <= 4096 && c->heads <= 128 && c->kv <= 128 &&
        c->dim <= 4096 && c->key_heads <= 128 && c->value_heads <= 128 &&
        c->key_dim <= 4096 && c->value_dim <= 4096 && c->kernel <= 64 &&
        c->experts <= 4096 && c->intermediate <= 65536 && c->shared <= 65536 &&
        c->heads % c->kv == 0 && c->value_heads % c->key_heads == 0 &&
        c->topk <= c->experts && c->rotary <= c->dim && c->rotary % 2 == 0 &&
        isfinite(c->epsilon) && c->epsilon > 0 && isfinite(c->theta) && c->theta > 0;
}
static int valid(Tensor a, uint64_t count) {
    uint64_t i;
    /* Immutable publisher matrices exceed the old fixture workspace ceiling.
       Their extent is caller-admitted; retain address-space overflow rejection.
       Scratch allocation keeps its separate bounded workspace policy below. */
    if (!a.data || a.count != count || count > SIZE_MAX / sizeof(float)) return 0;
    for (i = 0; i < count; ++i) if (!isfinite(a.data[i])) return 0;
    return 1;
}
static float *allocate(uint64_t count) {
    if (count > SIZE_MAX / sizeof(float) || count > 67108864) return NULL;
    return (float *)calloc((size_t)count, sizeof(float));
}
static void matvec(const float *x, const float *w, uint64_t rows, uint64_t cols, float *out) {
    uint64_t r, j;
    for (r = 0; r < rows; ++r) {
        double sum = 0;
        for (j = 0; j < cols; ++j) sum += (double)x[j] * w[r * cols + j];
        out[r] = (float)sum;
    }
}
static void norm(const float *x, const float *w, uint64_t n, double eps, int offset, float *out) {
    uint64_t j;
    double sum = 0, scale;
    for (j = 0; j < n; ++j) sum += (double)x[j] * x[j];
    scale = 1 / sqrt(sum / (double)n + eps);
    for (j = 0; j < n; ++j) out[j] = (float)(x[j] * scale * (w[j] + offset));
}
static float sigmoid(float x) {
    if (x >= 0) return (float)(1 / (1 + exp(-(double)x)));
    { double e = exp((double)x); return (float)(e / (1 + e)); }
}
static float silu(float x) { return x * sigmoid(x); }
static double softplus(double x) { return x > 0 ? x + log1p(exp(-x)) : log1p(exp(x)); }
static void rotate(float *x, uint64_t n, const int64_t *positions,
                   uint64_t row, uint64_t rows, const uint64_t *section, double theta) {
    uint64_t j;
    for (j = 0; j < n / 2; ++j) {
        uint64_t axis = 0;
        if (section && j % 3 == 1 && j / 3 < section[1]) axis = 1;
        if (section && j % 3 == 2 && j / 3 < section[2]) axis = 2;
        double angle = (double)positions[axis * rows + row] * pow(theta, -(double)(2 * j) / (double)n);
        float a = x[j], b = x[j + n / 2];
        x[j] = (float)(a * cos(angle) - b * sin(angle));
        x[j + n / 2] = (float)(a * sin(angle) + b * cos(angle));
    }
}

API uint64_t oa_qwen35_version(void) { return 1; }

API int oa_qwen35_embed(Tensor weight, uint64_t vocab, uint64_t hidden,
    const int64_t *ids, uint64_t tokens, Tensor out) {
    uint64_t i;
    if (!vocab || !hidden || !tokens || vocab > 1048576 || hidden > 4096 ||
        tokens > 1048576 || !ids || !out.data || out.count != tokens * hidden ||
        out.count > SIZE_MAX / sizeof(float) ||
        !valid(weight, vocab * hidden)) return 1;
    for (i = 0; i < tokens; ++i) if (ids[i] < 0 || (uint64_t)ids[i] >= vocab) return 1;
    for (i = 0; i < tokens; ++i)
        memcpy(out.data + i * hidden, weight.data + (uint64_t)ids[i] * hidden, (size_t)hidden * sizeof(float));
    return 0;
}

API int oa_qwen35_head(Tensor x, Tensor nw, Tensor weight, uint64_t hidden,
    uint64_t vocab, double epsilon, Tensor normalized, Tensor logits) {
    uint64_t i, tokens;
    if (!hidden || hidden > 4096 || !vocab || vocab > 1048576 || x.count % hidden ||
        !isfinite(epsilon) || epsilon <= 0) return 1;
    tokens = x.count / hidden;
    if (!tokens || tokens > 1048576 || !valid(x, tokens * hidden) || !valid(nw, hidden) ||
        !valid(weight, vocab * hidden) || !normalized.data || normalized.count != x.count ||
        !logits.data || logits.count != tokens * vocab || logits.count > SIZE_MAX / sizeof(float)) return 1;
    for (i = 0; i < tokens; ++i) {
        norm(x.data + i * hidden, nw.data, hidden, epsilon, 1, normalized.data + i * hidden);
        matvec(normalized.data + i * hidden, weight.data, vocab, hidden, logits.data + i * vocab);
    }
    return valid(logits, tokens * vocab) ? 0 : 3;
}

static int layer_run(const Config *c, Layer *l, uint64_t batch, uint64_t tokens,
    uint64_t start, const uint8_t *mask, uint64_t mask_count,
    const int64_t *positions, uint64_t position_count, Tensor x,
    Tensor routers, int64_t *indices, uint64_t index_count, Tensor routing, Tensor attention,
    ExpertFFN selected_expert, void *expert_user, uint64_t global_layer,
    const uint64_t *mrope_section, uint64_t position_bound) {
    uint64_t h, b, t, i, j, e, rank, end, rows, kw, vw, width, aw, scratch_count, work_width;
    uint64_t counts[19];
    float *scratch = NULL, *z, *qproj, *kproj, *vproj, *context, *temp, *expert, *shared, *prob;
    float *q, *k, *v, *gate, *state, *history;
    if (!geometry(c) || !position_bound || position_bound > INT64_MAX || !l || l->kind > 1 || !batch || batch > 128 ||
        !tokens || tokens > c->capacity || start > c->capacity - tokens) return 1;
    end = start + tokens; rows = batch * tokens; h = c->hidden;
    kw = c->key_heads * c->key_dim; vw = c->value_heads * c->value_dim;
    width = 2 * kw + vw; aw = c->heads * c->dim;
    if (mrope_section && (mrope_section[0] > c->rotary / 2 ||
        mrope_section[1] > c->rotary / 2 || mrope_section[2] > c->rotary / 2 ||
        mrope_section[0] + mrope_section[1] + mrope_section[2] != c->rotary / 2)) return 1;
    if (!mask || mask_count != batch * end || !positions || position_count != rows * (mrope_section ? 3 : 1) ||
        !indices || index_count != rows * c->topk || !valid(x, rows * h) ||
        !routers.data || routers.count != rows * c->experts ||
        !routing.data || routing.count != rows * c->topk) return 1;
    for (i = 0; i < mask_count; ++i) if (mask[i] > 1) return 1;
    for (i = 0; i < position_count; ++i) if (positions[i] < 0 || (uint64_t)positions[i] >= position_bound) return 1;
    counts[0] = h; counts[1] = h; counts[2] = c->experts * h;
    counts[3] = c->experts * 2 * c->intermediate * h;
    counts[4] = c->experts * h * c->intermediate;
    counts[5] = c->shared * h; counts[6] = c->shared * h; counts[7] = h * c->shared; counts[8] = h;
    if (l->kind == 0) {
        counts[9] = 2 * aw * h; counts[10] = c->kv * c->dim * h;
        counts[11] = counts[10]; counts[12] = h * aw;
        counts[13] = c->dim; counts[14] = c->dim;
        for (i = 15; i < 19; ++i) counts[i] = 0;
        if (!valid(l->left, batch * c->kv * c->capacity * c->dim) ||
            !valid(l->right, batch * c->kv * c->capacity * c->dim) ||
            !attention.data || attention.count != batch * c->heads * tokens * end) return 1;
    } else {
        counts[9] = width * h; counts[10] = vw * h;
        counts[11] = c->value_heads * h; counts[12] = c->value_heads * h;
        counts[13] = width * c->kernel; counts[14] = c->value_dim;
        counts[15] = c->value_heads; counts[16] = c->value_heads; counts[17] = h * vw; counts[18] = 0;
        if (!valid(l->left, batch * width * c->kernel) ||
            !valid(l->right, batch * c->value_heads * c->key_dim * c->value_dim) || attention.count) return 1;
    }
    for (i = 0; i < 19; ++i) {
        if (selected_expert && (i == 3 || i == 4)) {
            if (l->w[i].data || l->w[i].count) return 1;
        } else if (counts[i] && !valid(l->w[i], counts[i])) return 1;
    }
    /* One bounded allocation per layer; the facade separately bounds total buffers. */
    work_width = aw > 2 * vw ? aw : 2 * vw;
    if (work_width < h) work_width = h;
    scratch_count = rows * (h + 2 * aw + 2 * c->kv * c->dim) +
        width + work_width + h + 1 + 2 * c->intermediate + 2 * c->shared + c->experts +
        2 * c->value_heads + 2 * c->key_dim + 2 * c->value_dim;
    scratch = allocate(scratch_count);
    if (!scratch) return 2;
    z = scratch; qproj = z + rows * h; kproj = qproj + rows * 2 * aw;
    vproj = kproj + rows * c->kv * c->dim; context = vproj + rows * c->kv * c->dim;
    temp = context + width; expert = temp + work_width + h + 1;
    shared = expert + 2 * c->intermediate; prob = shared + 2 * c->shared;
    for (i = 0; i < rows; ++i) norm(x.data + i*h, l->w[0].data, h, c->epsilon, 1, z + i*h);
    if (l->kind == 0) {
        for (b = 0; b < batch; ++b) for (t = 0; t < tokens; ++t) {
            uint64_t row = b*tokens+t;
            matvec(z+row*h, l->w[9].data, 2*aw, h, qproj+row*2*aw);
            matvec(z+row*h, l->w[10].data, c->kv*c->dim, h, kproj+row*c->kv*c->dim);
            matvec(z+row*h, l->w[11].data, c->kv*c->dim, h, vproj+row*c->kv*c->dim);
            for (i = 0; i < c->heads; ++i) {
                q = qproj+row*2*aw+i*2*c->dim;
                norm(q, l->w[13].data, c->dim, c->epsilon, 1, q);
                rotate(q, c->rotary, positions, row, rows, mrope_section, c->theta);
            }
            for (i = 0; i < c->kv; ++i) {
                uint64_t offset = ((b*c->kv+i)*c->capacity+start+t)*c->dim;
                k = kproj+row*c->kv*c->dim+i*c->dim;
                norm(k, l->w[14].data, c->dim, c->epsilon, 1, k);
                rotate(k, c->rotary, positions, row, rows, mrope_section, c->theta);
                memcpy(l->left.data+offset, k, (size_t)c->dim*sizeof(float));
                memcpy(l->right.data+offset, vproj+row*c->kv*c->dim+i*c->dim, (size_t)c->dim*sizeof(float));
            }
        }
        for (b = 0; b < batch; ++b) for (t = 0; t < tokens; ++t) {
            uint64_t row = b*tokens+t;
            for (i = 0; i < c->heads; ++i) {
                uint64_t kh = i/(c->heads/c->kv);
                float *scores = attention.data+((b*c->heads+i)*tokens+t)*end;
                double maximum = -DBL_MAX, total = 0;
                q = qproj+row*2*aw+i*2*c->dim; gate = q+c->dim;
                for (j = 0; j < end; ++j) {
                    double sum = 0;
                    if (j <= start+t && mask[b*end+j]) {
                        k = l->left.data+((b*c->kv+kh)*c->capacity+j)*c->dim;
                        for (e = 0; e < c->dim; ++e) sum += (double)q[e]*k[e];
                        scores[j] = (float)(sum/sqrt((double)c->dim));
                    } else scores[j] = -FLT_MAX;
                    if (scores[j] > maximum) maximum = scores[j];
                }
                for (j = 0; j < end; ++j) { scores[j] = (float)exp((double)scores[j]-maximum); total += scores[j]; }
                for (j = 0; j < end; ++j) scores[j] = (float)(scores[j]/total);
                for (e = 0; e < c->dim; ++e) {
                    double sum = 0;
                    for (j = 0; j < end; ++j) sum += (double)scores[j]*l->right.data[((b*c->kv+kh)*c->capacity+j)*c->dim+e];
                    temp[i*c->dim+e] = (float)sum*sigmoid(gate[e]);
                }
            }
            matvec(temp, l->w[12].data, h, aw, temp+work_width);
            for (j = 0; j < h; ++j) x.data[row*h+j] += temp[work_width+j];
        }
    } else {
        float *beta = prob+c->experts, *decay = beta+c->value_heads;
        float *qn = decay+c->value_heads, *kn = qn+c->key_dim, *read = kn+c->key_dim;
        for (b = 0; b < batch; ++b) for (t = 0; t < tokens; ++t) {
            uint64_t row = b*tokens+t;
            float *input = z+row*h;
            if (!mask[b*end+start+t]) memset(input, 0, (size_t)h*sizeof(float));
            matvec(input, l->w[9].data, width, h, context);
            matvec(input, l->w[10].data, vw, h, temp);
            matvec(input, l->w[11].data, c->value_heads, h, beta);
            matvec(input, l->w[12].data, c->value_heads, h, decay);
            for (j = 0; j < width; ++j) {
                double sum = 0;
                history = l->left.data+(b*width+j)*c->kernel;
                memmove(history, history+1, (size_t)(c->kernel-1)*sizeof(float));
                history[c->kernel-1] = context[j];
                for (e = 0; e < c->kernel; ++e) sum += (double)history[e]*l->w[13].data[j*c->kernel+e];
                context[j] = silu((float)sum);
            }
            for (i = 0; i < c->value_heads; ++i) {
                uint64_t kh = i/(c->value_heads/c->key_heads);
                double qs = 1e-6, ks = 1e-6;
                float forget = (float)exp(-exp((double)l->w[16].data[i])*softplus((double)decay[i]+l->w[15].data[i]));
                float write = sigmoid(beta[i]);
                q = context+kh*c->key_dim; k = context+kw+kh*c->key_dim;
                v = context+2*kw+i*c->value_dim;
                state = l->right.data+(b*c->value_heads+i)*c->key_dim*c->value_dim;
                for (j = 0; j < c->key_dim; ++j) { qs += (double)q[j]*q[j]; ks += (double)k[j]*k[j]; }
                for (j = 0; j < c->key_dim; ++j) {
                    qn[j] = (float)(q[j]/sqrt(qs)/sqrt((double)c->key_dim)); kn[j] = (float)(k[j]/sqrt(ks));
                }
                for (j = 0; j < c->key_dim*c->value_dim; ++j) state[j] *= forget;
                for (j = 0; j < c->value_dim; ++j) {
                    double prediction = 0, result = 0;
                    float error;
                    for (e = 0; e < c->key_dim; ++e) prediction += (double)kn[e]*state[e*c->value_dim+j];
                    error = (v[j]-(float)prediction)*write;
                    for (e = 0; e < c->key_dim; ++e) {
                        state[e*c->value_dim+j] += kn[e]*error;
                        result += (double)qn[e]*state[e*c->value_dim+j];
                    }
                    read[j] = (float)result;
                }
                norm(read, l->w[14].data, c->value_dim, c->epsilon, 0, read);
                for (j = 0; j < c->value_dim; ++j) temp[vw+i*c->value_dim+j] = read[j]*silu(temp[i*c->value_dim+j]);
            }
            matvec(temp+vw, l->w[17].data, h, vw, temp+work_width);
            for (j = 0; j < h; ++j) x.data[row*h+j] += temp[work_width+j];
        }
    }
    for (i = 0; i < rows; ++i) {
        float *input = z+i*h, *router = routers.data+i*c->experts;
        double maximum = -DBL_MAX, total = 0, selected_sum = 0;
        norm(x.data+i*h, l->w[1].data, h, c->epsilon, 1, input);
        matvec(input, l->w[2].data, c->experts, h, router);
        for (e = 0; e < c->experts; ++e) if (router[e] > maximum) maximum = router[e];
        for (e = 0; e < c->experts; ++e) { prob[e] = (float)exp((double)router[e]-maximum); total += prob[e]; }
        for (e = 0; e < c->experts; ++e) prob[e] = (float)(prob[e]/total);
        for (rank = 0; rank < c->topk; ++rank) {
            uint64_t best = 0;
            float score = -1;
            for (e = 0; e < c->experts; ++e) {
                int used = 0;
                for (j = 0; j < rank; ++j) if ((uint64_t)indices[i*c->topk+j] == e) used = 1;
                if (!used && prob[e] > score) { score = prob[e]; best = e; }
            }
            indices[i*c->topk+rank] = (int64_t)best; routing.data[i*c->topk+rank] = score; selected_sum += score;
        }
        for (rank = 0; rank < c->topk; ++rank) {
            float scale = (float)(routing.data[i*c->topk+rank]/selected_sum);
            e = (uint64_t)indices[i*c->topk+rank]; routing.data[i*c->topk+rank] = scale;
            if (selected_expert) {
                if (selected_expert(expert_user,global_layer,e,input,h,temp,h)) {
                    free(scratch); return 4;
                }
                for (j = 0; j < h; ++j) if (!isfinite(temp[j])) { free(scratch); return 3; }
            } else {
                matvec(input, l->w[3].data+e*2*c->intermediate*h, 2*c->intermediate, h, expert);
                for (j = 0; j < c->intermediate; ++j) expert[j] = silu(expert[j])*expert[c->intermediate+j];
                matvec(expert, l->w[4].data+e*h*c->intermediate, h, c->intermediate, temp);
            }
            for (j = 0; j < h; ++j) x.data[i*h+j] += temp[j]*scale;
        }
        matvec(input, l->w[5].data, c->shared, h, shared);
        matvec(input, l->w[6].data, c->shared, h, shared+c->shared);
        for (j = 0; j < c->shared; ++j) shared[j] = silu(shared[j])*shared[c->shared+j];
        matvec(shared, l->w[7].data, h, c->shared, temp);
        matvec(input, l->w[8].data, 1, h, temp+h);
        for (j = 0; j < h; ++j) x.data[i*h+j] += temp[j]*sigmoid(temp[h]);
    }
    free(scratch);
    return valid(x, rows*h) && valid(l->left, l->left.count) && valid(l->right, l->right.count) ? 0 : 3;
}

API int oa_qwen35_layer(const Config *c, Layer *l, uint64_t batch, uint64_t tokens,
    uint64_t start, const uint8_t *mask, uint64_t mask_count,
    const int64_t *positions, uint64_t position_count, Tensor x,
    Tensor routers, int64_t *indices, uint64_t index_count, Tensor routing, Tensor attention) {
    return layer_run(c,l,batch,tokens,start,mask,mask_count,positions,position_count,x,
        routers,indices,index_count,routing,attention,NULL,NULL,0,NULL,c ? c->capacity : 0);
}

API uint64_t oa_qwen35_streamed_version(void) { return 1; }

API int oa_qwen35_layer_streamed(const Config *c, Layer *l, uint64_t batch, uint64_t tokens,
    uint64_t start, const uint8_t *mask, uint64_t mask_count,
    const int64_t *positions, uint64_t position_count, Tensor x,
    Tensor routers, int64_t *indices, uint64_t index_count, Tensor routing, Tensor attention,
    ExpertFFN selected_expert, void *expert_user, uint64_t global_layer) {
    if (!selected_expert) return 1;
    return layer_run(c,l,batch,tokens,start,mask,mask_count,positions,position_count,x,
        routers,indices,index_count,routing,attention,selected_expert,expert_user,global_layer,NULL,c ? c->capacity : 0);
}

API uint64_t oa_qwen35_mrope_version(void) { return 1; }

API int oa_qwen35_layer_mrope(const Config *c, Layer *l, uint64_t batch, uint64_t tokens,
    uint64_t start, const uint8_t *mask, uint64_t mask_count,
    const int64_t *positions, uint64_t position_count, Tensor x,
    Tensor routers, int64_t *indices, uint64_t index_count, Tensor routing, Tensor attention,
    const uint64_t *section, ExpertFFN selected_expert, void *expert_user, uint64_t global_layer) {
    if (!section) return 1;
    return layer_run(c,l,batch,tokens,start,mask,mask_count,positions,position_count,x,
        routers,indices,index_count,routing,attention,selected_expert,expert_user,global_layer,section,c ? c->capacity : 0);
}

API uint64_t oa_qwen35_context_version(void) { return 1; }

API int oa_qwen35_layer_context(const Config *c, Layer *l, uint64_t batch, uint64_t tokens,
    uint64_t start, const uint8_t *mask, uint64_t mask_count,
    const int64_t *positions, uint64_t position_count, Tensor x,
    Tensor routers, int64_t *indices, uint64_t index_count, Tensor routing, Tensor attention,
    const uint64_t *section, ExpertFFN selected_expert, void *expert_user,
    uint64_t global_layer, uint64_t position_bound) {
    return layer_run(c,l,batch,tokens,start,mask,mask_count,positions,position_count,x,
        routers,indices,index_count,routing,attention,selected_expert,expert_user,
        global_layer,section,position_bound);
}
