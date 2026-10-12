/* Independent GLM-MoE-DSA, compressed MLA cache and additive selected-expert ABI. */
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
    uint64_t hidden, heads, qr, kr, nope, rope, value, idim, iheads, itopk;
    uint64_t experts, topk, groups, gtop, dense_width, moe_width, shared_width, capacity, normalize;
    double epsilon, theta, route_scale;
    uint64_t yarn, original_context, truncate;
    double factor, beta_fast, beta_slow, mscale, mscale_all_dim, attention_factor;
} Config;
typedef struct { uint64_t full_indexer, dense; Tensor w[21], latent, rotary, index_keys; } Layer;
static int geometry(const Config *c) {
    uint64_t fields[18];
    size_t i;
    if (!c) return 0;
    memcpy(fields, c, sizeof(fields));
    for (i = 0; i < 18; ++i)
        if (!fields[i] || fields[i] > (i == 17 ? 1048576 : 65536)) return 0;
    if (c->yarn > 1 || c->truncate > 1 || !c->original_context || c->original_context > 1048576 ||
        !isfinite(c->factor) || c->factor < 1 || c->factor > 1e12 ||
        !isfinite(c->beta_fast) || !isfinite(c->beta_slow) || c->beta_slow < 1e-12 ||
        c->beta_fast < c->beta_slow || c->beta_fast > 1e12 ||
        !isfinite(c->mscale) || c->mscale < 0 || c->mscale > 1e12 ||
        !isfinite(c->mscale_all_dim) || c->mscale_all_dim < 0 || c->mscale_all_dim > 1e12 ||
        !isfinite(c->attention_factor) || c->attention_factor < 0 || c->attention_factor > 1e12 ||
        (c->attention_factor && c->attention_factor < 1e-12) ||
        c->theta < 1e-12 || c->theta > 1e12 || (c->yarn && c->theta <= 1)) return 0;
    return c->hidden <= 8192 && c->heads <= 128 && c->iheads <= 128 && c->experts <= 4096 &&
        c->qr <= 8192 && c->kr <= 4096 && c->nope <= 4096 && c->rope <= 4096 && c->value <= 4096 &&
        c->idim <= 4096 && c->rope <= c->idim && c->rope % 2 == 0 && c->experts % c->groups == 0 &&
        c->experts / c->groups >= 2 && c->gtop <= c->groups &&
        c->topk <= c->gtop * (c->experts / c->groups) && c->normalize <= 1 &&
        isfinite(c->epsilon) && c->epsilon > 0 && isfinite(c->theta) && c->theta > 0 &&
        isfinite(c->route_scale) && c->route_scale > 0;
}
static int valid(Tensor t, uint64_t count) {
    uint64_t i;
    if (!t.data || t.count != count || count > SIZE_MAX / sizeof(float)) return 0;
    for (i = 0; i < count; ++i) if (!isfinite(t.data[i])) return 0;
    return 1;
}
static int valid_cache_prefix(Tensor t, uint64_t batch, uint64_t capacity,
    uint64_t channels, uint64_t used) {
    uint64_t b, i, count;
    if (!batch || !capacity || !channels || used > capacity ||
        batch > UINT64_MAX / capacity || batch * capacity > UINT64_MAX / channels) return 0;
    count = batch * capacity * channels;
    if (!t.data || t.count != count || count > SIZE_MAX / sizeof(float)) return 0;
    for (b = 0; b < batch; ++b) for (i = 0; i < used * channels; ++i)
        if (!isfinite(t.data[b * capacity * channels + i])) return 0;
    return 1;
}
static float *allocate(uint64_t n) {
    if (n > SIZE_MAX / sizeof(float)) return NULL;
    return (float *)calloc((size_t)n, sizeof(float));
}
static void mv(const float *x, const float *w, uint64_t rows, uint64_t cols, float *out) {
    uint64_t i, j;
    for (i = 0; i < rows; ++i) {
        double sum = 0;
        for (j = 0; j < cols; ++j) sum += (double)x[j] * w[i * cols + j];
        out[i] = (float)sum;
    }
}
static void rms(const float *x, const float *w, uint64_t n, double eps, float *out) {
    uint64_t j;
    double sum = 0, scale;
    for (j = 0; j < n; ++j) sum += (double)x[j] * x[j];
    scale = 1 / sqrt(sum / (double)n + eps);
    for (j = 0; j < n; ++j) out[j] = (float)(x[j] * scale * w[j]);
}
static void layernorm(float *x, const float *w, const float *bias, uint64_t n) {
    uint64_t j;
    double mean = 0, variance = 0;
    for (j = 0; j < n; ++j) mean += x[j];
    mean /= (double)n;
    for (j = 0; j < n; ++j) variance += (x[j] - mean) * (x[j] - mean);
    variance = 1 / sqrt(variance / (double)n + 1e-6);
    for (j = 0; j < n; ++j) x[j] = (float)((x[j] - mean) * variance * w[j] + bias[j]);
}
static float sigmoid(float x) {
    if (x >= 0) return (float)(1 / (1 + exp(-(double)x)));
    { double e = exp((double)x); return (float)(e / (1 + e)); }
}
static float silu(float x) { return x * sigmoid(x); }
/* Adjacent input pairs; output stores all real components then all imaginary. */
static double magnitude(double factor, double multiplier) {
    return factor <= 1 ? 1 : 1 + 0.1 * multiplier * log(factor);
}
static void rotate(float *x, uint64_t n, int64_t position, const Config *c, float *work) {
    uint64_t j;
    double low = 0, high = 0, amplitude = 1;
    if (c->yarn) {
        double coefficient = (double)n / (2 * log(c->theta));
        double turns = (double)c->original_context / (2 * acos(-1.0));
        low = coefficient * log(turns / c->beta_fast);
        high = coefficient * log(turns / c->beta_slow);
        if (c->truncate) { low = floor(low); high = ceil(high); }
        if (low < 0) low = 0;
        if (high > (double)n - 1) high = (double)n - 1;
        if (low == high) high += 0.001;
        amplitude = c->attention_factor;
        if (!amplitude) amplitude = c->mscale && c->mscale_all_dim ?
            magnitude(c->factor, c->mscale) / magnitude(c->factor, c->mscale_all_dim) :
            magnitude(c->factor, 1);
    }
    for (j = 0; j < n / 2; ++j) {
        double angle = (double)position * pow(c->theta, -(double)(2 * j) / (double)n);
        float a = x[2 * j], b = x[2 * j + 1];
        if (c->yarn) {
            /* Match the FP32 frequency/phase contract; no Python frequency table. */
            float power = powf((float)c->theta, (float)(2 * j) / (float)n);
            float ramp = ((float)j - (float)low) / (float)(high - low);
            float extrapolation, frequency, phase, cosine, sine;
            if (ramp < 0) ramp = 0;
            if (ramp > 1) ramp = 1;
            extrapolation = 1 - ramp;
            frequency = (1 / ((float)c->factor * power)) * (1 - extrapolation) +
                (1 / power) * extrapolation;
            phase = (float)position * frequency;
            cosine = cosf(phase) * (float)amplitude;
            sine = sinf(phase) * (float)amplitude;
            work[j] = a * cosine - b * sine;
            work[j + n / 2] = a * sine + b * cosine;
            continue;
        }
        work[j] = (float)(a * cos(angle) - b * sin(angle));
        work[j + n / 2] = (float)(a * sin(angle) + b * cos(angle));
    }
    memcpy(x, work, (size_t)n * sizeof(float));
}
static uint64_t best_unused(const float *score, uint64_t n, const int64_t *used, uint64_t count) {
    uint64_t e, j, best = 0;
    double maximum = -DBL_MAX;
    for (e = 0; e < n; ++e) {
        int found = 0;
        for (j = 0; j < count; ++j) if ((uint64_t)used[j] == e) found = 1;
        if (!found && score[e] > maximum) { maximum = score[e]; best = e; }
    }
    return best;
}
API uint64_t oa_glm_dsa_version(void) { return 2; }
API uint64_t oa_glm_dsa_optional_diagnostics_version(void) { return 1; }
API uint64_t oa_glm_dsa_cache_expansion_version(void) { return 1; }
API int oa_glm_dsa_geometry(const Config *c) { return geometry(c) ? 0 : 1; }
API int oa_glm_dsa_expand_cache(const Config *c, Tensor projection, Tensor latent,
    Tensor rotary, uint64_t batch, uint64_t used, Tensor keys, Tensor values) {
    uint64_t b, head, t, i, rank, key_count, value_count, qdim;
    if (!geometry(c) || !batch || batch > 128 || used > c->capacity) return 1;
    qdim = c->nope + c->rope;
    key_count = batch * c->heads * used * qdim;
    value_count = batch * c->heads * used * c->value;
    if (keys.count != key_count || values.count != value_count ||
        key_count > SIZE_MAX / sizeof(float) || value_count > SIZE_MAX / sizeof(float)) return 1;
    if (!used) return 0;
    if (!keys.data || !values.data ||
        !valid(projection, c->heads * (c->nope + c->value) * c->kr) ||
        !valid_cache_prefix(latent, batch, c->capacity, c->kr, used) ||
        !valid_cache_prefix(rotary, batch, c->capacity, c->rope, used)) return 1;
    for (b = 0; b < batch; ++b) for (head = 0; head < c->heads; ++head)
        for (t = 0; t < used; ++t) {
            const float *input = latent.data + (b * c->capacity + t) * c->kr;
            const float *weight = projection.data + head * (c->nope + c->value) * c->kr;
            float *key = keys.data + ((b * c->heads + head) * used + t) * qdim;
            float *value = values.data + ((b * c->heads + head) * used + t) * c->value;
            for (i = 0; i < c->nope; ++i) {
                double sum = 0;
                for (rank = 0; rank < c->kr; ++rank) sum += (double)input[rank] * weight[i * c->kr + rank];
                key[i] = (float)sum;
            }
            memcpy(key + c->nope, rotary.data + (b * c->capacity + t) * c->rope,
                (size_t)c->rope * sizeof(float));
            for (i = 0; i < c->value; ++i) {
                double sum = 0;
                for (rank = 0; rank < c->kr; ++rank)
                    sum += (double)input[rank] * weight[(c->nope + i) * c->kr + rank];
                value[i] = (float)sum;
            }
        }
    return valid(keys, key_count) && valid(values, value_count) ? 0 : 3;
}
API int oa_glm_dsa_embed(Tensor weight, uint64_t vocab, uint64_t hidden,
    const int64_t *ids, uint64_t rows, Tensor out) {
    uint64_t i;
    if (!vocab || vocab > 1048576 || !hidden || hidden > 8192 || !rows ||
        rows > SIZE_MAX / sizeof(float) / hidden || rows > SIZE_MAX / sizeof(int64_t) ||
        !ids || !out.data || out.count != rows * hidden || !valid(weight, vocab * hidden)) return 1;
    for (i = 0; i < rows; ++i) if (ids[i] < 0 || (uint64_t)ids[i] >= vocab) return 1;
    for (i = 0; i < rows; ++i) memcpy(out.data + i * hidden,
        weight.data + (uint64_t)ids[i] * hidden, (size_t)hidden * sizeof(float));
    return 0;
}
API int oa_glm_dsa_head(Tensor x, Tensor nw, Tensor weight, uint64_t hidden,
    uint64_t vocab, double epsilon, Tensor normalized, Tensor logits) {
    uint64_t i, rows;
    if (!hidden || hidden > 8192 || !vocab || vocab > 1048576 || x.count % hidden ||
        !isfinite(epsilon) || epsilon <= 0) return 1;
    rows = x.count / hidden;
    if (!rows || rows > SIZE_MAX / sizeof(float) / hidden || rows > SIZE_MAX / sizeof(float) / vocab ||
        !valid(x, rows * hidden) || !valid(nw, hidden) ||
        !valid(weight, vocab * hidden) || !normalized.data || normalized.count != x.count ||
        !logits.data || logits.count != rows * vocab) return 1;
    for (i = 0; i < rows; ++i) {
        rms(x.data + i * hidden, nw.data, hidden, epsilon, normalized.data + i * hidden);
        mv(normalized.data + i * hidden, weight.data, vocab, hidden, logits.data + i * vocab);
    }
    return valid(logits, rows * vocab) ? 0 : 3;
}
typedef int (*DsaExpert)(void *, uint64_t, const float *, uint64_t, float *, uint64_t);
API uint64_t oa_glm_dsa_streamed_experts_version(void) { return 1; }
static int layer_execute(const Config *c, Layer *l, uint64_t batch, uint64_t tokens, uint64_t start,
    const uint8_t *mask, uint64_t mask_count, const int64_t *positions, uint64_t position_count,
    Tensor x, int64_t *selected, uint64_t selected_count, Tensor attention, Tensor routers,
    int64_t *experts, uint64_t expert_count, Tensor routing, DsaExpert expert, void *owner) {
    uint64_t b, t, row, head, i, j, e, rank, end, rows, h, qdim, indexk, group_width;
    uint64_t counts[21], scratch_count, ffwidth;
    float *scratch, *z, *qres, *q, *iq, *head_weights, *work, *query_latent, *context_latent;
    float *context, *output, *gate, *up, *scores, *group_scores, *choice;
    if (!geometry(c) || !l || l->full_indexer > 1 || l->dense > 1 || !batch || batch > 128 ||
        !tokens || tokens > c->capacity || start > c->capacity - tokens) return 1;
    h = c->hidden; rows = batch * tokens; end = start + tokens;
    qdim = c->nope + c->rope; indexk = c->itopk < end ? c->itopk : end;
    group_width = c->experts / c->groups;
    if (!mask || mask_count != batch * end || !positions || position_count != rows ||
        !selected || selected_count != rows * indexk || !valid(x, rows * h) ||
        (attention.data ? attention.count != batch * c->heads * tokens * end : attention.count != 0) ||
        !valid_cache_prefix(l->latent, batch, c->capacity, c->kr, start) ||
        !valid_cache_prefix(l->rotary, batch, c->capacity, c->rope, start)) return 1;
    for (i = 0; i < mask_count; ++i) if (mask[i] > 1) return 1;
    for (i = 0; i < rows; ++i) if (positions[i] < 0 || (uint64_t)positions[i] >= c->capacity) return 1;
    memset(counts, 0, sizeof(counts));
    counts[0] = h; counts[1] = h; counts[2] = c->qr * h; counts[3] = c->qr;
    counts[4] = c->heads * qdim * c->qr; counts[5] = (c->kr + c->rope) * h;
    counts[6] = c->kr; counts[7] = c->heads * (c->nope + c->value) * c->kr;
    counts[8] = h * c->heads * c->value;
    if (l->full_indexer) {
        counts[9] = c->iheads * c->idim * c->qr; counts[10] = c->idim * h;
        counts[11] = c->idim; counts[12] = c->idim; counts[13] = c->iheads * h;
        if (!valid_cache_prefix(l->index_keys, batch, c->capacity, c->idim, start)) return 1;
    } else {
        for (i = 0; i < selected_count; ++i) if (selected[i] < 0 || (uint64_t)selected[i] >= end) return 1;
        if (l->index_keys.count) return 1;
    }
    if (l->dense) {
        counts[14] = c->dense_width * h; counts[15] = counts[14]; counts[16] = h * c->dense_width;
        if (routers.count || expert_count || routing.count) return 1;
    } else {
        counts[14] = c->experts * h; counts[15] = c->experts;
        if (!expert) {
            counts[16] = c->experts * 2 * c->moe_width * h;
            counts[17] = c->experts * h * c->moe_width;
        } else if (l->w[16].count || l->w[17].count) return 1;
        counts[18] = c->shared_width * h; counts[19] = counts[18]; counts[20] = h * c->shared_width;
        if (!routers.data || routers.count != rows * c->experts || !experts ||
            expert_count != rows * c->topk || !routing.data || routing.count != rows * c->topk) return 1;
    }
    for (i = 0; i < 21; ++i) if (counts[i] && !valid(l->w[i], counts[i])) return 1;
    ffwidth = c->dense_width > 2 * c->moe_width ? c->dense_width : 2 * c->moe_width;
    if (ffwidth < c->shared_width) ffwidth = c->shared_width;
    scratch_count = rows * (h + c->qr + c->heads * qdim + c->iheads * c->idim + c->iheads) +
        c->kr + 2 * c->rope + c->idim + 2 * c->kr + c->heads * c->value + h +
        2 * ffwidth + end + c->groups + 2 * c->experts;
    scratch = allocate(scratch_count);
    if (!scratch) return 2;
    z = scratch; qres = z + rows * h; q = qres + rows * c->qr;
    iq = q + rows * c->heads * qdim; head_weights = iq + rows * c->iheads * c->idim;
    work = head_weights + rows * c->iheads; query_latent = work + c->kr + 2 * c->rope + c->idim;
    context_latent = query_latent + c->kr; context = context_latent + c->kr;
    output = context + c->heads * c->value; gate = output + h; up = gate + ffwidth;
    scores = up + ffwidth; group_scores = scores + end; choice = group_scores + c->groups;
    for (row = 0; row < rows; ++row) {
        int64_t position = positions[row];
        b = row / tokens; t = row % tokens;
        rms(x.data + row * h, l->w[0].data, h, c->epsilon, z + row * h);
        mv(z + row * h, l->w[2].data, c->qr, h, qres + row * c->qr);
        rms(qres + row * c->qr, l->w[3].data, c->qr, 1e-6, qres + row * c->qr);
        mv(qres + row * c->qr, l->w[4].data, c->heads * qdim, c->qr, q + row * c->heads * qdim);
        for (head = 0; head < c->heads; ++head)
            rotate(q + (row * c->heads + head) * qdim + c->nope, c->rope, position, c, work);
        mv(z + row * h, l->w[5].data, c->kr + c->rope, h, work);
        rms(work, l->w[6].data, c->kr, 1e-6, l->latent.data + (b * c->capacity + start + t) * c->kr);
        rotate(work + c->kr, c->rope, position, c, work + c->kr + c->rope);
        memcpy(l->rotary.data + (b * c->capacity + start + t) * c->rope, work + c->kr, (size_t)c->rope * sizeof(float));
        if (l->full_indexer) {
            float *ik = l->index_keys.data + (b * c->capacity + start + t) * c->idim;
            mv(qres + row * c->qr, l->w[9].data, c->iheads * c->idim, c->qr, iq + row * c->iheads * c->idim);
            for (head = 0; head < c->iheads; ++head)
                rotate(iq + (row * c->iheads + head) * c->idim, c->rope, position, c, work);
            mv(z + row * h, l->w[10].data, c->idim, h, ik);
            layernorm(ik, l->w[11].data, l->w[12].data, c->idim);
            rotate(ik, c->rope, position, c, work);
            mv(z + row * h, l->w[13].data, c->iheads, h, head_weights + row * c->iheads);
        }
    }
    for (i = 0; i < rows * (h + c->qr + c->heads * qdim + c->iheads * c->idim + c->iheads); ++i)
        if (!isfinite(scratch[i])) { free(scratch); return 3; }
    for (b = 0; b < batch; ++b) for (t = 0; t < tokens; ++t) {
        row = b * tokens + t;
        if (l->full_indexer) {
            for (j = 0; j < end; ++j) {
                double sum = 0;
                if (j <= start + t && mask[b * end + j]) {
                    for (head = 0; head < c->iheads; ++head) {
                        double dot = 0;
                        for (e = 0; e < c->idim; ++e) dot +=
                            (double)iq[(row * c->iheads + head) * c->idim + e] * l->index_keys.data[(b * c->capacity + j) * c->idim + e];
                        if (dot > 0) sum += head_weights[row * c->iheads + head] * dot / sqrt((double)c->idim * (double)c->iheads);
                    }
                    scores[j] = (float)sum;
                } else scores[j] = -FLT_MAX;
            }
            for (rank = 0; rank < indexk; ++rank)
                selected[row * indexk + rank] = (int64_t)best_unused(scores, end, selected + row * indexk, rank);
        }
        for (head = 0; head < c->heads; ++head) {
            float *query = q + (row * c->heads + head) * qdim;
            const float *projection = l->w[7].data + head * (c->nope + c->value) * c->kr;
            float *attention_row = attention.data ?
                attention.data + ((b * c->heads + head) * tokens + t) * end : scores;
            double maximum = -DBL_MAX, total = 0;
            for (e = 0; e < c->kr; ++e) {
                double sum = 0;
                for (i = 0; i < c->nope; ++i) sum += (double)query[i] * projection[i * c->kr + e];
                query_latent[e] = (float)sum;
            }
            for (j = 0; j < end; ++j) {
                int chosen = 0;
                double sum = 0;
                for (rank = 0; rank < indexk; ++rank) if ((uint64_t)selected[row * indexk + rank] == j) chosen = 1;
                if (chosen && j <= start + t && mask[b * end + j]) {
                    for (e = 0; e < c->kr; ++e) sum += (double)query_latent[e] * l->latent.data[(b * c->capacity + j) * c->kr + e];
                    for (e = 0; e < c->rope; ++e) sum += (double)query[c->nope + e] * l->rotary.data[(b * c->capacity + j) * c->rope + e];
                    double multiplier = c->yarn ? magnitude(c->factor, c->mscale_all_dim) : 1;
                    attention_row[j] = (float)(sum * multiplier * multiplier / sqrt((double)qdim));
                } else attention_row[j] = -FLT_MAX;
                if (attention_row[j] > maximum) maximum = attention_row[j];
            }
            for (j = 0; j < end; ++j) { attention_row[j] = (float)exp((double)attention_row[j] - maximum); total += attention_row[j]; }
            for (j = 0; j < end; ++j) attention_row[j] = (float)(attention_row[j] / total);
            if (!attention.data) for (j = 0; j < end; ++j)
                if (!isfinite(attention_row[j])) { free(scratch); return 3; }
            for (e = 0; e < c->kr; ++e) {
                double sum = 0;
                for (j = 0; j < end; ++j) sum += (double)attention_row[j] * l->latent.data[(b * c->capacity + j) * c->kr + e];
                context_latent[e] = (float)sum;
            }
            mv(context_latent, projection + c->nope * c->kr, c->value, c->kr, context + head * c->value);
        }
        mv(context, l->w[8].data, h, c->heads * c->value, output);
        for (j = 0; j < h; ++j) x.data[row * h + j] += output[j];
        rms(x.data + row * h, l->w[1].data, h, c->epsilon, z + row * h);
        if (l->dense) {
            mv(z + row * h, l->w[14].data, c->dense_width, h, gate);
            mv(z + row * h, l->w[15].data, c->dense_width, h, up);
            for (j = 0; j < c->dense_width; ++j) gate[j] = silu(gate[j]) * up[j];
            mv(gate, l->w[16].data, h, c->dense_width, output);
            for (j = 0; j < h; ++j) x.data[row * h + j] += output[j];
        } else {
            float *router = routers.data + row * c->experts;
            float *prob = choice + c->experts;
            double denominator = 1e-20;
            mv(z + row * h, l->w[14].data, c->experts, h, router);
            for (e = 0; e < c->experts; ++e) { prob[e] = sigmoid(router[e]); choice[e] = prob[e] + l->w[15].data[e]; }
            for (i = 0; i < c->groups; ++i) {
                double first = -DBL_MAX, second = -DBL_MAX;
                for (e = 0; e < group_width; ++e) {
                    double score = choice[i * group_width + e];
                    if (score > first) { second = first; first = score; } else if (score > second) second = score;
                }
                group_scores[i] = (float)(first + second);
                if (!isfinite(group_scores[i])) { free(scratch); return 3; }
            }
            /* Select groups without changing expert correction scores. */
            for (i = 0; i < c->gtop; ++i) {
                uint64_t best = 0;
                double maximum_group = -DBL_MAX;
                for (j = 0; j < c->groups; ++j)
                    if (group_scores[j] != FLT_MAX && group_scores[j] > maximum_group) {
                        maximum_group = group_scores[j]; best = j;
                    }
                group_scores[best] = FLT_MAX;
            }
            for (e = 0; e < c->experts; ++e) if (group_scores[e / group_width] != FLT_MAX) choice[e] = -FLT_MAX;
            for (rank = 0; rank < c->topk; ++rank) {
                e = best_unused(choice, c->experts, experts + row * c->topk, rank);
                experts[row * c->topk + rank] = (int64_t)e;
                routing.data[row * c->topk + rank] = prob[e]; denominator += prob[e];
            }
            for (rank = 0; rank < c->topk; ++rank) {
                float scale = (float)(routing.data[row * c->topk + rank] * c->route_scale / (c->normalize ? denominator : 1));
                e = (uint64_t)experts[row * c->topk + rank]; routing.data[row * c->topk + rank] = scale;
                if (expert) {
                    Tensor result;
                    result.data = output; result.count = h;
                    if (expert(owner, e, z + row*h, h, output, h) || !valid(result, h)) {
                        free(scratch); return 4;
                    }
                } else {
                    mv(z + row * h, l->w[16].data + e * 2 * c->moe_width * h, 2 * c->moe_width, h, gate);
                    for (j = 0; j < c->moe_width; ++j) gate[j] = silu(gate[j]) * gate[c->moe_width + j];
                    mv(gate, l->w[17].data + e * h * c->moe_width, h, c->moe_width, output);
                }
                for (j = 0; j < h; ++j) x.data[row * h + j] += output[j] * scale;
            }
            mv(z + row * h, l->w[18].data, c->shared_width, h, gate);
            mv(z + row * h, l->w[19].data, c->shared_width, h, up);
            for (j = 0; j < c->shared_width; ++j) gate[j] = silu(gate[j]) * up[j];
            mv(gate, l->w[20].data, h, c->shared_width, output);
            for (j = 0; j < h; ++j) x.data[row * h + j] += output[j];
        }
    }
    free(scratch);
    return valid(x, rows * h) && (!attention.data || valid(attention, attention.count)) &&
        (l->dense || (valid(routers, routers.count) && valid(routing, routing.count))) &&
        valid_cache_prefix(l->latent, batch, c->capacity, c->kr, end) &&
        valid_cache_prefix(l->rotary, batch, c->capacity, c->rope, end) &&
        (!l->full_indexer || valid_cache_prefix(l->index_keys, batch, c->capacity, c->idim, end)) ? 0 : 3;
}
API int oa_glm_dsa_layer(const Config *c, Layer *l, uint64_t batch, uint64_t tokens, uint64_t start,
    const uint8_t *mask, uint64_t mask_count, const int64_t *positions, uint64_t position_count,
    Tensor x, int64_t *selected, uint64_t selected_count, Tensor attention, Tensor routers,
    int64_t *experts, uint64_t expert_count, Tensor routing) {
    return layer_execute(c,l,batch,tokens,start,mask,mask_count,positions,position_count,
        x,selected,selected_count,attention,routers,experts,expert_count,routing,NULL,NULL);
}
API int oa_glm_dsa_layer_streamed(const Config *c, Layer *l, uint64_t batch, uint64_t tokens, uint64_t start,
    const uint8_t *mask, uint64_t mask_count, const int64_t *positions, uint64_t position_count,
    Tensor x, int64_t *selected, uint64_t selected_count, Tensor attention, Tensor routers,
    int64_t *experts, uint64_t expert_count, Tensor routing, DsaExpert expert, void *owner) {
    if (!expert) return 1;
    return layer_execute(c,l,batch,tokens,start,mask,mask_count,positions,position_count,
        x,selected,selected_count,attention,routers,experts,expert_count,routing,expert,owner);
}
