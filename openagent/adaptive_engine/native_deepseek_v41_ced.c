/* Original bounded dense CED/CSA2 equation execution, not publisher ABI. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

typedef struct {
    int hidden, layers, encoder, vocabulary, capacity, window;
    int ratio, topk, block_size, candidate_blocks, intermediate;
    double epsilon;
} Config;
typedef struct { char prefix; Config value; } ConfigAlignment;

static int extent(const void *p, size_t bytes, size_t alignment) {
    uintptr_t a = (uintptr_t)p;
    return p && !(a % alignment) && bytes <= UINTPTR_MAX - a;
}
static int overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x < y + nb && y < x + na;
}
static int finite_array(const float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (!isfinite(x[i])) return 0;
    return 1;
}
static int store(double x, float *out) {
    if (!isfinite(x) || fabs(x) > FLT_MAX) return 0;
    *out = (float)x;
    return 1;
}
static int matrix(const float *w, const float *x, int rows, int cols, float *out) {
    int r, c;
    for (r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (c = 0; c < cols; ++c) sum += (double)w[r * cols + c] * x[c];
        if (!store(sum, out + r)) return 0;
    }
    return 1;
}
static int norm(const float *x, const float *gain, int n, double eps, float *out) {
    double square = 0.0, factor;
    int i;
    for (i = 0; i < n; ++i) square += (double)x[i] * x[i];
    factor = 1.0 / sqrt(square / n + eps);
    for (i = 0; i < n; ++i)
        if (!store(x[i] * factor * gain[i], out + i)) return 0;
    return 1;
}
static double dot(const float *a, const float *b, int n) {
    double result = 0.0;
    int i;
    for (i = 0; i < n; ++i) result += (double)a[i] * b[i];
    return result;
}
static int geometry(const Config *c) {
    return c && c->hidden >= 1 && c->hidden <= 16 && c->layers >= 2 && c->layers <= 12 &&
        c->encoder >= 1 && c->encoder < c->layers && c->vocabulary >= 2 &&
        c->vocabulary <= 256 && c->capacity >= 1 && c->capacity <= 64 &&
        c->window >= 1 && c->window <= c->capacity && c->ratio >= 1 && c->ratio <= 4 &&
        c->topk >= 1 && c->topk <= 8 && c->block_size >= 1 && c->block_size <= 8 &&
        c->candidate_blocks >= 1 && c->candidate_blocks <= 8 &&
        c->intermediate >= 1 && c->intermediate <= 64 && isfinite(c->epsilon) && c->epsilon > 0;
}
static size_t weight_count(const Config *c) {
    size_t h = (size_t)c->hidden;
    return 2 * h * c->vocabulary + h + c->layers * (2 * h + 9 * h * h + 3 * h * c->intermediate);
}
API uint32_t oa_v41_ced_version(void) { return 1; }

/* modes: 0 SWA only, 1 Full, 2 Reindex, 3 Reuse. */
API int oa_v41_ced_forward(const Config *c, const int32_t *modes, size_t nm,
    const float *weights, size_t nw, const int32_t *ids, size_t n,
    float *output, size_t no, int32_t *trace, size_t trace_count,
    size_t budget, int fail_layer) {
    size_t h, stride, cache_count, trace_width, required, i;
    float *allocation, *local_k, *local_v, *raw, *zraw, *encoder, *staged;
    int32_t *staged_trace;
    int t, l, r, j, d, owner, entries, selected_count, candidate_count;
    int selected[8], candidates[8];
    float main_kv[64 * 16], index_keys[64 * 16];
    float x[16], normalized[16], q[16], gq[16], k[16], v[16], iq[16];
    float combined[16], update[16], gate[64], up[64], product[64];
    double score[64], attention_scores[128], maximum, denominator;
    const float *base, *final_gain, *head;
    int status = 3, seen_encoder = 0, seen_decoder = 0;
    if (!extent(c, sizeof(Config), offsetof(ConfigAlignment, value)) ||
        !geometry(c) || nm != (size_t)c->layers || !n || n > (size_t)c->capacity ||
        fail_layer < -1 || fail_layer >= c->layers) return 1;
    h = (size_t)c->hidden;
    stride = 2 * h + 9 * h * h + 3 * h * c->intermediate;
    trace_width = 2 + (size_t)c->topk;
    if (nw != weight_count(c) || no != n * c->vocabulary ||
        trace_count != (size_t)c->layers * trace_width ||
        !extent(modes, nm * sizeof(int32_t), sizeof(int32_t)) ||
        !extent(weights, nw * sizeof(float), sizeof(float)) ||
        !extent(ids, n * sizeof(int32_t), sizeof(int32_t)) ||
        !extent(output, no * sizeof(float), sizeof(float)) ||
        !extent(trace, trace_count * sizeof(int32_t), sizeof(int32_t))) return 1;
    if (overlap(output, no * sizeof(float), trace, trace_count * sizeof(int32_t)) ||
        overlap(output, no * sizeof(float), c, sizeof(Config)) ||
        overlap(trace, trace_count * sizeof(int32_t), c, sizeof(Config)) ||
        overlap(output, no * sizeof(float), weights, nw * sizeof(float)) ||
        overlap(trace, trace_count * sizeof(int32_t), weights, nw * sizeof(float)) ||
        overlap(output, no * sizeof(float), ids, n * sizeof(int32_t)) ||
        overlap(trace, trace_count * sizeof(int32_t), ids, n * sizeof(int32_t)) ||
        overlap(output, no * sizeof(float), modes, nm * sizeof(int32_t)) ||
        overlap(trace, trace_count * sizeof(int32_t), modes, nm * sizeof(int32_t))) return 1;
    for (l = 0; l < c->layers; ++l) {
        if (modes[l] < 0 || modes[l] > 3) return 1;
        if (l < c->encoder) {
            if (modes[l] == 1) seen_encoder = 1;
            if (modes[l] >= 2 && !seen_encoder) return 1;
        } else {
            if (modes[l] == 1) {
                if (seen_decoder) return 1;
                seen_decoder = 1;
            }
            if (modes[l] >= 2 && !seen_decoder) return 1;
        }
    }
    if (!seen_encoder || !seen_decoder || !finite_array(weights, nw)) return 1;
    for (i = 0; i < n; ++i) if (ids[i] < 0 || ids[i] >= c->vocabulary) return 1;
    cache_count = (size_t)c->layers * n * h;
    required = (4 * cache_count + n * h + no) * sizeof(float) + trace_count * sizeof(int32_t);
    /* Fixed stack arrays are separate from this heap-staging budget. */
    if (budget < required) return 2;
    allocation = (float *)calloc(4 * cache_count + n * h + no, sizeof(float));
    staged_trace = (int32_t *)malloc(trace_count * sizeof(int32_t));
    if (!allocation || !staged_trace) { free(allocation); free(staged_trace); return 4; }
    local_k = allocation; local_v = local_k + cache_count;
    raw = local_v + cache_count; zraw = raw + cache_count;
    encoder = zraw + cache_count; staged = encoder + n * h;
    base = weights + h * c->vocabulary;
    final_gain = base + stride * c->layers; head = final_gain + h;
    for (t = 0; t < (int)n; ++t) {
        memcpy(x, weights + (size_t)ids[t] * h, h * sizeof(float));
        owner = -1; entries = selected_count = candidate_count = 0;
        for (l = 0; l < c->layers; ++l) {
            const float *w = base + stride * l;
            const float *attn_gain = w; w += h;
            const float *wq = w; w += h * h;
            const float *wk = w; w += h * h;
            const float *wv = w; w += h * h;
            const float *wo = w; w += h * h;
            const float *mlp_gain = w; w += h;
            const float *wg = w; w += h * c->intermediate;
            const float *wu = w; w += h * c->intermediate;
            const float *wd = w; w += h * c->intermediate;
            const float *wgq = w; w += h * h;
            const float *wkv = w; w += h * h;
            const float *wz = w; w += h * h;
            const float *wiq = w; w += h * h;
            const float *wik = w;
            size_t offset = ((size_t)l * n + t) * h;
            int first_local = t + 1 > c->window ? t + 1 - c->window : 0;
            int local_count = t - first_local + 1, total;
            if (l == c->encoder) { owner = -1; entries = selected_count = candidate_count = 0; }
            if (l == fail_layer) { status = 5; goto done; }
            if (!norm(x, attn_gain, c->hidden, c->epsilon, normalized) ||
                !matrix(wq, normalized, c->hidden, c->hidden, q) ||
                !matrix(wk, normalized, c->hidden, c->hidden, k) ||
                !matrix(wv, normalized, c->hidden, c->hidden, v)) goto done;
            memcpy(local_k + offset, k, h * sizeof(float));
            memcpy(local_v + offset, v, h * sizeof(float));
            if (modes[l]) {
                if (!matrix(wgq, normalized, c->hidden, c->hidden, gq)) goto done;
                if (modes[l] == 1) {
                    const float *source = l < c->encoder ? x : encoder + (size_t)t * h;
                    if (!matrix(wkv, source, c->hidden, c->hidden, raw + offset) ||
                        !matrix(wz, source, c->hidden, c->hidden, zraw + offset)) goto done;
                    owner = l; entries = (t + 1) / c->ratio;
                    for (r = 0; r < entries; ++r) {
                        for (d = 0; d < c->hidden; ++d) {
                            maximum = -DBL_MAX;
                            for (j = 0; j < c->ratio; ++j) {
                                size_t p = ((size_t)l * n + r * c->ratio + j) * h + d;
                                if (zraw[p] > maximum) maximum = zraw[p];
                            }
                            denominator = 0.0;
                            {
                                double sum = 0.0;
                                for (j = 0; j < c->ratio; ++j) {
                                    size_t p = ((size_t)l * n + r * c->ratio + j) * h + d;
                                    double a = exp(zraw[p] - maximum);
                                    denominator += a; sum += a * raw[p];
                                }
                                if (!store(sum / denominator, main_kv + (size_t)r * h + d)) goto done;
                            }
                        }
                        if (!matrix(wik, main_kv + (size_t)r * h, c->hidden, c->hidden,
                                    index_keys + (size_t)r * h)) goto done;
                    }
                }
                if (modes[l] != 3) {
                    int order[64], order_count = 0;
                    if (!matrix(wiq, normalized, c->hidden, c->hidden, iq)) goto done;
                    for (r = 0; r < entries; ++r) {
                        int eligible = 1;
                        score[r] = fmax(0.0, dot(iq, index_keys + (size_t)r * h, c->hidden));
                        if (!isfinite(score[r])) goto done;
                        if (l >= c->encoder && modes[l] == 2) {
                            eligible = 0;
                            for (j = 0; j < candidate_count; ++j)
                                if (r / c->block_size == candidates[j]) eligible = 1;
                        }
                        if (eligible) {
                            int at = order_count;
                            while (at && (score[r] > score[order[at - 1]] ||
                                (score[r] == score[order[at - 1]] && r < order[at - 1]))) {
                                order[at] = order[at - 1]; --at;
                            }
                            order[at] = r; ++order_count;
                        }
                    }
                    selected_count = order_count < c->topk ? order_count : c->topk;
                    for (j = 0; j < selected_count; ++j) selected[j] = order[j];
                    if (l >= c->encoder && modes[l] == 1) {
                        int blocks = (entries + c->block_size - 1) / c->block_size;
                        int block_order[64]; double maxima[64];
                        candidate_count = blocks < c->candidate_blocks ? blocks : c->candidate_blocks;
                        for (r = 0; r < blocks; ++r) {
                            int at = r;
                            maxima[r] = -DBL_MAX;
                            for (j = r * c->block_size; j < entries && j < (r + 1) * c->block_size; ++j)
                                if (score[j] > maxima[r]) maxima[r] = score[j];
                            while (at && maxima[r] > maxima[block_order[at - 1]]) {
                                block_order[at] = block_order[at - 1]; --at;
                            }
                            block_order[at] = r;
                        }
                        for (j = 0; j < candidate_count; ++j) candidates[j] = block_order[j];
                    }
                }
            }
            total = local_count + (modes[l] ? selected_count : 0);
            maximum = -DBL_MAX;
            for (r = 0; r < total; ++r) {
                double value = r < local_count ? dot(q, local_k + ((size_t)l * n + first_local + r) * h, c->hidden) :
                    dot(gq, main_kv + (size_t)selected[r - local_count] * h, c->hidden);
                attention_scores[r] = value / sqrt((double)c->hidden);
                if (!isfinite(attention_scores[r])) goto done;
                if (attention_scores[r] > maximum) maximum = attention_scores[r];
            }
            denominator = 0.0;
            for (r = 0; r < total; ++r) { attention_scores[r] = exp(attention_scores[r] - maximum); denominator += attention_scores[r]; }
            for (d = 0; d < c->hidden; ++d) {
                double sum = 0.0;
                for (r = 0; r < total; ++r) {
                    float value = r < local_count ? local_v[((size_t)l * n + first_local + r) * h + d] :
                        main_kv[(size_t)selected[r - local_count] * h + d];
                    sum += attention_scores[r] * value;
                }
                if (!store(sum / denominator, combined + d)) goto done;
            }
            if (!matrix(wo, combined, c->hidden, c->hidden, update)) goto done;
            for (d = 0; d < c->hidden; ++d) if (!store((double)x[d] + update[d], x + d)) goto done;
            if (!norm(x, mlp_gain, c->hidden, c->epsilon, normalized) ||
                !matrix(wg, normalized, c->intermediate, c->hidden, gate) ||
                !matrix(wu, normalized, c->intermediate, c->hidden, up)) goto done;
            for (j = 0; j < c->intermediate; ++j) {
                double a = gate[j], sigmoid = a >= 0 ? 1.0 / (1.0 + exp(-a)) : exp(a) / (1.0 + exp(a));
                if (!store(a * sigmoid * up[j], product + j)) goto done;
            }
            if (!matrix(wd, product, c->hidden, c->intermediate, update)) goto done;
            for (d = 0; d < c->hidden; ++d) if (!store((double)x[d] + update[d], x + d)) goto done;
            if (l + 1 == c->encoder) memcpy(encoder + (size_t)t * h, x, h * sizeof(float));
            staged_trace[(size_t)l * trace_width] = modes[l] ? owner : -1;
            staged_trace[(size_t)l * trace_width + 1] = modes[l] ? selected_count : 0;
            for (j = 0; j < c->topk; ++j)
                staged_trace[(size_t)l * trace_width + 2 + j] = modes[l] && j < selected_count ? selected[j] : -1;
        }
        if (!norm(x, final_gain, c->hidden, c->epsilon, normalized) ||
            !matrix(head, normalized, c->vocabulary, c->hidden, staged + (size_t)t * c->vocabulary)) goto done;
    }
    memcpy(output, staged, no * sizeof(float));
    memcpy(trace, staged_trace, trace_count * sizeof(int32_t));
    status = 0;
done:
    free(allocation); free(staged_trace);
    return status;
}
