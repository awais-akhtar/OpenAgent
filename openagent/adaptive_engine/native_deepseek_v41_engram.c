/* Original bounded Engram equations. Private ABI; no vendor implementation. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct {
    uint32_t order, width, rows, reserved;
    uint64_t seed, multiplier[4];
    const float *table;
    size_t count;
} oa_engram_head;
typedef struct {
    uint32_t abi, heads, dim, branches, vocabulary, pad;
    float epsilon;
    uint32_t reserved;
    const oa_engram_head *head;
    const float *key, *value, *query_scale, *key_scale;
    size_t key_count, value_count, query_count, scale_count, weight_budget;
} oa_engram_config;
typedef struct {
    uint32_t abi, length, capacity, reserved;
    uint32_t tokens[4096];
} oa_engram_state;

API uint32_t oa_v41_engram_abi_version(void) { return 1; }
API int oa_v41_engram_state_init(oa_engram_state *s, uint32_t capacity) {
    if (!s || !capacity || capacity > 4096) return 1;
    memset(s, 0, sizeof(*s)); s->abi = 1; s->capacity = capacity; return 0;
}
API int oa_v41_engram_rollback(oa_engram_state *s, uint32_t length) {
    if (!s || s->abi != 1 || s->reserved || !s->capacity ||
        s->capacity > 4096 || s->length > s->capacity || length > s->length) return 1;
    memset(s->tokens + length, 0, (s->length - length) * sizeof(uint32_t));
    s->length = length; return 0;
}
static int finite_array(const float *p, size_t n) {
    size_t i;
    if (!p) return 0;
    for (i = 0; i < n; ++i) if (!isfinite(p[i])) return 0;
    return 1;
}
static int overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (!na || !nb) return 0;
    if (!a || !b || na > UINTPTR_MAX - x || nb > UINTPTR_MAX - y) return 1;
    return x < y + nb && y < x + na;
}
static int output_overlap(float *out, size_t no, uint32_t *ids, size_t ni,
                          const void *p, size_t bytes) {
    return overlap(out, no * sizeof(float), p, bytes) ||
           overlap(ids, ni * sizeof(uint32_t), p, bytes);
}
/* Counts are float elements. Bounds precede every product and allocation. */
API int oa_v41_engram_run(const oa_engram_config *c, oa_engram_state *s,
    const uint32_t *tokens, size_t nt, const float *hidden, size_t nh,
    float *out, size_t no, uint32_t *indices, size_t ni, size_t budget) {
    size_t m = 0, weights = 0, h, t, b, j, z, offset, outputs, addresses, work;
    float *allocation, *staged, *memory, *key, *value;
    uint32_t *ids;
    uint32_t orders = 0;
    if (sizeof(size_t) < 8 || !c || !s || c->abi != 1 || c->reserved || !c->head ||
        !c->heads || c->heads > 24 || !c->dim || c->dim > 256 ||
        !c->branches || c->branches > 8 || !c->vocabulary ||
        c->pad >= c->vocabulary || !isfinite(c->epsilon) || c->epsilon <= 0 ||
        s->abi != 1 || s->reserved || !s->capacity || s->capacity > 4096 ||
        s->length > s->capacity || nt > 256 || nt > s->capacity - s->length) return 1;
    for (h = 0; h < c->heads; ++h) {
        const oa_engram_head *a = c->head + h;
        if (a->reserved || a->order < 2 || a->order > 4 || !a->width ||
            a->width > 256 || !a->rows || a->rows > 33554432 || !a->table ||
            a->count != (size_t)a->rows * a->width) return 1;
        orders |= 1u << a->order;
        m += a->width; weights += a->count;
    }
    if (orders != 28 || c->key_count != c->branches * c->dim * m ||
        c->value_count != c->dim * m || c->query_count != c->branches * c->dim ||
        c->scale_count != c->query_count) return 1;
    weights += c->key_count + c->value_count + c->query_count + c->scale_count;
    if (weights > ((size_t)1 << 28) || weights * sizeof(float) > c->weight_budget) return 2;
    outputs = nt * c->branches * c->dim; addresses = nt * c->heads;
    if (nh != outputs || no < outputs || ni < addresses ||
        (nt && (!tokens || !hidden || !out || !indices))) return 1;
    if (overlap(s, sizeof(*s), c, sizeof(*c)) ||
        overlap(s, sizeof(*s), c->head, c->heads * sizeof(*c->head)) ||
        overlap(s, sizeof(*s), hidden, nh * sizeof(float)) ||
        overlap(s, sizeof(*s), c->key, c->key_count * sizeof(float)) ||
        overlap(s, sizeof(*s), c->value, c->value_count * sizeof(float)) ||
        overlap(s, sizeof(*s), c->query_scale, c->query_count * sizeof(float)) ||
        overlap(s, sizeof(*s), c->key_scale, c->scale_count * sizeof(float))) return 1;
    if (overlap(out, outputs * sizeof(float), indices, addresses * sizeof(uint32_t)) ||
        overlap(tokens, nt * sizeof(uint32_t), s, sizeof(*s)) ||
        output_overlap(out, outputs, indices, addresses, s, sizeof(*s)) ||
        output_overlap(out, outputs, indices, addresses, c, sizeof(*c)) ||
        output_overlap(out, outputs, indices, addresses, c->head, c->heads * sizeof(*c->head)) ||
        output_overlap(out, outputs, indices, addresses, hidden, nh * sizeof(float)) ||
        output_overlap(out, outputs, indices, addresses, tokens, nt * sizeof(uint32_t)) ||
        output_overlap(out, outputs, indices, addresses, c->key, c->key_count * sizeof(float)) ||
        output_overlap(out, outputs, indices, addresses, c->value, c->value_count * sizeof(float)) ||
        output_overlap(out, outputs, indices, addresses, c->query_scale, c->query_count * sizeof(float)) ||
        output_overlap(out, outputs, indices, addresses, c->key_scale, c->scale_count * sizeof(float))) return 1;
    for (h = 0; h < c->heads; ++h)
        if (overlap(s, sizeof(*s), c->head[h].table, c->head[h].count * sizeof(float)) ||
            output_overlap(out, outputs, indices, addresses, c->head[h].table,
                           c->head[h].count * sizeof(float))) return 1;
    for (t = 0; t < s->length; ++t) if (s->tokens[t] >= c->vocabulary) return 1;
    for (t = 0; t < nt; ++t) if (tokens[t] >= c->vocabulary) return 1;
    if (!finite_array(c->key, c->key_count) || !finite_array(c->value, c->value_count) ||
        !finite_array(c->query_scale, c->query_count) ||
        !finite_array(c->key_scale, c->scale_count) || (nt && !finite_array(hidden, nh))) return 3;
    if (!nt) return 0;
    work = (outputs + m + 2 * c->dim) * sizeof(float) + addresses * sizeof(uint32_t);
    if (work > budget) return 2;
    allocation = (float *)malloc(work);
    if (!allocation) return 2;
    staged = allocation; memory = staged + outputs; key = memory + m;
    value = key + c->dim; ids = (uint32_t *)(value + c->dim);
    for (t = 0; t < nt; ++t) {
        offset = 0;
        for (h = 0; h < c->heads; ++h) {
            const oa_engram_head *a = c->head + h;
            uint64_t hash = a->seed;
            for (j = 0; j < a->order; ++j) {
                size_t back = a->order - 1 - j;
                uint32_t token = c->pad;
                if (back <= t) token = tokens[t - back];
                else if (back - t <= s->length) token = s->tokens[s->length - (back - t)];
                hash ^= (uint64_t)token * a->multiplier[j];
            }
            z = (size_t)(hash % a->rows); ids[t * c->heads + h] = (uint32_t)z;
            for (j = 0; j < a->width; ++j) {
                float v = a->table[z * a->width + j];
                if (!isfinite(v)) goto nonfinite;
                memory[offset + j] = v;
            }
            offset += a->width;
        }
        for (j = 0; j < c->dim; ++j) {
            double v = 0;
            for (z = 0; z < m; ++z) v += (double)c->value[j * m + z] * memory[z];
            if (!isfinite(v) || fabs(v) > FLT_MAX) goto nonfinite;
            value[j] = (float)v;
        }
        for (b = 0; b < c->branches; ++b) {
            double qsum = 0, ksum = 0, dot = 0, gate;
            size_t base = (t * c->branches + b) * c->dim;
            for (j = 0; j < c->dim; ++j) {
                double v = 0, q = hidden[base + j];
                for (z = 0; z < m; ++z) v += (double)c->key[(b * c->dim + j) * m + z] * memory[z];
                if (!isfinite(v) || fabs(v) > FLT_MAX) goto nonfinite;
                key[j] = (float)v; qsum += q * q; ksum += (double)key[j] * key[j];
            }
            qsum = sqrt(qsum / c->dim + c->epsilon);
            ksum = sqrt(ksum / c->dim + c->epsilon);
            for (j = 0; j < c->dim; ++j)
                dot += ((double)hidden[base + j] / qsum) * c->query_scale[b * c->dim + j] *
                       ((double)key[j] / ksum) * c->key_scale[b * c->dim + j];
            dot /= sqrt((double)c->dim);
            if (!isfinite(dot)) goto nonfinite;
            gate = dot >= 0 ? 1 / (1 + exp(-dot)) : exp(dot) / (1 + exp(dot));
            for (j = 0; j < c->dim; ++j) staged[base + j] = (float)(gate * value[j]);
        }
    }
    /* Commit only after all tokens succeed. Buffers must be disjoint from state/weights. */
    memcpy(out, staged, outputs * sizeof(float)); memcpy(indices, ids, addresses * sizeof(uint32_t));
    memcpy(s->tokens + s->length, tokens, nt * sizeof(uint32_t)); s->length += (uint32_t)nt;
    free(allocation); return 0;
nonfinite:
    free(allocation); return 3;
}
