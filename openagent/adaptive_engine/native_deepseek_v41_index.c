/* Original bounded equation fixture, not an exact publisher indexer. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

API uint32_t oa_v41_index_abi_version(void) { return 1u; }

static int extent(const void *p, size_t n, size_t alignment) {
    uintptr_t a = (uintptr_t)p;
    return p && !(a % alignment) && n <= UINTPTR_MAX - a;
}
static int overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x < y + nb && y < x + na;
}

/* s[r] = sum_h weight[h] * max(0, dot(q[h], k[r,h])). */
API int oa_v41_index_score(const float *q, size_t nq, const float *keys,
    size_t nk, const float *weights, size_t nw, uint32_t rows,
    uint32_t heads, uint32_t width, float *out, size_t no, size_t budget) {
    float staged[32];
    size_t r, h, d, i, n;
    if (!rows || rows > 32u || !heads || heads > 8u || !width || width > 8u)
        return 1;
    n = (size_t)heads * width;
    if (nq != n || nk != (size_t)rows * n || nw != heads || no != rows ||
        !extent(q, nq * sizeof(float), sizeof(float)) ||
        !extent(keys, nk * sizeof(float), sizeof(float)) ||
        !extent(weights, nw * sizeof(float), sizeof(float)) ||
        !extent(out, no * sizeof(float), sizeof(float))) return 1;
    if (budget < no * sizeof(float)) return 2;
    for (i = 0; i < nq; ++i) if (!isfinite(q[i])) return 3;
    for (i = 0; i < nk; ++i) if (!isfinite(keys[i])) return 3;
    for (i = 0; i < nw; ++i) if (!isfinite(weights[i])) return 3;
    for (r = 0; r < rows; ++r) {
        double score = 0.0;
        for (h = 0; h < heads; ++h) {
            double dot = 0.0;
            for (d = 0; d < width; ++d)
                dot += (double)q[h * width + d] * keys[r * n + h * width + d];
            if (dot > 0.0) score += (double)weights[h] * dot;
        }
        if (!isfinite(score) || fabs(score) > FLT_MAX) return 3;
        staged[r] = (float)score;
    }
    memcpy(out, staged, no * sizeof(float));
    return 0;
}

/* mode 0 ranks admitted block maxima; mode 1 ranks admitted rows in candidates. */
API int oa_v41_index_select(const float *scores, size_t ns,
    const uint8_t *admitted, size_t na, uint32_t rows, uint32_t block_size,
    int mode, const int32_t *candidates, size_t nc, uint32_t limit,
    int32_t *out, size_t capacity, uint32_t *count, size_t budget) {
    float values[32];
    int32_t ids[32], staged[32];
    uint8_t blocks[32] = {0};
    size_t i, j, n = 0, selected, nb;
    if (!rows || rows > 32u || !block_size || block_size > 32u ||
        mode < 0 || mode > 1 || !limit || limit > 32u || capacity > 32u ||
        ns != rows || na != rows ||
        !extent(scores, ns * sizeof(float), sizeof(float)) ||
        !extent(admitted, na, 1) ||
        !extent(out, capacity * sizeof(int32_t), sizeof(int32_t)) ||
        !extent(count, sizeof(uint32_t), sizeof(uint32_t)) ||
        overlap(out, capacity * sizeof(int32_t), count, sizeof(uint32_t))) return 1;
    nb = ((size_t)rows + block_size - 1u) / block_size;
    if (mode == 0) {
        if (candidates || nc) return 1;
        for (i = 0; i < nb; ++i) blocks[i] = 1;
    } else {
        if (nc > nb || (nc && !extent(candidates, nc * sizeof(int32_t), sizeof(int32_t))) ||
            (!nc && candidates)) return 1;
        for (i = 0; i < nc; ++i) {
            if (candidates[i] < 0 || (size_t)candidates[i] >= nb || blocks[candidates[i]])
                return 1;
            blocks[candidates[i]] = 1;
        }
    }
    if (budget < (size_t)rows * (sizeof(float) + 2u * sizeof(int32_t)) + nb)
        return 2;
    /* Validate even excluded rows: masks never conceal corrupt score payloads. */
    for (i = 0; i < rows; ++i) {
        if (admitted[i] > 1u) return 1;
        if (!isfinite(scores[i])) return 3;
    }
    for (i = 0; i < rows; ++i) {
        size_t b = i / block_size;
        if (!admitted[i] || !blocks[b]) continue;
        if (mode == 1) {
            ids[n] = (int32_t)i; values[n++] = scores[i];
        } else {
            for (j = 0; j < n && ids[j] != (int32_t)b; ++j) { }
            if (j == n) { ids[n] = (int32_t)b; values[n++] = scores[i]; }
            else if (scores[i] > values[j]) values[j] = scores[i];
        }
    }
    selected = n < limit ? n : limit;
    if (capacity < selected) return 4;
    for (i = 0; i < selected; ++i) {
        size_t best = i;
        for (j = i + 1; j < n; ++j)
            if (values[j] > values[best] ||
                (values[j] == values[best] && ids[j] < ids[best])) best = j;
        staged[i] = ids[best];
        ids[best] = ids[i]; values[best] = values[i];
    }
    memcpy(out, staged, selected * sizeof(int32_t));
    *count = (uint32_t)selected;
    return 0;
}
