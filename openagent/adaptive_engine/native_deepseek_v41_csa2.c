/* Original bounded CSA2 ownership and raw CED projection primitives. */
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

static int extent(const void *pointer, size_t bytes, size_t alignment) {
    uintptr_t address = (uintptr_t)pointer;
    return pointer && !(address % alignment) && bytes <= UINTPTR_MAX - address;
}
static int overlaps(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x < y + nb && y < x + na;
}

/* Report equation (1), raw projections only: no quantization or compression. */
API int oa_v41_ced_project(const float *hidden, size_t hidden_count,
    const float *w_kv, size_t kv_weight_count,
    const float *w_z, size_t z_weight_count,
    uint32_t rows, uint32_t hidden_width, uint32_t kv_width, uint32_t z_width,
    float *kv, size_t kv_count, float *z, size_t z_count) {
    float projected_kv[32 * 8], projected_z[32 * 8];
    size_t r, c, h, i;
    if (!hidden || !w_kv || !w_z || !kv || !z ||
        rows < 1 || rows > 32 || hidden_width < 1 || hidden_width > 8 ||
        kv_width < 1 || kv_width > 8 || z_width < 1 || z_width > 8 ||
        hidden_count != (size_t)rows * hidden_width ||
        kv_weight_count != (size_t)hidden_width * kv_width ||
        z_weight_count != (size_t)hidden_width * z_width ||
        kv_count != (size_t)rows * kv_width ||
        z_count != (size_t)rows * z_width) return 1;
    if (!extent(hidden, hidden_count * sizeof(float), sizeof(float)) ||
        !extent(w_kv, kv_weight_count * sizeof(float), sizeof(float)) ||
        !extent(w_z, z_weight_count * sizeof(float), sizeof(float)) ||
        !extent(kv, kv_count * sizeof(float), sizeof(float)) ||
        !extent(z, z_count * sizeof(float), sizeof(float)) ||
        overlaps(kv, kv_count * sizeof(float), z, z_count * sizeof(float))) return 1;
    for (i = 0; i < hidden_count; ++i) if (!isfinite(hidden[i])) return 3;
    for (i = 0; i < kv_weight_count; ++i) if (!isfinite(w_kv[i])) return 3;
    for (i = 0; i < z_weight_count; ++i) if (!isfinite(w_z[i])) return 3;
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < kv_width; ++c) {
            double sum = 0.0;
            for (h = 0; h < hidden_width; ++h)
                sum += (double)hidden[r * hidden_width + h] * w_kv[h * kv_width + c];
            if (!isfinite(sum) || fabs(sum) > FLT_MAX) return 3;
            projected_kv[r * kv_width + c] = (float)sum;
        }
        for (c = 0; c < z_width; ++c) {
            double sum = 0.0;
            for (h = 0; h < hidden_width; ++h)
                sum += (double)hidden[r * hidden_width + h] * w_z[h * z_width + c];
            if (!isfinite(sum) || fabs(sum) > FLT_MAX) return 3;
            projected_z[r * z_width + c] = (float)sum;
        }
    }
    memcpy(kv, projected_kv, kv_count * sizeof(float));
    memcpy(z, projected_z, z_count * sizeof(float));
    return 0;
}

typedef struct {
    uint32_t initialized, rows, width, index_width, selected;
    int32_t source, selector, last_layer;
    float kv[32 * 8], keys[32 * 8];
    int32_t indices[8];
} State;

static int valid_state(const State *state) {
    size_t i, j;
    if (state->initialized != 1 || state->rows < 1 || state->rows > 32 ||
        state->width < 1 || state->width > 8 || state->index_width < 1 ||
        state->index_width > 8 || state->selected < 1 || state->selected > 8 ||
        state->selected > state->rows || state->source < 0 ||
        state->selector < state->source || state->last_layer < state->selector) return 0;
    for (i = 0; i < (size_t)state->rows * state->width; ++i)
        if (!isfinite(state->kv[i])) return 0;
    for (i = 0; i < (size_t)state->rows * state->index_width; ++i)
        if (!isfinite(state->keys[i])) return 0;
    for (i = 0; i < state->selected; ++i) {
        if (state->indices[i] < 0 || (uint32_t)state->indices[i] >= state->rows) return 0;
        for (j = 0; j < i; ++j)
            if (state->indices[j] == state->indices[i]) return 0;
    }
    return 1;
}

API size_t oa_v41_csa2_size(void) { return sizeof(State); }
API int oa_v41_csa2_reset(void *buffer, size_t bytes) {
    if (bytes != sizeof(State) || !extent(buffer, bytes, 1)) return 1;
    memset(buffer, 0, bytes);
    return 0;
}

/* Commit only after all validation. memcpy avoids alignment requirements. */
API int oa_v41_csa2_publish(void *buffer, size_t bytes, int mode,
    int32_t layer, int32_t source, const float *kv, size_t kv_count,
    const float *keys, size_t key_count, uint32_t rows,
    uint32_t width, uint32_t index_width, const int32_t *indices,
    size_t count) {
    State next;
    size_t i, j;
    if (bytes != sizeof(State) || !extent(buffer, bytes, 1) || mode < 0 || mode > 2 ||
        layer < 0 || source < 0) return 1;
    memcpy(&next, buffer, sizeof(next));
    if (next.initialized > 1) return 1;
    if (next.initialized && !valid_state(&next)) return 1;
    if (next.initialized && layer <= next.last_layer) return 2;
    if (mode == 0) {
        if (source != layer || !kv || !keys || rows < 1 || rows > 32 ||
            width < 1 || width > 8 || index_width < 1 || index_width > 8 ||
            kv_count != (size_t)rows * width ||
            key_count != (size_t)rows * index_width) return 1;
        if (!extent(kv, kv_count * sizeof(float), sizeof(float)) ||
            !extent(keys, key_count * sizeof(float), sizeof(float))) return 1;
        for (i = 0; i < kv_count; ++i) if (!isfinite(kv[i])) return 3;
        for (i = 0; i < key_count; ++i) if (!isfinite(keys[i])) return 3;
    } else {
        if (!next.initialized || source != next.source) return 2;
        if (kv || keys || kv_count || key_count || rows || width || index_width)
            return 1;
        rows = next.rows;
    }
    if (mode == 2) {
        if (indices || count) return 1;
    } else {
        if (!indices || count < 1 || count > 8 || count > rows) return 1;
        if (!extent(indices, count * sizeof(int32_t), sizeof(int32_t))) return 1;
        for (i = 0; i < count; ++i) {
            if (indices[i] < 0 || (uint32_t)indices[i] >= rows) return 1;
            for (j = 0; j < i; ++j) if (indices[j] == indices[i]) return 1;
        }
    }
    if (mode == 0) {
        memset(&next, 0, sizeof(next));
        next.initialized = 1;
        next.rows = rows;
        next.width = width;
        next.index_width = index_width;
        next.source = source;
        memcpy(next.kv, kv, kv_count * sizeof(float));
        memcpy(next.keys, keys, key_count * sizeof(float));
    }
    if (mode != 2) {
        memset(next.indices, 0, sizeof(next.indices));
        memcpy(next.indices, indices, count * sizeof(int32_t));
        next.selected = (uint32_t)count;
        next.selector = layer;
    }
    next.last_layer = layer;
    memcpy(buffer, &next, sizeof(next));
    return 0;
}

API int oa_v41_csa2_read(const void *buffer, size_t bytes,
    float *kv, size_t kv_count, float *keys, size_t key_count,
    int32_t *indices, size_t count, int32_t *metadata, size_t metadata_count) {
    State state;
    if (bytes != sizeof(State) || !extent(buffer, bytes, 1)) return 1;
    memcpy(&state, buffer, sizeof(state));
    if (!valid_state(&state)) return 2;
    if (!kv || !keys || !indices || !metadata ||
        kv_count != (size_t)state.rows * state.width ||
        key_count != (size_t)state.rows * state.index_width ||
        count != state.selected || metadata_count != 3) return 1;
    {
        const void *outputs[4] = {kv, keys, indices, metadata};
        size_t lengths[4] = {kv_count * sizeof(float), key_count * sizeof(float),
                             count * sizeof(int32_t), metadata_count * sizeof(int32_t)};
        size_t i, j;
        for (i = 0; i < 4; ++i) {
            if (!extent(outputs[i], lengths[i], sizeof(float)) ||
                overlaps(outputs[i], lengths[i], buffer, bytes)) return 1;
            for (j = 0; j < i; ++j)
                if (overlaps(outputs[i], lengths[i], outputs[j], lengths[j])) return 1;
        }
    }
    memcpy(kv, state.kv, kv_count * sizeof(float));
    memcpy(keys, state.keys, key_count * sizeof(float));
    memcpy(indices, state.indices, count * sizeof(int32_t));
    metadata[0] = state.source;
    metadata[1] = state.selector;
    metadata[2] = state.last_layer;
    return 0;
}
