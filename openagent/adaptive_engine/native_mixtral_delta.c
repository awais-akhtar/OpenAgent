/* Independent key-by-value gated-delta update. No Python callbacks. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define OA_EXPORT __declspec(dllexport)
#else
#define OA_EXPORT __attribute__((visibility("default")))
#endif

OA_EXPORT int oa_gated_delta_step(size_t heads, size_t keys, size_t values,
    const float *q, const float *k, const float *v, const float *beta,
    const float *decay, float *state, float *output) {
    size_t h, i, j, count, out_count;
    float *next, *result;
    if (!heads || !keys || !values || !q || !k || !v || !beta || !decay || !state || !output)
        return 1;
    if (heads > SIZE_MAX / keys || heads * keys > SIZE_MAX / values)
        return 1;
    count = heads * keys * values;
    out_count = heads * values;
    if (count > SIZE_MAX / sizeof(float) || count > 67108864)
        return 1;
    next = (float *)malloc(count * sizeof(float));
    result = (float *)malloc(out_count * sizeof(float));
    if (!next || !result) { free(next); free(result); return 2; }
    for (h = 0; h < heads; ++h) {
        if (!isfinite(beta[h]) || !isfinite(decay[h]) || beta[h] < 0 || beta[h] > 1 || decay[h] < 0 || decay[h] > 1)
            goto invalid;
        for (i = 0; i < keys; ++i) {
            if (!isfinite(q[h * keys + i]) || !isfinite(k[h * keys + i])) goto invalid;
            for (j = 0; j < values; ++j) {
                size_t index = (h * keys + i) * values + j;
                if (!isfinite(state[index])) goto invalid;
                next[index] = state[index] * decay[h];
            }
        }
        for (j = 0; j < values; ++j) {
            double prediction = 0, read = 0;
            float error;
            if (!isfinite(v[h * values + j])) goto invalid;
            for (i = 0; i < keys; ++i)
                prediction += (double)k[h * keys + i] * next[(h * keys + i) * values + j];
            error = (v[h * values + j] - (float)prediction) * beta[h];
            for (i = 0; i < keys; ++i) {
                size_t index = (h * keys + i) * values + j;
                next[index] += k[h * keys + i] * error;
                if (!isfinite(next[index])) goto invalid;
                read += (double)q[h * keys + i] * next[index];
            }
            result[h * values + j] = (float)read;
            if (!isfinite(result[h * values + j])) goto invalid;
        }
    }
    memcpy(state, next, count * sizeof(float));
    memcpy(output, result, out_count * sizeof(float));
    free(next); free(result); return 0;
invalid:
    free(next); free(result); return 3;
}
