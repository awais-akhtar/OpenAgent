/* Original dynamic FP32 transformer operations; private counted CPU ABI. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

static int valid(const void *p, size_t n, size_t width) {
    return p && n && n <= SIZE_MAX / width &&
        n * width <= UINTPTR_MAX - (uintptr_t)p && !((uintptr_t)p % width);
}
static int numbers(const float *p, size_t n) {
    size_t i;
    if (!valid(p, n, sizeof(float))) return 0;
    for (i = 0; i < n; ++i) if (!isfinite(p[i])) return 0;
    return 1;
}
static int put(double x, float *p) {
    if (!isfinite(x) || fabs(x) > FLT_MAX) return 0;
    *p = (float)x;
    return 1;
}
static double sigmoid(double x) {
    return x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
}
static float *stage(size_t n, size_t budget) {
    if (!n || n > SIZE_MAX / sizeof(float) || n * sizeof(float) > budget) return NULL;
    return (float *)malloc(n * sizeof(float));
}
API uint32_t oa_tensor_ops_version(void) {
    return sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 ? 1 : 0;
}

/* Status 1 invalid contract, 2 nonfinite arithmetic, 3 staging admission. */
API int oa_tensor_norm(const float *x, size_t count, const float *gain, size_t gn,
    size_t rows, size_t width, double epsilon, float *out, size_t on, size_t budget) {
    size_t r, j;
    float *tmp;
    if (!rows || !width || rows > SIZE_MAX / width || count != rows * width ||
        gn != width || on != count || !valid(out, on, sizeof(float)) ||
        !isfinite(epsilon) || epsilon <= 0 || !numbers(x, count) || !numbers(gain, gn)) return 1;
    tmp = stage(on, budget);
    if (!tmp) return 3;
    for (r = 0; r < rows; ++r) {
        double square = 0, scale;
        for (j = 0; j < width; ++j) square += (double)x[r * width + j] * x[r * width + j];
        scale = 1.0 / sqrt(square / (double)width + epsilon);
        for (j = 0; j < width; ++j)
            if (!put((double)x[r * width + j] * scale * gain[j], tmp + r * width + j)) {
                free(tmp); return 2;
            }
    }
    memcpy(out, tmp, on * sizeof(float)); free(tmp); return 0;
}

API int oa_tensor_sigmoid(const float *x, size_t count, float *out, size_t on, size_t budget) {
    size_t i;
    float *tmp;
    if (on != count || !numbers(x, count) || !valid(out, on, sizeof(float))) return 1;
    tmp = stage(on, budget);
    if (!tmp) return 3;
    for (i = 0; i < count; ++i) tmp[i] = (float)sigmoid(x[i]);
    memcpy(out, tmp, on * sizeof(float)); free(tmp); return 0;
}

API int oa_tensor_situ(const float *gate, const float *up, size_t count,
    double beta, double linear_beta, float *out, size_t on, size_t budget) {
    size_t i;
    float *tmp;
    if (on != count || !isfinite(beta) || beta <= 0 || !isfinite(linear_beta) ||
        linear_beta <= 0 || !numbers(gate, count) || !numbers(up, count) ||
        !valid(out, on, sizeof(float))) return 1;
    tmp = stage(on, budget);
    if (!tmp) return 3;
    for (i = 0; i < count; ++i) {
        double value = beta * tanh((double)gate[i] / beta) * sigmoid(gate[i]) *
            linear_beta * tanh((double)up[i] / linear_beta);
        if (!put(value, tmp + i)) { free(tmp); return 2; }
    }
    memcpy(out, tmp, on * sizeof(float)); free(tmp); return 0;
}

API int oa_tensor_attnres(const float *values, size_t count, const float *gain,
    const float *query, size_t width, size_t blocks, double epsilon,
    float *out, size_t on, size_t budget) {
    size_t b, j, score_bytes, out_bytes;
    double *scores, max_score = -INFINITY, denominator = 0;
    float *tmp;
    if (!blocks || !width || blocks > SIZE_MAX / width || count != blocks * width ||
        on != width || !numbers(values, count) || !numbers(gain, width) ||
        !numbers(query, width) || !valid(out, on, sizeof(float)) ||
        !isfinite(epsilon) || epsilon <= 0 || blocks > SIZE_MAX / sizeof(double) ||
        width > SIZE_MAX / sizeof(float)) return 1;
    score_bytes = blocks * sizeof(double); out_bytes = width * sizeof(float);
    if (score_bytes > budget || out_bytes > budget - score_bytes) return 3;
    scores = (double *)malloc(score_bytes); tmp = (float *)malloc(out_bytes);
    if (!scores || !tmp) { free(scores); free(tmp); return 3; }
    for (b = 0; b < blocks; ++b) {
        double square = 0, scale, score = 0;
        for (j = 0; j < width; ++j) square += (double)values[b * width + j] * values[b * width + j];
        scale = 1.0 / sqrt(square / (double)width + epsilon);
        for (j = 0; j < width; ++j) {
            float key;
            if (!put((double)values[b * width + j] * scale * gain[j], &key)) {
                free(scores); free(tmp); return 2;
            }
            score += (double)key * query[j];
        }
        if (!isfinite(score)) { free(scores); free(tmp); return 2; }
        scores[b] = score;
        if (score > max_score) max_score = score;
    }
    for (b = 0; b < blocks; ++b) { scores[b] = exp(scores[b] - max_score); denominator += scores[b]; }
    if (!isfinite(denominator) || denominator <= 0) { free(scores); free(tmp); return 2; }
    for (j = 0; j < width; ++j) {
        double value = 0;
        for (b = 0; b < blocks; ++b) value += scores[b] / denominator * values[b * width + j];
        if (!put(value, tmp + j)) { free(scores); free(tmp); return 2; }
    }
    memcpy(out, tmp, out_bytes); free(scores); free(tmp); return 0;
}

/* Split-half rotary on trailing channels. One absolute position per row. */
API int oa_tensor_rope(const float *x, size_t count, const uint64_t *positions,
    size_t pn, size_t rows, size_t width, size_t rotary, double theta,
    float *out, size_t on, size_t budget) {
    size_t r, j, half;
    float *tmp;
    if (!rows || !width || rows > SIZE_MAX / width || count != rows * width ||
        pn != rows || on != count || rotary > width || rotary % 2 ||
        !isfinite(theta) || theta <= 0 || !numbers(x, count) ||
        !valid(positions, pn, sizeof(uint64_t)) || !valid(out, on, sizeof(float))) return 1;
    for (r = 0; r < rows; ++r) if (positions[r] > UINT64_C(9007199254740991)) return 1;
    tmp = stage(on, budget);
    if (!tmp) return 3;
    memcpy(tmp, x, on * sizeof(float)); half = rotary / 2;
    for (r = 0; r < rows; ++r) for (j = 0; j < half; ++j) {
        size_t a = r * width + width - rotary + j, b = a + half;
        double frequency = pow(theta, -(double)(2 * j) / (double)rotary);
        double angle = (double)positions[r] * frequency, cosine = cos(angle), sine = sin(angle);
        if (!put((double)x[a] * cosine - (double)x[b] * sine, tmp + a) ||
            !put((double)x[b] * cosine + (double)x[a] * sine, tmp + b)) {
            free(tmp); return 2;
        }
    }
    memcpy(out, tmp, on * sizeof(float)); free(tmp); return 0;
}
