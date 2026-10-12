#include "segment_math.h"
#include <stddef.h>
#include <float.h>
#include <math.h>

static double row_scale(const float *input, size_t width, double epsilon) {
    size_t j;
    double sum = 0.0;
    for (j = 0; j < width; ++j) {
        double value = input[j];
        sum += value * value;
    }
    return 1.0 / sqrt(sum / (double)width + epsilon);
}

oa_status OA_CALL oa_segment_rmsnorm_f32(const float *input,
    const float *weight, float *output, uint64_t rows, uint64_t width,
    double epsilon, uint64_t max_bytes) {
    size_t r, j, n, w;
    uint64_t elements;
    if (!input || !weight || !output || !rows || !width ||
        !isfinite(epsilon) || epsilon <= 0.0) return OA_INVALID;
    if (!max_bytes || width > UINT64_MAX / rows) return OA_LIMIT;
    elements = rows * width;
    if (elements > SIZE_MAX / sizeof(float) ||
        elements > max_bytes / sizeof(float)) return OA_LIMIT;
    n = (size_t)elements;
    w = (size_t)width;
    for (j = 0; j < w; ++j) if (!isfinite(weight[j])) return OA_INVALID;
    for (j = 0; j < n; ++j) if (!isfinite(input[j])) return OA_INVALID;
    /* Preflight all rows before publishing any output, including FP32 overflow. */
    for (r = 0; r < (size_t)rows; ++r) {
        double scale = row_scale(input + r * w, w, epsilon);
        for (j = 0; j < w; ++j) {
            double value = (double)input[r * w + j] * scale * weight[j];
            if (!isfinite(value) || fabs(value) > FLT_MAX) return OA_INVALID;
        }
    }
    for (r = 0; r < (size_t)rows; ++r) {
        double scale = row_scale(input + r * w, w, epsilon);
        for (j = 0; j < w; ++j)
            output[r * w + j] = (float)((double)input[r * w + j] * scale * weight[j]);
    }
    return OA_OK;
}
