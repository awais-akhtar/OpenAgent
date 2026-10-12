/* Original rowwise MXFP4 decoder. No allocation and no vendor implementation. */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

static int overlap(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < an : x - y < bn;
}

static int extent(const void *p, size_t n) {
    return p && n <= UINTPTR_MAX - (uintptr_t)p;
}

static float element(unsigned code, unsigned scale) {
    unsigned e = (code >> 1) & 3, m = code & 1;
    float v = e ? ldexpf(1.0f + 0.5f * (float)m, (int)e - 1) : 0.5f * (float)m;
    v = ldexpf(v, (int)scale - 127);
    return code & 8 ? -v : v;
}

API int oa_mxfp4_version(void) { return 1; }

/* Lengths are bytes. Status: 1 contract, 2 alias, 3 nonfinite, 4 budget.
   Exact dense rows, columns divisible by 32; no implicit padding or swizzle. */
API int oa_mxfp4_decode(const uint8_t *packed, size_t pn,
                       const uint8_t *scales, size_t sn,
                       size_t rows, size_t cols, void *output, size_t on,
                       size_t budget) {
    size_t n, i, r, c;
    if (!packed || !scales || !output || !rows || !cols || cols % 32 ||
        rows > SIZE_MAX / cols) return 1;
    n = rows * cols;
    if (n > SIZE_MAX / sizeof(float) || sizeof(float) != 4 || FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24 || pn != n / 2 || sn != n / 32 ||
        on != n * sizeof(float)) return 1;
    if (!extent(packed, pn) || !extent(scales, sn) || !extent(output, on)) return 1;
    if (pn > budget || sn > budget - pn || on > budget - pn - sn) return 4;
    if (overlap(output, on, packed, pn) || overlap(output, on, scales, sn)) return 2;
    /* Preflight all bytes and FP32 results before committing any output. */
    for (i = 0; i < sn; ++i) if (scales[i] == 255) return 3;
    for (r = 0; r < rows; ++r) for (c = 0; c < cols; ++c) {
        unsigned code = (packed[r * (cols / 2) + c / 2] >> (4 * (c & 1))) & 15;
        if (!isfinite(element(code, scales[r * (cols / 32) + c / 32]))) return 3;
    }
    for (i = 0; i < n; ++i) {
        unsigned code = (packed[i / 2] >> (4 * (i & 1))) & 15;
        float v = element(code, scales[i / 32]);
        memcpy((uint8_t *)output + i * sizeof(float), &v, sizeof(float));
    }
    return 0;
}

/* Direct packed matvec; stage rows, never expand a dense weight matrix. */
API int oa_mxfp4_matvec(const uint8_t *packed, size_t pn,
    const uint8_t *scales, size_t sn, size_t rows, size_t cols,
    const void *input, size_t xn, void *output, size_t yn, size_t budget) {
    size_t n, r, c, required;
    float *staged;
    if (!packed || !scales || !input || !output || !rows || !cols || cols % 32 ||
        rows > SIZE_MAX / cols || cols > SIZE_MAX / 4 || rows > SIZE_MAX / 4 ||
        sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24) return 1;
    n = rows * cols;
    if (pn != n / 2 || sn != n / 32 || xn != 4 * cols || yn != 4 * rows) return 1;
    if (!extent(packed, pn) || !extent(scales, sn) || !extent(input, xn) ||
        !extent(output, yn)) return 1;
    required = pn;
    if (sn > SIZE_MAX - required) return 1;
    required += sn;
    if (xn > SIZE_MAX - required) return 1;
    required += xn;
    if (yn > (SIZE_MAX - required) / 2) return 1;
    required += 2 * yn;
    if (budget < required) return 4;
    if (overlap(output, yn, packed, pn) || overlap(output, yn, scales, sn) ||
        overlap(output, yn, input, xn)) return 2;
    for (c = 0; c < cols; ++c) {
        float x;
        memcpy(&x, (const uint8_t *)input + c * 4, 4);
        if (!isfinite(x)) return 3;
    }
    for (c = 0; c < sn; ++c) if (scales[c] == 255) return 3;
    staged = (float *)malloc(yn);
    if (!staged) return 5;
    for (r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (c = 0; c < cols; ++c) {
            unsigned code = (packed[r * (cols / 2) + c / 2] >> (4 * (c & 1))) & 15;
            float w = element(code, scales[r * (cols / 32) + c / 32]), x;
            if (!isfinite(w)) { free(staged); return 3; }
            memcpy(&x, (const uint8_t *)input + c * 4, 4);
            sum += (double)w * x;
        }
        if (!isfinite(sum) || fabs(sum) > FLT_MAX) { free(staged); return 3; }
        staged[r] = (float)sum;
    }
    memcpy(output, staged, yn);
    free(staged);
    return 0;
}
