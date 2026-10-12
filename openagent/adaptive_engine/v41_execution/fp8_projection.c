/* Direct row-major E4M3 weights with one UE8M0 scale per 32x32 tile. */
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

static int extent(const void *p, size_t n, size_t alignment) {
    return p && (uintptr_t)p % alignment == 0 && n <= UINTPTR_MAX - (uintptr_t)p;
}

static int overlaps(const void *a, size_t na, const void *b, size_t nb) {
    return (uintptr_t)a < (uintptr_t)b + nb && (uintptr_t)b < (uintptr_t)a + na;
}

static float e4m3(uint8_t code) {
    unsigned e = (code >> 3) & 15, f = code & 7;
    float value = e == 0 ? ldexpf((float)f, -9) : ldexpf((float)(8 + f), (int)e - 10);
    return code & 128 ? -value : value;
}

API uint32_t oa_v41_fp8_projection_version(void) { return 1; }

/* Budget covers activation transport and staged output, not borrowed weights.
   All failures leave caller output unchanged. No dense matrix is allocated. */
API int oa_v41_fp8_projection_matvec(const uint8_t *weights, size_t nw,
    const uint8_t *scales, size_t ns, size_t rows, size_t cols,
    const float *x, size_t nx, float *out, size_t no, size_t budget) {
    size_t tr, tc, r, c;
    float *staged;
    if (!rows || !cols || cols % 32 || rows > SIZE_MAX / cols ||
        cols > SIZE_MAX / sizeof(float) || rows > SIZE_MAX / (2 * sizeof(float))) return 1;
    tr = rows / 32 + (rows % 32 != 0);
    tc = cols / 32 + (cols % 32 != 0);
    if (tr > SIZE_MAX / tc || nw != rows * cols || ns != tr * tc ||
        nx != cols * sizeof(float) || no != rows * sizeof(float) ||
        !extent(weights, nw, 1) || !extent(scales, ns, 1) ||
        !extent(x, nx, sizeof(float)) || !extent(out, no, sizeof(float))) return 1;
    if (overlaps(out, no, weights, nw) || overlaps(out, no, scales, ns) ||
        overlaps(out, no, x, nx)) return 2;
    if (nx > budget || no > (budget - nx) / 2) return 4;
    for (c = 0; c < cols; ++c) if (!isfinite(x[c])) return 3;
    for (c = 0; c < ns; ++c) if (scales[c] == 255) return 3;
    staged = (float *)malloc(no);
    if (!staged) return 5;
    for (r = 0; r < rows; ++r) {
        double total = 0.0;
        for (c = 0; c < cols; ++c) {
            uint8_t code = weights[r * cols + c];
            float weight;
            if ((code & 127) == 127) { free(staged); return 3; }
            weight = ldexpf(e4m3(code), (int)scales[(r / 32) * tc + c / 32] - 127);
            if (!isfinite(weight)) { free(staged); return 3; }
            total += (double)weight * (double)x[c];
        }
        if (!isfinite(total) || fabs(total) > FLT_MAX) { free(staged); return 3; }
        staged[r] = (float)total;
    }
    memcpy(out, staged, no);
    free(staged);
    return 0;
}
