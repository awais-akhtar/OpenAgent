/* Original local codec; NOT a publisher checkpoint layout. See the codec doc. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif
#define MAX_N 16384u
#define MAX_B 1024u

/* Private local ABI identity, independent of model/checkpoint versioning. */
API uint32_t oa_v41_quant_abi_version(void) { return 1u; }

static double e4(unsigned code) {
    unsigned e = code >> 3, m = code & 7u;
    return e ? ldexp(1.0 + (double)m / 8.0, (int)e - 7) : (double)m / 512.0;
}
static double e2(unsigned code) {
    unsigned e = (code & 7u) >> 1, m = code & 1u;
    double v = e ? ldexp(1.0 + (double)m / 2.0, (int)e - 1) : (double)m / 2.0;
    return code & 8u ? -v : v;
}
static unsigned nearest(double x, unsigned last, int fp4) {
    unsigned k, best = 0;
    double d = fabs(x);
    if (x >= (fp4 ? e2(last) : e4(last))) return last;
    for (k = 1; k <= last; ++k) {
        double distance = fabs(x - (fp4 ? e2(k) : e4(k)));
        if (distance < d || (distance == d && !(k & 1u))) {
            d = distance; best = k;
        }
    }
    return best;
}
static int geometry(uint32_t rows, uint32_t channels, size_t *n, size_t *b) {
    if (!rows || rows > 32u || !channels || channels > 512u || channels % 16u) return 0;
    *n = (size_t)rows * channels; *b = *n / 16u;
    return 1;
}
/* Validate address arithmetic before dereferencing; allocation truth is caller-owned. */
static int extent(const void *p, size_t bytes, size_t alignment) {
    uintptr_t a = (uintptr_t)p;
    return p && !(a % alignment) && bytes <= UINTPTR_MAX - a;
}

API int oa_v41_quant_scales(const float *x, size_t nx, uint32_t rows,
    uint32_t channels, uint8_t *scales, size_t ns, size_t budget) {
    uint8_t staged[MAX_B]; size_t n, b, g, i;
    if (!geometry(rows, channels, &n, &b) || nx != n || ns != b ||
        !extent(x, n * sizeof(float), sizeof(float)) || !extent(scales, b, 1)) return 1;
    if (budget < b) return 2;
    for (g = 0; g < b; ++g) {
        double maximum = 0.0;
        for (i = 0; i < 16; ++i) {
            double v = x[g * 16 + i];
            if (!isfinite(v)) return 3;
            if (fabs(v) > maximum) maximum = fabs(v);
        }
        staged[g] = (uint8_t)nearest(maximum / 6.0, 126, 0);
        if (maximum > 0.0 && staged[g] == 0) staged[g] = 1;
    }
    memcpy(scales, staged, b); return 0;
}

API int oa_v41_quant_encode(const float *x, size_t nx, const uint8_t *scales,
    size_t ns, uint32_t rows, uint32_t channels, uint8_t *out, size_t no, size_t budget) {
    uint8_t staged[MAX_B * 9u]; size_t n, b, g, i;
    if (!geometry(rows, channels, &n, &b) || nx != n || ns != b || no != b * 9u ||
        !extent(x, n * sizeof(float), sizeof(float)) || !extent(scales, b, 1) ||
        !extent(out, b * 9u, 1)) return 1;
    if (budget < b * 9u) return 2;
    for (g = 0; g < b; ++g) {
        unsigned s = scales[g]; double scale;
        if (s > 126u) return 3;
        scale = e4(s); staged[g * 9] = (uint8_t)s;
        for (i = 0; i < 16; ++i) {
            double v = x[g * 16 + i]; unsigned code;
            if (!isfinite(v) || (scale == 0.0 && v != 0.0)) return 3;
            code = scale == 0.0 ? 0u : nearest(fabs(v) / scale, 7, 1);
            if (signbit(v)) code |= 8u;
            if (!(i & 1u)) staged[g * 9 + 1 + i / 2] = (uint8_t)code;
            else staged[g * 9 + 1 + i / 2] |= (uint8_t)(code << 4);
        }
    }
    memcpy(out, staged, b * 9u); return 0;
}

API int oa_v41_quant_decode(const uint8_t *packed, size_t np, uint32_t rows,
    uint32_t channels, float *out, size_t no, size_t budget) {
    float staged[MAX_N]; size_t n, b, g, i;
    if (!geometry(rows, channels, &n, &b) || np != b * 9u || no != n ||
        !extent(packed, b * 9u, 1) || !extent(out, n * sizeof(float), sizeof(float))) return 1;
    if (budget < n * sizeof(float)) return 2;
    for (g = 0; g < b; ++g) {
        unsigned s = packed[g * 9];
        if (s > 126u) return 3;
        for (i = 0; i < 16; ++i) {
            unsigned code = (packed[g * 9 + 1 + i / 2] >> ((i & 1u) * 4u)) & 15u;
            staged[g * 16 + i] = (float)(e2(code) * e4(s));
        }
    }
    memcpy(out, staged, n * sizeof(float)); return 0;
}
