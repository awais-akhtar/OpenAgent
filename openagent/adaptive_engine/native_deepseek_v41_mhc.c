/* Original bounded equation-(6) fixture. No publisher checkpoint ABI. */
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
#define NX 512u
#define NA 32u
#define NQ 192u
typedef struct {
    uint32_t magic, rows, streams, width, depth, capacity;
    float x[NX], previous[NA];
} State;
static int region(const void *p, size_t bytes, size_t align) {
    uintptr_t a = (uintptr_t)p;
    return p != NULL && a % align == 0u && bytes <= UINTPTR_MAX - a;
}
static int overlaps(const void *a, size_t n, const void *b, size_t m) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x < y + m && y < x + n;
}
static int floats(const float *p, size_t count, size_t wanted) {
    return count == wanted && region(p, wanted * sizeof(float), sizeof(float));
}
static int finite_array(const float *p, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (!isfinite(p[i])) return 0;
    return 1;
}
static int store(double v, float *p) {
    if (!isfinite(v) || fabs(v) > FLT_MAX) return 3;
    *p = (float)v;
    return 0;
}
static int valid(const State *s, size_t bytes) {
    size_t i, j, nx, na;
    if (bytes != sizeof(State) || !region(s, bytes, sizeof(uint32_t))) return 1;
    if (s->magic != 0x4d484331u || !s->rows || s->rows > 8u ||
        !s->streams || s->streams > 4u || !s->width || s->width > 16u ||
        !s->capacity || s->capacity > 64u || s->depth > s->capacity) return 1;
    na = (size_t)s->rows * s->streams; nx = na * s->width;
    if (!finite_array(s->x, nx) || !finite_array(s->previous, na)) return 3;
    for (i = 0; i < na; ++i) if (s->previous[i] < 0.0f || s->previous[i] > 1.0f) return 1;
    /* Canonical unused tails make snapshots portable within this private ABI. */
    for (j = nx; j < NX; ++j) if (s->x[j] != 0.0f) return 1;
    for (j = na; j < NA; ++j) if (s->previous[j] != 0.0f) return 1;
    return 0;
}
API uint32_t oa_v41_mhc_abi_version(void) { return 1u; }
API size_t oa_v41_mhc_state_bytes(void) { return sizeof(State); }
API size_t oa_v41_mhc_scratch_bytes(void) {
    /* Payload staging including nested update/predict, not compiler stack overhead. */
    return sizeof(State)+(NQ+256u+NX)*sizeof(float)+40u*sizeof(double);
}
API int oa_v41_mhc_validate(const void *state, size_t bytes) {
    return valid((const State *)state, bytes);
}
API int oa_v41_mhc_init(void *state, size_t bytes, uint32_t rows,
    uint32_t streams, uint32_t width, uint32_t capacity,
    const float *x, size_t nx, const float *initial, size_t na) {
    State t;
    size_t i, n = (size_t)rows * streams;
    if (!rows || rows > 8u || !streams || streams > 4u || !width || width > 16u ||
        !capacity || capacity > 64u || bytes != sizeof(State) ||
        !region(state, bytes, sizeof(uint32_t)) || !floats(x, nx, n * width) ||
        !floats(initial, na, n) || overlaps(state, bytes, x, nx * sizeof(float)) ||
        overlaps(state, bytes, initial, na * sizeof(float))) return 1;
    if (!finite_array(x, nx) || !finite_array(initial, na)) return 3;
    for (i = 0; i < na; ++i) if (initial[i] < 0.0f || initial[i] > 1.0f) return 1;
    memset(&t, 0, sizeof(t));
    t.magic = 0x4d484331u; t.rows = rows; t.streams = streams;
    t.width = width; t.capacity = capacity;
    memcpy(t.x, x, nx * sizeof(float)); memcpy(t.previous, initial, na * sizeof(float));
    memcpy(state, &t, sizeof(t)); return 0;
}
static int mix(const State *s, const float *a, float *out) {
    size_t r, d, j, n = s->streams, w = s->width;
    for (r = 0; r < s->rows; ++r) for (d = 0; d < w; ++d) {
        double v = 0.0;
        for (j = 0; j < n; ++j) v += (double)a[r*n+j] * s->x[(r*n+j)*w+d];
        if (store(v, &out[r*w+d])) return 3;
    }
    return 0;
}
API int oa_v41_mhc_mix(const void *state, size_t bytes, float *out, size_t count) {
    const State *s = (const State *)state;
    float t[128]; int status = valid(s, bytes);
    if (status) return status;
    if (!floats(out, count, (size_t)s->rows*s->width) ||
        overlaps(state, bytes, out, count*sizeof(float))) return 1;
    status = mix(s, s->previous, t);
    if (!status) memcpy(out, t, count*sizeof(float));
    return status;
}
static double sigmoid(double v) {
    if (v >= 0.0) return 1.0 / (1.0 + exp(-v));
    v = exp(v); return v / (1.0 + v);
}
/* H fixture: flattened RMSNorm -> linear -> sigmoid A, 2sigmoid C,
   positive B with fixed row/column Sinkhorn sweeps. */
static int predict(const State *s, const float *weight, const float *bias,
    double eps, uint32_t sweeps, float *out) {
    size_t r, k, j, i, n = s->streams, h = n*s->width, q = n*n+2u*n;
    uint32_t sweep;
    double logits[24], b[16], rms, v, total, maximum;
    for (r = 0; r < s->rows; ++r) {
        rms = eps;
        for (j = 0; j < h; ++j) rms += (double)s->x[r*h+j]*s->x[r*h+j]/(double)h;
        rms = sqrt(rms);
        for (k = 0; k < q; ++k) {
            v = bias[k];
            for (j = 0; j < h; ++j) v += (double)weight[k*h+j]*(s->x[r*h+j]/rms);
            if (!isfinite(v)) return 3;
            logits[k] = v;
        }
        for (j = 0; j < n; ++j) {
            out[r*q+j] = (float)sigmoid(logits[j]);
            out[r*q+n+n*n+j] = (float)(2.0*sigmoid(logits[n+n*n+j]));
        }
        maximum = logits[n];
        for (j = 0; j < n*n; ++j) if (logits[n+j] > maximum) maximum = logits[n+j];
        for (j = 0; j < n*n; ++j) b[j] = exp(logits[n+j]-maximum)+1e-12;
        for (sweep = 0; sweep < sweeps; ++sweep) {
            for (i = 0; i < n; ++i) {
                total = 0.0; for (j = 0; j < n; ++j) total += b[i*n+j];
                for (j = 0; j < n; ++j) b[i*n+j] /= total;
            }
            for (j = 0; j < n; ++j) {
                total = 0.0; for (i = 0; i < n; ++i) total += b[i*n+j];
                for (i = 0; i < n; ++i) b[i*n+j] /= total;
            }
        }
        for (j = 0; j < n*n; ++j) out[r*q+n+j] = (float)b[j];
    }
    return 0;
}
API int oa_v41_mhc_predict(const void *state, size_t bytes,
    const float *weight, size_t nw, const float *bias, size_t nb,
    double eps, uint32_t sweeps, float *out, size_t no) {
    const State *s = (const State *)state;
    float t[NQ]; size_t q, h; int status = valid(s, bytes);
    if (status) return status;
    q = (size_t)s->streams*(s->streams+2u); h = (size_t)s->streams*s->width;
    if (!floats(weight, nw, q*h) || !floats(bias, nb, q) ||
        !floats(out, no, (size_t)s->rows*q) || !isfinite(eps) || eps <= 0.0 ||
        !sweeps || sweeps > 32u || overlaps(state, bytes, out, no*sizeof(float)) ||
        overlaps(out, no*sizeof(float), weight, nw*sizeof(float)) ||
        overlaps(out, no*sizeof(float), bias, nb*sizeof(float))) return 1;
    if (!finite_array(weight, nw) || !finite_array(bias, nb)) return 3;
    status = predict(s, weight, bias, eps, sweeps, t);
    if (!status) memcpy(out, t, no*sizeof(float));
    return status;
}
static int update(State *s, const float *coeff, const float *f) {
    float x[NX]; size_t r, i, j, d, n = s->streams, w = s->width, q = n*(n+2u);
    if (!finite_array(coeff, (size_t)s->rows*q) || !finite_array(f, (size_t)s->rows*w)) return 3;
    for (r = 0; r < s->rows; ++r) {
        const float *c = coeff+r*q;
        for (j = 0; j < n; ++j) if (c[j] < 0.0f || c[j] > 1.0f ||
            c[n+n*n+j] < 0.0f || c[n+n*n+j] > 2.0f) return 1;
        for (j = 0; j < n*n; ++j) if (c[n+j] < 0.0f || c[n+j] > 1.0f) return 1;
        for (i = 0; i < n; ++i) for (d = 0; d < w; ++d) {
            double v = (double)c[n+n*n+i]*f[r*w+d];
            for (j = 0; j < n; ++j) v += (double)c[n+i*n+j]*s->x[(r*n+j)*w+d];
            if (store(v, &x[(r*n+i)*w+d])) return 3;
        }
    }
    memcpy(s->x, x, (size_t)s->rows*n*w*sizeof(float));
    for (r = 0; r < s->rows; ++r) memcpy(s->previous+r*n, coeff+r*q, n*sizeof(float));
    ++s->depth; return 0;
}
API int oa_v41_mhc_commit(void *state, size_t bytes, uint32_t expected_depth,
    const float *coeff, size_t nc, const float *f, size_t nf) {
    State t; State *s = (State *)state; int status = valid(s, bytes);
    if (status) return status;
    if (s->depth != expected_depth) return 1;
    if (s->depth == s->capacity) return 4;
    if (!floats(coeff, nc, (size_t)s->rows*s->streams*(s->streams+2u)) ||
        !floats(f, nf, (size_t)s->rows*s->width) ||
        overlaps(state, bytes, coeff, nc*sizeof(float)) || overlaps(state, bytes, f, nf*sizeof(float))) return 1;
    memcpy(&t, s, sizeof(t)); status = update(&t, coeff, f);
    if (!status) memcpy(s, &t, sizeof(t));
    return status;
}
/* Contiguous depth-major matrices. All chunk work is staged before commit. */
API int oa_v41_mhc_run(void *state, size_t bytes, uint32_t expected_depth, uint32_t blocks,
    const float *hw, size_t nhw, const float *hb, size_t nhb,
    const float *fw, size_t nfw, const float *fb, size_t nfb,
    double eps, uint32_t sweeps, size_t budget) {
    State t; State *s = (State *)state; float a[NQ], input[128], f[128];
    size_t q, h, w, b, r, i, j; int status = valid(s, bytes);
    if (status) return status;
    if (s->depth != expected_depth || !blocks || blocks > 16u || !isfinite(eps) || eps <= 0.0 ||
        !sweeps || sweeps > 32u) return 1;
    if (blocks > s->capacity-s->depth) return 4;
    q = (size_t)s->streams*(s->streams+2u); w = s->width; h = (size_t)s->streams*w;
    if (!floats(hw, nhw, blocks*q*h) || !floats(hb, nhb, blocks*q) ||
        !floats(fw, nfw, blocks*w*w) || !floats(fb, nfb, blocks*w) ||
        overlaps(state, bytes, hw, nhw*sizeof(float)) || overlaps(state, bytes, hb, nhb*sizeof(float)) ||
        overlaps(state, bytes, fw, nfw*sizeof(float)) || overlaps(state, bytes, fb, nfb*sizeof(float))) return 1;
    if (budget < oa_v41_mhc_scratch_bytes()) return 2;
    if (!finite_array(hw, nhw) || !finite_array(hb, nhb) ||
        !finite_array(fw, nfw) || !finite_array(fb, nfb)) return 3;
    memcpy(&t, s, sizeof(t));
    for (b = 0; b < blocks; ++b) {
        status = mix(&t, t.previous, input); if (status) return status;
        status = predict(&t, hw+b*q*h, hb+b*q, eps, sweeps, a); if (status) return status;
        for (r = 0; r < t.rows; ++r) for (i = 0; i < w; ++i) {
            double v = fb[b*w+i];
            for (j = 0; j < w; ++j) v += (double)fw[b*w*w+i*w+j]*input[r*w+j];
            if (store(v, &f[r*w+i])) return 3;
        }
        status = update(&t, a, f); if (status) return status;
    }
    memcpy(s, &t, sizeof(t)); return 0;
}
/* Explicit final head: separate supplied stream mix, RMSNorm, linear logits. */
API int oa_v41_mhc_head(const void *state, size_t bytes,
    const float *a, size_t na, const float *norm, size_t nn,
    const float *weight, size_t nw, const float *bias, size_t nb,
    uint32_t vocab, double eps, float *out, size_t capacity) {
    const State *s = (const State *)state; float mixed[128], result[256];
    size_t r, i, d, w, count; int status = valid(s, bytes);
    if (status) return status;
    w = s->width; count = (size_t)s->rows*vocab;
    if (!vocab || vocab > 32u || !isfinite(eps) || eps <= 0.0 ||
        !floats(a, na, (size_t)s->rows*s->streams) || !floats(norm, nn, w) ||
        !floats(weight, nw, vocab*w) || !floats(bias, nb, vocab) ||
        capacity > 256u || !region(out, capacity*sizeof(float), sizeof(float))) return 1;
    if (capacity < count) return 4;
    if (overlaps(state, bytes, out, capacity*sizeof(float)) ||
        overlaps(a, na*sizeof(float), out, capacity*sizeof(float)) ||
        overlaps(norm, nn*sizeof(float), out, capacity*sizeof(float)) ||
        overlaps(weight, nw*sizeof(float), out, capacity*sizeof(float)) ||
        overlaps(bias, nb*sizeof(float), out, capacity*sizeof(float))) return 1;
    if (!finite_array(a, na) || !finite_array(norm, nn) || !finite_array(weight, nw) || !finite_array(bias, nb)) return 3;
    status = mix(s, a, mixed); if (status) return status;
    for (r = 0; r < s->rows; ++r) {
        double rms = eps;
        for (d = 0; d < w; ++d) rms += (double)mixed[r*w+d]*mixed[r*w+d]/(double)w;
        rms = sqrt(rms);
        for (i = 0; i < vocab; ++i) {
            double v = bias[i];
            for (d = 0; d < w; ++d) v += weight[i*w+d]*(mixed[r*w+d]/rms)*norm[d];
            if (store(v, &result[r*vocab+i])) return 3;
        }
    }
    memcpy(out, result, count*sizeof(float)); return 0;
}
