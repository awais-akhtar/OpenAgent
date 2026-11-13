/* Fused dequantize-and-dot kernels for ggml Q4_K and Q6_K weight rows.
 *
 * Each kernel computes y[r] = dot(row r of W, x) for a contiguous range of
 * rows, decoding the quantized blocks on the fly so no float copy of the
 * matrix is made. The layouts match the scalar reference loops in
 * tests/test_qwen3moe_gguf.py. Built by qwen3moe_native.py with MSVC.
 */
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#define QK_K 256

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* subnormal: normalize */
            int e = -1;
            do { e++; mant <<= 1; } while (!(mant & 0x400));
            mant &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

static void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* One Q4_K block (144 bytes, 256 weights) dotted with x[0..255].
   Element = d*sc*nibble - dmin*m, so the block is d*sc*sum(q*x) - dmin*m*sum(x). */
static float dot_q4k_block(const uint8_t *b, const float *x) {
    float d = fp16_to_fp32((uint16_t)(b[0] | (b[1] << 8)));
    float dmin = fp16_to_fp32((uint16_t)(b[2] | (b[3] << 8)));
    const uint8_t *scales = b + 4;
    const uint8_t *qs = b + 16;
    float acc = 0.0f;
    int is = 0;
    const __m256i mask = _mm256_set1_epi32(0xF);
    for (int chunk = 0; chunk < 4; chunk++) {
        uint8_t sc1, m1, sc2, m2;
        scale_min_k4(is, scales, &sc1, &m1);
        scale_min_k4(is + 1, scales, &sc2, &m2);
        const uint8_t *q = qs + 32 * chunk;
        const float *xl = x + 64 * chunk;
        const float *xh = xl + 32;
        __m256 vsl = _mm256_setzero_ps(), vxl = _mm256_setzero_ps();
        __m256 vsh = _mm256_setzero_ps(), vxh = _mm256_setzero_ps();
        for (int k = 0; k < 4; k++) {
            __m128i b8 = _mm_loadl_epi64((const __m128i *)(q + 8 * k));
            __m256i ints = _mm256_cvtepu8_epi32(b8);
            __m256 lo = _mm256_cvtepi32_ps(_mm256_and_si256(ints, mask));
            __m256 hi = _mm256_cvtepi32_ps(_mm256_srli_epi32(ints, 4));
            __m256 xlv = _mm256_loadu_ps(xl + 8 * k);
            __m256 xhv = _mm256_loadu_ps(xh + 8 * k);
            vsl = _mm256_fmadd_ps(lo, xlv, vsl);
            vxl = _mm256_add_ps(vxl, xlv);
            vsh = _mm256_fmadd_ps(hi, xhv, vsh);
            vxh = _mm256_add_ps(vxh, xhv);
        }
        acc += d * sc1 * hsum256(vsl) - dmin * m1 * hsum256(vxl);
        acc += d * sc2 * hsum256(vsh) - dmin * m2 * hsum256(vxh);
        is += 2;
    }
    return acc;
}

/* One Q6_K block (210 bytes, 256 weights) dotted with x[0..255]. */
static float dot_q6k_block(const uint8_t *b, const float *x) {
    const uint8_t *ql = b;
    const uint8_t *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    float d = fp16_to_fp32((uint16_t)(b[208] | (b[209] << 8)));
    const __m256i mask4 = _mm256_set1_epi32(0xF);
    const __m256i mask2 = _mm256_set1_epi32(3);
    const __m256i off = _mm256_set1_epi32(32);
    __m256 vacc = _mm256_setzero_ps();
    for (int half = 0; half < 2; half++) {
        const uint8_t *qlh = ql + 64 * half;
        const uint8_t *qhh = qh + 32 * half;
        const int8_t *sch = sc + 8 * half;
        const float *xh = x + 128 * half;
        for (int l0 = 0; l0 < 32; l0 += 8) {
            int is = l0 / 16;
            __m256i lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qlh + l0)));
            __m256i hi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qlh + 32 + l0)));
            __m256i hb = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qhh + l0)));
            __m256i q1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(lo, mask4),
                             _mm256_slli_epi32(_mm256_and_si256(hb, mask2), 4)), off);
            __m256i q2 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(hi, mask4),
                             _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 2), mask2), 4)), off);
            __m256i q3 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(lo, 4),
                             _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 4), mask2), 4)), off);
            __m256i q4 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(hi, 4),
                             _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hb, 6), mask2), 4)), off);
            __m256 s1 = _mm256_set1_ps(d * sch[is]);
            __m256 s2 = _mm256_set1_ps(d * sch[is + 2]);
            __m256 s3 = _mm256_set1_ps(d * sch[is + 4]);
            __m256 s4 = _mm256_set1_ps(d * sch[is + 6]);
            vacc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q1), s1), _mm256_loadu_ps(xh + l0), vacc);
            vacc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q2), s2), _mm256_loadu_ps(xh + l0 + 32), vacc);
            vacc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q3), s3), _mm256_loadu_ps(xh + l0 + 64), vacc);
            vacc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q4), s4), _mm256_loadu_ps(xh + l0 + 96), vacc);
        }
    }
    return hsum256(vacc);
}

/* y[r] for r in [row0, row1): rows of `cols` weights, each row cols/256 blocks. */
EXPORT void q4k_matvec_rows(const uint8_t *w, int64_t cols, int64_t row0, int64_t row1,
                     const float *x, float *y) {
    int64_t blocks = cols / QK_K;
    for (int64_t r = row0; r < row1; r++) {
        const uint8_t *row = w + r * blocks * 144;
        float acc = 0.0f;
        for (int64_t k = 0; k < blocks; k++) {
            acc += dot_q4k_block(row + k * 144, x + k * QK_K);
        }
        y[r] = acc;
    }
}

EXPORT void q6k_matvec_rows(const uint8_t *w, int64_t cols, int64_t row0, int64_t row1,
                     const float *x, float *y) {
    int64_t blocks = cols / QK_K;
    for (int64_t r = row0; r < row1; r++) {
        const uint8_t *row = w + r * blocks * 210;
        float acc = 0.0f;
        for (int64_t k = 0; k < blocks; k++) {
            acc += dot_q6k_block(row + k * 210, x + k * QK_K);
        }
        y[r] = acc;
    }
}


/* ---- int8 activations (Q8 blocks of 256) and integer dot products ---- */
#ifdef _WIN32
static int32_t hsum_epi32_256(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

/* Quantize n floats (n multiple of 256) to int8 blocks with a per-block scale
   (amax/127) and per-32-value sums. */
EXPORT void quantize_q8(const float *x, int64_t n, int8_t *q, float *d, int32_t *bsums) {
    int64_t blocks = n / QK_K;
    for (int64_t b = 0; b < blocks; b++) {
        const float *xb = x + b * QK_K;
        float amax = 0.0f;
        for (int i = 0; i < QK_K; i++) {
            float a = xb[i] < 0 ? -xb[i] : xb[i];
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        d[b] = scale;
        for (int j = 0; j < 8; j++) {
            int32_t sum = 0;
            for (int i = 0; i < 32; i++) {
                float v = xb[j * 32 + i] * inv;
                int32_t r = (int32_t)(v < 0 ? v - 0.5f : v + 0.5f);
                if (r > 127) r = 127;
                if (r < -127) r = -127;
                q[b * QK_K + j * 32 + i] = (int8_t)r;
                sum += r;
            }
            bsums[b * 8 + j] = sum;
        }
    }
}

/* Q4_K weights times Q8 activations, rows [row0, row1). */
static void q4k_q8_rows(const uint8_t *w, int64_t cols, int64_t row0, int64_t row1,
                        const int8_t *xq, const float *xd, const int32_t *xbs, float *y) {
    int64_t blocks = cols / QK_K;
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int64_t r = row0; r < row1; r++) {
        const uint8_t *row = w + r * blocks * 144;
        float acc = 0.0f;
        for (int64_t b = 0; b < blocks; b++) {
            const uint8_t *blk = row + b * 144;
            float d = fp16_to_fp32((uint16_t)(blk[0] | (blk[1] << 8)));
            float dmin = fp16_to_fp32((uint16_t)(blk[2] | (blk[3] << 8)));
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; j++) scale_min_k4(j, blk + 4, &sc[j], &mn[j]);
            const uint8_t *qs = blk + 16;
            const int8_t *xb = xq + b * QK_K;
            int32_t msum = 0;
            int32_t ssum = 0;
            for (int c = 0; c < 4; c++) {
                __m256i v = _mm256_loadu_si256((const __m256i *)(qs + 32 * c));
                __m256i lo = _mm256_and_si256(v, mask);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
                __m256i xl = _mm256_loadu_si256((const __m256i *)(xb + 64 * c));
                __m256i xh = _mm256_loadu_si256((const __m256i *)(xb + 64 * c + 32));
                __m256i pl = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xl), ones);
                __m256i ph = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xh), ones);
                ssum += (int32_t)sc[2 * c] * hsum_epi32_256(pl) + (int32_t)sc[2 * c + 1] * hsum_epi32_256(ph);
                msum += (int32_t)mn[2 * c] * xbs[b * 8 + 2 * c] + (int32_t)mn[2 * c + 1] * xbs[b * 8 + 2 * c + 1];
            }
            acc += xd[b] * (d * (float)ssum - dmin * (float)msum);
        }
        y[r] = acc;
    }
}

/* Q6_K weights times Q8 activations, rows [row0, row1), 16-value groups in 16-bit lanes. */
static void q6k_q8_rows(const uint8_t *w, int64_t cols, int64_t row0, int64_t row1,
                        const int8_t *xq, const float *xd, float *y) {
    int64_t blocks = cols / QK_K;
    const __m256i m4 = _mm256_set1_epi16(0xF);
    const __m256i m2 = _mm256_set1_epi16(3);
    const __m256i off = _mm256_set1_epi16(32);
    for (int64_t r = row0; r < row1; r++) {
        const uint8_t *row = w + r * blocks * 210;
        float acc = 0.0f;
        for (int64_t b = 0; b < blocks; b++) {
            const uint8_t *blk = row + b * 210;
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            const int8_t *sc = (const int8_t *)(blk + 192);
            float d = fp16_to_fp32((uint16_t)(blk[208] | (blk[209] << 8)));
            const int8_t *xb = xq + b * QK_K;
            __m256i vsum = _mm256_setzero_si256();
            for (int half = 0; half < 2; half++) {
                const uint8_t *qlh = ql + 64 * half;
                const uint8_t *qhh = qh + 32 * half;
                const int8_t *sch = sc + 8 * half;
                const int8_t *xh = xb + 128 * half;
                for (int l0 = 0; l0 < 32; l0 += 16) {
                    int is = l0 / 16;
                    __m256i lo = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(qlh + l0)));
                    __m256i hi = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(qlh + 32 + l0)));
                    __m256i hb = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(qhh + l0)));
                    __m256i q1 = _mm256_sub_epi16(_mm256_or_si256(_mm256_and_si256(lo, m4),
                                     _mm256_slli_epi16(_mm256_and_si256(hb, m2), 4)), off);
                    __m256i q2 = _mm256_sub_epi16(_mm256_or_si256(_mm256_and_si256(hi, m4),
                                     _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2), m2), 4)), off);
                    __m256i q3 = _mm256_sub_epi16(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo, 4), m4),
                                     _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 4), m2), 4)), off);
                    __m256i q4 = _mm256_sub_epi16(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(hi, 4), m4),
                                     _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 6), m2), 4)), off);
                    __m256i x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xh + l0)));
                    __m256i x2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xh + 32 + l0)));
                    __m256i x3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xh + 64 + l0)));
                    __m256i x4 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xh + 96 + l0)));
                    vsum = _mm256_add_epi32(vsum, _mm256_mullo_epi32(_mm256_madd_epi16(q1, x1), _mm256_set1_epi32(sch[is])));
                    vsum = _mm256_add_epi32(vsum, _mm256_mullo_epi32(_mm256_madd_epi16(q2, x2), _mm256_set1_epi32(sch[is + 2])));
                    vsum = _mm256_add_epi32(vsum, _mm256_mullo_epi32(_mm256_madd_epi16(q3, x3), _mm256_set1_epi32(sch[is + 4])));
                    vsum = _mm256_add_epi32(vsum, _mm256_mullo_epi32(_mm256_madd_epi16(q4, x4), _mm256_set1_epi32(sch[is + 6])));
                }
            }
            acc += xd[b] * d * (float)hsum_epi32_256(vsum);
        }
        y[r] = acc;
    }
}
#endif



/* Q8-activation matvec for dense matrices: quantizes x once, then computes rows [r0, r1). */
EXPORT void q8_matvec_rows(int ggml_type, const uint8_t *w, int64_t cols, int64_t r0, int64_t r1,
                           const float *x, float *y) {
    int8_t xq[8192];
    float xd[32];
    int32_t xbs[32 * 8];
    quantize_q8(x, cols, xq, xd, xbs);
    if (ggml_type == 12) {
        q4k_q8_rows(w, cols, r0, r1, xq, xd, xbs, y);
    } else {
        q6k_q8_rows(w, cols, r0, r1, xq, xd, y);
    }
}

/* ---- routed-expert pass with a persistent worker pool (Windows threads) ---- */
#ifdef _WIN32
#include <windows.h>

#define MOE_MAX_TASKS 8
#define MOE_MAX_THREADS 16

typedef struct {
    const uint8_t *gate[MOE_MAX_TASKS];
    const uint8_t *up[MOE_MAX_TASKS];
    const uint8_t *down[MOE_MAX_TASKS];
    const float *h;
    float *scratch; /* MOE_MAX_TASKS x n_embd */
    float *ffn;     /* MOE_MAX_TASKS x n_ff */
    int64_t n_embd, n_ff;
    int gate_type, up_type, down_type; /* per tensor: 12 = Q4_K, 14 = Q6_K */
    int tasks;
} moe_job;

static HANDLE moe_start[MOE_MAX_THREADS];
static HANDLE moe_done;
static HANDLE moe_threads[MOE_MAX_THREADS];
static volatile LONG moe_remaining;
static moe_job moe_current;
static int moe_thread_count = 0;
static volatile LONG moe_stop = 0;
static CRITICAL_SECTION moe_gate_lock;
static int moe_initialised = 0;

static void moe_run_task(int t) {
    const moe_job *j = &moe_current;
    float *ffn_gate = j->ffn + (int64_t)t * 2 * j->n_ff;
    float *ffn_up = ffn_gate + j->n_ff;
    float *scratch = j->scratch + (int64_t)t * j->n_embd;
    /* Activations are quantized to int8 once per matrix input, as llama.cpp does,
       so each row is an integer dot product. */
    int8_t xq[4096];
    float xd[16];
    int32_t xbs[16 * 8];
    quantize_q8(j->h, j->n_embd, xq, xd, xbs);
    if (j->gate_type == 12) {
        q4k_q8_rows(j->gate[t], j->n_embd, 0, j->n_ff, xq, xd, xbs, ffn_gate);
    } else {
        q6k_q8_rows(j->gate[t], j->n_embd, 0, j->n_ff, xq, xd, ffn_gate);
    }
    if (j->up_type == 12) {
        q4k_q8_rows(j->up[t], j->n_embd, 0, j->n_ff, xq, xd, xbs, ffn_up);
    } else {
        q6k_q8_rows(j->up[t], j->n_embd, 0, j->n_ff, xq, xd, ffn_up);
    }
    for (int64_t i = 0; i < j->n_ff; i++) {
        float g = ffn_gate[i];
        float silu = g / (1.0f + expf(-g));
        ffn_gate[i] = silu * ffn_up[i];
    }
    quantize_q8(ffn_gate, j->n_ff, xq, xd, xbs);
    if (j->down_type == 12) {
        q4k_q8_rows(j->down[t], j->n_ff, 0, j->n_embd, xq, xd, xbs, scratch);
    } else {
        q6k_q8_rows(j->down[t], j->n_ff, 0, j->n_embd, xq, xd, scratch);
    }
}

static DWORD WINAPI moe_worker(LPVOID arg) {
    int index = (int)(intptr_t)arg;
    for (;;) {
        WaitForSingleObject(moe_start[index], INFINITE);
        if (moe_stop) return 0;
        for (int t = index; t < moe_current.tasks; t += moe_thread_count) moe_run_task(t);
        if (InterlockedDecrement(&moe_remaining) == 0) SetEvent(moe_done);
    }
}

EXPORT int moe_pool_init(int threads) {
    if (moe_initialised) return moe_thread_count;
    if (threads < 1) threads = 1;
    if (threads > MOE_MAX_THREADS) threads = MOE_MAX_THREADS;
    /* the calling thread works as worker 0, so only threads-1 are created */
    moe_thread_count = threads;
    moe_done = CreateEventW(NULL, FALSE, FALSE, NULL);
    InitializeCriticalSection(&moe_gate_lock);
    moe_stop = 0;
    for (int i = 1; i < threads; i++) {
        moe_start[i] = CreateEventW(NULL, FALSE, FALSE, NULL);
        moe_threads[i] = CreateThread(NULL, 0, moe_worker, (LPVOID)(intptr_t)i, 0, NULL);
    }
    moe_initialised = 1;
    return threads;
}

EXPORT void moe_pool_shutdown(void) {
    if (!moe_initialised) return;
    moe_stop = 1;
    for (int i = 1; i < moe_thread_count; i++) {
        SetEvent(moe_start[i]);
        WaitForSingleObject(moe_threads[i], INFINITE);
        CloseHandle(moe_threads[i]);
        CloseHandle(moe_start[i]);
    }
    CloseHandle(moe_done);
    DeleteCriticalSection(&moe_gate_lock);
    moe_initialised = 0;
    moe_thread_count = 0;
}

/* Routed experts for one token and one layer:
   out = sum_t weights[t] * down_t(silu(gate_t h) * up_t h).
   Tasks run in parallel; each writes its own scratch slot, then the caller sums. */
EXPORT int moe_layer(const uint8_t *const *gate, const uint8_t *const *up, const uint8_t *const *down,
                     int gate_type, int up_type, int down_type, int64_t n_embd, int64_t n_ff, int tasks,
                     const float *weights, const float *h, float *scratch, float *ffn, float *out) {
    if (!moe_initialised || tasks < 1 || tasks > MOE_MAX_TASKS) return -1;
    EnterCriticalSection(&moe_gate_lock);
    for (int t = 0; t < tasks; t++) {
        moe_current.gate[t] = gate[t];
        moe_current.up[t] = up[t];
        moe_current.down[t] = down[t];
    }
    moe_current.h = h;
    moe_current.scratch = scratch;
    moe_current.ffn = ffn;
    moe_current.n_embd = n_embd;
    moe_current.n_ff = n_ff;
    moe_current.gate_type = gate_type;
    moe_current.up_type = up_type;
    moe_current.down_type = down_type;
    moe_current.tasks = tasks;
    int workers = moe_thread_count < tasks ? moe_thread_count : tasks;
    moe_remaining = workers;
    for (int i = 1; i < workers; i++) SetEvent(moe_start[i]);
    /* worker 0 is this thread */
    for (int t = 0; t < tasks; t += moe_thread_count) moe_run_task(t);
    if (InterlockedDecrement(&moe_remaining) != 0) WaitForSingleObject(moe_done, INFINITE);
    /* weighted sum in task order: deterministic results */
    for (int64_t i = 0; i < n_embd; i++) out[i] = 0.0f;
    for (int t = 0; t < tasks; t++) {
        const float *row = scratch + (int64_t)t * n_embd;
        float w = weights[t];
        for (int64_t i = 0; i < n_embd; i++) out[i] += w * row[i];
    }
    LeaveCriticalSection(&moe_gate_lock);
    return 0;
}
#endif

/* ---- native token loop: attention, KV update, routing, experts, output ----
 * One call per generated token. Python supplies the weight pointers, KV caches and
 * two callbacks: acquire(layer, experts) leases the routed expert matrices and
 * returns their addresses; release(layer) returns the leases after the layer.
 * Dense matrices are read in their GGUF quantization with Q8 activations.
 */
#ifdef _WIN32

typedef int (*qwen_acquire_fn)(int64_t layer, const int32_t *experts, int64_t n,
                               int64_t *gate, int64_t *up, int64_t *down);
typedef void (*qwen_release_fn)(int64_t layer);

typedef struct {
    int64_t attn_norm, q_norm, k_norm, ffn_norm, router; /* float32 pointers */
    int64_t wq, wk, wv, wo;                              /* quantized matrices */
    int32_t tq, tk, tv, to;                              /* ggml types (12 or 14) */
    int32_t tgate, tup, tdown, tpad;                     /* expert types vary by layer */
} qwen_layer;

typedef struct {
    int64_t n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff, n_expert, n_used, ctx, vocab;
    double eps, scale;
    int64_t output_norm, output_w, rope_inv; /* float32, quantized, double[] */
    int32_t output_type, gate_type, down_type, pad;
    int64_t layers;                          /* qwen_layer[n_layer] */
    int64_t k_cache, v_cache;                /* float32 [n_layer][ctx][n_head_kv][head_dim] */
    int64_t acquire_cb, release_cb;
    /* scratch, allocated by qwen_init */
    int64_t buf;
} qwen_ctx;

static void rms_norm_vec(const float *x, const float *w, int64_t n, double eps, float *out) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float scale = (float)(1.0 / sqrt(ss / (double)n + eps));
    for (int64_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void rope_heads(float *x, int64_t heads, int64_t hd, int64_t pos, const double *inv) {
    int64_t half = hd / 2;
    for (int64_t h = 0; h < heads; h++) {
        float *v = x + h * hd;
        for (int64_t i = 0; i < half; i++) {
            double angle = (double)pos * inv[i];
            float c = (float)cos(angle), s = (float)sin(angle);
            float a = v[i], b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = b * c + a * s;
        }
    }
}

static void matvec_q8_prepared(int type, const uint8_t *w, int64_t cols, int64_t rows,
                               const int8_t *xq, const float *xd, const int32_t *xbs, float *y) {
    if (type == 12) q4k_q8_rows(w, cols, 0, rows, xq, xd, xbs, y);
    else q6k_q8_rows(w, cols, 0, rows, xq, xd, y);
}

static float *qwen_f(int64_t p) { return (float *)(intptr_t)p; }

/* Scratch layout inside ctx->buf, allocated once by qwen3moe_token_init. */
typedef struct {
    float *x, *h, *q, *k, *v, *att, *h2, *moe, *logits_router, *probs, *moe_scratch, *moe_ffn;
    int8_t *xq;
    float xd[64];
    int32_t xbs[64 * 8];
    int32_t ids[16];
    float weights[16];
    double *inv;
} qwen_work;

EXPORT int qwen3moe_token_init(qwen_ctx *c) {
    qwen_work *w = (qwen_work *)calloc(1, sizeof(qwen_work));
    if (!w) return -1;
    int64_t E = c->n_embd, QD = c->n_head * c->head_dim, KD = c->n_head_kv * c->head_dim;
    int64_t P = c->ctx > c->n_expert ? c->ctx : c->n_expert;
    w->x = (float *)calloc(E, sizeof(float));
    w->h = (float *)calloc(E, sizeof(float));
    w->h2 = (float *)calloc(E, sizeof(float));
    w->moe = (float *)calloc(E, sizeof(float));
    w->q = (float *)calloc(QD, sizeof(float));
    w->att = (float *)calloc(QD, sizeof(float));
    w->k = (float *)calloc(KD, sizeof(float));
    w->v = (float *)calloc(KD, sizeof(float));
    w->logits_router = (float *)calloc(c->n_expert, sizeof(float));
    w->probs = (float *)calloc(P, sizeof(float));
    w->moe_scratch = (float *)calloc(8 * E, sizeof(float));
    w->moe_ffn = (float *)calloc(8 * 2 * c->n_ff, sizeof(float));
    w->xq = (int8_t *)calloc(8192, 1);
    if (!w->x || !w->h || !w->h2 || !w->moe || !w->q || !w->att || !w->k || !w->v ||
        !w->logits_router || !w->probs || !w->moe_scratch || !w->moe_ffn || !w->xq) return -1;
    c->buf = (int64_t)(intptr_t)w;
    return 0;
}

EXPORT void qwen3moe_token_free(qwen_ctx *c) {
    qwen_work *w = (qwen_work *)(intptr_t)c->buf;
    if (!w) return;
    free(w->x); free(w->h); free(w->h2); free(w->moe); free(w->q); free(w->att);
    free(w->k); free(w->v); free(w->logits_router); free(w->probs);
    free(w->moe_scratch); free(w->moe_ffn); free(w->xq); free(w);
    c->buf = 0;
}


EXPORT int qwen3moe_token(qwen_ctx *c, const float *emb, int64_t pos, float *logits) {
    if (!c || !emb || !logits || pos < 0 || pos >= c->ctx) return -1;
    qwen_work *w = (qwen_work *)(intptr_t)c->buf;
    if (!w) return -4;
    w->inv = (double *)(intptr_t)c->rope_inv;
    int64_t E = c->n_embd, HD = c->head_dim, NH = c->n_head, NK = c->n_head_kv;
    int64_t group = NH / NK, QD = NH * HD, KD = NK * HD;
    float scale = (float)c->scale;
    float *x = w->x;
    memcpy(x, emb, sizeof(float) * E);
    qwen_layer *layers = (qwen_layer *)(intptr_t)c->layers;
    for (int64_t L = 0; L < c->n_layer; L++) {
        const qwen_layer *ly = &layers[L];
        /* ---- attention ---- */
        rms_norm_vec(x, qwen_f(ly->attn_norm), E, c->eps, w->h);
        int8_t *xq = w->xq;
        float *xd = w->xd;
        int32_t *xbs = w->xbs;
        quantize_q8(w->h, E, xq, xd, xbs);
        matvec_q8_prepared(ly->tq, (const uint8_t *)(intptr_t)ly->wq, E, QD, xq, xd, xbs, w->q);
        matvec_q8_prepared(ly->tk, (const uint8_t *)(intptr_t)ly->wk, E, KD, xq, xd, xbs, w->k);
        matvec_q8_prepared(ly->tv, (const uint8_t *)(intptr_t)ly->wv, E, KD, xq, xd, xbs, w->v);
        for (int64_t hI = 0; hI < NH; hI++) {
            float tmp[1024];
            rms_norm_vec(w->q + hI * HD, qwen_f(ly->q_norm), HD, c->eps, tmp);
            memcpy(w->q + hI * HD, tmp, sizeof(float) * HD);
        }
        for (int64_t kI = 0; kI < NK; kI++) {
            float tmp[1024];
            rms_norm_vec(w->k + kI * HD, qwen_f(ly->k_norm), HD, c->eps, tmp);
            memcpy(w->k + kI * HD, tmp, sizeof(float) * HD);
        }
        rope_heads(w->q, NH, HD, pos, w->inv);
        rope_heads(w->k, NK, HD, pos, w->inv);
        float *kc = (float *)(intptr_t)c->k_cache + (L * c->ctx) * KD;
        float *vc = (float *)(intptr_t)c->v_cache + (L * c->ctx) * KD;
        memcpy(kc + pos * KD, w->k, sizeof(float) * KD);
        memcpy(vc + pos * KD, w->v, sizeof(float) * KD);
        for (int64_t hI = 0; hI < NH; hI++) {
            int64_t kv = hI / group;
            const float *q = w->q + hI * HD;
            float *out = w->att + hI * HD;
            double maxs = -1e300;
            float *sc = w->probs; /* reused: one score per position */
            for (int64_t t = 0; t <= pos; t++) {
                const float *kt = kc + t * KD + kv * HD;
                float dot = 0.0f;
                for (int64_t d = 0; d < HD; d++) dot += q[d] * kt[d];
                sc[t] = dot * scale;
                if (sc[t] > maxs) maxs = sc[t];
            }
            double denom = 0.0;
            for (int64_t t = 0; t <= pos; t++) {
                sc[t] = (float)exp((double)sc[t] - maxs);
                denom += sc[t];
            }
            for (int64_t d = 0; d < HD; d++) out[d] = 0.0f;
            for (int64_t t = 0; t <= pos; t++) {
                float p = (float)(sc[t] / denom);
                const float *vt = vc + t * KD + kv * HD;
                for (int64_t d = 0; d < HD; d++) out[d] += p * vt[d];
            }
        }
        quantize_q8(w->att, QD, xq, xd, xbs);
        float *proj = w->h2;
        matvec_q8_prepared(ly->to, (const uint8_t *)(intptr_t)ly->wo, QD, E, xq, xd, xbs, proj);
        for (int64_t i = 0; i < E; i++) x[i] += proj[i];

        /* ---- routed experts ---- */
        rms_norm_vec(x, qwen_f(ly->ffn_norm), E, c->eps, w->h2);
        const float *rw = qwen_f(ly->router);
        for (int64_t e = 0; e < c->n_expert; e++) {
            double s = 0.0;
            const float *row = rw + e * E;
            for (int64_t i = 0; i < E; i++) s += (double)row[i] * (double)w->h2[i];
            w->logits_router[e] = (float)s;
        }
        double mx = -1e300;
        for (int64_t e = 0; e < c->n_expert; e++) if (w->logits_router[e] > mx) mx = w->logits_router[e];
        double tot = 0.0;
        for (int64_t e = 0; e < c->n_expert; e++) {
            w->probs[e] = (float)exp((double)w->logits_router[e] - mx);
            tot += w->probs[e];
        }
        for (int64_t e = 0; e < c->n_expert; e++) w->probs[e] = (float)(w->probs[e] / tot);
        /* top-k by repeated maximum (k = 8 of 128: cheaper than a full sort) */
        int64_t used = c->n_used;
        double wsum = 0.0;
        for (int64_t k = 0; k < used; k++) {
            int64_t best = -1;
            float bv = -1.0f;
            for (int64_t e = 0; e < c->n_expert; e++) {
                if (w->probs[e] > bv) { bv = w->probs[e]; best = e; }
            }
            w->ids[k] = (int32_t)best;
            w->weights[k] = bv;
            wsum += bv;
            w->probs[best] = -1.0f;
        }
        for (int64_t k = 0; k < used; k++) w->weights[k] = (float)(w->weights[k] / wsum);
        int64_t gate[8], up[8], down[8];
        if (used > 8) return -2;
        if (((int (*)(int64_t, const int32_t *, int64_t, int64_t *, int64_t *, int64_t *))(intptr_t)c->acquire_cb)(
                L, w->ids, used, gate, up, down) != 0) return -3;
        moe_layer((const uint8_t *const *)gate, (const uint8_t *const *)up, (const uint8_t *const *)down,
                  ly->tgate, ly->tup, ly->tdown, E, c->n_ff, (int)used, w->weights, w->h2,
                  w->moe_scratch, w->moe_ffn, w->moe);
        ((void (*)(int64_t))(intptr_t)c->release_cb)(L);
        for (int64_t i = 0; i < E; i++) x[i] += w->moe[i];
    }
    rms_norm_vec(x, qwen_f(c->output_norm), E, c->eps, w->h);
    quantize_q8(w->h, E, w->xq, w->xd, w->xbs);
    matvec_q8_prepared(c->output_type, (const uint8_t *)(intptr_t)c->output_w, E, c->vocab,
                       w->xq, w->xd, w->xbs, logits);
    return 0;
}

#endif
