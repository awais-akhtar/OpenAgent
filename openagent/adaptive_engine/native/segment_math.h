#ifndef OA_SEGMENT_MATH_H
#define OA_SEGMENT_MATH_H
#include "segment_bridge.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Native host-endian FP32 RMSNorm, no callbacks or retained storage.
 * Buffers must be valid, tightly packed and mutually non-overlapping.
 * max_bytes bounds each buffer independently. Output is unchanged on error.
 * Positive finite epsilon; finite inputs/weights; unrepresentable output
 * returns OA_INVALID. Accumulation and square root use double precision. */
OA_API oa_status OA_CALL oa_segment_rmsnorm_f32(const float *input,
    const float *weight, float *output, uint64_t rows, uint64_t width,
    double epsilon, uint64_t max_bytes);
#ifdef __cplusplus
}
#endif
#endif
