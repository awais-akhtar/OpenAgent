#ifndef OA_SEGMENT_EXECUTOR_H
#define OA_SEGMENT_EXECUTOR_H
#include "segment_bridge.h"
#ifdef __cplusplus
extern "C" {
#endif
#define OA_EXECUTOR_VERSION 1u
typedef enum {
    OA_PACKED_INT8_ROW = 0,
    OA_PACKED_Q8_0 = 1,
    OA_PACKED_Q4_0 = 2,
    OA_PACKED_Q4_1 = 3,
    OA_PACKED_Q5_0 = 4,
    OA_PACKED_Q5_1 = 5,
    OA_PACKED_Q8_1 = 6
} oa_packed_format;
typedef struct { const float *data; uint64_t count; } oa_tensor;
typedef struct {
    uint32_t version, heads, kv_heads, topk, normalize_topk, reserved;
    uint64_t hidden, intermediate, experts, layers, capacity;
    double epsilon, theta;
} oa_range_config;
typedef struct {
    oa_tensor input_norm, q, k, v, o, q_norm, k_norm, post_norm, router;
    oa_tensor gate, up, down;
} oa_range_layer;
/* Synchronous, callback-free host FP32 execution. Weights are row-major,
 * expert-major. KV is [layer,kv_head,capacity,head_dim]. All pointers must
 * reference valid, non-overlapping storage; lengths are checked, pointers
 * are not sandboxed. No storage retained. Caller serializes session access.
 * max_bytes bounds EACH tensor and aggregate temporary allocations.
 * Output, KV and position are unchanged on every failure. */
OA_API uint32_t OA_CALL oa_executor_version(void);
OA_API oa_status OA_CALL oa_executor_range(const oa_range_config *config,
    const oa_range_layer *layers, uint64_t layer_count,
    oa_tensor input, uint64_t rows, float *output, uint64_t output_count,
    float *keys, float *values, uint64_t cache_count, uint64_t *position,
    uint64_t max_bytes);
/* Embedding copies rows by signed token ID. Head performs RMSNorm + W.T.
 * Edge output is also transactional; no callbacks or Python arithmetic. */
OA_API oa_status OA_CALL oa_executor_embed(oa_tensor embedding,
    uint64_t vocab, uint64_t hidden, const int64_t *ids, uint64_t rows,
    float *output, uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_head(oa_tensor input, uint64_t rows,
    uint64_t hidden, oa_tensor norm, oa_tensor weight, uint64_t vocab,
    double epsilon, float *output, uint64_t output_count, uint64_t max_bytes);
/* Compose embedding -> range -> head as ONE transaction. Output [rows,vocab],
 * KV and position publish only after the head succeeds. Same storage/alias
 * obligations as range. max_bytes bounds each tensor and the peak aggregate
 * heap scratch, including nested primitive allocations. No callbacks.
 * The caller supplies the complete desired layer sequence; this C function
 * cannot infer model coverage from descriptors. */
OA_API oa_status OA_CALL oa_executor_decode(const oa_range_config *config,
    const oa_range_layer *layers, uint64_t layer_count,
    oa_tensor embedding, oa_tensor final_norm, oa_tensor head, uint64_t vocab,
    const int64_t *ids, uint64_t rows, float *output, uint64_t output_count,
    float *keys, float *values, uint64_t cache_count, uint64_t *position,
    uint64_t max_bytes);
/* Phased leased execution. config.layers MUST be 1; gate/up/down descriptors
 * MUST be empty. Prepare performs attention + post norm + routing, returning
 * residual/normalized [rows,H] and ids/weights [rows,topk]. KV is [K,T,D].
 * No expert storage or callbacks. Position is an input, not advanced here.
 * All outputs and KV remain unchanged on failure. Caller stages a whole range
 * across phases before committing its session. Same non-alias/storage contract.
 * ids are uint64; max_bytes bounds each buffer and aggregate C heap scratch. */
OA_API oa_status OA_CALL oa_executor_prepare_layer(const oa_range_config *config,
    const oa_range_layer *dense, oa_tensor input, uint64_t rows,
    float *residual, float *normalized, uint64_t output_count,
    uint64_t *ids, float *weights, uint64_t route_count,
    float *keys, float *values, uint64_t cache_count, uint64_t position,
    uint64_t max_bytes);
/* Exact existing INT8-row byte layout: signed gate[I,H], up[I,H], down[H,I],
 * followed by little-endian FP32 row scales gate[I], up[I], down[H]. Scales
 * need not be aligned; finite and positive. No FP32 weight-bank materialization.
 * Dot accumulation is double, intermediates/output FP32. */
OA_API oa_status OA_CALL oa_executor_swiglu_int8_row(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const void *payload, uint64_t payload_bytes, float *output,
    uint64_t output_count, uint64_t max_bytes);
/* Apply ONE leased expert to selected rows, updating a separate FP32 accumulator
 * transactionally. Routes must contain unique IDs per row in [0,experts),
 * finite weights in [0,1]; shape [rows,topk]. Unselected rows are copied intact.
 * No storage pointers retained. Caller owns checksums and lease lifetimes. */
OA_API oa_status OA_CALL oa_executor_apply_int8_row(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const uint64_t *ids, const float *weights, uint64_t route_count,
    uint64_t topk, uint64_t experts, uint64_t expert,
    const void *payload, uint64_t payload_bytes, oa_tensor accumulator,
    float *output, uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_residual_f32(oa_tensor left, oa_tensor right,
    float *output, uint64_t output_count, uint64_t max_bytes);
/* Additive v1 Q8_0 exports: gate/up/down row-major 34-byte blocks,
 * little-endian binary16 scale followed by 32 signed codes. Both widths
 * must be multiples of 32. Finite signed/zero scales and -128 are valid.
 * Direct block products; no decoded weight banks. Same transactional rules. */
OA_API oa_status OA_CALL oa_executor_swiglu_q8_0(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const void *payload, uint64_t payload_bytes, float *output,
    uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_apply_q8_0(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const uint64_t *ids, const float *weights, uint64_t route_count,
    uint64_t topk, uint64_t experts, uint64_t expert,
    const void *payload, uint64_t payload_bytes, oa_tensor accumulator,
    float *output, uint64_t output_count, uint64_t max_bytes);
/* Additive v1 GGML Q4_0 exports: gate/up/down row-major 18-byte blocks.
 * Each block has a little-endian binary16 d and 16 nibble-pair bytes:
 * low nibble minus 8 for positions 0..15, high minus 8 for 16..31.
 * Both widths are multiples of 32; finite signed/zero d is valid.
 * Direct double-accumulated block products, FP32 activations/output;
 * no decoded weight banks. Same admission and transactional rules. */
OA_API oa_status OA_CALL oa_executor_swiglu_q4_0(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const void *payload, uint64_t payload_bytes, float *output,
    uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_apply_q4_0(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const uint64_t *ids, const float *weights, uint64_t route_count,
    uint64_t topk, uint64_t experts, uint64_t expert,
    const void *payload, uint64_t payload_bytes, oa_tensor accumulator,
    float *output, uint64_t output_count, uint64_t max_bytes);
/* Additive GGML blocks, little-endian half metadata, gate/up/down order:
 * Q4_1: 20 bytes, d[0:2], m[2:4], nibble pairs[4:20], value=d*q+m.
 * Q5_0: 22 bytes, d[0:2], lane high bits[2:6], pairs[6:22], value=d*(q-16).
 * Q5_1: 24 bytes, d[0:2], m[2:4], high bits[4:8], pairs[8:24], value=d*q+m.
 * Q5 q includes bit 4 from the little-endian high-bit word at its lane index.
 * Nibble lows are lanes 0..15, highs 16..31, as for Q4_0.
 * Q8_1: 36 bytes, d[0:2], s[2:4], signed codes[4:36], value=d*q.
 * s is validated finite metadata, not an extra term in an FP32-vector dot.
 * Signed/zero finite metadata is legal. Widths must be multiples of 32.
 * Same bounded scratch/staged output contract; no decoded weight banks. */
OA_API oa_status OA_CALL oa_executor_swiglu_q4_1(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const void *payload, uint64_t payload_bytes, float *output,
    uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_apply_q4_1(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const uint64_t *ids, const float *weights, uint64_t route_count,
    uint64_t topk, uint64_t experts, uint64_t expert,
    const void *payload, uint64_t payload_bytes, oa_tensor accumulator,
    float *output, uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_swiglu_q5_0(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const void *payload, uint64_t payload_bytes, float *output,
    uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_apply_q5_0(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const uint64_t *ids, const float *weights, uint64_t route_count,
    uint64_t topk, uint64_t experts, uint64_t expert,
    const void *payload, uint64_t payload_bytes, oa_tensor accumulator,
    float *output, uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_swiglu_q5_1(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const void *payload, uint64_t payload_bytes, float *output,
    uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_apply_q5_1(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const uint64_t *ids, const float *weights, uint64_t route_count,
    uint64_t topk, uint64_t experts, uint64_t expert,
    const void *payload, uint64_t payload_bytes, oa_tensor accumulator,
    float *output, uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_swiglu_q8_1(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const void *payload, uint64_t payload_bytes, float *output,
    uint64_t output_count, uint64_t max_bytes);
OA_API oa_status OA_CALL oa_executor_apply_q8_1(oa_tensor input,
    uint64_t rows, uint64_t hidden, uint64_t intermediate,
    const uint64_t *ids, const float *weights, uint64_t route_count,
    uint64_t topk, uint64_t experts, uint64_t expert,
    const void *payload, uint64_t payload_bytes, oa_tensor accumulator,
    float *output, uint64_t output_count, uint64_t max_bytes);
#ifdef __cplusplus
}
#endif
#endif
