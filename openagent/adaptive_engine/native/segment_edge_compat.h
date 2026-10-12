#ifndef OA_SEGMENT_EDGE_COMPAT_H
#define OA_SEGMENT_EDGE_COMPAT_H
#include <stddef.h>
#include <stdint.h>
#include "native_exports.h"

/* Public Segment v1 / Edge v2 data layouts. These declarations describe the
 * wire interface; dispatcher and adapter implementations are OpenAgent code.
 * Buffers and callbacks remain trusted in-process pointers, not a sandbox. */
#ifdef __cplusplus
extern "C" {
#endif
typedef int (*oa_se_cancel)(void *);
typedef int (*oa_se_write)(void *, const void *, size_t);
typedef int (*oa_se_read)(void *, void *, size_t);
typedef struct {
    uint32_t size, version;
    uint64_t flags;
    char engine[64], schema[128], numeric[96];
    uint32_t dtype, width, rows, context, layers, reserved32[7];
    uint64_t reserved64[4];
} oa_se_segment_caps;
typedef struct {
    uint32_t size;
    const char *model;
    uint32_t begin, end, context, reserved;
    uint64_t memory, backend;
    const void *plan;
    size_t plan_size;
} oa_se_segment_options;
typedef struct {
    uint32_t size, context;
    uint64_t memory, reserved[4];
} oa_se_session_options;
typedef struct {
    uint32_t size, rows;
    uint64_t position;
    const int32_t *tokens;
    size_t token_count;
    const void *input;
    size_t input_bytes;
    void *output;
    size_t output_bytes;
    oa_se_cancel cancel;
    void *cancel_data;
    uint64_t reserved[4];
} oa_se_run;
typedef struct {
    uint32_t size, version;
    const char *engine;
    int (*open)(void **, oa_se_segment_caps *, const oa_se_segment_options *, char *, size_t);
    void (*destroy)(void *);
    int (*create)(void *, void **, const oa_se_session_options *, char *, size_t);
    void (*destroy_session)(void *);
    int (*run)(void *, const oa_se_run *, char *, size_t);
    int (*snapshot)(void *, oa_se_write, void *, char *, size_t);
    int (*restore)(void *, oa_se_read, void *, char *, size_t);
    void (*reserved[8])(void);
} oa_se_segment_adapter;
typedef struct {
    uint32_t size, version;
    uint64_t flags;
    char engine[64], schema[128], numeric[96], tokenizer[64];
    uint32_t dtype, width, vocab, rows, context, layers;
    int32_t bos, eos;
    uint32_t reserved32[4];
    uint64_t resident_bytes, reserved64[3];
} oa_se_edge_caps;
typedef struct {
    uint32_t size;
    const char *model;
    uint64_t memory, backend;
    const void *plan;
    size_t plan_size;
    uint64_t reserved[3];
} oa_se_edge_options;
typedef struct {
    uint32_t size, rows;
    const int32_t *tokens;
    size_t token_count;
    void *output;
    size_t output_bytes;
    oa_se_cancel cancel;
    void *cancel_data;
    uint64_t reserved[3];
} oa_se_embed;
typedef struct {
    uint32_t size, rows;
    const void *input;
    size_t input_bytes;
    int32_t *tokens;
    size_t token_capacity;
    float *scores;
    size_t score_capacity;
    oa_se_cancel cancel;
    void *cancel_data;
    uint64_t reserved[3];
} oa_se_select;
typedef struct {
    uint32_t size, rows;
    const void *input;
    size_t input_bytes;
    float *logits;
    size_t logits_capacity;
    oa_se_cancel cancel;
    void *cancel_data;
    uint64_t reserved[3];
} oa_se_logits;
typedef struct {
    uint32_t size, version;
    const char *engine;
    int (*open)(void **, oa_se_edge_caps *, const oa_se_edge_options *, char *, size_t);
    void (*destroy)(void *);
    int (*tokenize)(void *, const char *, size_t, int32_t *, size_t, size_t *, char *, size_t);
    int (*detokenize)(void *, const int32_t *, size_t, char *, size_t, size_t *, char *, size_t);
    int (*embed)(void *, const oa_se_embed *, char *, size_t);
    int (*select)(void *, const oa_se_select *, char *, size_t);
    int (*logits)(void *, const oa_se_logits *, char *, size_t);
    void (*reserved[7])(void);
} oa_se_edge_adapter;

/* OpenAgent-specific optional layout probe. Returns record bytes, field count,
 * then field offsets. Zero means an unknown record or insufficient capacity. */
OA_API size_t oa_segment_edge_layout(uint32_t record, uint64_t *values, size_t capacity);

OA_API int coli_segment_adapter_register(const oa_se_segment_adapter *);
OA_API const oa_se_segment_adapter *coli_segment_adapter_lookup(const char *);
OA_API int coli_segment_adapter_count(void);
OA_API int coli_segment_engine_open(const char *, const oa_se_segment_options *, void **, char *, size_t);
OA_API int coli_segment_engine_capabilities(const void *, oa_se_segment_caps *, char *, size_t);
OA_API int coli_segment_engine_close(void *, char *, size_t);
OA_API int coli_segment_session_create(void *, const oa_se_session_options *, void **, char *, size_t);
OA_API void coli_segment_session_destroy(void *);
OA_API int coli_segment_run(void *, const oa_se_run *, char *, size_t);
OA_API int coli_segment_snapshot(void *, oa_se_write, void *, char *, size_t);
OA_API int coli_segment_restore(void *, oa_se_read, void *, char *, size_t);
OA_API int coli_edge_adapter_register(const oa_se_edge_adapter *);
OA_API const oa_se_edge_adapter *coli_edge_adapter_lookup(const char *);
OA_API int coli_edge_adapter_count(void);
OA_API int coli_edge_engine_open(const char *, const oa_se_edge_options *, void **, char *, size_t);
OA_API int coli_edge_engine_capabilities(const void *, oa_se_edge_caps *, char *, size_t);
OA_API void coli_edge_engine_close(void *);
OA_API int coli_edge_tokenize(void *, const char *, size_t, int32_t *, size_t, size_t *, char *, size_t);
OA_API int coli_edge_detokenize(void *, const int32_t *, size_t, char *, size_t, size_t *, char *, size_t);
OA_API int coli_edge_embed(void *, const oa_se_embed *, char *, size_t);
OA_API int coli_edge_select(void *, const oa_se_select *, char *, size_t);
OA_API int coli_edge_logits(void *, const oa_se_logits *, char *, size_t);
#ifdef __cplusplus
}
#endif
#endif
