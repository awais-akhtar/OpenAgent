#ifndef OA_QWEN4_PUBLIC_INTERNAL_H
#define OA_QWEN4_PUBLIC_INTERNAL_H
#include "qwen4_public_model.h"
#include <stdatomic.h>
struct oa_q4_public_model {
    atomic_uint_fast64_t refs;
    atomic_flag guard;
    uint64_t resident,used,limit,work;
    uint32_t rows;
    char *root,identity[65];
    oa_q4_compose_config config;
    oa_q4_compose_layer *layers;
    oa_q4_tensor embedding;
    oa_q4_wire_function wire;
    oa_q4_head_function head,hidden;
    oa_original_experts *experts;
    oa_original_rows **ple;
};
#endif
