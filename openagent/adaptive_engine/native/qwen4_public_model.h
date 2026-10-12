#ifndef OA_QWEN4_PUBLIC_MODEL_H
#define OA_QWEN4_PUBLIC_MODEL_H
#include "native_exports.h"
#include "qwen4_composition_abi.h"
#include "original_experts.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct oa_q4_public_model oa_q4_public_model;
typedef struct {
    uint32_t size,version,rows,reserved;
    const char *root,*identity;
    uint64_t max_resource_bytes,max_work_bytes,external_store_bytes;
    oa_q4_compose_config config;
    const oa_q4_compose_layer *layers;
    oa_q4_tensor embedding;
    oa_q4_wire_function wire;
    oa_q4_head_function head;
    oa_original_experts *experts;
    oa_original_rows *const *ple;
    oa_q4_head_function hidden;
} oa_q4_public_descriptor;
/* Executable functions remain borrowed from a pinned kernel DLL. The caller
 * retains that library until model release and every child release succeed. */
OA_API int oa_qwen4_public_model_create(const oa_q4_public_descriptor *,oa_q4_public_model **,char *,size_t);
OA_API int oa_qwen4_public_model_retain(oa_q4_public_model *);
OA_API int oa_qwen4_public_model_release(oa_q4_public_model *);
/* Owner close refuses while child sessions/handles retain executable pointers. */
OA_API int oa_qwen4_public_model_close(oa_q4_public_model *);
OA_API size_t oa_qwen4_public_model_layout(uint64_t *,size_t);
OA_API int oa_qwen4_public_initialize(void);
OA_API int oa_qwen4_public_register(const oa_q4_public_descriptor *,uint64_t *,char *,size_t);
OA_API int oa_qwen4_public_unregister(uint64_t,char *,size_t);
OA_API int oa_qwen4_public_hidden(uint64_t,uint32_t,const float *,uint64_t,float *,uint64_t,oa_q4_cancel,void *,char *,size_t);
#ifdef __cplusplus
}
#endif
#endif
