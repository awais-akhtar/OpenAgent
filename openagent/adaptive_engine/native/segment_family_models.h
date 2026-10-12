#ifndef OA_SEGMENT_FAMILY_MODELS_H
#define OA_SEGMENT_FAMILY_MODELS_H
#include "segment_edge_compat.h"
#include "original_experts.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Native public adapters use immutable admitted weights. No Python execution
 * callback is installed. Function pointers must refer to the corresponding
 * native decoder library and remain loaded until model release succeeds. */
typedef struct {float *data;uint64_t count;} oa_family_tensor;
typedef struct {
    uint64_t hidden,vocab,output_vocab,layers,width,kernel,relative,capacity,floor;
    uint64_t moe_width,experts,shared,topk;
    float epsilon,multiplier,alpha,route_scale;
} oa_family_inkling_config;
typedef struct {
    uint64_t heads,kv_heads,dim,extent,window,sparse;
    oa_family_tensor weights[26],state;
} oa_family_inkling_layer;
typedef int (*oa_family_glm_step)(const int32_t *,const float *,const int32_t *,
    const oa_family_tensor *,oa_family_tensor,oa_family_tensor,oa_family_tensor,
    oa_family_tensor *,int32_t,int64_t,oa_family_tensor,oa_family_tensor,oa_family_tensor,
    int (*)(void *,int32_t,int32_t,const float *,float *),void *);
typedef int (*oa_family_inkling_run)(const oa_family_inkling_config *,oa_family_inkling_layer *,
    const oa_family_tensor *,const int64_t *,uint64_t,uint64_t,uint64_t,uint64_t,
    oa_family_tensor,oa_family_tensor,oa_family_tensor,oa_family_tensor);
typedef int (*oa_family_glm_range)(const int32_t *,const float *,const int32_t *,
    const oa_family_tensor *,oa_family_tensor,oa_family_tensor,oa_family_tensor *,
    int32_t,int64_t,int32_t,int32_t,oa_family_tensor,oa_family_tensor);
typedef int (*oa_family_inkling_range)(const oa_family_inkling_config *,oa_family_inkling_layer *,
    const oa_family_tensor *,const int64_t *,uint64_t,uint64_t,uint64_t,uint64_t,
    uint64_t,uint64_t,oa_family_tensor,oa_family_tensor,oa_family_tensor);
typedef int (*oa_family_inkling_expert)(void *,uint64_t,uint64_t,const float *,float *);
typedef int (*oa_family_glm_range_streamed)(const int32_t *,const float *,const int32_t *,
    const oa_family_tensor *,oa_family_tensor,oa_family_tensor,oa_family_tensor *,
    int32_t,int64_t,int32_t,int32_t,oa_family_tensor,oa_family_tensor,
    int (*)(void *,int32_t,int32_t,const float *,float *),void *);
typedef int (*oa_family_inkling_streamed)(const oa_family_inkling_config *,oa_family_inkling_layer *,
    const oa_family_tensor *,const int64_t *,uint64_t,uint64_t,uint64_t,uint64_t,
    oa_family_tensor,oa_family_tensor,oa_family_tensor,oa_family_tensor,oa_family_inkling_expert,void *);
typedef int (*oa_family_inkling_range_streamed)(const oa_family_inkling_config *,oa_family_inkling_layer *,
    const oa_family_tensor *,const int64_t *,uint64_t,uint64_t,uint64_t,uint64_t,
    uint64_t,uint64_t,oa_family_tensor,oa_family_tensor,oa_family_tensor,oa_family_inkling_expert,void *);
typedef struct {
    uint32_t size,version,family,rows;
    const char *model,*identity;
    uint64_t max_resource_bytes,max_work_bytes;
    int32_t glm_config[26];float glm_floats[6];
    const int32_t *glm_schedule;
    const oa_family_tensor *glm_weights;
    oa_family_glm_step glm_step;
    oa_family_inkling_config inkling_config;
    const oa_family_inkling_layer *inkling_layers;
    oa_family_inkling_run inkling_run;
    oa_family_tensor globals[4];
} oa_family_model_descriptor;
/* family=1: GLM5-next private ABI3 with embeddings ABI1; family=2: Inkling
 * private ABI2 with embeddings ABI1. Descriptor ABI1 admits complete models;
 * additive range registration supplies native partition execution.
 * Incoming rows are normalized Inkling embeddings or raw GLM embeddings;
 * outgoing rows already include final normalization. Edge projects logits. */
OA_API int oa_segment_family_register(const oa_family_model_descriptor *,uint64_t *,char *,size_t);
OA_API int oa_segment_family_release(uint64_t,char *,size_t);
/* Additive extension: install before opening handles, preserving descriptor ABI1. */
OA_API int oa_segment_family_enable_inkling_ranges(uint64_t,oa_family_inkling_range,char *,size_t);
OA_API int oa_segment_family_enable_glm_ranges(uint64_t,oa_family_glm_range,char *,size_t);
OA_API int oa_segment_family_glm_final_hidden(uint64_t,const float *,uint32_t,uint64_t,
    float *,uint64_t,char *,size_t);
OA_API int oa_segment_family_attach_original_experts(uint64_t,oa_original_experts *,
    oa_family_glm_range_streamed,oa_family_inkling_streamed,oa_family_inkling_range_streamed,char *,size_t);
OA_API uint32_t oa_segment_family_version(void);
OA_API int oa_segment_family_initialize(void);
OA_API size_t oa_segment_family_layout(uint32_t,uint64_t *,size_t);
#ifdef __cplusplus
}
#endif
#endif
