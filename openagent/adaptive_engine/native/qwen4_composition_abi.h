#ifndef OA_QWEN4_COMPOSITION_ABI_H
#define OA_QWEN4_COMPOSITION_ABI_H
#include <stdint.h>
#include <stddef.h>
/* Private kernel inclusion may use its existing counted Tensor/Config aliases.
 * Both aliases have these exact member types and order. Runtime admission also
 * probes every composition record before accepting executable function pointers.
 */
#ifndef OA_Q4_TENSOR
typedef struct {float *data;uint64_t count;} oa_q4_tensor;
#define OA_Q4_TENSOR oa_q4_tensor
#define OA_Q4_LOCAL_TENSOR 1
#endif
#ifndef OA_Q4_GEOMETRY
typedef struct {
    uint64_t hidden,heads,kv,dim,key_heads,value_heads,key_dim,value_dim;
    uint64_t kernel,experts,topk,intermediate,shared,capacity,rotary;
    double epsilon,theta;
} oa_q4_geometry;
#define OA_Q4_GEOMETRY oa_q4_geometry
#define OA_Q4_LOCAL_GEOMETRY 1
#endif
typedef struct {
    uint64_t kind,ple_enabled;
    OA_Q4_TENSOR attention_hyper[4],mlp_hyper[4],attention[9],mlp[7];
    OA_Q4_TENSOR cache[3],ple_weights[7],ple_history;
    const int64_t *multipliers,*sizes,*offsets;
    int64_t *previous;
    uint64_t multiplier_count,lane_count,previous_count,ple_table_rows;
    void *ple_user;
} oa_q4_compose_layer;
typedef struct {
    uint32_t size,version;
    OA_Q4_GEOMETRY text;
    uint64_t layers,streams,rank,index_heads,index_dim,index_budget,index_ratio;
    uint64_t ple_embedding,ple_kernel,ple_ngram,ple_lanes,eos;
    uint64_t vocabulary,max_work_bytes;
    double rope_scale;
    int32_t mrope_sections[3];
    uint32_t mrope_enabled,sigmoid_gate,normalize_topk,finalize;
    OA_Q4_TENSOR final_hyper[3],head;
} oa_q4_compose_config;
typedef struct {
    OA_Q4_TENSOR router,scores,attention,incoming_hidden;
    int64_t *selected;uint8_t *chosen;
    uint64_t selected_count,chosen_count;
} oa_q4_compose_trace;
typedef int (*oa_q4_cancel)(void *);
typedef int (*oa_q4_expert)(void *,uint64_t,uint64_t,const float *,uint64_t,float *,uint64_t);
typedef int (*oa_q4_lookup)(void *,uint64_t,uint64_t,float *);
typedef int (*oa_q4_wire_function)(const oa_q4_compose_config *,oa_q4_compose_layer *,oa_q4_compose_trace *,
    uint64_t,uint64_t,uint64_t,uint64_t,OA_Q4_TENSOR,OA_Q4_TENSOR,uint8_t *,uint64_t,int64_t *,uint64_t,
    oa_q4_expert,void *,oa_q4_lookup,void *,oa_q4_cancel,void *);
typedef int (*oa_q4_head_function)(const oa_q4_compose_config *,uint64_t,OA_Q4_TENSOR,OA_Q4_TENSOR,oa_q4_cancel,void *);
#ifdef OA_Q4_LOCAL_TENSOR
#undef OA_Q4_TENSOR
#undef OA_Q4_LOCAL_TENSOR
#endif
#ifdef OA_Q4_LOCAL_GEOMETRY
#undef OA_Q4_GEOMETRY
#undef OA_Q4_LOCAL_GEOMETRY
#endif
#endif
