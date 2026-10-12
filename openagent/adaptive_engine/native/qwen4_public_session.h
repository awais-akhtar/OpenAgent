#ifndef OA_QWEN4_PUBLIC_SESSION_H
#define OA_QWEN4_PUBLIC_SESSION_H
#include "qwen4_public_model.h"
#include "segment_edge_compat.h"
typedef struct oa_q4_public_session oa_q4_public_session;
/* Scoped, thread-local observation of already computed routing tensors. */
typedef int (*oa_q4_routing_observer)(void *,uint32_t,uint64_t,uint64_t,uint64_t,
    const float *,const int64_t *,const float *);
OA_API int oa_qwen4_public_observation_begin(const char *,oa_q4_routing_observer,void *);
OA_API int oa_qwen4_public_observation_end(const char *,oa_q4_routing_observer,void *);
OA_API int oa_qwen4_public_session_create(oa_q4_public_model *,uint32_t,uint32_t,uint64_t,
    oa_q4_public_session **,char *,size_t);
OA_API int oa_qwen4_public_session_close(oa_q4_public_session *);
OA_API int oa_qwen4_public_session_run(oa_q4_public_session *,const oa_se_run *,char *,size_t);
OA_API int oa_qwen4_public_session_snapshot(oa_q4_public_session *,oa_se_write,void *,char *,size_t);
OA_API int oa_qwen4_public_session_restore(oa_q4_public_session *,oa_se_read,void *,char *,size_t);
#endif
