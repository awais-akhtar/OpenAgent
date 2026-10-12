#ifndef OA_ORIGINAL_EXPERTS_H
#define OA_ORIGINAL_EXPERTS_H
#include "native_exports.h"
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct oa_original_experts oa_original_experts;
typedef struct oa_original_projections oa_original_projections;
/* dtype: 1=F32, 2=F16, 3=BF16, 4=E4M3, 5=E8M0 (scales only). */
typedef struct {
    const char *path;
    uint64_t size;
    int64_t mtime_ns;
} oa_original_file;
typedef struct {
    uint32_t file,dtype,scale_file,scale_dtype;
    uint64_t offset,elements,element_offset,rows,columns,parent_rows;
    uint64_t scale_offset,scale_elements,block_rows,block_columns;
} oa_original_matrix;
typedef struct {
    uint32_t layer,expert;
    oa_original_matrix gate,up,down;
} oa_original_expert;
OA_API uint32_t oa_original_experts_version(void);
OA_API size_t oa_original_experts_layout(uint32_t,uint64_t *,size_t);
OA_API int oa_original_experts_create(const oa_original_file *,uint32_t,
    const oa_original_expert *,uint32_t,uint64_t,uint64_t,float,uint64_t,
    oa_original_experts **,char *,size_t);
OA_API int oa_original_experts_retain(oa_original_experts *);
OA_API int oa_original_experts_release(oa_original_experts *);
OA_API int oa_original_experts_supports(oa_original_experts *,uint32_t,uint32_t,uint64_t,uint64_t);
/* Native GLM int32 and Inkling uint64 callback entrypoints. */
OA_API int oa_original_expert_evaluate(void *,int32_t,int32_t,const float *,float *);
OA_API int oa_original_expert_evaluate64(void *,uint64_t,uint64_t,const float *,float *);
/* Qwen counted selected-expert ABI: exact hidden input and output extents. */
OA_API int oa_original_expert_evaluate_counted64(void *,uint64_t,uint64_t,const float *,uint64_t,float *,uint64_t);
OA_API int oa_original_expert_evaluate_cancel(void *,int32_t,int32_t,const float *,float *,int (*)(void *),void *);
/* Dense original checkpoint rows. The inventory index is stable for the
 * owner's lifetime; evaluation commits its caller output only on success. */
OA_API uint32_t oa_original_projections_version(void);
OA_API int oa_original_projections_create(const oa_original_file *,uint32_t,
    const oa_original_matrix *,uint32_t,uint64_t,oa_original_projections **,char *,size_t);
OA_API int oa_original_projections_create_rows(const oa_original_file *,uint32_t,
    const oa_original_matrix *,uint32_t,uint64_t,oa_original_projections **,char *,size_t);
OA_API int oa_original_projections_retain(oa_original_projections *);
OA_API int oa_original_projections_release(oa_original_projections *);
OA_API int oa_original_projection_shape(oa_original_projections *,uint32_t,uint64_t *,uint64_t *);
OA_API int oa_original_projection_evaluate(oa_original_projections *,uint32_t,
    const float *,uint64_t,float *,uint64_t,int (*)(void *),void *);
OA_API int oa_original_projection_evaluate_policy(oa_original_projections *,uint32_t,
    const float *,uint64_t,float *,uint64_t,int (*)(void *),void *,uint32_t);
OA_API int oa_original_projection_read_row(oa_original_projections *,uint32_t,uint64_t,float *,uint64_t);
typedef struct oa_original_rows oa_original_rows;
OA_API int oa_original_rows_create(oa_original_projections *,const uint32_t *,uint32_t,uint64_t,
    oa_original_rows **,char *,size_t);
OA_API int oa_original_rows_release(oa_original_rows *);
OA_API int oa_original_rows_retain(oa_original_rows *);
OA_API int oa_original_rows_shape(oa_original_rows *,uint64_t *,uint64_t *);
OA_API int oa_original_rows_lookup(void *,uint64_t,uint64_t,float *);
#ifdef __cplusplus
}
#endif
#endif
