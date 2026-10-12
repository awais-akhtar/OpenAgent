/* Original page-wise online softmax, shared by long-context MLA and MoonViT.
 * Private ABI: Python supplies disjoint correctly sized contiguous buffers.
 * Running numerator/denominator stay FP64; call on private transaction scratch.
 */
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

/* Compressed MLA keys are [tokens,rank+rope], shared by all heads. Query
 * NoPE has already been multiplied by the head's key expansion transpose.
 * Numerators accumulate latent vectors, not expanded per-head values.
 */
API int oa_k3_latent_segment(uint64_t tokens, uint64_t heads, uint64_t rank,
    uint64_t rope, double scaling, const float *q_latent, const float *q_rope,
    const float *cache, double *peak, double *denominator, double *numerator)
{
    uint64_t h,t,d;
    if (!tokens || tokens>4096 || !heads || heads>128 || !rank || rank>65536 ||
        rope>512 || !isfinite(scaling) || scaling<=0 || !q_latent ||
        (rope && !q_rope) || !cache || !peak || !denominator || !numerator) return 1;
    for (h=0; h<heads; ++h) for (t=0; t<tokens; ++t) {
        const float *latent=cache+t*(rank+rope);
        double score=0,next_peak,old_scale,new_scale;
        for (d=0; d<rank; ++d) score+=(double)q_latent[h*rank+d]*latent[d];
        for (d=0; d<rope; ++d) score+=(double)q_rope[h*rope+d]*latent[rank+d];
        score*=scaling;
        if (!isfinite(score)) return 2;
        next_peak=fmax(peak[h],score);
        old_scale=denominator[h]==0 ? 0 : exp(peak[h]-next_peak);
        new_scale=exp(score-next_peak);
        for (d=0; d<rank; ++d) {
            double next=numerator[h*rank+d]*old_scale+new_scale*latent[d];
            if (!isfinite(next)) return 2;
            numerator[h*rank+d]=next;
        }
        denominator[h]=denominator[h]*old_scale+new_scale;
        peak[h]=next_peak;
        if (!isfinite(denominator[h])) return 2;
    }
    return 0;
}

API int oa_k3_attention_segment(uint64_t tokens, uint64_t heads,
    uint64_t key_dim, uint64_t value_dim, const float *q, const float *k,
    const float *v, double *peak, double *denominator, double *numerator)
{
    uint64_t h,t,d;
    if (!tokens || tokens>4096 || !heads || heads>128 || !key_dim ||
        key_dim>512 || !value_dim || value_dim>256 || !q || !k || !v ||
        !peak || !denominator || !numerator) return 1;
    for (h=0; h<heads; ++h) {
        for (t=0; t<tokens; ++t) {
            double score=0, old_scale, new_scale, next_peak;
            for (d=0; d<key_dim; ++d)
                score+=(double)q[h*key_dim+d]*k[(t*heads+h)*key_dim+d];
            score/=sqrt((double)key_dim);
            if (!isfinite(score)) return 2;
            next_peak=fmax(peak[h],score);
            old_scale=denominator[h]==0 ? 0 : exp(peak[h]-next_peak);
            new_scale=exp(score-next_peak);
            for (d=0; d<value_dim; ++d) {
                double next=numerator[h*value_dim+d]*old_scale+
                    new_scale*v[(t*heads+h)*value_dim+d];
                if (!isfinite(next)) return 2;
                numerator[h*value_dim+d]=next;
            }
            denominator[h]=denominator[h]*old_scale+new_scale;
            peak[h]=next_peak;
            if (!isfinite(denominator[h])) return 2;
        }
    }
    return 0;
}

/* Tower tanh-GELU and projector erf-GELU. Private scratch budget in bytes. */
API int oa_k3_vision_gelu(const float *x, uint64_t n, int approximate,
    float *out, uint64_t budget)
{
    uint64_t i;
    float *stage;
    if (!x || !out || !n || n>SIZE_MAX/sizeof(float) ||
        n*sizeof(float)>budget || (approximate!=0 && approximate!=1)) return 1;
    stage=(float *)malloc((size_t)n*sizeof(float));
    if (!stage) return 3;
    for (i=0; i<n; ++i) {
        double z=x[i], y;
        if (!isfinite(z)) { free(stage); return 2; }
        y=approximate ? 0.5*z*(1+tanh(0.7978845608028654*(z+0.044715*z*z*z))) :
            0.5*z*(1+erf(z/1.4142135623730951));
        if (!isfinite(y) || fabs(y)>FLT_MAX) { free(stage); return 2; }
        stage[i]=(float)y;
    }
    memcpy(out,stage,(size_t)n*sizeof(float));
    free(stage);
    return 0;
}

/* Complex-pair vision RoPE; phases are token-major x/y-interleaved angles.
 * This is not the split-half text rotary convention.
 */
API int oa_k3_vision_rope(const float *x, const float *phase,
    uint64_t tokens, uint64_t heads, uint64_t dim, float *out, uint64_t budget)
{
    uint64_t t,h,d,n;
    float *stage;
    if (!x || !phase || !out || !tokens || !heads || heads>128 ||
        !dim || dim>256 || dim%4 || tokens>SIZE_MAX/(heads*dim*sizeof(float))) return 1;
    n=tokens*heads*dim;
    if (n*sizeof(float)>budget) return 1;
    stage=(float *)malloc((size_t)n*sizeof(float));
    if (!stage) return 3;
    for (t=0; t<tokens; ++t) for (h=0; h<heads; ++h) for (d=0; d<dim; d+=2) {
        uint64_t index=(t*heads+h)*dim+d;
        double angle=phase[t*(dim/2)+d/2], real=x[index], imag=x[index+1];
        double a=real*cos(angle)-imag*sin(angle), b=real*sin(angle)+imag*cos(angle);
        if (!isfinite(a) || !isfinite(b) || fabs(a)>FLT_MAX || fabs(b)>FLT_MAX) {
            free(stage); return 2;
        }
        stage[index]=(float)a; stage[index+1]=(float)b;
    }
    memcpy(out,stage,(size_t)n*sizeof(float));
    free(stage);
    return 0;
}

/* Publisher noaux_tc group selection: top-two corrected scores per group,
 * group mask, expert top-k, then ORIGINAL sigmoid-score renormalization.
 * Ties fail closed (status 4), rather than inventing Torch topk tie order.
 */
API int oa_k3_grouped_route(uint64_t n, uint64_t topk, uint64_t groups,
    uint64_t group_topk, double scaling, const float *scores, const float *bias,
    uint64_t *ids, float *weights, uint64_t budget)
{
    uint64_t i,j,g,width,selected;
    float *choice=NULL,*group_scores=NULL;
    unsigned char *allowed=NULL,*used=NULL;
    double denominator=0;
    int rc=0;
    if (!n || !topk || topk>n || !groups || n%groups || !group_topk ||
        group_topk>groups || !scores || !bias || !ids || !weights ||
        !isfinite(scaling) || scaling<=0 || n>SIZE_MAX/6 || groups>SIZE_MAX/5 ||
        n*6>SIZE_MAX-groups*5 || n*6>budget || groups*5>budget-n*6) return 1;
    width=n/groups;
    if (groups>group_topk && (width<2 || topk>width*group_topk)) return 1;
    choice=(float *)malloc((size_t)n*sizeof(float));
    group_scores=(float *)malloc((size_t)groups*sizeof(float));
    allowed=(unsigned char *)calloc((size_t)groups,1);
    used=(unsigned char *)calloc((size_t)n,1);
    if (!choice || !group_scores || !allowed || !used) { rc=3; goto done; }
    for (i=0; i<n; ++i) {
        if (!isfinite(scores[i]) || scores[i]<0 || scores[i]>1 || !isfinite(bias[i])) { rc=2; goto done; }
        choice[i]=scores[i]+bias[i];
        if (!isfinite(choice[i])) { rc=2; goto done; }
    }
    if (groups>group_topk) {
        for (g=0; g<groups; ++g) {
            float first=-INFINITY,second=-INFINITY;
            for (i=g*width; i<(g+1)*width; ++i) {
                if (choice[i]>first) { second=first; first=choice[i]; }
                else if (choice[i]>second) second=choice[i];
            }
            group_scores[g]=first+second;
            if (!isfinite(group_scores[g])) { rc=2; goto done; }
            for (j=0; j<g; ++j) if (group_scores[j]==group_scores[g]) { rc=4; goto done; }
        }
        for (selected=0; selected<group_topk; ++selected) {
            uint64_t best=groups;
            float peak=-INFINITY;
            for (g=0; g<groups; ++g) if (!allowed[g] && group_scores[g]>peak) { best=g; peak=group_scores[g]; }
            allowed[best]=1;
        }
    } else memset(allowed,1,(size_t)groups);
    for (i=0; i<n; ++i) if (allowed[i/width])
        for (j=0; j<i; ++j) if (allowed[j/width] && choice[i]==choice[j]) { rc=4; goto done; }
    for (selected=0; selected<topk; ++selected) {
        uint64_t best=n;
        float peak=-INFINITY;
        for (i=0; i<n; ++i) if (allowed[i/width] && !used[i] && choice[i]>peak) { best=i; peak=choice[i]; }
        if (best==n) { rc=2; goto done; }
        ids[selected]=best; used[best]=1;
        denominator+=scores[best];
    }
    denominator=topk>1 ? denominator+1e-20 : 1;
    for (i=0; i<topk; ++i) {
        double value=scores[ids[i]]/denominator*scaling;
        if (!isfinite(value) || value>FLT_MAX) { rc=2; goto done; }
        weights[i]=(float)value;
    }
done:
    free(choice); free(group_scores); free(allowed); free(used);
    return rc;
}
