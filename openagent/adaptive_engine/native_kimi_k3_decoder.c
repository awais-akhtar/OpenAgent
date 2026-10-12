/* Original bounded dense NoPE-MLA K3 dataflow. No vendor code or KDA fallback.
 * Packed row-major FP32 matrices, expanded head-major-in-token KV storage.
 * Sized disjoint buffers are required. All observable writes are transactional.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
typedef struct {
    int hidden, layers, heads, nope, rope, value, qrank, kvrank;
    int intermediate, block, vocab, capacity;
    double epsilon, beta, linear_beta;
} K3Config;
int oa_kimi_k3_attnres(const float *, const float *, const float *, int, int, double, float *);

static int valid(const K3Config *c) {
    return c && c->hidden >= 1 && c->hidden <= 64 && c->layers >= 1 && c->layers <= 8 &&
        c->heads >= 1 && c->heads <= 4 && c->nope >= 1 && c->nope <= 16 &&
        c->rope >= 0 && c->rope <= 16 && c->value >= 1 && c->value <= 16 &&
        c->qrank >= 1 && c->qrank <= 64 && c->kvrank >= 1 && c->kvrank <= 64 &&
        c->intermediate >= 1 && c->intermediate <= 128 && c->block >= 1 &&
        c->block <= 8 && c->vocab >= 1 && c->vocab <= 256 &&
        c->capacity >= 1 && c->capacity <= 128 && isfinite(c->epsilon) &&
        c->epsilon > 0 && isfinite(c->beta) && c->beta > 0 &&
        isfinite(c->linear_beta) && c->linear_beta > 0;
}
static size_t weight_count(const K3Config *c) {
    size_t h = (size_t)c->hidden, q = (size_t)c->heads * (c->nope + c->rope);
    size_t v = (size_t)c->heads * c->value, f = (size_t)c->intermediate;
    size_t layer = 6*h + c->qrank*h + c->qrank + q*c->qrank +
        (c->kvrank+c->rope)*h + c->kvrank +
        (size_t)c->heads*(c->nope+c->value)*c->kvrank + v*h + h*v + 3*h*f;
    return 2*(size_t)c->vocab*h + (size_t)c->layers*layer + 3*h;
}
static const float *take(const float **p, int n) {
    const float *r = *p; *p += n; return r;
}
static void mat(const float *w, const float *x, int rows, int cols, float *out) {
    int i, j;
    for (i=0; i<rows; ++i) {
        double s=0;
        for (j=0; j<cols; ++j) s += (double)w[i*cols+j]*x[j];
        out[i]=(float)s;
    }
}
static void rms(const float *x, const float *gain, int n, double eps, float *out) {
    int j; double s=0;
    for (j=0; j<n; ++j) s += (double)x[j]*x[j];
    s=1.0/sqrt(s/n+eps);
    for (j=0; j<n; ++j) out[j]=(float)(x[j]*s*gain[j]);
}
static double sigmoid(double x) {
    return x>=0 ? 1.0/(1.0+exp(-x)) : exp(x)/(1.0+exp(x));
}
static int finite_array(const float *x, size_t n) {
    size_t i; for (i=0; i<n; ++i) if (!isfinite(x[i])) return 0; return 1;
}

EXPORT int oa_kimi_k3_decode(const K3Config *c, const float *weights, size_t nw,
    const int *tokens, int count, float *keys, float *values, int *length,
    float *logits, int fail_after_token)
{
    float *k=NULL, *v=NULL, *out=NULL;
    size_t nk, nv; int t, l, j, a, d, s, old, status=0;
    if (!valid(c) || !weights || !tokens || !keys || !values || !length || !logits ||
        nw!=weight_count(c) || count<1 || count>128 || *length<0 ||
        *length>c->capacity || fail_after_token < -1 || fail_after_token>=count) return 1;
    if (count>c->capacity-*length) return 4;
    for (t=0; t<count; ++t) if (tokens[t]<0 || tokens[t]>=c->vocab) return 1;
    nk=(size_t)c->layers*c->capacity*c->heads*(c->nope+c->rope);
    nv=(size_t)c->layers*c->capacity*c->heads*c->value;
    if (!finite_array(weights,nw) || !finite_array(keys,nk) || !finite_array(values,nv)) return 2;
    k=(float *)malloc(nk*sizeof(float)); v=(float *)malloc(nv*sizeof(float));
    out=(float *)malloc((size_t)count*c->vocab*sizeof(float));
    if (!k || !v || !out) { status=5; goto done; }
    memcpy(k,keys,nk*sizeof(float)); memcpy(v,values,nv*sizeof(float)); old=*length;
    for (t=0; t<count; ++t) {
        float sources[9*64], prefix[64], x[64], norm[64], qa[64], query[128];
        float kv[80], expanded[128], attended[64], gate[64], branch[64];
        float fg[128], fu[128], activation[128];
        int blocks=0, h=c->hidden, qdim=c->nope+c->rope, pos=old+t;
        const float *p=weights+(size_t)c->vocab*h;
        memcpy(prefix,weights+(size_t)tokens[t]*h,(size_t)h*sizeof(float));
        for (l=0; l<c->layers; ++l) {
            const float *in_gain=take(&p,h), *post_gain=take(&p,h);
            const float *ag=take(&p,h), *aq=take(&p,h), *mg=take(&p,h), *mq=take(&p,h);
            const float *wa=take(&p,c->qrank*h), *qg=take(&p,c->qrank);
            const float *wb=take(&p,c->heads*qdim*c->qrank);
            const float *wka=take(&p,(c->kvrank+c->rope)*h), *kg=take(&p,c->kvrank);
            const float *wkb=take(&p,c->heads*(c->nope+c->value)*c->kvrank);
            const float *wg=take(&p,c->heads*c->value*h), *wo=take(&p,h*c->heads*c->value);
            const float *wfg=take(&p,c->intermediate*h), *wfu=take(&p,c->intermediate*h);
            const float *wfd=take(&p,h*c->intermediate);
            memcpy(sources+blocks*h,prefix,(size_t)h*sizeof(float));
            if (blocks) {
                status=oa_kimi_k3_attnres(sources,aq,ag,blocks+1,h,c->epsilon,x);
                if (status) goto done;
            } else memcpy(x,prefix,(size_t)h*sizeof(float));
            if (l%c->block==0) { ++blocks; memset(prefix,0,(size_t)h*sizeof(float)); }
            rms(x,in_gain,h,c->epsilon,norm);
            mat(wa,norm,c->qrank,h,qa); rms(qa,qg,c->qrank,1e-6,qa);
            mat(wb,qa,c->heads*qdim,c->qrank,query);
            mat(wka,norm,c->kvrank+c->rope,h,kv);
            rms(kv,kg,c->kvrank,1e-6,qa);
            mat(wkb,qa,c->heads*(c->nope+c->value),c->kvrank,expanded);
            mat(wg,norm,c->heads*c->value,h,gate);
            for (a=0; a<c->heads; ++a) {
                size_t ki=((size_t)l*c->capacity+pos)*c->heads*qdim+(size_t)a*qdim;
                size_t vi=((size_t)l*c->capacity+pos)*c->heads*c->value+(size_t)a*c->value;
                double scores[128], max_score=-HUGE_VAL, denom=0;
                for (d=0; d<c->nope; ++d) k[ki+d]=expanded[a*(c->nope+c->value)+d];
                for (d=0; d<c->rope; ++d) k[ki+c->nope+d]=kv[c->kvrank+d];
                for (d=0; d<c->value; ++d) v[vi+d]=expanded[a*(c->nope+c->value)+c->nope+d];
                for (s=0; s<=pos; ++s) {
                    size_t base=((size_t)l*c->capacity+s)*c->heads*qdim+(size_t)a*qdim;
                    double score=0;
                    for (d=0; d<qdim; ++d) score+=(double)query[a*qdim+d]*k[base+d];
                    scores[s]=score/sqrt((double)qdim);
                    if (!isfinite(scores[s])) { status=2; goto done; }
                    if (scores[s]>max_score) max_score=scores[s];
                }
                for (s=0; s<=pos; ++s) { scores[s]=exp(scores[s]-max_score); denom+=scores[s]; }
                for (d=0; d<c->value; ++d) {
                    double sum=0;
                    for (s=0; s<=pos; ++s) {
                        size_t base=((size_t)l*c->capacity+s)*c->heads*c->value+(size_t)a*c->value;
                        sum+=scores[s]*v[base+d];
                    }
                    attended[a*c->value+d]=(float)(sum/denom*sigmoid(gate[a*c->value+d]));
                }
            }
            mat(wo,attended,h,c->heads*c->value,branch);
            for (j=0; j<h; ++j) prefix[j]+=branch[j];
            memcpy(sources+blocks*h,prefix,(size_t)h*sizeof(float));
            status=oa_kimi_k3_attnres(sources,mq,mg,blocks+1,h,c->epsilon,x);
            if (status) goto done;
            rms(x,post_gain,h,c->epsilon,norm);
            mat(wfg,norm,c->intermediate,h,fg); mat(wfu,norm,c->intermediate,h,fu);
            for (j=0; j<c->intermediate; ++j)
                activation[j]=(float)(c->beta*tanh(fg[j]/c->beta)*sigmoid(fg[j])*
                    c->linear_beta*tanh(fu[j]/c->linear_beta));
            mat(wfd,activation,h,c->intermediate,branch);
            for (j=0; j<h; ++j) prefix[j]+=branch[j];
            if (!finite_array(prefix,(size_t)h)) { status=2; goto done; }
        }
        {
            const float *gain=take(&p,c->hidden), *rq=take(&p,c->hidden), *rg=take(&p,c->hidden);
            memcpy(sources+blocks*c->hidden,prefix,(size_t)c->hidden*sizeof(float));
            status=oa_kimi_k3_attnres(sources,rq,rg,blocks+1,c->hidden,c->epsilon,x);
            if (status) goto done;
            rms(x,gain,c->hidden,c->epsilon,norm);
            mat(p,norm,c->vocab,c->hidden,out+(size_t)t*c->vocab);
        }
        if (t==fail_after_token) { status=3; goto done; }
    }
    if (!finite_array(out,(size_t)count*c->vocab) || !finite_array(k,nk) || !finite_array(v,nv)) {
        status=2; goto done;
    }
    memcpy(keys,k,nk*sizeof(float)); memcpy(values,v,nv*sizeof(float));
    memcpy(logits,out,(size_t)count*c->vocab*sizeof(float)); *length=old+count;
done:
    free(k); free(v); free(out); return status;
}
