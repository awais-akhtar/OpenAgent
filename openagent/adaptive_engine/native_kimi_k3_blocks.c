/* Bounded native building blocks for an explicitly configured K3 scheduler. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
typedef struct {
    int hidden, layers, heads, nope, rope, value, qrank, kvrank;
    int intermediate, block, vocab, capacity;
    double epsilon, beta, linear_beta;
} K3Config;

static int finite_values(const float *x,size_t n) {
    size_t i; if (!x) return 0;
    for (i=0;i<n;++i) if (!isfinite(x[i])) return 0;
    return 1;
}
static double sigmoid(double x) {
    if (x>=0) return 1.0/(1.0+exp(-x));
    x=exp(x); return x/(1.0+x);
}
static int disjoint(const float *a,size_t na,const float *b,size_t nb) {
    uintptr_t aa=(uintptr_t)a,bb=(uintptr_t)b;
    return aa<=bb ? bb-aa>=na*sizeof(float) : aa-bb>=nb*sizeof(float);
}
static int mat(const float *w,const float *x,int r,int c,float *out) {
    int i,j;
    for (i=0;i<r;++i) {
        double sum=0;
        for (j=0;j<c;++j) sum+=(double)w[i*c+j]*x[j];
        if (!isfinite(sum) || fabs(sum)>FLT_MAX) return 2;
        out[i]=(float)sum;
    }
    return 0;
}
static int norm(const float *x,const float *g,int n,double eps,float *out) {
    int i; double sum=0;
    for (i=0;i<n;++i) sum+=(double)x[i]*x[i];
    sum=1.0/sqrt(sum/n+eps);
    for (i=0;i<n;++i) {
        double v=x[i]*sum*g[i];
        if (!isfinite(v) || fabs(v)>FLT_MAX) return 2;
        out[i]=(float)v;
    }
    return 0;
}

API int oa_k3_blocks_version(void) { return 1; }
API int oa_k3_blocks_norm(const float *x,const float *gain,int n,double eps,float *out) {
    float stage[256]; int rc;
    if (n<1 || n>256 || !out || !isfinite(eps) || eps<=0) return 1;
    if (!finite_values(x,(size_t)n) || !finite_values(gain,(size_t)n)) return 2;
    rc=norm(x,gain,n,eps,stage);
    if (!rc) memcpy(out,stage,(size_t)n*sizeof(float));
    return rc;
}
API int oa_k3_blocks_add(const float *x,const float *y,int n,float *out) {
    float stage[64]; int i;
    if (n<1 || n>64 || !out) return 1;
    if (!finite_values(x,(size_t)n) || !finite_values(y,(size_t)n)) return 2;
    for (i=0;i<n;++i) {
        double value=(double)x[i]+y[i];
        if (!isfinite(value) || fabs(value)>FLT_MAX) return 2;
        stage[i]=(float)value;
    }
    memcpy(out,stage,(size_t)n*sizeof(float)); return 0;
}
API int oa_k3_blocks_matrix(const float *w,size_t nw,const float *x,int r,int c,float *out) {
    float stage[256]; int rc;
    if (r<1 || r>256 || c<1 || c>256 || nw!=(size_t)r*c || !out) return 1;
    if (!finite_values(w,nw) || !finite_values(x,(size_t)c)) return 2;
    rc=mat(w,x,r,c,stage);
    if (!rc) memcpy(out,stage,(size_t)r*sizeof(float));
    return rc;
}

API int oa_k3_blocks_mla(const K3Config *c,const float *w,size_t nw,
    const float *x,size_t nx,float *keys,size_t nk,float *values,size_t nv,
    int length,float *out,size_t no)
{
    float *sk=NULL,*sv=NULL; int a,d,s,rc=0,qdim,h,vwidth;
    float qa[64],query[128],latent[80],expanded[128],read[64],result[64];
    double gate[64];
    const float *wa,*qg,*wb,*wka,*kg,*wkb,*wg,*wo; size_t expected;
    if (!c || c->hidden<1 || c->hidden>64 || c->heads<1 || c->heads>4 ||
        c->nope<1 || c->nope>16 || c->rope<0 || c->rope>16 ||
        c->value<1 || c->value>16 || c->qrank<1 || c->qrank>64 ||
        c->kvrank<1 || c->kvrank>64 || c->capacity<1 || c->capacity>128 ||
        c->layers<1 || c->layers>8 || c->intermediate<1 || c->intermediate>128 ||
        c->block<1 || c->block>8 || c->vocab<1 || c->vocab>256 ||
        !isfinite(c->epsilon) || c->epsilon<=0 ||
        !isfinite(c->beta) || c->beta<=0 ||
        !isfinite(c->linear_beta) || c->linear_beta<=0 ||
        length<0 || length>=c->capacity || !out) return 1;
    h=c->hidden; qdim=c->nope+c->rope; vwidth=c->heads*c->value;
    expected=(size_t)c->qrank*h+c->qrank+(size_t)c->heads*qdim*c->qrank+
        (size_t)(c->kvrank+c->rope)*h+c->kvrank+
        (size_t)c->heads*(c->nope+c->value)*c->kvrank+2*(size_t)vwidth*h;
    if (nw!=expected || nx!=(size_t)h || no!=(size_t)h ||
        nk!=(size_t)c->capacity*c->heads*qdim || nv!=(size_t)c->capacity*vwidth) return 1;
    if (!finite_values(w,nw) || !finite_values(x,nx) ||
        !finite_values(keys,nk) || !finite_values(values,nv)) return 2;
    if (!disjoint(keys,nk,values,nv) || !disjoint(keys,nk,out,no) ||
        !disjoint(values,nv,out,no) ||
        !disjoint(w,nw,keys,nk) || !disjoint(w,nw,values,nv) ||
        !disjoint(w,nw,out,no) || !disjoint(x,nx,keys,nk) ||
        !disjoint(x,nx,values,nv) || !disjoint(x,nx,out,no)) return 1;
    sk=(float *)malloc(nk*sizeof(float)); sv=(float *)malloc(nv*sizeof(float));
    if (!sk || !sv) { rc=3; goto done; }
    memcpy(sk,keys,nk*sizeof(float)); memcpy(sv,values,nv*sizeof(float));
    wa=w; qg=wa+c->qrank*h; wb=qg+c->qrank;
    wka=wb+c->heads*qdim*c->qrank; kg=wka+(c->kvrank+c->rope)*h;
    wkb=kg+c->kvrank; wg=wkb+c->heads*(c->nope+c->value)*c->kvrank;
    wo=wg+vwidth*h;
    if ((rc=mat(wa,x,c->qrank,h,qa)) || (rc=norm(qa,qg,c->qrank,1e-6,qa)) ||
        (rc=mat(wb,qa,c->heads*qdim,c->qrank,query)) ||
        (rc=mat(wka,x,c->kvrank+c->rope,h,latent)) ||
        (rc=norm(latent,kg,c->kvrank,1e-6,qa)) ||
        (rc=mat(wkb,qa,c->heads*(c->nope+c->value),c->kvrank,expanded))) goto done;
    /* Probability logits need not fit the published FP32 stream range. */
    for (a=0;a<vwidth;++a) {
        gate[a]=0;
        for (d=0;d<h;++d) gate[a]+=(double)wg[a*h+d]*x[d];
    }
    for (a=0;a<c->heads;++a) {
        size_t ki=((size_t)length*c->heads+a)*qdim;
        size_t vi=(size_t)length*vwidth+(size_t)a*c->value;
        double scores[128],peak=-HUGE_VAL,total=0;
        for (d=0;d<c->nope;++d) sk[ki+d]=expanded[a*(c->nope+c->value)+d];
        for (d=0;d<c->rope;++d) sk[ki+c->nope+d]=latent[c->kvrank+d];
        for (d=0;d<c->value;++d) sv[vi+d]=expanded[a*(c->nope+c->value)+c->nope+d];
        for (s=0;s<=length;++s) {
            double sum=0; size_t base=((size_t)s*c->heads+a)*qdim;
            for (d=0;d<qdim;++d) sum+=(double)query[a*qdim+d]*sk[base+d];
            scores[s]=sum/sqrt((double)qdim);
            if (!isfinite(scores[s])) { rc=2; goto done; }
            if (scores[s]>peak) peak=scores[s];
        }
        for (s=0;s<=length;++s) { scores[s]=exp(scores[s]-peak); total+=scores[s]; }
        for (d=0;d<c->value;++d) {
            double sum=0,value;
            for (s=0;s<=length;++s) sum+=scores[s]*sv[(size_t)s*vwidth+(size_t)a*c->value+d];
            value=sum/total*sigmoid(gate[a*c->value+d]);
            if (!isfinite(value) || fabs(value)>FLT_MAX) { rc=2; goto done; }
            read[a*c->value+d]=(float)value;
        }
    }
    rc=mat(wo,read,h,vwidth,result);
    if (!rc) {
        memcpy(keys,sk,nk*sizeof(float)); memcpy(values,sv,nv*sizeof(float));
        memcpy(out,result,(size_t)h*sizeof(float));
    }
done:
    free(sk); free(sv); return rc;
}
