/* K3 projection equations (2), (5), (6); shared KDA recurrence is separate.
 * Private FP32 ABI. Direct callers supply correctly sized, disjoint outputs.
 */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct { int hidden, heads, key, value, rank; } K3Projection;

API int oa_k3_kda_projection_version(void) { return 1; }

static int valid(const K3Projection *c) {
    return c && c->hidden>=1 && c->hidden<=64 && c->heads>=1 && c->heads<=8 &&
        c->key>=1 && c->key<=32 && c->value>=1 && c->value<=32 &&
        c->rank>=1 && c->rank<=64;
}
static int finite_values(const float *x, size_t n) {
    size_t i;
    if (!x) return 0;
    for (i=0; i<n; ++i) if (!isfinite(x[i])) return 0;
    return 1;
}
static double sigmoid(double x) {
    if (x>=0) return 1.0/(1.0+exp(-x));
    x=exp(x); return x/(1.0+x);
}
static int project(const float *w, const float *x, int rows, int cols, float *out) {
    int i,j;
    for (i=0; i<rows; ++i) {
        double sum=0;
        for (j=0; j<cols; ++j) sum+=(double)w[i*cols+j]*x[j];
        if (!isfinite(sum) || fabs(sum)>FLT_MAX) return 3;
        out[i]=(float)sum;
    }
    return 0;
}
static double dot(const float *w, const float *x, int cols) {
    int j; double sum=0;
    for (j=0; j<cols; ++j) sum+=(double)w[j]*x[j];
    return sum;
}

API int oa_k3_kda_project(const K3Projection *c, int n,
    const float *weights, size_t nw, const float *x, size_t nx,
    float *q, float *k, float *v, float *decay, float *beta, float *gate,
    const size_t *counts)
{
    float *stage=NULL; size_t kw,vw,required,total; int t,i,jj,rc=0;
    float *outputs[6];
    const float *wq,*wk,*wv,*wd,*wu,*bias,*alog,*wb,*wg;
    float *sq,*sk,*sv,*sd,*sb,*sg;
    if (!valid(c) || n<1 || n>128 || !counts || !q || !k || !v ||
        !decay || !beta || !gate) return 1;
    kw=(size_t)c->heads*c->key; vw=(size_t)c->heads*c->value;
    required=(2*kw+2*vw+(size_t)c->rank+c->heads)*c->hidden+
        kw*c->rank+kw+c->heads;
    if (nw!=required || nx!=(size_t)n*c->hidden ||
        counts[0]!=(size_t)n*kw || counts[1]!=counts[0] || counts[3]!=counts[0] ||
        counts[2]!=(size_t)n*vw || counts[5]!=counts[2] ||
        counts[4]!=(size_t)n*c->heads) return 1;
    outputs[0]=q; outputs[1]=k; outputs[2]=v;
    outputs[3]=decay; outputs[4]=beta; outputs[5]=gate;
    for (i=0; i<6; ++i) for (jj=i+1; jj<6; ++jj) {
        uintptr_t a=(uintptr_t)outputs[i], b=(uintptr_t)outputs[jj];
        if (a<=b ? b-a<counts[i]*sizeof(float) : a-b<counts[jj]*sizeof(float)) return 1;
    }
    if (!finite_values(weights,nw) || !finite_values(x,nx)) return 2;
    wq=weights; wk=wq+kw*c->hidden; wv=wk+kw*c->hidden;
    wd=wv+vw*c->hidden; wu=wd+c->rank*c->hidden;
    bias=wu+kw*c->rank; alog=bias+kw; wb=alog+c->heads;
    wg=wb+c->heads*c->hidden;
    /* Reject unusable scales before allocating or publishing any result. */
    for (i=0; i<c->heads; ++i) if (!isfinite(exp((double)alog[i]))) return 3;
    total=(size_t)n*(3*kw+2*vw+c->heads);
    stage=(float *)malloc(total*sizeof(float));
    if (!stage) return 4;
    sq=stage; sk=sq+n*kw; sv=sk+n*kw; sd=sv+n*vw;
    sb=sd+n*kw; sg=sb+(size_t)n*c->heads;
    for (t=0; t<n; ++t) {
        double latent[64]; int j;
        const float *xt=x+(size_t)t*c->hidden;
        if ((rc=project(wq,xt,(int)kw,c->hidden,sq+t*kw)) ||
            (rc=project(wk,xt,(int)kw,c->hidden,sk+t*kw)) ||
            (rc=project(wv,xt,(int)vw,c->hidden,sv+t*vw))) goto done;
        /* Only published streams require FP32 range; logits stay in double. */
        for (i=0; i<c->rank; ++i)
            latent[i]=dot(wd+(size_t)i*c->hidden,xt,c->hidden);
        for (i=0; i<(int)kw; ++i) {
            double z=bias[i];
            for (j=0; j<c->rank; ++j) z+=(double)wu[(size_t)i*c->rank+j]*latent[j];
            z*=exp((double)alog[i/c->key]);
            if (!isfinite(z)) { rc=3; goto done; }
            sd[t*kw+i]=(float)(-5.0*sigmoid(z));
        }
        for (i=0; i<c->heads; ++i)
            sb[(size_t)t*c->heads+i]=(float)sigmoid(dot(wb+(size_t)i*c->hidden,xt,c->hidden));
        for (i=0; i<(int)vw; ++i)
            sg[t*vw+i]=(float)sigmoid(dot(wg+(size_t)i*c->hidden,xt,c->hidden));
    }
    memcpy(q,sq,counts[0]*sizeof(float)); memcpy(k,sk,counts[1]*sizeof(float));
    memcpy(v,sv,counts[2]*sizeof(float)); memcpy(decay,sd,counts[3]*sizeof(float));
    memcpy(beta,sb,counts[4]*sizeof(float)); memcpy(gate,sg,counts[5]*sizeof(float));
done:
    free(stage); return rc;
}

API int oa_k3_kda_output(const K3Projection *c, int n,
    const float *weights, size_t nw, const float *x, size_t nx,
    float *output, size_t no)
{
    float *stage; size_t width; int t,rc=0;
    if (!valid(c) || n<1 || n>128 || !output) return 1;
    width=(size_t)c->heads*c->value;
    if (nw!=(size_t)c->hidden*width || nx!=(size_t)n*width ||
        no!=(size_t)n*c->hidden) return 1;
    if (!finite_values(weights,nw) || !finite_values(x,nx)) return 2;
    stage=(float *)malloc(no*sizeof(float)); if (!stage) return 4;
    for (t=0; t<n; ++t) {
        rc=project(weights,x+t*width,c->hidden,(int)width,stage+(size_t)t*c->hidden);
        if (rc) break;
    }
    if (!rc) memcpy(output,stage,no*sizeof(float));
    free(stage); return rc;
}
