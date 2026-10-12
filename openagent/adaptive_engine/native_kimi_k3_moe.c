/* Original scalar inference from K3 report equations 11-13. No vendor runtime. */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
typedef struct {
    int hidden, latent, intermediate, experts, selected, shared;
    float epsilon, beta, linear_beta;
} Config;
static int finite_array(const float *a, size_t n) {
    size_t i;
    for (i=0; i<n; ++i) if (!isfinite(a[i])) return 0;
    return 1;
}
static float sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f/(1.0f+z) : z/(1.0f+z);
}
static void matvec(const float *w, const float *x, float *y, int rows, int cols) {
    int i, j;
    for (i=0; i<rows; ++i) {
        double sum=0.0;
        for (j=0; j<cols; ++j) sum += (double)w[i*cols+j]*x[j];
        y[i]=(!isfinite(sum) || fabs(sum)>FLT_MAX) ? (float)NAN : (float)sum;
    }
}
static int expert(const Config *c, const float *g, const float *u, const float *d,
                  const float *x, int width, float *out) {
    float gate[128], up[128];
    int i;
    matvec(g,x,gate,c->intermediate,width);
    matvec(u,x,up,c->intermediate,width);
    if (!finite_array(gate,(size_t)c->intermediate) ||
        !finite_array(up,(size_t)c->intermediate)) return 0;
    for (i=0; i<c->intermediate; ++i)
        gate[i]=c->beta*tanhf(gate[i]/c->beta)*sigmoid(gate[i])*
                c->linear_beta*tanhf(up[i]/c->linear_beta);
    matvec(d,gate,out,width,c->intermediate);
    return finite_array(out,(size_t)width);
}
EXPORT int oa_kimi_k3_moe(const Config *c, const float *w, size_t nw,
    const float *x, size_t nx, int tokens, float *out, size_t no,
    int *routes, size_t nr, size_t memory_budget, size_t mac_budget, int fault) {
    int h,l,m,e,k,s,t,i,j,a,b,order[64];
    size_t expected, memory, work, rh, eh;
    const float *router,*bias,*ld,*norm,*lu,*rg,*ru,*rd,*sg,*su,*sd;
    float *result;
    int *chosen;
    float z[64], scores[64], biased[64], aggregate[64], temp[64], mapped[64];
    int status=3;
    if (!c || !w || !x || !out || !routes) return 1;
    h=c->hidden; l=c->latent; m=c->intermediate; e=c->experts;
    k=c->selected; s=c->shared;
    if (h<1 || h>64 || l<1 || l>64 || m<1 || m>128 || e<2 || e>64 ||
        k<1 || k>=e || s!=2 || tokens<1 || tokens>128 || fault < -1 || fault>=tokens ||
        !isfinite(c->epsilon) || c->epsilon<1e-12f || c->epsilon>1e6f ||
        !isfinite(c->beta) || c->beta<1e-12f || c->beta>1e6f ||
        !isfinite(c->linear_beta) || c->linear_beta<1e-12f || c->linear_beta>1e6f) return 1;
    rh=(size_t)m*l; eh=(size_t)m*h;
    expected=(size_t)e*h+e+2*(size_t)h*l+l+3*(size_t)e*rh+3*(size_t)s*eh;
    memory=12*expected+4096+(size_t)tokens*(12*(size_t)h+8*(size_t)k);
    work=(size_t)tokens*((size_t)e*h+2*(size_t)h*l+3*(size_t)k*l*m+3*(size_t)s*h*m);
    if (nw!=expected || nx!=(size_t)tokens*h || no!=nx || nr!=(size_t)tokens*k) return 1;
    /* Publishing overlapping output streams cannot preserve either contract. */
    if ((uintptr_t)out <= (uintptr_t)routes ?
        (uintptr_t)routes-(uintptr_t)out < no*sizeof(float) :
        (uintptr_t)out-(uintptr_t)routes < nr*sizeof(int)) return 1;
    if (memory>memory_budget || work>mac_budget) return 2;
    if (!finite_array(w,nw) || !finite_array(x,nx)) return 3;
    router=w; bias=router+(size_t)e*h; ld=bias+e; norm=ld+(size_t)l*h;
    lu=norm+l; rg=lu+(size_t)h*l; ru=rg+(size_t)e*rh;
    rd=ru+(size_t)e*rh; sg=rd+(size_t)e*rh;
    su=sg+(size_t)s*eh; sd=su+(size_t)s*eh;
    result=(float *)malloc(nx*sizeof(float)); chosen=(int *)malloc(nr*sizeof(int));
    if (!result || !chosen) { free(result); free(chosen); return 5; }
    for (t=0; t<tokens; ++t) {
        const float *input=x+(size_t)t*h;
        float denom=0.0f;
        double square=0.0, inverse;
        matvec(router,input,scores,e,h);
        if (!finite_array(scores,(size_t)e)) goto cleanup;
        for (i=0; i<e; ++i) {
            scores[i]=sigmoid(scores[i]); biased[i]=scores[i]+bias[i]; order[i]=i;
        }
        if (!finite_array(biased,(size_t)e)) goto cleanup;
        /* Descending score, ascending index inside the selected set only. */
        for (i=1; i<e; ++i) {
            a=order[i]; j=i;
            while (j>0 && biased[order[j-1]]<biased[a]) { order[j]=order[j-1]; --j; }
            order[j]=a;
        }
        if (biased[order[k-1]]==biased[order[k]]) { status=4; goto cleanup; }
        for (i=0; i<k; ++i) denom+=scores[order[i]];
        if (!(denom>0.0f) || !isfinite(denom)) goto cleanup;
        matvec(ld,input,z,l,h);
        if (!finite_array(z,(size_t)l)) goto cleanup;
        memset(aggregate,0,(size_t)l*sizeof(float));
        for (i=0; i<k; ++i) {
            a=order[i]; chosen[t*k+i]=a;
            if (!expert(c,rg+(size_t)a*rh,ru+(size_t)a*rh,rd+(size_t)a*rh,z,l,temp)) goto cleanup;
            for (j=0; j<l; ++j) aggregate[j]+=scores[a]/denom*temp[j];
        }
        if (!finite_array(aggregate,(size_t)l)) goto cleanup;
        for (i=0; i<l; ++i) square+=(double)aggregate[i]*aggregate[i];
        if (!isfinite(square)) goto cleanup;
        inverse=1.0/sqrt(square/(double)l+c->epsilon);
        for (i=0; i<l; ++i) {
            double normalized=aggregate[i]*inverse*norm[i];
            if (!isfinite(normalized) || fabs(normalized)>FLT_MAX) goto cleanup;
            aggregate[i]=(float)normalized;
        }
        matvec(lu,aggregate,mapped,h,l);
        for (b=0; b<s; ++b) {
            if (!expert(c,sg+(size_t)b*eh,su+(size_t)b*eh,sd+(size_t)b*eh,input,h,temp)) goto cleanup;
            for (i=0; i<h; ++i) mapped[i]+=temp[i];
        }
        if (!finite_array(mapped,(size_t)h) || t==fault) goto cleanup;
        memcpy(result+(size_t)t*h,mapped,(size_t)h*sizeof(float));
    }
    memcpy(out,result,nx*sizeof(float)); memcpy(routes,chosen,nr*sizeof(int)); status=0;
cleanup:
    free(result); free(chosen); return status;
}
