/* Resident Qwen4 grouped normalization and low-rank gated stream mixer. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define OA_API __declspec(dllexport)
#else
#define OA_API __attribute__((visibility("default")))
#endif

static double logistic(double x) {
    if (x >= 0) return 1.0/(1.0+exp(-x));
    { double e = exp(x); return e/(1.0+e); }
}
static int finite_array(const float *x, uint64_t n) {
    uint64_t i;
    if (!x) return 0;
    for (i=0;i<n;i++) if (!isfinite(x[i])) return 0;
    return 1;
}
OA_API int oa_qwen4_hyper(uint64_t rows, uint64_t heads, uint64_t hidden,
        uint64_t rank, double epsilon, const float *x, const float *norm,
        const float *down, const float *up, const float *inject,
        float *mixed, float *injection) {
    uint64_t width, n, scratch, row, g, j, k, q;
    double *mem, *z, *low;
    float *pending, *pending_gate;
    if (!rows || heads<2 || !hidden || !rank || !isfinite(epsilon) || epsilon<=0 ||
        heads>SIZE_MAX/hidden || !mixed || (inject && !injection)) return 1;
    width=heads*hidden;
    if (rows>SIZE_MAX/width || rank>SIZE_MAX/width ||
        heads>SIZE_MAX/width || rows>SIZE_MAX/hidden || rows>SIZE_MAX/heads) return 1;
    n=rows*hidden;
    if (width>SIZE_MAX-rank || n>SIZE_MAX-rows*heads) return 1;
    scratch=width+rank;
    if (scratch>SIZE_MAX/sizeof(double) || n+rows*heads>SIZE_MAX/sizeof(float)) return 1;
    if (!finite_array(x, rows*width) || !finite_array(norm,width) ||
        !finite_array(down,rank*width) || !finite_array(up,width*rank) ||
        (inject && !finite_array(inject,heads*width))) return 2;
    mem=(double *)malloc((size_t)scratch*sizeof(double));
    pending=(float *)malloc((size_t)(n+rows*heads)*sizeof(float));
    if (!mem || !pending) {free(mem);free(pending);return 3;}
    z=mem;low=mem+width;pending_gate=pending+n;
    for (row=0;row<rows;row++) {
        for (g=0;g<heads;g++) {
            double energy=0, scale;
            for (j=0;j<hidden;j++) {double v=x[row*width+g*hidden+j];energy+=v*v;}
            scale=1.0/sqrt(energy/(double)hidden+epsilon);
            for (j=0;j<hidden;j++) {
                q=g*hidden+j;
                z[q]=(double)x[row*width+q]*scale*(1.0+(double)norm[q]);
            }
        }
        for (k=0;k<rank;k++) {
            double value=0;
            for (q=0;q<width;q++) value+=z[q]*(double)down[k*width+q];
            value/=(double)heads;low[k]=value*logistic(value);
        }
        for (j=0;j<hidden;j++) {
            double sum=0;
            for (g=0;g<heads;g++) {
                double value=0;
                q=g*hidden+j;
                for (k=0;k<rank;k++) value+=low[k]*(double)up[q*rank+k];
                sum+=z[q]*logistic(value);
            }
            pending[row*hidden+j]=(float)(sum/(double)heads);
            if (!isfinite(pending[row*hidden+j])) {free(mem);free(pending);return 2;}
        }
        if (inject) for (g=0;g<heads;g++) {
            double value=0;
            for (q=0;q<width;q++) value+=z[q]*(double)inject[g*width+q];
            pending_gate[row*heads+g]=(float)(2.0*logistic(value/(double)heads));
            if (!isfinite(pending_gate[row*heads+g])) {free(mem);free(pending);return 2;}
        }
    }
    memcpy(mixed,pending,(size_t)n*sizeof(float));
    if (inject) memcpy(injection,pending_gate,(size_t)(rows*heads)*sizeof(float));
    free(mem);free(pending);return 0;
}
