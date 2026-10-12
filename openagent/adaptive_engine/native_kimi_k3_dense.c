/* Optional bounded FP32 SwiGLU. Output is committed only on success. */
#include <math.h>
#include <float.h>
#include <string.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
API int oa_k3_dense(const float *gate, const float *up, const float *down,
                    const float *x, int h, int m, float *out) {
    float activation[128], stage[64];
    int i, j;
    if (!gate || !up || !down || !x || !out || h<1 || h>64 || m<1 || m>128)
        return 1;
    for (i=0; i<h; ++i) if (!isfinite(x[i])) return 2;
    for (i=0; i<h*m; ++i)
        if (!isfinite(gate[i]) || !isfinite(up[i]) || !isfinite(down[i])) return 2;
    for (i=0; i<m; ++i) {
        double g=0, u=0, s, v;
        for (j=0; j<h; ++j) {
            g+=(double)gate[i*h+j]*x[j];
            u+=(double)up[i*h+j]*x[j];
        }
        if (fabs(g)>FLT_MAX || fabs(u)>FLT_MAX) return 2;
        g=(float)g; u=(float)u;
        s=g>=0 ? 1.0/(1.0+exp(-g)) : exp(g)/(1.0+exp(g));
        v=g*s*u;
        if (!isfinite(v) || fabs(v)>FLT_MAX) return 2;
        activation[i]=(float)v;
    }
    for (i=0; i<h; ++i) {
        double v=0;
        for (j=0; j<m; ++j) v+=(double)down[i*m+j]*activation[j];
        if (!isfinite(v) || fabs(v)>FLT_MAX) return 2;
        stage[i]=(float)v;
    }
    memcpy(out, stage, (size_t)h*sizeof(float));
    return 0;
}
