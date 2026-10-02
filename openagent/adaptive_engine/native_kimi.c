/* Original resident projected KDA CPU kernel. Private ABI, not a decoder ABI. */
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

API uint64_t oa_kimi_kda_version(void) { return 1; }

static int finite_buffer(const float *p, uint64_t n) {
    uint64_t i;
    if (!p) return 0;
    for (i=0; i<n; ++i) if (!isfinite(p[i])) return 0;
    return 1;
}

/* History is time-major; weights are head/channel/tap, oldest tap first. */
static void filter(const float *x, const float *w, float *past, double *y,
                   uint64_t width, uint64_t kernel) {
    uint64_t c, j;
    for (c=0; c<width; ++c) {
        double z = (double)x[c]*w[c*kernel+kernel-1];
        for (j=0; j+1<kernel; ++j) z += (double)past[j*width+c]*w[c*kernel+j];
        y[c] = z >= 0 ? z/(1+exp(-z)) : z*exp(z)/(1+exp(z));
    }
    if (kernel>1) {
        memmove(past, past+width, (size_t)((kernel-2)*width)*sizeof(float));
        memcpy(past+(kernel-2)*width, x, (size_t)width*sizeof(float));
    }
}

/* All counts checked before access. Caller supplies disjoint writable buffers.
   Mutation on arithmetic failure is confined to caller-provided staging state. */
API int oa_kimi_kda(uint64_t n, uint64_t heads, uint64_t key, uint64_t value,
    uint64_t kernel, double epsilon, double scale, double l2_epsilon,
    const float *q, const float *k, const float *v, const float *decay,
    const float *beta, const float *gate, const float *norm,
    const float *tq, const float *tk, const float *tv,
    float *state, float *hq, float *hk, float *hv, float *out,
    const uint64_t *counts) {
    uint64_t a, t, h, i, j, kw, vw, sn, expected[15];
    double *work, *fq, *fk, *fv, *error, *result;
    const float *buffers[15] = {q,k,v,decay,beta,gate,norm,tq,tk,tv,state,hq,hk,hv,out};
    if (!counts || !n || n>4096 || !heads || heads>128 || !key || key>256 ||
        !value || value>256 || !kernel || kernel>64 ||
        !isfinite(epsilon) || epsilon<=0 || !isfinite(scale) || scale<=0 ||
        !isfinite(l2_epsilon) || l2_epsilon<=0) return 1;
    kw=heads*key; vw=heads*value; sn=kw*value;
    expected[0]=expected[1]=expected[3]=n*kw;
    expected[2]=expected[5]=expected[14]=n*vw;
    expected[4]=n*heads; expected[6]=value;
    expected[7]=expected[8]=kw*kernel; expected[9]=vw*kernel;
    expected[10]=sn; expected[11]=expected[12]=(kernel-1)*kw;
    expected[13]=(kernel-1)*vw;
    for (a=0; a<15; ++a) {
        if (counts[a]!=expected[a] || !buffers[a]) return 2;
        if (a<14 && !finite_buffer(buffers[a], counts[a])) return 3;
    }
    for (a=0; a<n*kw; ++a) if (decay[a]>0) return 3;
    for (a=0; a<n*heads; ++a) if (beta[a]<0 || beta[a]>1) return 3;
    for (a=0; a<n*vw; ++a) if (gate[a]<0 || gate[a]>1) return 3;
    work=(double *)malloc((size_t)(2*kw+vw+2*value)*sizeof(double));
    if (!work) return 4;
    fq=work; fk=fq+kw; fv=fk+kw; error=fv+vw; result=error+value;
    for (t=0; t<n; ++t) {
        filter(q+t*kw,tq,hq,fq,kw,kernel);
        filter(k+t*kw,tk,hk,fk,kw,kernel);
        filter(v+t*vw,tv,hv,fv,vw,kernel);
        for (h=0; h<heads; ++h) {
            double qn=l2_epsilon, kn=l2_epsilon, rms=epsilon;
            float *s=state+h*key*value;
            for (i=0; i<key; ++i) { qn+=fq[h*key+i]*fq[h*key+i]; kn+=fk[h*key+i]*fk[h*key+i]; }
            qn=scale/sqrt(qn); kn=1/sqrt(kn);
            /* Decay precedes delta correction: the two operations do not commute. */
            for (i=0; i<key; ++i) {
                double alpha=exp(decay[t*kw+h*key+i]);
                fq[h*key+i]*=qn; fk[h*key+i]*=kn;
                for (j=0; j<value; ++j) s[i*value+j]=(float)(s[i*value+j]*alpha);
            }
            for (j=0; j<value; ++j) {
                double prediction=0;
                for (i=0; i<key; ++i) prediction+=fk[h*key+i]*s[i*value+j];
                error[j]=beta[t*heads+h]*(fv[h*value+j]-prediction);
            }
            for (j=0; j<value; ++j) {
                result[j]=0;
                for (i=0; i<key; ++i) {
                    double updated=s[i*value+j]+fk[h*key+i]*error[j];
                    if (!isfinite(updated) || fabs(updated)>FLT_MAX) { free(work); return 5; }
                    s[i*value+j]=(float)updated;
                    result[j]+=fq[h*key+i]*s[i*value+j];
                }
                rms+=result[j]*result[j]/(double)value;
            }
            for (j=0; j<value; ++j) {
                double y=result[j]/sqrt(rms)*norm[j]*gate[t*vw+h*value+j];
                if (!isfinite(y) || fabs(y)>FLT_MAX) { free(work); return 5; }
                out[t*vw+h*value+j]=(float)y;
            }
        }
    }
    free(work);
    return 0;
}

/* Small checked arithmetic ABI used by the checkpoint decoder's scheduler. */
API int oa_kimi_matvec(uint64_t rows, uint64_t cols, const float *w,
    const float *x, float *y) {
    uint64_t i,j;
    if (!rows || rows>1048576 || !cols || cols>65536 || rows*cols>67108864 ||
        !w || !x || !y) return 1;
    for (i=0;i<rows;++i) {
        double sum=0;
        for (j=0;j<cols;++j) sum+=(double)w[i*cols+j]*x[j];
        if (!isfinite(sum) || fabs(sum)>FLT_MAX) return 5;
        y[i]=(float)sum;
    }
    return 0;
}

API int oa_kimi_vector(uint64_t n, uint64_t op, double param,
    const float *x, const float *w, float *y) {
    uint64_t i;
    double rms=0;
    if (!n || n>1048576 || !x || !y || op>7 ||
        ((op==0 || op==1 || op==2) && !w)) return 1;
    if (op==0) {
        if (!isfinite(param) || param<=0) return 1;
        for (i=0;i<n;++i) rms+=(double)x[i]*x[i]/(double)n;
        rms=1/sqrt(rms+param);
    }
    for (i=0;i<n;++i) {
        double z=x[i], r;
        switch(op) {
            case 0: r=z*rms*w[i]; break;
            case 1: r=z+w[i]; break;
            case 2: r=z*w[i]; break;
            case 3: r=z>=0 ? 1/(1+exp(-z)) : exp(z)/(1+exp(z)); break;
            case 4: r=z>=0 ? z/(1+exp(-z)) : z*exp(z)/(1+exp(z)); break;
            case 5: r=fmax(z,0)+log1p(exp(-fabs(z))); break;
            case 6: r=exp(z); break;
            default: r=z*param; break;
        }
        if (!isfinite(r) || fabs(r)>FLT_MAX) return 5;
        y[i]=(float)r;
    }
    return 0;
}

/* Expanded NoPE MLA: token-major K/V, no RoPE and no output gate. */
API int oa_kimi_attention(uint64_t tokens, uint64_t heads, uint64_t kd,
    uint64_t vd, const float *q, const float *k, const float *v, float *out) {
    uint64_t h,t,i,j;
    double *score;
    if (!tokens || tokens>65536 || !heads || heads>128 || !kd || kd>512 ||
        !vd || vd>256 || !q || !k || !v || !out) return 1;
    score=(double *)malloc((size_t)tokens*sizeof(double));
    if (!score) return 4;
    for (h=0;h<heads;++h) {
        double maximum=-DBL_MAX,total=0;
        for (t=0;t<tokens;++t) {
            double s=0;
            for (i=0;i<kd;++i) s+=(double)q[h*kd+i]*k[(t*heads+h)*kd+i];
            score[t]=s/sqrt((double)kd);
            if (score[t]>maximum) maximum=score[t];
        }
        for (t=0;t<tokens;++t) { score[t]=exp(score[t]-maximum); total+=score[t]; }
        for (j=0;j<vd;++j) {
            double r=0;
            for (t=0;t<tokens;++t) r+=score[t]/total*v[(t*heads+h)*vd+j];
            if (!isfinite(r) || fabs(r)>FLT_MAX) { free(score); return 5; }
            out[h*vd+j]=(float)r;
        }
    }
    free(score); return 0;
}

/* Single-group router, stable lowest-index tie convention. Published vendor
   code mutates the score view with correction bias before gathering weights. */
API int oa_kimi_route(uint64_t experts, uint64_t topk, uint64_t renorm,
    double scale, const float *scores, const float *bias, uint64_t *ids, float *weights) {
    uint64_t i,j,best;
    double sum=0;
    if (experts<2 || experts>4096 || !topk || topk>experts || renorm>1 ||
        !isfinite(scale) || scale<=0 || !scores || !bias || !ids || !weights) return 1;
    for (j=0;j<topk;++j) {
        double maximum=-DBL_MAX;
        best=experts;
        for (i=0;i<experts;++i) {
            uint64_t seen=0,a;
            double s=(double)(float)(scores[i]+bias[i]);
            for (a=0;a<j;++a) if (ids[a]==i) seen=1;
            if (!seen && s>maximum) { maximum=s; best=i; }
        }
        if (best==experts || !isfinite(maximum)) return 5;
        ids[j]=best; weights[j]=(float)maximum; sum+=maximum;
    }
    if (renorm && topk>1) sum+=1e-20; else sum=1;
    for (j=0;j<topk;++j) {
        double r=weights[j]/sum*scale;
        if (!isfinite(r) || fabs(r)>FLT_MAX) return 5;
        weights[j]=(float)r;
    }
    return 0;
}
