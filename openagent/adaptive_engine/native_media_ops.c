/* Original variable-geometry CPU media primitives. No framework dispatcher. */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
typedef struct {float *data; uint64_t count;} Tensor;
static int valid(Tensor t,uint64_t n) {
    uint64_t i;
    if(!t.data||t.count!=n) return 0;
    for(i=0;i<n;i++) if(!isfinite(t.data[i])) return 0;
    return 1;
}
API uint64_t oa_media_ops_version(void) {return 1;}
API uint64_t oa_media_swiglu_version(void) {return 1;}
API int oa_media_swiglu_clamped(float limit,Tensor gate,Tensor up,Tensor output) {
    uint64_t i;
    if(!isfinite(limit)||limit<=0||!valid(gate,gate.count)||!valid(up,gate.count)||
        !output.data||output.count!=gate.count) return 1;
    for(i=0;i<gate.count;i++) {
        float g=gate.data[i]<limit?gate.data[i]:limit;
        float u=up.data[i]>limit?limit:up.data[i]<-limit?-limit:up.data[i];
        float sigmoid=g>=0?1.0f/(1.0f+expf(-g)):expf(g)/(1.0f+expf(g));
        output.data[i]=g*sigmoid*u;
    }
    return valid(output,output.count)?0:3;
}
API int oa_media_audio_embed(uint64_t rows,uint64_t bins,uint64_t vocabulary,uint64_t hidden,
    const int64_t *ids,Tensor table,Tensor output) {
    uint64_t r,b,h;
    if(!rows||rows>1048576||!bins||bins>512||!vocabulary||vocabulary>65536||!hidden||hidden>16384||
        !ids||!valid(table,bins*vocabulary*hidden)||!output.data||output.count!=rows*hidden) return 1;
    for(r=0;r<rows*bins;r++) if(ids[r]<0||(uint64_t)ids[r]>=vocabulary) return 1;
    for(r=0;r<rows;r++) for(h=0;h<hidden;h++) {
        double sum=0;
        for(b=0;b<bins;b++) sum+=table.data[(b*vocabulary+(uint64_t)ids[r*bins+b])*hidden+h];
        output.data[r*hidden+h]=(float)sum;
    }
    return valid(output,output.count)?0:3;
}
API int oa_media_linear(uint64_t rows,uint64_t input,uint64_t output,
    Tensor x,Tensor weight,Tensor bias,Tensor y) {
    uint64_t r,i,j;
    if(!rows||rows>1048576||!input||input>1048576||!output||output>1048576||
        !valid(x,rows*input)||!valid(weight,input*output)||
        (bias.count&&!valid(bias,output))||(!bias.count&&bias.data)||
        !y.data||y.count!=rows*output) return 1;
    for(r=0;r<rows;r++) for(i=0;i<output;i++) {
        double sum=bias.count?bias.data[i]:0;
        for(j=0;j<input;j++) sum+=(double)x.data[r*input+j]*weight.data[i*input+j];
        y.data[r*output+i]=(float)sum;
    }
    return valid(y,y.count)?0:3;
}
API int oa_media_norm(uint64_t rows,uint64_t width,double epsilon,int rms,
    Tensor x,Tensor weight,Tensor bias,Tensor y) {
    uint64_t r,j;
    if(!rows||rows>1048576||!width||width>1048576||!isfinite(epsilon)||epsilon<=0||
        (rms!=0&&rms!=1)||!valid(x,rows*width)||!valid(weight,width)||
        (bias.count&&!valid(bias,width))||(!bias.count&&bias.data)||!y.data||y.count!=x.count) return 1;
    for(r=0;r<rows;r++) {
        double mean=0,variance=0;
        if(!rms) {for(j=0;j<width;j++) mean+=x.data[r*width+j];mean/=width;}
        for(j=0;j<width;j++) {double d=x.data[r*width+j]-mean;variance+=d*d;}
        variance=1/sqrt(variance/width+epsilon);
        for(j=0;j<width;j++) y.data[r*width+j]=(float)((x.data[r*width+j]-mean)*variance)*
            weight.data[j]+(bias.count?bias.data[j]:0);
    }
    return valid(y,y.count)?0:3;
}
API int oa_media_activation(int kind,Tensor x,Tensor y) {
    uint64_t j;
    if((kind<0||kind>2)||!valid(x,x.count)||!y.data||y.count!=x.count) return 1;
    for(j=0;j<x.count;j++) {
        double v=x.data[j];
        if(kind==0) y.data[j]=(float)(0.5*v*(1+erf(v/sqrt(2.0))));
        else if(kind==1) y.data[j]=(float)(0.5*v*(1+tanh(sqrt(2/3.14159265358979323846)*(v+0.044715*v*v*v))));
        else y.data[j]=(float)(v/(1+exp(-v)));
    }
    return valid(y,y.count)?0:3;
}
/* qkv is [token,3,head,dimension], positions [token,2]. Split-half
   rotary pairs concatenate H and W frequencies. Each frame attends itself. */
API int oa_media_vision_attention(uint64_t tokens,uint64_t heads,uint64_t dim,
    Tensor qkv,const int64_t *positions,const uint64_t *offsets,uint64_t frames,Tensor output) {
    uint64_t frame,start,end,i,j,h,k,l,quarter;
    float *memory,*q,*key,*score;
    if(!tokens||tokens>1048576||!heads||heads>256||!dim||dim>4096||dim%4||
        !frames||frames>65536||!positions||!offsets||offsets[0]!=0||offsets[frames]!=tokens||
        !valid(qkv,tokens*3*heads*dim)||!output.data||output.count!=tokens*heads*dim) return 1;
    for(i=0;i<frames;i++) if(offsets[i]>=offsets[i+1]||offsets[i+1]>tokens) return 1;
    for(i=0;i<tokens*2;i++) if(positions[i]<0||positions[i]>1048576) return 1;
    memory=(float *)malloc((size_t)(2*dim+tokens)*sizeof(float));if(!memory) return 2;
    q=memory;key=q+dim;score=key+dim;quarter=dim/4;
    for(frame=0;frame<frames;frame++) {
        start=offsets[frame];end=offsets[frame+1];
        for(i=start;i<end;i++) for(h=0;h<heads;h++) {
            double maximum=-DBL_MAX,total=0;
            for(k=0;k<dim/2;k++) {
                double angle=positions[2*i+(k/quarter)]*pow(10000.0,-(double)(2*(k%quarter))/(double)(dim/2));
                float a=qkv.data[(i*3*heads+h)*dim+k],b=qkv.data[(i*3*heads+h)*dim+k+dim/2];
                q[k]=(float)(a*cos(angle)-b*sin(angle));q[k+dim/2]=(float)(b*cos(angle)+a*sin(angle));
            }
            for(j=start;j<end;j++) {
                double dot=0;
                for(k=0;k<dim/2;k++) {
                    double angle=positions[2*j+(k/quarter)]*pow(10000.0,-(double)(2*(k%quarter))/(double)(dim/2));
                    float a=qkv.data[(j*3*heads+heads+h)*dim+k],b=qkv.data[(j*3*heads+heads+h)*dim+k+dim/2];
                    key[k]=(float)(a*cos(angle)-b*sin(angle));key[k+dim/2]=(float)(b*cos(angle)+a*sin(angle));
                }
                for(k=0;k<dim;k++) dot+=(double)q[k]*key[k];
                score[j]=(float)(dot/sqrt((double)dim));if(score[j]>maximum) maximum=score[j];
            }
            for(j=start;j<end;j++) {score[j]=(float)exp(score[j]-maximum);total+=score[j];}
            for(k=0;k<dim;k++) {
                double value=0;
                for(l=start;l<end;l++) value+=(double)score[l]/total*qkv.data[(l*3*heads+2*heads+h)*dim+k];
                output.data[(i*heads+h)*dim+k]=(float)value;
            }
        }
    }
    free(memory);return valid(output,output.count)?0:3;
}
