/* Original bounded Qwen3.5 vision kernels, not a pretrained encoder. */
#include <stdint.h>
#include <math.h>
#include <string.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
typedef struct { const float *data; uint64_t count; } Tensor;
API uint64_t oa_qwen35_vision_version(void) { return 1; }
static float dot(const float *a, const float *b, int n) {
    double s=0; int i; for(i=0;i<n;i++) s+=(double)a[i]*b[i]; return (float)s;
}
static void norm8(const float *x, float *y, const Tensor *w) {
    double mean=0, var=0; int j;
    for(j=0;j<8;j++) mean+=x[j]; mean/=8;
    for(j=0;j<8;j++) { double d=x[j]-mean; var+=d*d; } var/=8;
    for(j=0;j<8;j++) y[j]=(float)((x[j]-mean)/sqrt(var+1e-6))*w[0].data[j]+w[1].data[j];
}
static void block(float *x, const Tensor *w) {
    float y[32], qkv[96], attention[32], hidden[16]; int i,j,h,k,r;
    for(i=0;i<4;i++) {
        norm8(x+i*8,y+i*8,w);
        for(j=0;j<24;j++) qkv[i*24+j]=dot(y+i*8,w[4].data+j*8,8)+w[5].data[j];
        /* Two spatial axes, half-split rotary pairs in each four-wide head. */
        for(r=0;r<2;r++) for(h=0;h<2;h++) for(j=0;j<2;j++) {
            int p=i*24+r*8+h*4+j;
            double angle=j==0 ? i/2 : i%2;
            float a=qkv[p], b=qkv[p+2];
            qkv[p]=(float)(a*cos(angle)-b*sin(angle));
            qkv[p+2]=(float)(b*cos(angle)+a*sin(angle));
        }
    }
    for(i=0;i<4;i++) for(h=0;h<2;h++) {
        double score[4], maximum=-1e300, total=0;
        for(k=0;k<4;k++) {
            score[k]=dot(qkv+i*24+h*4,qkv+k*24+8+h*4,4)*0.5;
            if(score[k]>maximum) maximum=score[k];
        }
        for(k=0;k<4;k++) { score[k]=exp(score[k]-maximum); total+=score[k]; }
        for(j=0;j<4;j++) {
            double s=0;
            for(k=0;k<4;k++) s+=score[k]/total*qkv[k*24+16+h*4+j];
            attention[i*8+h*4+j]=(float)s;
        }
    }
    for(i=0;i<4;i++) {
        for(j=0;j<8;j++) x[i*8+j]+=dot(attention+i*8,w[6].data+j*8,8)+w[7].data[j];
        norm8(x+i*8,y+i*8,w+2);
        for(j=0;j<16;j++) {
            double v=dot(y+i*8,w[8].data+j*8,8)+w[9].data[j];
            hidden[j]=(float)(0.5*v*(1+tanh(sqrt(2.0/3.14159265358979323846)*(v+0.044715*v*v*v))));
        }
        for(j=0;j<8;j++) x[i*8+j]+=dot(hidden,w[10].data+j*16,16)+w[11].data[j];
    }
}
API int oa_qwen35_vision_encode(Tensor pixels, const Tensor *w, uint64_t count,
                               Tensor output) {
    const uint64_t sizes[9]={24,8,32,8,8,1024,32,768,24};
    const uint64_t block_sizes[12]={8,8,8,8,192,24,64,8,128,16,128,8};
    float x[32], y[32], z[32], out[24]; int i,j; double mean,var;
    uint64_t b;
    if(!w || count<9 || count>57 || (count-9)%12 || !pixels.data || pixels.count!=12 ||
       !output.data || output.count!=24) return 1;
    for(i=0;i<9;i++) if(!w[i].data || w[i].count!=sizes[i]) return 1;
    for(b=9;b<count;b++) if(!w[b].data || w[b].count!=block_sizes[(b-9)%12]) return 1;
    for(i=0;i<12;i++) if(!isfinite(pixels.data[i])) return 2;
    for(i=0;i<9;i++) for(j=0;j<(int)sizes[i];j++)
        if(!isfinite(w[i].data[j])) return 2;
    for(b=9;b<count;b++) for(j=0;j<(int)w[b].count;j++) if(!isfinite(w[b].data[j])) return 2;
    for(i=0;i<4;i++) {
        for(j=0;j<8;j++) x[i*8+j]=dot(pixels.data+i*3,w[0].data+j*3,3)+
            w[1].data[j]+w[2].data[i*8+j];
    }
    for(b=9;b<count;b+=12) block(x,w+b);
    for(i=0;i<4;i++) {
        mean=0; for(j=0;j<8;j++) mean+=x[i*8+j]; mean/=8;
        var=0; for(j=0;j<8;j++) { double d=x[i*8+j]-mean; var+=d*d; } var/=8;
        for(j=0;j<8;j++) y[i*8+j]=(float)((x[i*8+j]-mean)/sqrt(var+1e-6))*
            w[3].data[j]+w[4].data[j];
    }
    for(i=0;i<32;i++) {
        float v=dot(y,w[5].data+i*32,32)+w[6].data[i];
        z[i]=(float)(0.5*v*(1+erf(v/sqrt(2.0))));
    }
    for(i=0;i<24;i++) { out[i]=dot(z,w[7].data+i*32,32)+w[8].data[i];
        if(!isfinite(out[i])) return 2; }
    memcpy((void*)output.data,out,sizeof(out)); return 0;
}
/* One merged 1x1 image token has T=H=W, delta=0. Scalar text RoPE is exact. */
API int oa_qwen35_vision_positions(int64_t *out, uint64_t count,
                                  uint64_t start, uint64_t capacity) {
    uint64_t i;
    if(!out || !count || count>65536 || capacity>65536 || start>capacity ||
       count>capacity-start) return 1;
    for(i=0;i<count;i++) out[i]=(int64_t)(start+i);
    return 0;
}
API int oa_qwen35_vision_merge(Tensor embedding, uint64_t vocab,
                              uint64_t image_id, Tensor image, Tensor output) {
    if(!embedding.data || !image.data || !output.data || !vocab || vocab>1048576 ||
       image_id>=vocab || embedding.count!=vocab*24 || output.count!=embedding.count ||
       image.count!=24) return 1;
    memcpy((void*)output.data,embedding.data,(size_t)embedding.count*sizeof(float));
    memcpy((float*)output.data+image_id*24,image.data,24*sizeof(float)); return 0;
}
