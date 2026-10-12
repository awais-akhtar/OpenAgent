/* Original row-major kernels. Private ops 0-15 remain unchanged;
   publisher contract ops 16-25 have a separate capability handshake. */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif
static int put(double x, float *p) {
    if (!isfinite(x) || fabs(x)>FLT_MAX) return 3;
    *p=(float)x; return 0;
}
static double dot(const float *x,const float *y,int n) {
    double z=0; int i;
    for(i=0;i<n;++i) z+=(double)x[i]*y[i];
    return z;
}
typedef int (*oa_v41_read_row)(void *,uint64_t,float *,uint64_t);
typedef int (*oa_v41_cancel_rows)(void *);
static float publisher_bf16(float);
static double publisher_e4m3(int);
typedef int (*oa_v41_read_packed)(void *,uint64_t,int *,unsigned char *,uint64_t,unsigned char *,uint64_t);
typedef struct {oa_v41_read_packed read;void *owner;uint64_t width;unsigned char *values,*scales;} oa_v41_packed_reader;
static int oa_v41_decode_packed_row(void *opaque,uint64_t index,float *output,uint64_t width){
    static const double fp4[]={0,.5,1,1.5,2,3,4,6};
    oa_v41_packed_reader *r=(oa_v41_packed_reader *)opaque;uint64_t axis,groups;int encoding,group,bits;
    if(!r||width!=r->width||r->read(r->owner,index,&encoding,r->values,width,r->scales,width/16))return 1;
    if(encoding<1||encoding>3)return 1;group=encoding==2?16:32;bits=encoding==1?8:4;
    if(width%(uint64_t)group)return 1;groups=width/(uint64_t)group;
    for(axis=0;axis<groups;axis++){
        unsigned code=r->scales[axis];
        if((encoding==2&&(!code||code>=127))||(encoding!=2&&code==255))return 1;
    }
    for(axis=0;axis<width;axis++){
        unsigned code=bits==8?r->values[axis]:(r->values[axis/2]>>(4*(axis&1)))&15;
        unsigned magnitude=code&(bits==8?127:7),sign=code&(bits==8?128:8);
        unsigned scale_code=r->scales[axis/(uint64_t)group];double scale,value;
        if(bits==8&&magnitude==127)return 1;
        scale=encoding==2?publisher_e4m3((int)scale_code):ldexp(1.0,(int)scale_code-127);
        value=copysign((bits==8?publisher_e4m3((int)magnitude):fp4[magnitude])*scale,sign?-1.0:1.0);
        if(put(value,output+axis))return 1;
        output[axis]=publisher_bf16(output[axis]);if(!isfinite(output[axis]))return 1;
    }
    return 0;
}
API uint32_t oa_v41_attention_rows_version(void){return 1;}
API uint32_t oa_v41_packed_rows_version(void){return 1;}
API int oa_v41_attention_rows(const float *query,uint32_t heads,uint32_t width,uint64_t count,
    const float *sinks,double scale,oa_v41_read_row read_row,void *owner,
    oa_v41_cancel_rows cancel,void *cancel_owner,uint64_t max_bytes,float *output){
    double *memory,*maximum,*sum,*numerator;float *row,*pending;uint64_t words,bytes,index,axis;uint32_t head;int status=1;
    if(!query||!sinks||!output||!read_row||!heads||heads>1024||!width||width>16384||!count||
        count>INT32_MAX||!isfinite(scale)||scale<=0)return 1;
    words=(uint64_t)heads*width;bytes=words*12+(uint64_t)heads*16+(uint64_t)width*4;
    if(bytes>max_bytes||bytes>SIZE_MAX)return 2;
    for(index=0;index<words;index++)if(!isfinite(query[index]))return 3;
    for(head=0;head<heads;head++)if(!isfinite(sinks[head]))return 3;
    memory=(double *)calloc((size_t)(words+2*heads),sizeof(double));
    row=(float *)malloc((size_t)width*sizeof(float));pending=(float *)malloc((size_t)words*sizeof(float));
    if(!memory||!row||!pending){free(memory);free(row);free(pending);return 2;}
    maximum=memory;sum=maximum+heads;numerator=sum+heads;
    for(head=0;head<heads;head++)maximum[head]=sinks[head];
    for(index=0;index<count;index++){
        if(cancel&&cancel(cancel_owner)){status=4;goto finish;}
        if(read_row(owner,index,row,width)){status=5;goto finish;}
        for(axis=0;axis<width;axis++)if(!isfinite(row[axis])){status=3;goto finish;}
        for(head=0;head<heads;head++){
            double score=dot(query+(uint64_t)head*width,row,(int)width)*scale;
            if(!isfinite(score)){status=3;goto finish;}if(score>maximum[head])maximum[head]=score;
        }
    }
    for(head=0;head<heads;head++)sum[head]=exp(sinks[head]-maximum[head]);
    for(index=0;index<count;index++){
        if(cancel&&cancel(cancel_owner)){status=4;goto finish;}
        if(read_row(owner,index,row,width)){status=5;goto finish;}
        for(axis=0;axis<width;axis++)if(!isfinite(row[axis])){status=3;goto finish;}
        for(head=0;head<heads;head++){
            double score=dot(query+(uint64_t)head*width,row,(int)width)*scale;
            double probability=exp(score-maximum[head]);
            if(!isfinite(score)||!isfinite(probability)){status=3;goto finish;}
            sum[head]+=probability;
            for(axis=0;axis<width;axis++)numerator[(uint64_t)head*width+axis]+=probability*row[axis];
        }
    }
    for(head=0;head<heads;head++){
        if(!isfinite(sum[head])||sum[head]<=0){status=3;goto finish;}
        for(axis=0;axis<width;axis++)if(put(numerator[(uint64_t)head*width+axis]/sum[head],
            pending+(uint64_t)head*width+axis)){status=3;goto finish;}
    }
    if(cancel&&cancel(cancel_owner)){status=4;goto finish;}
    memcpy(output,pending,(size_t)words*sizeof(float));status=0;
finish:free(memory);free(row);free(pending);return status;
}
API int oa_v41_attention_packed_rows(const float *query,uint32_t heads,uint32_t width,uint64_t count,
    const float *sinks,double scale,oa_v41_read_packed read,void *owner,
    oa_v41_cancel_rows cancel,void *cancel_owner,uint64_t max_bytes,float *output){
    oa_v41_packed_reader r;uint64_t bytes;int status;
    if(!read||!width||width>16384||width%16)return 1;bytes=(uint64_t)width+width/16;
    if(bytes>=max_bytes)return 2;
    r.read=read;r.owner=owner;r.width=width;r.values=(unsigned char *)malloc((size_t)bytes);
    if(!r.values)return 2;r.scales=r.values+width;
    status=oa_v41_attention_rows(query,heads,width,count,sinks,scale,oa_v41_decode_packed_row,&r,
        cancel,cancel_owner,max_bytes-bytes,output);free(r.values);return status;
}
API int oa_v41_index_packed_rows(const float *query,uint32_t heads,uint32_t width,uint64_t count,
    const float *weights,double scale,oa_v41_read_packed read,void *owner,
    oa_v41_cancel_rows cancel,void *cancel_owner,uint64_t max_bytes,float *output){
    oa_v41_packed_reader r;float *row,*pending;uint64_t bytes,index,axis;uint32_t head;int status=1;
    if(!query||!weights||!output||!read||!heads||heads>1024||!width||width>16384||width%16||
        !count||count>INT32_MAX||!isfinite(scale)||scale<=0)return 1;
    bytes=(uint64_t)width*5+width/16+count*4;if(bytes>max_bytes||bytes>SIZE_MAX)return 2;
    for(axis=0;axis<(uint64_t)heads*width;axis++)if(!isfinite(query[axis]))return 3;
    for(head=0;head<heads;head++)if(!isfinite(weights[head]))return 3;
    r.read=read;r.owner=owner;r.width=width;r.values=(unsigned char *)malloc((size_t)width+width/16);
    row=(float *)malloc((size_t)width*4);pending=(float *)malloc((size_t)count*4);
    if(!r.values||!row||!pending){free(r.values);free(row);free(pending);return 2;}r.scales=r.values+width;
    for(index=0;index<count;index++){
        double score=0;
        if(cancel&&cancel(cancel_owner)){status=4;goto finish_index;}
        if(oa_v41_decode_packed_row(&r,index,row,width)){status=5;goto finish_index;}
        for(head=0;head<heads;head++)score+=fmax(0,dot(query+(uint64_t)head*width,row,(int)width))*weights[head]*scale;
        if(put(score,pending+index)){status=3;goto finish_index;}
    }
    if(cancel&&cancel(cancel_owner)){status=4;goto finish_index;}
    memcpy(output,pending,(size_t)count*4);status=0;
finish_index:free(r.values);free(row);free(pending);return status;
}
static double sigmoid(double x) { return x>=0?1/(1+exp(-x)):exp(x)/(1+exp(x)); }
static float publisher_bf16(float value) {
    uint32_t bits; memcpy(&bits,&value,4);
    bits+=0x7fff+((bits>>16)&1); bits&=0xffff0000;
    memcpy(&value,&bits,4); return value;
}
static double publisher_e4m3(int bits) {
    int e=(bits>>3)&15,f=bits&7;
    return e?ldexp(8.0+f,e-10):ldexp((double)f,-9);
}
static double publisher_round(double x,int fp4) {
    static const double levels[8]={0,.5,1,1.5,2,3,4,6};
    double best=0,error=DBL_MAX,value=fabs(x); int code,selected=0;
    for(code=0;code<(fp4?8:127);++code) {
        double level=fp4?levels[code]:publisher_e4m3(code),distance=fabs(value-level);
        if(distance<error || (distance==error && !(code&1))) { best=level; error=distance; selected=code; }
    }
    (void)selected; return copysign(best,x);
}
static int extent(const void *p,size_t n) {
    uintptr_t address=(uintptr_t)p;
    return p && !(address%sizeof(float)) && n<=UINTPTR_MAX-address;
}
static int overlap(const void *a,size_t na,const void *b,size_t nb) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return na && nb && x<y+nb && y<x+na;
}
API uint32_t oa_v41_execution_version(void) { return 2; }
API uint32_t oa_v41_publisher_contract_version(void) { return 1; }

/* Publisher Engram metadata ABI: newest-first compressed tokens, one
   multiplier per lookback, and order-major prime buckets with offsets. */
API int oa_v41_publisher_hash(const uint64_t *tokens,const uint64_t *multipliers,
    size_t order,const uint64_t *primes,const uint64_t *offsets,size_t heads,
    uint64_t *out,size_t count) {
    size_t j,h; uint64_t rolling=0;
    if(!tokens || !multipliers || !primes || !offsets || !out || order<2 || order>32 ||
       !heads || heads>4096 || count!=(order-1)*heads) return 1;
    for(j=0;j<order;++j) {
        if(multipliers[j]>INT64_MAX || (tokens[j] && multipliers[j]>(uint64_t)INT64_MAX/tokens[j])) return 1;
        rolling^=tokens[j]*multipliers[j];
        if(j) for(h=0;h<heads;++h) {
            size_t slot=(j-1)*heads+h;
            if(!primes[slot] || offsets[slot]>UINT64_MAX-primes[slot]) return 1;
            out[slot]=offsets[slot]+rolling%primes[slot];
        }
    } return 0;
}

typedef struct { double score; int index; } Ranked;
static int better(Ranked a, Ranked b) {
    return a.score>b.score || (a.score==b.score && a.index<b.index);
}
static void worst_down(Ranked *heap,int count,int at) {
    for(;;) {
        int left=2*at+1,right=left+1,child;
        Ranked value;
        if(left>=count) return;
        child=left;
        if(right<count && better(heap[left],heap[right])) child=right;
        if(!better(heap[at],heap[child])) return;
        value=heap[at]; heap[at]=heap[child]; heap[child]=value; at=child;
    }
}
/* Worst-first bounded heap; ties preserve the lower input index. */
static int topk(const float *f,const double *d,int count,int k,float *out) {
    Ranked *heap=(Ranked *)malloc((size_t)k*sizeof(Ranked));
    int i,used=0;
    if(!heap) return 4;
    for(i=0;i<count;++i) {
        Ranked next={f?(double)f[i]:d[i],i};
        if(used<k) {
            int at=used++;
            heap[at]=next;
            while(at && better(heap[(at-1)/2],heap[at])) {
                Ranked value=heap[at]; heap[at]=heap[(at-1)/2]; heap[(at-1)/2]=value;
                at=(at-1)/2;
            }
        } else if(better(next,heap[0])) { heap[0]=next; worst_down(heap,k,0); }
    }
    for(i=k-1;i>0;--i) {
        Ranked value=heap[0]; heap[0]=heap[i]; heap[i]=value;
        worst_down(heap,i,0);
    }
    for(i=0;i<k;++i) out[i]=(float)heap[i].index;
    free(heap); return 0;
}

/* Lengths are float counts. Dispatcher never retains input/output pointers. */
API int oa_v41_execution_op(int op,int a,int b,int c,double parameter,
    const float *x,size_t nx,const float *w,size_t nw,
    const float *z,size_t nz,float *out,size_t no) {
    size_t i; int r,d;
    /* Admission is element-count based; never trust geometry to imply extent. */
    if(sizeof(size_t)<8 || !x || !out || !nx || !no || nx>268435456 || nw>268435456 || nz>268435456 ||
       no>268435456 || (nw && !w) || (nz && !z) || !isfinite(parameter)) return 1;
    if(!extent(x,nx*sizeof(float)) || !extent(out,no*sizeof(float)) ||
       (nw && !extent(w,nw*sizeof(float))) || (nz && !extent(z,nz*sizeof(float))) ||
       overlap(out,no*sizeof(float),x,nx*sizeof(float)) ||
       overlap(out,no*sizeof(float),w,nw*sizeof(float)) ||
       overlap(out,no*sizeof(float),z,nz*sizeof(float))) return 1;
    for(i=0;i<nx;++i) if(!isfinite(x[i])) return 3;
    for(i=0;i<nw;++i) if(!isfinite(w[i])) return 3;
    for(i=0;i<nz;++i) if(!isfinite(z[i])) return 3;
    switch(op) {
    case 0: /* linear [out,in] */
        if(a<1 || a>65536 || b<1 || b>65536 || nx!=(size_t)b || nw!=(size_t)a*b || no!=(size_t)a) return 1;
        for(r=0;r<a;++r) if(put(dot(w+r*b,x,b),out+r)) return 3;
        return 0;
    case 1: /* weighted RMSNorm */
        if(nx!=no || nw!=nx || parameter<=0) return 1;
        { double scale=1/sqrt(dot(x,x,(int)nx)/nx+parameter);
          for(i=0;i<nx;++i) if(put(x[i]*scale*w[i],out+i)) return 3; }
        return 0;
    case 2: /* residual addition */
        if(nx!=nw || nx!=no) return 1;
        for(i=0;i<nx;++i) if(put((double)x[i]+w[i],out+i)) return 3;
        return 0;
    case 3: /* V4 trailing-channel interleaved RoPE */
        if(nx!=no || b<0 || b>(int)nx || b%2 || parameter<=1 || a<0) return 1;
        memcpy(out,x,nx*sizeof(float));
        for(d=0;d<b;d+=2) {
            size_t at=nx-(size_t)b+(size_t)d;
            double angle=a/pow(parameter,(double)d/b),cs=cos(angle),sn=sin(angle);
            if(put(x[at]*cs-x[at+1]*sn,out+at) || put(x[at]*sn+x[at+1]*cs,out+at+1)) return 3;
        }
        return 0;
    case 4: /* complete-block channel-wise gated compression */
        if(a<1 || a>1048576 || b<1 || b>16384 || nx!=(size_t)a*b || nw!=nx || no!=(size_t)b) return 1;
        for(d=0;d<b;++d) {
            double maximum=-DBL_MAX,sum=0,den=0;
            for(r=0;r<a;++r) if(w[r*b+d]>maximum) maximum=w[r*b+d];
            for(r=0;r<a;++r) { double p=exp(w[r*b+d]-maximum); den+=p; sum+=p*x[r*b+d]; }
            if(put(sum/den,out+d)) return 3;
        }
        return 0;
    case 5: /* joint local/global attention: two queries, concatenated K and V */
        if(a<1 || a>2097152 || b<1 || b>16384 || c<1 || c>a || nx!=(size_t)2*b ||
           nw!=(size_t)a*b || nz!=nw || no!=(size_t)b) return 1;
        { double *scores=(double *)malloc((size_t)a*sizeof(double)),maximum=-DBL_MAX,den=0;
          if(!scores) return 4;
          for(r=0;r<a;++r) { scores[r]=dot(x+(r<c?0:b),w+r*b,b)/sqrt((double)b);
              if(!isfinite(scores[r])) { free(scores); return 3; }
              if(scores[r]>maximum) maximum=scores[r]; }
          for(r=0;r<a;++r) { scores[r]=exp(scores[r]-maximum); den+=scores[r]; }
          for(d=0;d<b;++d) { double sum=0;
              for(r=0;r<a;++r) sum+=scores[r]*z[r*b+d];
              if(put(sum/den,out+d)) { free(scores); return 3; } }
          free(scores); }
        return 0;
    case 6: /* single-head ReLU index */
        if(a<1 || a>1048576 || b<1 || b>16384 || nx!=(size_t)b || nw!=(size_t)a*b || no!=(size_t)a) return 1;
        for(r=0;r<a;++r) if(put(fmax(0,dot(x,w+r*b,b)),out+r)) return 3;
        return 0;
    case 7: /* stable descending top-k, indices exactly representable as FP32 */
        if(a<1 || a>1048576 || b<1 || b>a || nx!=(size_t)a || no!=(size_t)b) return 1;
        return topk(x,NULL,a,b,out);
    case 8: /* SwiGLU with explicit activation clipping */
        if(nx!=nw || nx!=no || parameter<=0) return 1;
        for(i=0;i<nx;++i) {
            double gate=x[i]>parameter?parameter:x[i];
            double up=w[i]>parameter?parameter:(w[i]<-parameter?-parameter:w[i]);
            if(put(gate*sigmoid(gate)*up,out+i)) return 3;
        }
        return 0;
    case 9: /* V4 affinities, corrected selection, uncorrected route weights */
        if(a<1 || a>4096 || b<1 || b>a || nx!=(size_t)a || nw!=nx || no!=(size_t)2*b || parameter<0) return 1;
        { double *affinity=(double *)malloc((size_t)a*2*sizeof(double)),*selection,total=0;
          int *order=(int *)malloc((size_t)a*sizeof(int)),j;
          if(!affinity || !order) { free(affinity); free(order); return 4; }
          selection=affinity+a;
          for(r=0;r<a;++r) { double v=x[r]; affinity[r]=sqrt(v>20?v:log1p(exp(v)));
              selection[r]=affinity[r]+w[r]; j=r;
              while(j && selection[r]>selection[order[j-1]]) { order[j]=order[j-1]; --j; } order[j]=r; }
          for(r=0;r<b;++r) total+=affinity[order[r]];
          for(r=0;r<b;++r) { out[r]=(float)order[r];
              if(put(affinity[order[r]]/(total+1e-20)*parameter,out+b+r)) {
                  free(affinity); free(order); return 3; } }
          free(affinity); free(order); }
        return 0;
    case 10: /* expert accumulation */
        if(nx!=nw || no!=nx) return 1;
        for(i=0;i<nx;++i) if(put(x[i]+parameter*w[i],out+i)) return 3;
        return 0;
    case 11: /* block maxima for candidate admission */
        if(a<1 || a>1048576 || b<1 || b>1048576 || nx!=(size_t)a || no!=(size_t)((a+b-1)/b)) return 1;
        for(r=0;r<(a+b-1)/b;++r) {
            float maximum=-FLT_MAX;
            for(d=r*b;d<a && d<(r+1)*b;++d) if(x[d]>maximum) maximum=x[d];
            out[r]=maximum;
        }
        return 0;
    case 12: /* Dynamic-width mHC H: flattened RMS, affine, sigmoid, Sinkhorn. */
        if(a<1 || a>8 || b<1 || b>16384 || c<1 || c>32 || parameter<=0 ||
           nx!=(size_t)a*b || nw!=(size_t)a*(a+2)*nx || nz!=(size_t)a*(a+2) || no!=nz) return 1;
        { double logits[80],mix[64],maximum=-DBL_MAX;
          double rms=sqrt(dot(x,x,(int)nx)/nx+parameter);
          int q=a*(a+2),j,k,sweep;
          for(r=0;r<q;++r) {
              logits[r]=z[r]+dot(w+(size_t)r*nx,x,(int)nx)/rms;
              if(!isfinite(logits[r])) return 3;
          }
          for(r=0;r<a;++r) { out[r]=(float)sigmoid(logits[r]); out[a+a*a+r]=(float)(2*sigmoid(logits[a+a*a+r])); }
          for(r=0;r<a*a;++r) if(logits[a+r]>maximum) maximum=logits[a+r];
          for(r=0;r<a*a;++r) mix[r]=exp(logits[a+r]-maximum)+1e-12;
          for(sweep=0;sweep<c;++sweep) {
              for(j=0;j<a;++j) { double sum=0; for(k=0;k<a;++k) sum+=mix[j*a+k];
                  for(k=0;k<a;++k) mix[j*a+k]/=sum; }
              for(k=0;k<a;++k) { double sum=0; for(j=0;j<a;++j) sum+=mix[j*a+k];
                  for(j=0;j<a;++j) mix[j*a+k]/=sum; }
          }
          for(r=0;r<a*a;++r) out[a+r]=(float)mix[r]; }
        return 0;
    case 13: /* Stream reduction using explicitly supplied (delayed or final) A. */
        if(a<1 || a>8 || b<1 || b>16384 || nx!=(size_t)a*b || nw!=(size_t)a || no!=(size_t)b) return 1;
        for(r=0;r<a;++r) if(w[r]<0 || w[r]>1) return 1;
        for(d=0;d<b;++d) { double sum=0;
            for(r=0;r<a;++r) sum+=(double)w[r]*x[r*b+d];
            if(put(sum,out+d)) return 3; }
        return 0;
    case 14: /* X' = B X + C F. Caller delays A to the next block. */
        if(a<1 || a>8 || b<1 || b>16384 || nx!=(size_t)a*b || nw!=(size_t)a*(a+2) || nz!=(size_t)b || no!=nx) return 1;
        for(r=0;r<a;++r) if(w[r]<0 || w[r]>1 || w[a+a*a+r]<0 || w[a+a*a+r]>2) return 1;
        for(r=0;r<a*a;++r) if(w[a+r]<0 || w[a+r]>1) return 1;
        for(r=0;r<a;++r) for(d=0;d<b;++d) { int j;
            double sum=(double)w[a+a*a+r]*z[d];
            for(j=0;j<a;++j) sum+=(double)w[a+r*a+j]*x[j*b+d];
            if(put(sum,out+r*b+d)) return 3; }
        return 0;
    case 15: /* Retrieved Engram: full-width RMS key/query gate, shared value. */
        if(a<1 || a>8 || b<1 || b>16384 || parameter<=0 || nx!=(size_t)a*b ||
           nw!=nx || nz!=(size_t)b+2*nx || no!=nx) return 1;
        for(r=0;r<a;++r) {
            const float *hidden=x+(size_t)r*b,*key=w+(size_t)r*b;
            const float *qs=z+b+(size_t)r*b,*ks=z+b+nx+(size_t)r*b;
            double qr=sqrt(dot(hidden,hidden,b)/b+parameter);
            double kr=sqrt(dot(key,key,b)/b+parameter),score=0,gate;
            for(d=0;d<b;++d) score+=(hidden[d]/qr)*qs[d]*(key[d]/kr)*ks[d];
            score/=sqrt((double)b);
            if(!isfinite(score)) return 3;
            gate=sigmoid(score);
            for(d=0;d<b;++d) if(put(gate*z[d],out+(size_t)r*b+d)) return 3;
        }
        return 0;
    case 16: /* Publisher mHC: x is flattened stream; w is projected mixes.
                z holds base, three scales, norm epsilon, Sinkhorn epsilon. */
        if(a<1 || a>8 || c<1 || c>64 || nx<1 || nw!=(size_t)a*(a+2) ||
           nz!=nw+5 || no!=nw || z[nw+3]<=0 || z[nw+4]<=0) return 1;
        { double mix[64],rms=sqrt(dot(x,x,(int)nx)/nx+z[nw+3]),eps=z[nw+4];
          int j,k,sweep;
          for(j=0;j<a;++j) {
              if(put(sigmoid(w[j]/rms*z[nw]+z[j])+eps,out+j)) return 3;
              if(put(2*sigmoid(w[a+j]/rms*z[nw+1]+z[a+j]),out+a+a*a+j)) return 3;
              double maximum=-DBL_MAX,sum=0;
              for(k=0;k<a;++k) { double v=w[2*a+j*a+k]/rms*z[nw+2]+z[2*a+j*a+k];
                  mix[j*a+k]=v; if(v>maximum) maximum=v; }
              for(k=0;k<a;++k) { mix[j*a+k]=exp(mix[j*a+k]-maximum); sum+=mix[j*a+k]; }
              for(k=0;k<a;++k) mix[j*a+k]=mix[j*a+k]/sum+eps;
          }
          for(sweep=0;sweep<c;++sweep) {
              if(sweep) for(j=0;j<a;++j) { double sum=eps;
                  for(k=0;k<a;++k) sum+=mix[j*a+k];
                  for(k=0;k<a;++k) mix[j*a+k]/=sum; }
              for(k=0;k<a;++k) { double sum=eps;
                  for(j=0;j<a;++j) sum+=mix[j*a+k];
                  for(j=0;j<a;++j) mix[j*a+k]/=sum; }
          }
          for(j=0;j<a*a;++j) if(put(mix[j],out+a+j)) return 3;
        } return 0;
    case 17: /* Publisher trailing adjacent-pair YaRN, inverse for attention output.
                w = theta, original context, factor, beta fast, beta slow. */
        if(a<0 || b<2 || b%2 || nx!=no || nx%(size_t)b || nw!=5 ||
           w[0]<=1 || w[1]<0 || w[2]<=0 || w[3]<=0 || w[4]<=0 || (c!=1 && c!=-1)) return 1;
        memcpy(out,x,nx*sizeof(float));
        for(i=0;i<nx;i+=(size_t)b) for(d=0;d<b;d+=2) {
            double freq=pow(w[0],-(double)d/b),angle;
            if(w[1]>0) {
                double low=fmax(floor(b*log(w[1]/(w[3]*2*3.141592653589793))/(2*log(w[0]))),0);
                double high=fmin(ceil(b*log(w[1]/(w[4]*2*3.141592653589793))/(2*log(w[0]))),b-1);
                double ramp=fmin(1,fmax(0,(d/2-low)/fmax(high-low,1e-3)));
                freq*=1-ramp+ramp/w[2];
            }
            angle=c*a*freq;
            if(put(x[i+d]*cos(angle)-x[i+d+1]*sin(angle),out+i+d) ||
               put(x[i+d]*sin(angle)+x[i+d+1]*cos(angle),out+i+d+1)) return 3;
        } return 0;
    case 18: /* Multihead shared latent K=V with one zero-value sink per head. */
        if(a<1 || b<1 || c<1 || nx!=(size_t)a*b || nw!=(size_t)c*b || nz!=(size_t)a || no!=nx || parameter<=0) return 1;
        for(r=0;r<a;++r) {
            double maximum=z[r],sum,value;
            for(d=0;d<c;++d) { value=dot(x+r*b,w+d*b,b)*parameter; if(value>maximum) maximum=value; }
            sum=exp(z[r]-maximum);
            for(d=0;d<c;++d) sum+=exp(dot(x+r*b,w+d*b,b)*parameter-maximum);
            for(d=0;d<b;++d) { int j; value=0;
                for(j=0;j<c;++j) value+=exp(dot(x+r*b,w+j*b,b)*parameter-maximum)*w[j*b+d];
                if(put(value/sum,out+r*b+d)) return 3; }
        } return 0;
    case 19: /* Index: sum over weighted rectified per-head dot products. */
        if(a<1 || b<1 || c<1 || nx!=(size_t)a*b || nw!=(size_t)c*b || nz!=(size_t)a || no!=(size_t)c) return 1;
        for(d=0;d<c;++d) { double score=0;
            for(r=0;r<a;++r) score+=fmax(0,dot(x+r*b,w+d*b,b))*z[r]*parameter;
            if(put(score,out+d)) return 3; } return 0;
    case 20: /* Publisher Engram residual update: signed square root before sigmoid. */
        if(a<1 || b<1 || nx!=(size_t)a*b || nw!=nx || nz!=(size_t)b+2*nx || no!=nx || parameter<=0) return 1;
        for(r=0;r<a;++r) { double score=0,gate;
            double denom=sqrt(dot(x+r*b,x+r*b,b)/b+parameter)*sqrt(dot(w+r*b,w+r*b,b)/b+parameter)*sqrt((double)b);
            for(d=0;d<b;++d) score+=(double)x[r*b+d]*w[r*b+d]*z[b+r*b+d]*z[b+nx+r*b+d];
            score/=denom; gate=sigmoid(copysign(sqrt(fmax(fabs(score),1e-6)),score));
            for(d=0;d<b;++d) if(put(x[r*b+d]+gate*z[d],out+r*b+d)) return 3;
        } return 0;
    case 21: /* Publisher route affinity; w=selection correction, z=temp,scale,norm.
                c: 0 sqrtsoftplus, 1 sigmoid, 2 softmax. */
        if(a<1 || b<1 || b>a || nx!=(size_t)a || nw!=nx || nz!=3 || no!=(size_t)2*b || z[0]<=0 || z[1]<=0 || c<0 || c>2) return 1;
        { double *scores=malloc((size_t)a*2*sizeof(double)),total=0,maximum=-DBL_MAX;
          float *ids=malloc((size_t)b*sizeof(float)); int j;
          if(!scores || !ids) { free(scores); free(ids); return 4; }
          for(j=0;j<a;++j) if(x[j]/z[0]>maximum) maximum=x[j]/z[0];
          for(j=0;j<a;++j) { double v=x[j]/z[0];
              scores[j]=c==2?exp(v-maximum):c==1?sigmoid(v):sqrt(fmax(v,0)+log1p(exp(-fabs(v)))); total+=scores[j]; }
          for(j=0;j<a;++j) { if(c==2) scores[j]/=total; scores[a+j]=scores[j]+w[j]; }
          int status=topk(NULL,scores+a,a,b,ids);
          total=1e-20; for(j=0;j<b;++j) total+=scores[(int)ids[j]];
          for(j=0;!status && j<b;++j) { out[j]=ids[j]; status=put(scores[(int)ids[j]]*z[1]/((z[2]!=0 && b>1)?total:1),out+b+j); }
          free(scores); free(ids); return status;
        }
    case 22: /* Route weight is applied BEFORE the down projection. */
        if(nx!=nw || no!=nx || nz!=1 || parameter<0) return 1;
        for(i=0;i<nx;++i) { double gate=x[i],up=w[i];
            if(parameter>0) { gate=fmin(gate,parameter); up=fmin(parameter,fmax(-parameter,up)); }
            if(put(gate*sigmoid(gate)*up*z[0],out+i)) return 3;
        } return 0;
    case 23: /* Publisher stream reduction/update, epsilon-offset pre coefficients allowed. */
        if(a<1 || a>8 || b<1 || nx!=(size_t)a*b) return 1;
        if(c==0) {
            if(nw!=(size_t)a || no!=(size_t)b) return 1;
            for(d=0;d<b;++d) { double sum=0; for(r=0;r<a;++r) sum+=(double)x[r*b+d]*w[r]; if(put(sum,out+d)) return 3; }
        } else {
            if(nw!=(size_t)a*(a+2) || nz!=(size_t)b || no!=nx) return 1;
            for(r=0;r<a;++r) for(d=0;d<b;++d) { int j; double sum=w[a+a*a+r]*z[d];
                for(j=0;j<a;++j) sum+=(double)w[a+r*a+j]*x[j*b+d]; if(put(sum,out+r*b+d)) return 3; }
        } return 0;
    case 24: /* Candidate block mask with compulsory newest reachable block. */
        if(a<1 || b<1 || no!=nx || nx>16777216) return 1;
        { int blocks=(int)((nx+a-1)/a),k=b<blocks?b:blocks,j,status;
          double *scores=malloc((size_t)blocks*sizeof(double));
          float *ids=malloc((size_t)k*sizeof(float));
          if(!scores || !ids) { free(scores); free(ids); return 4; }
          for(j=0;j<blocks;++j) { size_t t,end=(size_t)(j+1)*a; scores[j]=-DBL_MAX;
              if(end>nx) end=nx;
              for(t=(size_t)j*a;t<end;++t) if(x[t]>scores[j]) scores[j]=x[t]; }
          scores[blocks-1]=DBL_MAX;
          status=topk(NULL,scores,blocks,k,ids);
          memset(out,0,no*sizeof(float));
          if(!status) for(j=0;j<k;++j) { size_t t,end=(size_t)((int)ids[j]+1)*a;
              if(end>nx) end=nx;
              for(t=(size_t)(int)ids[j]*a;t<end;++t) out[t]=1; }
          free(scores); free(ids); return status;
        }
    case 25: /* Masked top-k, sorted by cache position; -1 fills missing picks. */
        if(nx>16777216 || nw!=nx || b<1 || (size_t)b>nx || no!=(size_t)b) return 1;
        { double *scores=malloc(nx*sizeof(double)); float *ids=malloc(no*sizeof(float)); int j,status;
          if(!scores || !ids) { free(scores); free(ids); return 4; }
          for(i=0;i<nx;++i) { if(w[i]!=0 && w[i]!=1) { free(scores); free(ids); return 1; }
              scores[i]=w[i]?x[i]:-DBL_MAX; }
          status=topk(NULL,scores,(int)nx,b,ids);
          if(!status) {
              for(j=0;j<b;++j) if(!w[(int)ids[j]]) ids[j]=-1;
              for(j=1;j<b;++j) { int k=j; float value=ids[j];
                  while(k && ids[k-1]>value) { ids[k]=ids[k-1]; --k; } ids[k]=value; }
              memcpy(out,ids,no*sizeof(float));
          }
          free(scores); free(ids); return status;
        }
    case 26: /* Concrete publisher BF16 / FP8 / FP4 activation roundtrips.
                a=0 BF16; 1 window FP8; 2 compressed FP4; 3 index FP4;
                4 FP8 activation (dequantized FP32, not BF16). */
        if(no!=nx || a<0 || a>4) return 1;
        if(a==0) { for(i=0;i<nx;++i) { out[i]=publisher_bf16(x[i]); if(!isfinite(out[i])) return 3; } return 0; }
        { int group=a==2?16:32,fp4=a==2 || a==3;
          if(nx%(size_t)group) return 1;
          for(i=0;i<nx;i+=(size_t)group) { int j; float maximum=0,scale;
              for(j=0;j<group;++j) if(fabsf(x[i+j])>maximum) maximum=fabsf(x[i+j]);
              if(a==2) scale=(float)publisher_round(fmaxf(maximum,6*ldexpf(1,-9))/6,0);
              else { float raw=fmaxf(maximum,fp4?6*ldexpf(1,-126):1e-4f)/(fp4?6:448);
                  scale=ldexpf(1,(int)ceil(log2((double)raw))); }
              if(!isfinite(scale) || scale<=0) return 3;
              for(j=0;j<group;++j) {
                  float normalized=x[i+j]/scale;
                  float value=(float)publisher_round(normalized,fp4)*scale;
                  if(a!=4) value=publisher_bf16(value);
                  if(!isfinite(value)) return 3; out[i+j]=value;
              }
          }
        } return 0;
    default: return 1;
    }
}

typedef struct {
    uint32_t order,width,rows,reserved;
    uint64_t seed,multiplier[4];
} ExecutionEngramHead;
typedef struct { char prefix; ExecutionEngramHead value; } HeadAlignment;

/* History is chronological, at most last three tokens plus current token.
   This matches seed XOR uint64(token*multiplier[j]), modulo rows, with pad
   for unavailable history. Table reads belong exclusively to the provider. */
API int oa_v41_execution_engram_hash(const ExecutionEngramHead *heads,size_t nh,
    const uint32_t *history,size_t nt,uint32_t vocabulary,uint32_t pad,
    uint32_t *out,size_t no) {
    size_t h,j; uint32_t staged[24],orders=0;
    uintptr_t hp=(uintptr_t)heads,tp=(uintptr_t)history,op=(uintptr_t)out;
    if(sizeof(size_t)<8 || !heads || !history || !out || !nh || nh>24 || !nt || nt>4 ||
       no!=nh || !vocabulary || pad>=vocabulary ||
       hp%offsetof(HeadAlignment,value) || tp%sizeof(uint32_t) || op%sizeof(uint32_t) ||
       nh*sizeof(*heads)>UINTPTR_MAX-hp || nt*sizeof(uint32_t)>UINTPTR_MAX-tp ||
       no*sizeof(uint32_t)>UINTPTR_MAX-op ||
       overlap(out,no*sizeof(uint32_t),heads,nh*sizeof(*heads)) ||
       overlap(out,no*sizeof(uint32_t),history,nt*sizeof(uint32_t))) return 1;
    for(j=0;j<nt;++j) if(history[j]>=vocabulary) return 1;
    for(h=0;h<nh;++h) {
        const ExecutionEngramHead *a=heads+h;
        uint64_t hash=a->seed;
        if(a->reserved || a->order<2 || a->order>4 || !a->width || a->width>16384 || !a->rows) return 1;
        orders|=1u<<a->order;
        for(j=0;j<a->order;++j) {
            size_t back=a->order-1-j;
            uint32_t token=back<nt?history[nt-1-back]:pad;
            hash^=(uint64_t)token*a->multiplier[j];
        }
        staged[h]=(uint32_t)(hash%a->rows);
    }
    if(orders!=28) return 1;
    memcpy(out,staged,nh*sizeof(uint32_t));
    return 0;
}
