/* Original scalar execution of the GLM5-next text equations. Private ABI v3.
 * Token-major evaluation is intentional: prefill and decode share one state
 * transition, independent of chunk-parallel implementations in other engines.
 */
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

typedef struct { float *p; uint64_t n; } Buf;
typedef int (*ExpertCall)(void *,int32_t,int32_t,const float *,float *);
enum { H,V,L,R,IT,M,EM,E,SE,TOP,N,QR,KR,KD,VD,IN,ID,IK,POOL,LN,LD,CONV,CAP,NORMTOP,GROUP,GROUPTOP };
enum { EPS,HEPS,LIMIT,ROUTE,LOWER,SAFE };
#define SLOT 42
/* Workspace is derived from admitted geometry, not a tiny-model constant. */
static size_t workspace(const int *c) {
    size_t width=(size_t)c[LN]*c[LD], h=c[H], r=c[R];
    size_t kda_size=8*width+c[LD]+c[LN];
    size_t dsa_size=(size_t)c[QR]+c[KR]+(size_t)c[N]*c[KD]+
        (size_t)c[N]*(c[KD]+c[VD])+(size_t)c[IN]*c[ID]+c[IN]+
        2*(size_t)c[CAP]+c[ID]+c[POOL]+(size_t)c[N]*c[VD];
    size_t feed_size=2*(size_t)c[E]+h+2*(size_t)c[SE]*c[EM];
    size_t scratch=kda_size>dsa_size?kda_size:dsa_size;
    if(feed_size>scratch) scratch=feed_size;
    if(2*(size_t)c[M]>scratch) scratch=2*(size_t)c[M];
    return 3*r*h+2*h+r+r*r+(r+2)*r+scratch;
}

static int valid(Buf b, uint64_t n) {
    uint64_t i;
    if (!b.p || b.n != n) return 0;
    for (i=0;i<n;i++) if (!isfinite(b.p[i])) return 0;
    return 1;
}
static float sig(float x) {
    if (x >= 0) return 1.0f/(1.0f+expf(-x));
    { float a=expf(x); return a/(1.0f+a); }
}
static float act(float x) { return x*sig(x); }
static void linear(const float *w, const float *x, float *y, int rows, int cols) {
    int i,j;
    for (i=0;i<rows;i++) {
        float s=0;
        for (j=0;j<cols;j++) s += w[(size_t)i*cols+j]*x[j];
        y[i]=s;
    }
}
static void rms(const float *x, const float *w, float *y, int n, float eps) {
    int i; float s=0;
    for (i=0;i<n;i++) s+=x[i]*x[i];
    if(!isfinite(s)) { for(i=0;i<n;i++) y[i]=NAN; return; }
    s=1.0f/sqrtf(s/(float)n+eps);
    for (i=0;i<n;i++) y[i]=x[i]*s*(w?w[i]:1.0f);
}
static void softmax(float *x, int n) {
    int i; float hi=x[0],s=0;
    for(i=0;i<n;i++) if(!isfinite(x[i])) {
        for(i=0;i<n;i++) x[i]=NAN;
        return;
    }
    for(i=1;i<n;i++) if(x[i]>hi) hi=x[i];
    for(i=0;i<n;i++) { x[i]=expf(x[i]-hi); s+=x[i]; }
    for(i=0;i<n;i++) x[i]/=s;
}
static float *take(float *mem, size_t *offset, size_t n, size_t capacity) {
    float *p;
    if (*offset>capacity || n>capacity-*offset) return NULL;
    p=mem+*offset; *offset+=n; return p;
}

static void collapse(const int *c,const float *f,const Buf *w,int base,
                     const float *streams,float *x,float *post,float *mix,float *temp) {
    int r=c[R],h=c[H],i,j,d,k; float eps=f[HEPS];
    rms(streams,NULL,temp,r*h,f[EPS]);
    linear(w[base].p,temp,temp+r*h,(r+2)*r,r*h);
    for(i=0;i<r;i++) {
        float pre=sig(temp[r*h+i]*w[base+2].p[0]+w[base+1].p[i])+eps;
        post[i]=2*sig(temp[r*h+r+i]*w[base+2].p[1]+w[base+1].p[r+i]);
        for(d=0;d<h;d++) {
            if(i==0) x[d]=0;
            x[d]+=pre*streams[i*h+d];
        }
        for(j=0;j<r;j++) mix[i*r+j]=temp[r*h+2*r+i*r+j]*w[base+2].p[2]+w[base+1].p[2*r+i*r+j];
        softmax(mix+i*r,r);
        for(j=0;j<r;j++) mix[i*r+j]+=eps;
    }
    for(k=0;k<c[IT];k++) {
        if(k>0) for(i=0;i<r;i++) {
            float s=eps;
            for(j=0;j<r;j++) s+=mix[i*r+j];
            for(j=0;j<r;j++) mix[i*r+j]/=s;
        }
        for(j=0;j<r;j++) {
            float s=eps;
            for(i=0;i<r;i++) s+=mix[i*r+j];
            for(i=0;i<r;i++) mix[i*r+j]/=s;
        }
    }
}
static void expand(const int *c,const float *old,const float *value,
                   const float *post,const float *mix,float *out) {
    int i,j,d;
    for(j=0;j<c[R];j++) for(d=0;d<c[H];d++) {
        float s=post[j]*value[d];
        for(i=0;i<c[R];i++) s+=mix[i*c[R]+j]*old[i*c[H]+d];
        out[j*c[H]+d]=s;
    }
}

static int kda(const int *c,const float *f,const Buf *w,const float *x,
               Buf history,Buf memory,float *out,float *mem) {
    int h=c[H],d=c[LD],n=c[LN],width=d*n,k=c[CONV],a,i,j,t;
    float *qkv=mem,*z=qkv+3*width,*fg=z+d,*gate=fg+width,*beta=gate+width;
    float *core=beta+n,*delta=core+width;
    linear(w[8].p,x,qkv,width,h); linear(w[9].p,x,qkv+width,width,h);
    linear(w[10].p,x,qkv+2*width,width,h);
    for(a=0;a<3*width;a++) {
        float sum=0; float *hist=history.p+(size_t)a*k;
        for(t=0;t<k-1;t++) hist[t]=hist[t+1];
        hist[k-1]=qkv[a];
        for(t=0;t<k;t++) sum+=hist[t]*w[11].p[a*k+t];
        qkv[a]=act(sum);
    }
    linear(w[12].p,x,z,d,h); linear(w[13].p,z,fg,width,d);
    linear(w[17].p,x,z,d,h); linear(w[18].p,z,gate,width,d);
    linear(w[16].p,x,beta,n,h);
    for(a=0;a<n;a++) {
        float qs=1e-6f,ks=1e-6f,b=sig(beta[a]);
        float *q=qkv+a*d,*key=qkv+width+a*d,*v=qkv+2*width+a*d;
        float *s=memory.p+(size_t)a*d*d;
        for(i=0;i<d;i++) { qs+=q[i]*q[i]; ks+=key[i]*key[i]; }
        qs=sqrtf(qs); ks=sqrtf(ks);
        for(i=0;i<d;i++) { q[i]=q[i]/qs/sqrtf((float)d); key[i]/=ks; }
        for(i=0;i<d;i++) {
            float g=fg[a*d+i]+w[14].p[a*d+i],rate=expf(w[15].p[a]);
            float logdec=f[SAFE]?f[LOWER]*sig(rate*g):-rate*(g>20?g:log1pf(expf(g)));
            float decay=expf(logdec);
            for(j=0;j<d;j++) s[i*d+j]*=decay;
        }
        for(j=0;j<d;j++) {
            float prediction=0;
            for(i=0;i<d;i++) prediction+=key[i]*s[i*d+j];
            delta[j]=b*(v[j]-prediction);
        }
        for(i=0;i<d;i++) for(j=0;j<d;j++) s[i*d+j]+=key[i]*delta[j];
        for(j=0;j<d;j++) {
            float sum=0;
            for(i=0;i<d;i++) sum+=q[i]*s[i*d+j];
            core[a*d+j]=sum;
        }
        rms(core+a*d,w[19].p,core+a*d,d,f[EPS]);
        for(j=0;j<d;j++) core[a*d+j]*=sig(gate[a*d+j]);
    }
    linear(w[20].p,core,out,h,width);
    return 0;
}

/* Boundary ties are rejected instead of assuming a framework-specific topk
 * ordering. Ties fully within the selected set do not affect the mask/sum. */
static int choose(const float *scores,int n,int k,unsigned char *selected) {
    int i,j,best=-1; float boundary=0;
    if(k<0 || k>n) return 3;
    for(i=0;i<n;i++) if(!isfinite(scores[i])) return 5;
    memset(selected,0,(size_t)n);
    for(j=0;j<k;j++) {
        best=-1;
        for(i=0;i<n;i++) if(!selected[i] && (best<0 || scores[i]>scores[best])) best=i;
        if(best<0) return 3;
        selected[best]=1; boundary=scores[best];
    }
    if(k>0 && k<n) for(i=0;i<n;i++) if(!selected[i] && scores[i]==boundary) return 4;
    return 0;
}
static int dsa(const int *c,const float *f,const Buf *w,const float *x,int pos,int shared,
               Buf kv,Buf index,unsigned char *mask,float *out,float *mem) {
    int n=c[N],d=c[KD],v=c[VD],id=c[ID],in=c[IN],pool=c[POOL];
    int t,a,j,z,p,pc=(pos+1)/pool,status;
    float *qr=mem,*latent=qr+c[QR],*query=latent+c[KR],*expanded=query+n*d;
    float *iq=expanded+n*(d+v),*iw=iq+in*id,*scores=iw+in;
    float *pk=scores+c[CAP],*tmp=pk+id,*att=tmp+pool,*core=att+c[CAP];
    unsigned char picked[65536];
    linear(w[8].p,x,qr,c[QR],c[H]); rms(qr,w[9].p,qr,c[QR],f[EPS]);
    linear(w[10].p,qr,query,n*d,c[QR]);
    linear(w[11].p,x,latent,c[KR],c[H]); rms(latent,w[12].p,latent,c[KR],f[EPS]);
    linear(w[13].p,latent,expanded,n*(d+v),c[KR]);
    memcpy(kv.p+(size_t)pos*n*(d+v),expanded,(size_t)n*(d+v)*sizeof(float));
    if(!shared) {
        float mean=0,var=0,*key=index.p+(size_t)pos*2*id;
        linear(w[15].p,qr,iq,in*id,c[QR]);
        linear(w[16].p,x,key,id,c[H]);
        for(j=0;j<id;j++) mean+=key[j];
        mean/=(float)id;
        for(j=0;j<id;j++) var+=(key[j]-mean)*(key[j]-mean);
        var=1/sqrtf(var/(float)id+1e-6f);
        for(j=0;j<id;j++) key[j]=(key[j]-mean)*var*w[17].p[j]+w[18].p[j];
        linear(w[21].p,x,key+id,id,c[H]);
        linear(w[19].p,x,iw,in,c[H]);
        for(p=0;p<pc;p++) {
            for(j=0;j<id;j++) {
                for(z=0;z<pool;z++) tmp[z]=index.p[((size_t)p*pool+z)*2*id+id+j]+w[20].p[z*id+j];
                softmax(tmp,pool); pk[j]=0;
                for(z=0;z<pool;z++) pk[j]+=tmp[z]*index.p[((size_t)p*pool+z)*2*id+j];
            }
            scores[p]=0;
            for(a=0;a<in;a++) {
                float dot=0;
                for(j=0;j<id;j++) dot+=iq[a*id+j]*pk[j];
                dot/=sqrtf((float)id);
                if(dot>0) scores[p]+=dot*iw[a]/sqrtf((float)in);
            }
        }
        status=choose(scores,pc,pc<c[IK]/pool?pc:c[IK]/pool,picked);
        if(status) return status;
        memset(mask,0,(size_t)c[CAP]);
        for(p=0;p<pc;p++) if(picked[p]) for(z=0;z<pool;z++) mask[p*pool+z]=1;
        for(t=pc*pool;t<=pos;t++) mask[t]=1;
    }
    for(a=0;a<n;a++) {
        for(t=0;t<=pos;t++) {
            float dot=0;
            if(mask[t]) for(j=0;j<d;j++) dot+=query[a*d+j]*kv.p[((size_t)t*n+a)*(d+v)+j];
            att[t]=mask[t]?dot/sqrtf((float)d):-FLT_MAX;
        }
        softmax(att,pos+1);
        for(j=0;j<v;j++) {
            float sum=0;
            for(t=0;t<=pos;t++) sum+=att[t]*kv.p[((size_t)t*n+a)*(d+v)+d+j];
            core[a*v+j]=sum;
        }
    }
    linear(w[14].p,core,out,c[H],n*v);
    return 0;
}

static void mlp(const float *x,const float *wg,const float *wu,const float *wd,
                int h,int m,float limit,float *out,float *tmp) {
    int j;
    linear(wg,x,tmp,m,h); linear(wu,x,tmp+m,m,h);
    for(j=0;j<m;j++) {
        float g=tmp[j]<limit?tmp[j]:limit,up=tmp[m+j];
        if(!isfinite(tmp[j]) || !isfinite(up)) {
            int i;
            for(i=0;i<h;i++) out[i]=NAN;
            return;
        }
        up=up>limit?limit:up<-limit?-limit:up;
        tmp[j]=act(g)*up;
    }
    linear(wd,tmp,out,h,m);
}
static int feed(const int *c,const float *f,const Buf *w,const float *x,int sparse,float *out,float *mem,
                int layer,ExpertCall expert,void *user,float *observed_logits,float *observed_weights) {
    int h=c[H],m=c[EM],e=c[E],a,j,status;
    float *scores=mem,*choice=scores+e,*value=choice+e,*tmp=value+h; float sum=1e-20f;
    unsigned char picked[4096],groups[4096];
    float group_scores[4096];
    if(!sparse) { mlp(x,w[24].p,w[25].p,w[26].p,h,c[M],f[LIMIT],out,mem); return 0; }
    linear(w[24].p,x,scores,e,h);
    if(observed_logits) memcpy(observed_logits,scores,(size_t)e*sizeof(float));
    for(a=0;a<e;a++) {
        if(!isfinite(scores[a])) return 5;
        scores[a]=sig(scores[a]); choice[a]=scores[a]+w[25].p[a];
    }
    /* Only the sum matters within each group, so equal top-two values are
       unambiguous. Group and expert selection boundaries remain fail-closed. */
    for(a=0;a<c[GROUP];a++) {
        int width=e/c[GROUP],start=a*width;
        float first=-FLT_MAX,second=-FLT_MAX;
        for(j=start;j<start+width;j++) {
            float score=choice[j];
            if(!isfinite(score)) return 5;
            if(score>first) { second=first; first=score; }
            else if(score>second) second=score;
        }
        group_scores[a]=first+second;
    }
    status=choose(group_scores,c[GROUP],c[GROUPTOP],groups); if(status) return status;
    for(a=0;a<e;a++) if(!groups[a/(e/c[GROUP])]) choice[a]=-FLT_MAX;
    status=choose(choice,e,c[TOP],picked); if(status) return status;
    for(a=0;a<e;a++) if(picked[a]) sum+=scores[a];
    mlp(x,w[28].p,w[29].p,w[30].p,h,c[SE]*m,f[LIMIT],out,tmp);
    for(a=0;a<e;a++) if(picked[a]) {
        float weight=scores[a]*f[ROUTE]/(c[NORMTOP]?sum:1.0f);
        if(observed_weights) observed_weights[a]=weight;
        if(expert) {
            if(expert(user,layer,a,x,value)) return 6;
            for(j=0;j<h;j++) if(!isfinite(value[j])) return 5;
        } else {
            const float *wg=w[26].p+(size_t)a*2*m*h;
            mlp(x,wg,wg+(size_t)m*h,w[27].p+(size_t)a*h*m,h,m,f[LIMIT],value,tmp);
        }
        for(j=0;j<h;j++) out[j]+=weight*value[j];
    }
    return 0;
}

static int geometry(const int *c,const float *f) {
    int i;
    for(i=0;i<23;i++) if(c[i]<1) return 0;
    if(c[H]>16384 || c[V]>1048576 || c[L]>128 || c[R]>8 || c[IT]>64 || c[CAP]>65536 ||
       c[M]>65536 || c[EM]>65536 || c[E]>4096 || c[SE]>32 || c[TOP]>c[E] || c[E]<2 ||
       c[N]>256 || c[QR]>8192 || c[KR]>8192 || c[KD]>512 || c[VD]>512 ||
       c[IN]>256 || c[ID]>512 || c[IK]>65536 || c[POOL]>32 || c[IK]%c[POOL] ||
       c[LN]>256 || c[LD]>512 || c[CONV]>32 ||
       (c[NORMTOP]!=0 && c[NORMTOP]!=1)) return 0;
    if(c[GROUP]<1 || c[GROUP]>4096 || c[GROUPTOP]<1 || c[GROUPTOP]>c[GROUP] ||
       c[E]%c[GROUP] || c[E]/c[GROUP]<2 || c[TOP]>c[GROUPTOP]*(c[E]/c[GROUP])) return 0;
    for(i=0;i<6;i++) if(!isfinite(f[i])) return 0;
    return f[EPS]>0 && f[HEPS]>0 && f[LIMIT]>0 && f[ROUTE]>0 &&
           (f[SAFE]==0 || (f[SAFE]==1 && f[LOWER]<0));
}
static int layer_valid(const int *c,const Buf *w,int linear_kind,int sparse,int shared,int streamed) {
    uint64_t expect[SLOT]={0}; int i,h=c[H],r=c[R],width=c[LN]*c[LD],d=c[LD];
    expect[0]=expect[3]=(uint64_t)(r+2)*r*r*h;
    expect[1]=expect[4]=(uint64_t)(r+2)*r; expect[2]=expect[5]=3;
    expect[6]=expect[7]=(uint64_t)h;
    if(linear_kind) {
        expect[8]=expect[9]=expect[10]=(uint64_t)width*h;
        expect[11]=(uint64_t)3*width*c[CONV]; expect[12]=expect[17]=(uint64_t)d*h;
        expect[13]=expect[18]=(uint64_t)width*d; expect[14]=(uint64_t)width;
        expect[15]=(uint64_t)c[LN]; expect[16]=(uint64_t)c[LN]*h;
        expect[19]=(uint64_t)d; expect[20]=(uint64_t)h*width;
    } else {
        expect[8]=(uint64_t)c[QR]*h; expect[9]=(uint64_t)c[QR]; expect[10]=(uint64_t)c[N]*c[KD]*c[QR];
        expect[11]=(uint64_t)c[KR]*h; expect[12]=(uint64_t)c[KR];
        expect[13]=(uint64_t)c[N]*(c[KD]+c[VD])*c[KR]; expect[14]=(uint64_t)h*c[N]*c[VD];
        if(!shared) {
            expect[15]=(uint64_t)c[IN]*c[ID]*c[QR]; expect[16]=expect[21]=(uint64_t)c[ID]*h;
            expect[17]=expect[18]=(uint64_t)c[ID]; expect[19]=(uint64_t)c[IN]*h;
            expect[20]=(uint64_t)c[POOL]*c[ID];
        }
    }
    if(!sparse) {
        expect[24]=expect[25]=expect[26]=(uint64_t)c[M]*h;
    } else {
        expect[24]=(uint64_t)c[E]*h; expect[25]=(uint64_t)c[E];
        if(!streamed) {
            expect[26]=(uint64_t)c[E]*2*c[EM]*h; expect[27]=(uint64_t)c[E]*h*c[EM];
        }
        expect[28]=expect[29]=expect[30]=(uint64_t)c[SE]*c[EM]*h;
    }
    for(i=0;i<SLOT;i++) {
        if(expect[i]) { if(!valid(w[i],expect[i])) return 0; }
        else if(w[i].n || w[i].p) return 0;
    }
    return 1;
}

API uint32_t oa_glm5_next_version(void) { return 3; }
API uint64_t oa_glm5_next_workspace_floats(const int32_t *c,const float *f) {
    return c && f && geometry(c,f)?(uint64_t)workspace(c):0;
}
static int glm5_next_step(const int32_t *c,const float *f,const int32_t *schedule,
                         const Buf *weights,Buf embed,Buf norm,Buf head,Buf *cache,
                         int32_t pos,int64_t token,Buf hidden,Buf logits,ExpertCall expert,void *user,Buf incoming,
                         Buf routing_logits,Buf routing_weights,int begin,int end,int range_wire) {
    float *mem,*streams,*next,*x,*value,*post,*mix,*temp,*scratch;
    size_t offset=0,capacity; int layer,status=0,i,j;
    unsigned char mask[65536];
    if(!c || !f || !schedule || !weights || !cache || !geometry(c,f)) return 1;
    if(end<0)end=c[L];
    if(begin<0 || begin>=end || end>c[L] || (range_wire!=0 && range_wire!=1))return 1;
    if(routing_logits.p || routing_weights.p || routing_logits.n || routing_weights.n) {
        uint64_t extent=(uint64_t)c[L]*c[E];
        if(!routing_logits.p || !routing_weights.p || routing_logits.n!=extent || routing_weights.n!=extent) return 1;
        memset(routing_logits.p,0,(size_t)extent*sizeof(float));
        memset(routing_weights.p,0,(size_t)extent*sizeof(float));
    }
    if(pos<0 || pos>=c[CAP] || token<0 || token>=c[V]) return 1;
    if(!valid(embed,(uint64_t)c[V]*c[H]) || !valid(norm,(uint64_t)c[H]) ||
       !hidden.p || hidden.n!=(range_wire?(uint64_t)c[R]*c[H]+c[CAP]:(uint64_t)c[H])) return 1;
    if(incoming.p) {
        if(!valid(incoming,range_wire?(uint64_t)c[R]*c[H]+c[CAP]:(uint64_t)c[H]))return 1;
        if(range_wire)for(i=0;i<c[CAP];i++) {
            float selected=incoming.p[(size_t)c[R]*c[H]+i];
            if((selected!=0 && selected!=1) || (i>pos && selected!=0))return 1;
        }
    }
    else if(incoming.n || range_wire) return 1;
    if(range_wire && (head.p || head.n || logits.p || logits.n))return 1;
    if(head.p) {
        if(!valid(head,(uint64_t)c[V]*c[H]) || !logits.p || logits.n!=(uint64_t)c[V]) return 1;
    } else if(head.n || logits.n || logits.p) return 1;
    for(layer=begin;layer<end;layer++) {
        int kind=schedule[3*layer],sparse=schedule[3*layer+1],shared=schedule[3*layer+2];
        uint64_t a,b;
        if((kind!=0 && kind!=1) || (sparse!=0 && sparse!=1) || (shared!=0 && shared!=1)) return 1;
        if(!kind && shared && (layer==0 || schedule[3*(layer-1)] || schedule[3*(layer-1)+2])) return 1;
        if(!layer_valid(c,weights+layer*SLOT,kind,sparse,shared,expert!=NULL)) return 1;
        a=kind?(uint64_t)3*c[LN]*c[LD]*c[CONV]:(uint64_t)c[CAP]*c[N]*(c[KD]+c[VD]);
        b=kind?(uint64_t)c[LN]*c[LD]*c[LD]:(uint64_t)c[CAP]*2*c[ID];
        if(!valid(cache[2*layer],a) || !valid(cache[2*layer+1],b)) return 1;
    }
    capacity=workspace(c);
    if(capacity>SIZE_MAX/sizeof(float)) return 2;
    mem=(float *)calloc(capacity,sizeof(float)); if(!mem) return 2;
    streams=take(mem,&offset,(size_t)c[R]*c[H],capacity); next=take(mem,&offset,(size_t)c[R]*c[H],capacity);
    x=take(mem,&offset,(size_t)c[H],capacity); value=take(mem,&offset,(size_t)c[H],capacity);
    post=take(mem,&offset,(size_t)c[R],capacity); mix=take(mem,&offset,(size_t)c[R]*c[R],capacity);
    temp=take(mem,&offset,(size_t)c[R]*c[H]+(size_t)(c[R]+2)*c[R],capacity);
    scratch=take(mem,&offset,capacity-offset,capacity);
    if(!streams || !next || !x || !value || !post || !mix || !temp || !scratch) { free(mem); return 2; }
    if(range_wire)memcpy(streams,incoming.p,(size_t)c[R]*c[H]*sizeof(float));
    else for(i=0;i<c[R];i++) memcpy(streams+i*c[H],incoming.p?incoming.p:embed.p+(size_t)token*c[H],(size_t)c[H]*sizeof(float));
    memset(mask,0,sizeof(mask));
    if(range_wire)for(i=0;i<c[CAP];i++)mask[i]=(unsigned char)incoming.p[(size_t)c[R]*c[H]+i];
    for(layer=begin;layer<end;layer++) {
        const Buf *w=weights+layer*SLOT;
        collapse(c,f,w,0,streams,x,post,mix,temp); rms(x,w[6].p,x,c[H],f[EPS]);
        if(schedule[3*layer]) status=kda(c,f,w,x,cache[2*layer],cache[2*layer+1],value,scratch);
        else status=dsa(c,f,w,x,pos,schedule[3*layer+2],cache[2*layer],cache[2*layer+1],mask,value,scratch);
        if(status) break;
        expand(c,streams,value,post,mix,next);
        collapse(c,f,w,3,next,x,post,mix,temp); rms(x,w[7].p,x,c[H],f[EPS]);
        status=feed(c,f,w,x,schedule[3*layer+1],value,scratch,layer,expert,user,
            routing_logits.p?routing_logits.p+(size_t)layer*c[E]:NULL,
            routing_weights.p?routing_weights.p+(size_t)layer*c[E]:NULL); if(status) break;
        expand(c,next,value,post,mix,streams);
        for(j=0;j<c[R]*c[H];j++) if(!isfinite(streams[j])) { status=5; break; }
        if(status) break;
    }
    if(!status) {
        if(range_wire) {
            memcpy(hidden.p,streams,(size_t)c[R]*c[H]*sizeof(float));
            for(i=0;i<c[CAP];i++)hidden.p[(size_t)c[R]*c[H]+i]=(float)mask[i];
        } else {
        for(j=0;j<c[H];j++) {
            float sum=0;
            for(i=0;i<c[R];i++) sum+=streams[i*c[H]+j];
            x[j]=sum/(float)c[R];
        }
        rms(x,norm.p,hidden.p,c[H],f[EPS]);
        if(head.p) linear(head.p,hidden.p,logits.p,c[V],c[H]);
        }
        if(!valid(hidden,hidden.n) || (head.p && !valid(logits,logits.n))) status=5;
        for(layer=2*begin;layer<2*end;layer++) if(!valid(cache[layer],cache[layer].n)) status=5;
    }
    free(mem); return status;
}

API int oa_glm5_next_step(const int32_t *c,const float *f,const int32_t *schedule,
    const Buf *weights,Buf embed,Buf norm,Buf head,Buf *cache,
    int32_t pos,int64_t token,Buf hidden,Buf logits) {
    Buf empty={NULL,0};
    return glm5_next_step(c,f,schedule,weights,embed,norm,head,cache,pos,token,hidden,logits,NULL,NULL,empty,empty,empty,0,-1,0);
}
API uint32_t oa_glm5_next_streamed_version(void) { return 1; }
API int oa_glm5_next_step_streamed(const int32_t *c,const float *f,const int32_t *schedule,
    const Buf *weights,Buf embed,Buf norm,Buf head,Buf *cache,
    int32_t pos,int64_t token,Buf hidden,Buf logits,ExpertCall expert,void *user) {
    if(!expert) return 1;
    Buf empty={NULL,0};
    return glm5_next_step(c,f,schedule,weights,embed,norm,head,cache,pos,token,hidden,logits,expert,user,empty,empty,empty,0,-1,0);
}
API uint32_t oa_glm5_next_embeddings_version(void) {return 1;}
API int oa_glm5_next_step_embeddings(const int32_t *c,const float *f,const int32_t *schedule,
    const Buf *weights,Buf embed,Buf norm,Buf head,Buf *cache,
    int32_t pos,int64_t token,Buf hidden,Buf logits,Buf incoming,ExpertCall expert,void *user) {
    if(!incoming.p) return 1;
    Buf empty={NULL,0};
    return glm5_next_step(c,f,schedule,weights,embed,norm,head,cache,pos,token,hidden,logits,expert,user,incoming,empty,empty,0,-1,0);
}
API uint32_t oa_glm5_next_observation_version(void) {return 1;}
API uint32_t oa_glm5_next_range_version(void) {return 1;}
API int oa_glm5_next_step_range(const int32_t *c,const float *f,const int32_t *schedule,
    const Buf *weights,Buf embed,Buf norm,Buf *cache,int32_t pos,int64_t token,
    int32_t begin,int32_t end,Buf incoming,Buf outgoing) {
    Buf empty={NULL,0};
    if(begin<0||end<0)return 1;
    return glm5_next_step(c,f,schedule,weights,embed,norm,empty,cache,pos,token,
        outgoing,empty,NULL,NULL,incoming,empty,empty,begin,end,1);
}
API int oa_glm5_next_step_range_streamed(const int32_t *c,const float *f,const int32_t *schedule,
    const Buf *weights,Buf embed,Buf norm,Buf *cache,int32_t pos,int64_t token,
    int32_t begin,int32_t end,Buf incoming,Buf outgoing,ExpertCall expert,void *user) {
    Buf empty={NULL,0};
    if(begin<0||end<0||!expert)return 1;
    return glm5_next_step(c,f,schedule,weights,embed,norm,empty,cache,pos,token,
        outgoing,empty,expert,user,incoming,empty,empty,begin,end,1);
}
API int oa_glm5_next_step_observed(const int32_t *c,const float *f,const int32_t *schedule,
    const Buf *weights,Buf embed,Buf norm,Buf head,Buf *cache,
    int32_t pos,int64_t token,Buf hidden,Buf logits,Buf incoming,
    ExpertCall expert,void *user,Buf routing_logits,Buf routing_weights) {
    if(!routing_logits.p || !routing_weights.p) return 1;
    return glm5_next_step(c,f,schedule,weights,embed,norm,head,cache,pos,token,hidden,logits,
        expert,user,incoming,routing_logits,routing_weights,0,-1,0);
}
