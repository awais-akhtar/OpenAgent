/* Original token-stream Mixtral decoder. Private counted ABI, staged KV only. */
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
typedef struct { float *data; uint64_t count; } Tensor;
typedef struct {
    uint64_t hidden,width,layers,heads,kv_heads,dim,experts,topk,vocab,capacity,window;
    float epsilon,theta;
} Config;
typedef struct { Tensor weights[9],keys,values; } Layer;
typedef int (*ExpertFFN)(void *,uint64_t,uint64_t,const float *,float *,uint64_t,uint64_t);
API uint64_t oa_mixtral_decoder_version(void) { return 1; }
API uint64_t oa_mixtral_streamed_version(void) { return 1; }
API uint64_t oa_mixtral_input_version(void) { return 1; }
API uint64_t oa_mixtral_scaled_rope_version(void) { return 1; }

static int finite_values(const float *a, uint64_t n) {
    uint64_t i;
    if (!a) return 0;
    for (i=0;i<n;++i) if (!isfinite(a[i])) return 0;
    return 1;
}
static int checked(Tensor a,uint64_t count,int finite) {
    return a.count==count && a.data && (!finite || finite_values(a.data,count));
}
static int store(float *out,double value) {
    if (!isfinite(value) || fabs(value)>FLT_MAX) return 0;
    *out=(float)value; return 1;
}
static int mat(const float *w,const float *x,float *y,uint64_t rows,uint64_t cols) {
    uint64_t i,j;
    for (i=0;i<rows;++i) {
        double sum=0;
        for (j=0;j<cols;++j) sum+=(double)w[i*cols+j]*x[j];
        if (!store(y+i,sum)) return 0;
    }
    return 1;
}
static int norm(const float *x,const float *w,float *out,uint64_t n,float eps) {
    uint64_t i;
    double variance=0,scale;
    for (i=0;i<n;++i) {
        double square=(double)x[i]*x[i];
        if (square>FLT_MAX) return 0;
        variance+=square;
        if (variance>FLT_MAX) return 0;
    }
    variance/=(double)n;
    scale=1/sqrt(variance+eps);
    for (i=0;i<n;++i) if (!store(out+i,x[i]*scale*w[i])) return 0;
    return 1;
}
static int add(float *x,const float *y,uint64_t n) {
    uint64_t i;
    for (i=0;i<n;++i) if (!store(x+i,(double)x[i]+y[i])) return 0;
    return 1;
}
static int rotate(float *x,uint64_t heads,uint64_t dim,uint64_t pos,float theta,double factor) {
    uint64_t h,i,half=dim/2;
    for (h=0;h<heads;++h) for (i=0;i<half;++i) {
        float frequency=(float)(1/pow((double)theta,(double)(2*i)/(double)dim));
        frequency=(float)((double)frequency/factor);
        float angle=(float)pos*frequency,cs=cosf(angle),sn=sinf(angle);
        float a=x[h*dim+i],b=x[h*dim+i+half];
        if (!store(x+h*dim+i,(double)a*cs-(double)b*sn) ||
            !store(x+h*dim+i+half,(double)b*cs+(double)a*sn)) return 0;
    }
    return 1;
}

static int mixtral_run(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor hidden,Tensor logits,Tensor routers,Tensor routes,ExpertFFN expert_fn,void *expert_user,
    const uint8_t *mask,uint64_t mask_count,const int64_t *positions,uint64_t position_count,
    Tensor incoming,int range_mode,double rope_factor) {
    uint64_t h,m,qw,kw,e,b,t,l,j,a,head,slot,pos,first,n,chosen;
    uint64_t expected[9],*selected;
    float *work,*x,*z,*att,*moe,*temp,*q,*context,*gu,*act,*prob,*route;
    double *score;
    int status=5;
    if (!c || !layers || (!range_mode && (!global || !ids)) || !batch || batch>16 || !tokens || tokens>4096 ||
        id_count!=batch*tokens || !c->hidden || c->hidden>16384 || !c->width || c->width>65536 ||
        !c->layers || c->layers>128 || !c->heads || c->heads>128 || !c->kv_heads ||
        c->kv_heads>128 || c->heads%c->kv_heads || !c->dim || c->dim>256 || c->dim%2 ||
        !c->experts || c->experts>4096 || !c->topk || c->topk>c->experts ||
        !c->vocab || c->vocab>1048576 || !c->capacity || c->capacity>1048576 ||
        start>c->capacity || tokens>c->capacity-start || c->window>1048576 ||
        !isfinite(c->epsilon) || c->epsilon<1e-12f || c->epsilon>1 ||
        !isfinite(c->theta) || c->theta<1 || c->theta>1e12f ||
        !isfinite(rope_factor) || rope_factor<1 || rope_factor>1048576) return 1;
    if ((mask && mask_count!=batch*(start+tokens)) || (!mask && mask_count) ||
        (positions && position_count!=id_count) || (!positions && position_count)) return 1;
    for (j=0;j<mask_count;++j) if (mask[j]>1) return 1;
    for (j=0;j<position_count;++j)
        if (positions[j]<0 || (uint64_t)positions[j]>=1048576) return 1;
    h=c->hidden; m=c->width; e=c->experts; qw=c->heads*c->dim; kw=c->kv_heads*c->dim;
    expected[0]=expected[1]=h; expected[2]=qw*h;
    expected[3]=expected[4]=kw*h; expected[5]=h*qw; expected[6]=e*h;
    expected[7]=e*2*m*h; expected[8]=e*h*m;
    if ((!range_mode && (!checked(global[0],c->vocab*h,1) || !checked(global[1],h,1) ||
        !checked(global[2],c->vocab*h,1) || !checked(logits,batch*tokens*c->vocab,0))) ||
        (range_mode && (!checked(incoming,batch*tokens*h,1) || logits.data || logits.count)) ||
        !checked(hidden,batch*tokens*h,0) ||
        !checked(routers,batch*tokens*c->layers*e,0) ||
        !checked(routes,batch*tokens*c->layers*e,0)) return 1;
    for (l=0;l<c->layers;++l) {
        for (a=0;a<(expert_fn ? 7u : 9u);++a)
            if (!checked(layers[l].weights[a],expected[a],1)) return 1;
        if (expert_fn && (layers[l].weights[7].data || layers[l].weights[7].count ||
                          layers[l].weights[8].data || layers[l].weights[8].count)) return 1;
        if (!checked(layers[l].keys,batch*c->capacity*kw,1) ||
            !checked(layers[l].values,batch*c->capacity*kw,1)) return 1;
    }
    if (!range_mode) for (j=0;j<id_count;++j) if (ids[j]<0 || (uint64_t)ids[j]>=c->vocab) return 1;
    n=5*h+2*qw+3*m+2*e;
    if (n>SIZE_MAX/sizeof(float) || c->capacity>SIZE_MAX/sizeof(double) || e>SIZE_MAX/sizeof(uint64_t)) return 1;
    work=(float *)malloc((size_t)n*sizeof(float));
    score=(double *)malloc((size_t)c->capacity*sizeof(double));
    selected=(uint64_t *)malloc((size_t)e*sizeof(uint64_t));
    if (!work || !score || !selected) { free(work); free(score); free(selected); return 4; }
    x=work; z=x+h; att=z+h; moe=att+h; temp=moe+h; q=temp+h;
    context=q+qw; gu=context+qw; act=gu+2*m; prob=act+m; route=prob+e;
    for (b=0;b<batch;++b) for (t=0;t<tokens;++t) {
        uint64_t rotary_pos=positions ? (uint64_t)positions[b*tokens+t] : start+t;
        pos=start+t;
        memcpy(x,range_mode ? incoming.data+(b*tokens+t)*h :
            global[0].data+(uint64_t)ids[b*tokens+t]*h,(size_t)h*sizeof(float));
        for (l=0;l<c->layers;++l) {
            const Tensor *w=layers[l].weights;
            float *keys=layers[l].keys.data+b*c->capacity*kw;
            float *values=layers[l].values.data+b*c->capacity*kw;
            float *router=routers.data+((b*tokens+t)*c->layers+l)*e;
            float *gamma=routes.data+((b*tokens+t)*c->layers+l)*e;
            if (!norm(x,w[0].data,z,h,c->epsilon) || !mat(w[2].data,z,q,qw,h) ||
                !mat(w[3].data,z,keys+pos*kw,kw,h) || !mat(w[4].data,z,values+pos*kw,kw,h) ||
                !rotate(q,c->heads,c->dim,rotary_pos,c->theta,rope_factor) ||
                !rotate(keys+pos*kw,c->kv_heads,c->dim,rotary_pos,c->theta,rope_factor)) goto done;
            first=c->window && pos+1>c->window ? pos+1-c->window : 0;
            for (head=0;head<c->heads;++head) {
                uint64_t kh=head/(c->heads/c->kv_heads);
                double maximum=-DBL_MAX,total=0;
                for (slot=first;slot<=pos;++slot) {
                    double dot=0;
                    if (mask && !mask[b*(start+tokens)+slot]) {
                        score[slot]=-DBL_MAX; continue;
                    }
                    for (j=0;j<c->dim;++j) dot+=(double)q[head*c->dim+j]*keys[slot*kw+kh*c->dim+j];
                    if (!isfinite(dot/sqrt((double)c->dim))) goto done;
                    score[slot]=(double)(float)(dot/sqrt((double)c->dim));
                    if (score[slot]>maximum) maximum=score[slot];
                }
                for (slot=first;slot<=pos;++slot) {
                    score[slot]=(mask && !mask[b*(start+tokens)+slot]) ? 0 : exp(score[slot]-maximum);
                    total+=score[slot];
                }
                if (!total) {
                    memset(context+head*c->dim,0,(size_t)c->dim*sizeof(float));
                    continue;
                }
                for (slot=first;slot<=pos;++slot) score[slot]=(double)(float)(score[slot]/total);
                for (j=0;j<c->dim;++j) {
                    double sum=0;
                    for (slot=first;slot<=pos;++slot) sum+=score[slot]*values[slot*kw+kh*c->dim+j];
                    if (!store(context+head*c->dim+j,sum)) goto done;
                }
            }
            if (!mat(w[5].data,context,att,h,qw) || !add(x,att,h) ||
                !norm(x,w[1].data,z,h,c->epsilon) || !mat(w[6].data,z,router,e,h)) goto done;
            {
                double maximum=-DBL_MAX,total=0,selected_sum=0;
                for (j=0;j<e;++j) if (router[j]>maximum) maximum=router[j];
                for (j=0;j<e;++j) { prob[j]=(float)exp(router[j]-maximum); total+=prob[j]; }
                for (j=0;j<e;++j) prob[j]=(float)(prob[j]/total);
                memset(route,0,(size_t)e*sizeof(float));
                for (a=0;a<c->topk;++a) {
                    float best=-1;
                    chosen=e;
                    for (j=0;j<e;++j) if (!route[j] && prob[j]>best) { best=prob[j]; chosen=j; }
                    if (chosen==e) goto done;
                    selected[a]=chosen; route[chosen]=1; selected_sum+=prob[chosen];
                }
                /* Unspecified vendor tie order must not masquerade as parity. */
                if (c->topk<e) for (j=0;j<e;++j)
                    if (!route[j] && prob[j]==prob[selected[c->topk-1]]) { status=6; goto done; }
                for (j=0;j<e;++j) gamma[j]=route[j] ? (float)(prob[j]/selected_sum) : 0;
            }
            memset(moe,0,(size_t)h*sizeof(float));
            for (a=0;a<e;++a) if (route[a]) {
                if (expert_fn) {
                    if (expert_fn(expert_user,l,a,z,temp,h,m) || !finite_values(temp,h)) {
                        status=7; goto done;
                    }
                } else {
                    if (!mat(w[7].data+a*2*m*h,z,gu,2*m,h)) goto done;
                    for (j=0;j<m;++j) {
                        double g=gu[j],sigmoid=g>=0 ? 1/(1+exp(-g)) : exp(g)/(1+exp(g));
                        if (!store(act+j,g*sigmoid*gu[m+j])) goto done;
                    }
                    if (!mat(w[8].data+a*h*m,act,temp,h,m)) goto done;
                }
                for (j=0;j<h;++j) {
                    float weighted;
                    if (!store(&weighted,(double)temp[j]*gamma[a]) || !store(moe+j,(double)moe[j]+weighted)) goto done;
                }
            }
            if (!add(x,moe,h)) goto done;
        }
        if (range_mode) memcpy(hidden.data+(b*tokens+t)*h,x,(size_t)h*sizeof(float));
        else if (!norm(x,global[1].data,hidden.data+(b*tokens+t)*h,h,c->epsilon) ||
            !mat(global[2].data,hidden.data+(b*tokens+t)*h,logits.data+(b*tokens+t)*c->vocab,c->vocab,h)) goto done;
    }
    status=0;
done:
    free(work); free(score); free(selected); return status;
}

API int oa_mixtral_decoder_run(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor hidden,Tensor logits,Tensor routers,Tensor routes) {
    return mixtral_run(c,layers,global,ids,id_count,batch,tokens,start,
        hidden,logits,routers,routes,NULL,NULL,NULL,0,NULL,0,(Tensor){NULL,0},0,1);
}

API int oa_mixtral_decoder_run_streamed(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor hidden,Tensor logits,Tensor routers,Tensor routes,ExpertFFN expert_fn,void *expert_user) {
    if (!expert_fn) return 1;
    return mixtral_run(c,layers,global,ids,id_count,batch,tokens,start,
        hidden,logits,routers,routes,expert_fn,expert_user,NULL,0,NULL,0,(Tensor){NULL,0},0,1);
}

API int oa_mixtral_decoder_run_inputs(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor hidden,Tensor logits,Tensor routers,Tensor routes,
    const uint8_t *mask,uint64_t mask_count,const int64_t *positions,uint64_t position_count,
    ExpertFFN expert_fn,void *expert_user) {
    return mixtral_run(c,layers,global,ids,id_count,batch,tokens,start,
        hidden,logits,routers,routes,expert_fn,expert_user,mask,mask_count,positions,position_count,
        (Tensor){NULL,0},0,1);
}

API int oa_mixtral_decoder_run_scaled_rope(const Config *c,Layer *layers,const Tensor *global,
    const int64_t *ids,uint64_t id_count,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor hidden,Tensor logits,Tensor routers,Tensor routes,
    const uint8_t *mask,uint64_t mask_count,const int64_t *positions,uint64_t position_count,
    ExpertFFN expert_fn,void *expert_user,double factor) {
    return mixtral_run(c,layers,global,ids,id_count,batch,tokens,start,
        hidden,logits,routers,routes,expert_fn,expert_user,mask,mask_count,positions,position_count,
        (Tensor){NULL,0},0,factor);
}

/* Additive hidden-to-hidden range ABI. Caller supplies staged continuation
 * banks and commits them only after success; no embedding/head is required. */
API uint64_t oa_mixtral_range_version(void) { return 1; }
API uint64_t oa_mixtral_edge_version(void) { return 1; }
API int oa_mixtral_edge_embed(uint64_t hidden,uint64_t vocab,Tensor embedding,
    const int64_t *ids,uint64_t count,Tensor output) {
    uint64_t row;
    if (!hidden || hidden>16384 || !vocab || vocab>1048576 || !count || count>65536 ||
        !ids || !checked(embedding,vocab*hidden,1) || !checked(output,count*hidden,0)) return 1;
    for (row=0;row<count;++row) if (ids[row]<0 || (uint64_t)ids[row]>=vocab) return 1;
    for (row=0;row<count;++row)
        memcpy(output.data+row*hidden,embedding.data+(uint64_t)ids[row]*hidden,
            (size_t)hidden*sizeof(float));
    return 0;
}
API int oa_mixtral_edge_logits(uint64_t hidden,uint64_t vocab,float epsilon,
    Tensor final_norm,Tensor head,Tensor incoming,uint64_t count,Tensor normalized,Tensor logits) {
    uint64_t row;
    if (!hidden || hidden>16384 || !vocab || vocab>1048576 || !count || count>65536 ||
        !isfinite(epsilon) || epsilon<1e-12f || epsilon>1 ||
        !checked(final_norm,hidden,1) || !checked(head,vocab*hidden,1) ||
        !checked(incoming,count*hidden,1) || !checked(normalized,count*hidden,0) ||
        !checked(logits,count*vocab,0)) return 1;
    for (row=0;row<count;++row)
        if (!norm(incoming.data+row*hidden,final_norm.data,normalized.data+row*hidden,hidden,epsilon) ||
            !mat(head.data,normalized.data+row*hidden,logits.data+row*vocab,vocab,hidden)) return 5;
    return 0;
}
API int oa_mixtral_range_run(const Config *c,Layer *layers,
    Tensor incoming,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor outgoing,Tensor routers,Tensor routes,
    const uint8_t *mask,uint64_t mask_count,const int64_t *positions,uint64_t position_count,
    ExpertFFN expert_fn,void *expert_user) {
    return mixtral_run(c,layers,NULL,NULL,batch*tokens,batch,tokens,start,
        outgoing,(Tensor){NULL,0},routers,routes,expert_fn,expert_user,
        mask,mask_count,positions,position_count,incoming,1,1);
}

API int oa_mixtral_range_run_scaled_rope(const Config *c,Layer *layers,
    Tensor incoming,uint64_t batch,uint64_t tokens,uint64_t start,
    Tensor outgoing,Tensor routers,Tensor routes,
    const uint8_t *mask,uint64_t mask_count,const int64_t *positions,uint64_t position_count,
    ExpertFFN expert_fn,void *expert_user,double factor) {
    return mixtral_run(c,layers,NULL,NULL,batch*tokens,batch,tokens,start,
        outgoing,(Tensor){NULL,0},routers,routes,expert_fn,expert_user,
        mask,mask_count,positions,position_count,incoming,1,factor);
}
