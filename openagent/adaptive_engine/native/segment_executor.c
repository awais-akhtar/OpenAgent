#include "segment_executor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stddef.h>

static int mul(uint64_t a, uint64_t b, uint64_t *n) {
    if (b && a > UINT64_MAX / b) return 0;
    *n = a * b; return 1;
}
static int add(uint64_t *total, uint64_t n) {
    if (n > UINT64_MAX - *total) return 0;
    *total += n; return 1;
}
static int bounded(uint64_t n, uint64_t limit) {
    return n && n <= SIZE_MAX / sizeof(float) && n <= limit / sizeof(float);
}
static int finite_array(const float *p, uint64_t n) {
    uint64_t i;
    if (!p) return 0;
    for (i=0; i<n; ++i) if (!isfinite(p[i])) return 0;
    return 1;
}
static int tensor(oa_tensor t, uint64_t n, uint64_t limit) {
    return t.count == n && bounded(n, limit) && finite_array(t.data, n);
}
static void norm(const float *x, const float *w, float *y, uint64_t n, double eps) {
    uint64_t i; double sum=0, scale;
    for (i=0; i<n; ++i) sum += (double)x[i]*x[i];
    scale = 1 / sqrt(sum/(double)n + eps);
    for (i=0; i<n; ++i) y[i]=(float)(x[i]*scale*w[i]);
}
static void linear(const float *x, const float *w, float *y, uint64_t in, uint64_t out) {
    uint64_t i,j;
    for (i=0; i<out; ++i) {
        double sum=0;
        for (j=0; j<in; ++j) sum += (double)x[j]*w[i*in+j];
        y[i]=(float)sum;
    }
}
static void rope(float *x, uint64_t heads, uint64_t d, uint64_t pos, double theta) {
    uint64_t h,j;
    for (h=0; h<heads; ++h) for (j=0; j<d/2; ++j) {
        double angle=(double)pos*pow(theta,-2.0*(double)j/(double)d);
        double a=x[h*d+j], b=x[h*d+j+d/2];
        x[h*d+j]=(float)(a*cos(angle)-b*sin(angle));
        x[h*d+j+d/2]=(float)(b*cos(angle)+a*sin(angle));
    }
}
static double sigmoid(double x) {
    if (x>=0) return 1/(1+exp(-x));
    { double e=exp(x); return e/(1+e); }
}
static int prepare_row(const oa_range_config *c, const oa_range_layer *w,
    float *row, float *z, float *q, float *k, float *v, float *ctx,
    float *att, float *prob, float *ks, float *vs, uint64_t layer_index,
    uint64_t position) {
    uint64_t h=c->hidden,d=h/c->heads,kv=c->kv_heads*d,head,j,t;
    norm(row,w->input_norm.data,z,h,c->epsilon);
    linear(z,w->q.data,q,h,h); linear(z,w->k.data,k,h,kv); linear(z,w->v.data,v,h,kv);
    norm(q,w->q_norm.data,q,h,c->epsilon); norm(k,w->k_norm.data,k,kv,c->epsilon);
    rope(q,c->heads,d,position,c->theta); rope(k,c->kv_heads,d,position,c->theta);
    for (head=0;head<c->kv_heads;++head) {
        uint64_t base=((layer_index*c->kv_heads+head)*c->capacity+position)*d;
        memcpy(ks+base,k+head*d,(size_t)d*sizeof(float));
        memcpy(vs+base,v+head*d,(size_t)d*sizeof(float));
    }
    for (head=0;head<c->heads;++head) {
        uint64_t base=(layer_index*c->kv_heads+head/(c->heads/c->kv_heads))*c->capacity*d;
        double maximum=-DBL_MAX,sum=0;
        for (t=0;t<=position;++t) {
            double score=0;
            for (j=0;j<d;++j) score+=(double)q[head*d+j]*ks[base+t*d+j];
            att[t]=(float)(score/sqrt((double)d));
            if (att[t]>maximum) maximum=att[t];
        }
        for (t=0;t<=position;++t) { att[t]=(float)exp(att[t]-maximum); sum+=att[t]; }
        for (j=0;j<d;++j) {
            double value=0;
            for (t=0;t<=position;++t) value+=(double)att[t]/sum*vs[base+t*d+j];
            ctx[head*d+j]=(float)value;
        }
    }
    linear(ctx,w->o.data,z,h,h);
    for (j=0;j<h;++j) row[j]+=z[j];
    norm(row,w->post_norm.data,z,h,c->epsilon);
    linear(z,w->router.data,prob,h,c->experts);
    return finite_array(row,h) && finite_array(z,h) && finite_array(q,h) &&
        finite_array(k,kv) && finite_array(v,kv) && finite_array(prob,c->experts);
}
uint32_t OA_CALL oa_executor_version(void) { return OA_EXECUTOR_VERSION; }

oa_status OA_CALL oa_executor_range(const oa_range_config *c,
    const oa_range_layer *ls, uint64_t layer_count, oa_tensor input,
    uint64_t rows, float *output, uint64_t output_count,
    float *keys, float *values, uint64_t cache_count, uint64_t *position,
    uint64_t limit) {
    uint64_t h,d,kv,n,cache,hh,kh,eh,ih,eih,ehi,l,r,j,head,a,e,best;
    uint64_t scratch,total,tmp,start,stop;
    float *mem,*x,*ks,*vs,*z,*q,*k,*v,*ctx,*att,*prob,*gate,*up,*mix;
    oa_status status=OA_INVALID;
    if (!c || !ls || !output || !keys || !values || !position) return OA_INVALID;
    if (c->version!=OA_EXECUTOR_VERSION) return OA_VERSION;
    h=c->hidden;
    if (!h || !c->heads || !c->kv_heads || h%c->heads ||
        c->heads%c->kv_heads || !c->layers || layer_count!=c->layers ||
        !c->intermediate || !c->experts || !c->topk || c->topk>c->experts ||
        c->normalize_topk>1 || c->reserved || !rows ||
        !isfinite(c->epsilon) || c->epsilon<=0 || !isfinite(c->theta) || c->theta<=0)
        return OA_INVALID;
    d=h/c->heads; if (d%2) return OA_INVALID;
    start=*position;
    if (start>c->capacity || rows>c->capacity-start) return OA_LIMIT;
    stop=start+rows;
    if (!mul(c->kv_heads,d,&kv) || !mul(rows,h,&n) ||
        !mul(c->layers,c->capacity,&tmp) || !mul(tmp,kv,&cache) ||
        !mul(h,h,&hh) || !mul(kv,h,&kh) || !mul(c->experts,h,&eh) ||
        !mul(c->intermediate,h,&ih) || !mul(c->experts,ih,&eih)) return OA_LIMIT;
    ehi=eih;
    if (!bounded(n,limit) || !bounded(cache,limit) || !bounded(eih,limit) ||
        !bounded(hh,limit) || !bounded(eh,limit)) return OA_LIMIT;
    if (output_count!=n || cache_count!=cache || !tensor(input,n,limit)) return OA_INVALID;
    for (l=0;l<c->layers;++l) {
        const oa_range_layer *w=ls+l;
        if (!tensor(w->input_norm,h,limit) || !tensor(w->post_norm,h,limit) ||
            !tensor(w->q,hh,limit) || !tensor(w->k,kh,limit) || !tensor(w->v,kh,limit) ||
            !tensor(w->o,hh,limit) || !tensor(w->q_norm,h,limit) ||
            !tensor(w->k_norm,kv,limit) || !tensor(w->router,eh,limit) ||
            !tensor(w->gate,eih,limit) || !tensor(w->up,eih,limit) ||
            !tensor(w->down,ehi,limit)) return OA_INVALID;
    }
    /* Prefix alone is live; uninitialized future cache slots are never read. */
    for (l=0;l<c->layers;++l) for (head=0;head<c->kv_heads;++head)
        if (!finite_array(keys+(l*c->kv_heads+head)*c->capacity*d,start*d) ||
            !finite_array(values+(l*c->kv_heads+head)*c->capacity*d,start*d)) return OA_INVALID;
    /* Sum checked separately from tensor products, before any allocation. */
    scratch=0; total=n;
    { uint64_t sizes[9]={cache,cache,h,h,kv,kv,h,c->capacity,c->experts};
      for (j=0;j<9;++j) { if (sizes[j]>UINT64_MAX-total) return OA_LIMIT; total+=sizes[j]; } }
    if (!mul(c->intermediate,2,&scratch) || scratch>UINT64_MAX-total) return OA_LIMIT;
    total+=scratch;
    if (h>UINT64_MAX-total) return OA_LIMIT;
    total+=h;
    if (!bounded(total,limit)) return OA_LIMIT;
    mem=(float*)malloc((size_t)total*sizeof(float)); if (!mem) return OA_LIMIT;
    x=mem; ks=x+n; vs=ks+cache; z=vs+cache; q=z+h; k=q+h; v=k+kv;
    ctx=v+kv; att=ctx+h; prob=att+c->capacity; gate=prob+c->experts;
    up=gate+c->intermediate; mix=up+c->intermediate;
    memcpy(x,input.data,(size_t)n*sizeof(float));
    memcpy(ks,keys,(size_t)cache*sizeof(float)); memcpy(vs,values,(size_t)cache*sizeof(float));
    for (l=0;l<c->layers;++l) for (r=0;r<rows;++r) {
        const oa_range_layer *w=ls+l; float *row=x+r*h;
        if (!prepare_row(c,w,row,z,q,k,v,ctx,att,prob,ks,vs,l,start+r)) goto done;
        { double maximum=-DBL_MAX,sum=0,selected=0;
          for (e=0;e<c->experts;++e) if (prob[e]>maximum) maximum=prob[e];
          for (e=0;e<c->experts;++e) { prob[e]=(float)exp(prob[e]-maximum); sum+=prob[e]; }
          /* Stable ties: first expert wins. Selection weights kept unnormalized
           * until the final weighted sum; no callback expert dispatch. */
          memset(mix,0,(size_t)h*sizeof(float));
          for (a=0;a<c->topk;++a) {
              best=0; for (e=1;e<c->experts;++e) if (prob[e]>prob[best]) best=e;
              { double weight=prob[best]/sum; selected+=weight; prob[best]=-1;
                linear(z,w->gate.data+best*ih,gate,h,c->intermediate);
                linear(z,w->up.data+best*ih,up,h,c->intermediate);
                for (j=0;j<c->intermediate;++j) gate[j]=(float)(gate[j]*sigmoid(gate[j])*up[j]);
                linear(gate,w->down.data+best*ih,ctx,c->intermediate,h);
                for (j=0;j<h;++j) mix[j]+=(float)(weight*ctx[j]); }
          }
          for (j=0;j<h;++j) row[j]+=(float)(mix[j]/(c->normalize_topk ? selected : 1)); }
        if (!finite_array(row,h) || !finite_array(k,kv) || !finite_array(v,kv)) goto done;
    }
    memcpy(output,x,(size_t)n*sizeof(float));
    /* Publish only newly appended slots, preserving future storage. */
    for (l=0;l<c->layers;++l) for (head=0;head<c->kv_heads;++head) {
        uint64_t base=((l*c->kv_heads+head)*c->capacity+start)*d;
        memcpy(keys+base,ks+base,(size_t)(rows*d)*sizeof(float));
        memcpy(values+base,vs+base,(size_t)(rows*d)*sizeof(float));
    }
    *position=stop; status=OA_OK;
done:
    free(mem); return status;
}

oa_status OA_CALL oa_executor_embed(oa_tensor w, uint64_t vocab, uint64_t h,
    const int64_t *ids, uint64_t rows, float *output, uint64_t count, uint64_t limit) {
    uint64_t n,wn,r;
    if (!ids || !output || !rows || !h || !vocab) return OA_INVALID;
    if (!mul(rows,h,&n) || !mul(vocab,h,&wn) || !bounded(n,limit) ||
        !bounded(wn,limit) || rows>limit/sizeof(int64_t)) return OA_LIMIT;
    if (count!=n || !tensor(w,wn,limit)) return OA_INVALID;
    for (r=0;r<rows;++r) if (ids[r]<0 || (uint64_t)ids[r]>=vocab) return OA_INVALID;
    for (r=0;r<rows;++r) memcpy(output+r*h,w.data+(uint64_t)ids[r]*h,(size_t)h*sizeof(float));
    return OA_OK;
}
oa_status OA_CALL oa_executor_head(oa_tensor input, uint64_t rows, uint64_t h,
    oa_tensor nw, oa_tensor w, uint64_t vocab, double eps,
    float *output, uint64_t count, uint64_t limit) {
    uint64_t n,wn,on,total,r; float *mem,*z;
    if (!output || !rows || !h || !vocab || !isfinite(eps) || eps<=0) return OA_INVALID;
    if (!mul(rows,h,&n) || !mul(vocab,h,&wn) || !mul(rows,vocab,&on) ||
        h>UINT64_MAX-on) return OA_LIMIT;
    total=on+h;
    if (!bounded(total,limit) || !bounded(wn,limit) || !bounded(n,limit)) return OA_LIMIT;
    if (count!=on || !tensor(input,n,limit) || !tensor(nw,h,limit) || !tensor(w,wn,limit)) return OA_INVALID;
    mem=(float*)malloc((size_t)total*sizeof(float)); if (!mem) return OA_LIMIT;
    z=mem+on;
    for (r=0;r<rows;++r) { norm(input.data+r*h,nw.data,z,h,eps); linear(z,w.data,mem+r*vocab,h,vocab); }
    if (!finite_array(mem,on)) { free(mem); return OA_INVALID; }
    memcpy(output,mem,(size_t)on*sizeof(float)); free(mem); return OA_OK;
}

oa_status OA_CALL oa_executor_decode(const oa_range_config *c,
    const oa_range_layer *ls, uint64_t layer_count,
    oa_tensor embedding, oa_tensor final_norm, oa_tensor head_weight, uint64_t vocab,
    const int64_t *ids, uint64_t rows, float *output, uint64_t output_count,
    float *keys, float *values, uint64_t cache_count, uint64_t *position,
    uint64_t limit) {
    uint64_t h,d,kv,n,cache,logits,tmp,outer=0,range_scratch=0,head_scratch=0,peak;
    uint64_t staged_position,layer_index,kv_head,base;
    float *mem,*hidden,*result,*ks,*vs,*scores;
    oa_tensor input;
    oa_status status;
    if (!c || !ls || !ids || !output || !keys || !values || !position) return OA_INVALID;
    if (c->version != OA_EXECUTOR_VERSION) return OA_VERSION;
    h=c->hidden;
    if (!rows || !vocab || !h || !c->heads || !c->kv_heads || h%c->heads ||
        c->heads%c->kv_heads || !c->layers || c->layers!=layer_count ||
        !c->intermediate || !c->experts || !c->topk || c->topk>c->experts ||
        c->normalize_topk>1 || c->reserved || !isfinite(c->epsilon) || c->epsilon<=0 ||
        !isfinite(c->theta) || c->theta<=0) return OA_INVALID;
    d=h/c->heads; if (d%2) return OA_INVALID;
    if (*position>c->capacity || rows>c->capacity-*position) return OA_LIMIT;
    if (!mul(c->kv_heads,d,&kv) || !mul(rows,h,&n) ||
        !mul(c->layers,c->capacity,&tmp) || !mul(tmp,kv,&cache) ||
        !mul(rows,vocab,&logits)) return OA_LIMIT;
    if (!bounded(n,limit) || !bounded(cache,limit) || !bounded(logits,limit) ||
        rows>limit/sizeof(int64_t)) return OA_LIMIT;
    if (output_count!=logits || cache_count!=cache) return OA_INVALID;
    /* Outer buffers remain live across each nested primitive. Admit their sum
     * plus the larger nested allocation before allocating any heap storage. */
    if (!add(&outer,n) || !add(&outer,n) || !add(&outer,cache) ||
        !add(&outer,cache) || !add(&outer,logits)) return OA_LIMIT;
    if (!add(&range_scratch,n) || !add(&range_scratch,cache) ||
        !add(&range_scratch,cache) || !add(&range_scratch,h) ||
        !add(&range_scratch,h) || !add(&range_scratch,h) || !add(&range_scratch,h) ||
        !add(&range_scratch,kv) || !add(&range_scratch,kv) ||
        !add(&range_scratch,c->capacity) || !add(&range_scratch,c->experts) ||
        !add(&range_scratch,c->intermediate) || !add(&range_scratch,c->intermediate) ||
        !add(&head_scratch,logits) || !add(&head_scratch,h)) return OA_LIMIT;
    peak=outer;
    if (!add(&peak,range_scratch>head_scratch ? range_scratch : head_scratch) ||
        !bounded(peak,limit)) return OA_LIMIT;
    mem=(float*)malloc((size_t)outer*sizeof(float)); if (!mem) return OA_LIMIT;
    hidden=mem; result=hidden+n; ks=result+n; vs=ks+cache; scores=vs+cache;
    staged_position=*position;
    memcpy(ks,keys,(size_t)cache*sizeof(float));
    memcpy(vs,values,(size_t)cache*sizeof(float));
    status=oa_executor_embed(embedding,vocab,h,ids,rows,hidden,n,limit);
    if (status!=OA_OK) goto decode_done;
    input.data=hidden; input.count=n;
    status=oa_executor_range(c,ls,layer_count,input,rows,result,n,
        ks,vs,cache,&staged_position,limit);
    if (status!=OA_OK) goto decode_done;
    input.data=result;
    status=oa_executor_head(input,rows,h,final_norm,head_weight,vocab,c->epsilon,
        scores,logits,limit);
    if (status!=OA_OK) goto decode_done;
    memcpy(output,scores,(size_t)logits*sizeof(float));
    for (layer_index=0;layer_index<c->layers;++layer_index)
        for (kv_head=0;kv_head<c->kv_heads;++kv_head) {
            base=((layer_index*c->kv_heads+kv_head)*c->capacity+*position)*d;
            memcpy(keys+base,ks+base,(size_t)(rows*d)*sizeof(float));
            memcpy(values+base,vs+base,(size_t)(rows*d)*sizeof(float));
        }
    *position=staged_position;
decode_done:
    free(mem); return status;
}

oa_status OA_CALL oa_executor_prepare_layer(const oa_range_config *c,
    const oa_range_layer *w, oa_tensor input, uint64_t rows,
    float *residual, float *normalized, uint64_t output_count,
    uint64_t *ids, float *weights, uint64_t route_count,
    float *keys, float *values, uint64_t cache_count, uint64_t position,
    uint64_t limit) {
    uint64_t h,d,kv,n,cache,hh,kh,eh,routes,total=0,bytes=0,tmp,r,e,a,best,head;
    float *mem,*x,*zn,*ks,*vs,*z,*q,*k,*v,*ctx,*att,*prob,*rw;
    uint64_t *ri;
    oa_status status=OA_INVALID;
    if (!c || !w || !residual || !normalized || !ids || !weights || !keys || !values)
        return OA_INVALID;
    if (c->version!=OA_EXECUTOR_VERSION) return OA_VERSION;
    h=c->hidden;
    if (!h || !c->heads || !c->kv_heads || h%c->heads || c->heads%c->kv_heads ||
        c->layers!=1 || !c->intermediate || !c->experts || !c->topk || c->topk>c->experts ||
        c->normalize_topk>1 || c->reserved || !rows || !isfinite(c->epsilon) ||
        c->epsilon<=0 || !isfinite(c->theta) || c->theta<=0 ||
        w->gate.data || w->gate.count || w->up.data || w->up.count || w->down.data || w->down.count)
        return OA_INVALID;
    d=h/c->heads; if (d%2) return OA_INVALID;
    if (position>c->capacity || rows>c->capacity-position) return OA_LIMIT;
    if (!mul(c->kv_heads,d,&kv) || !mul(rows,h,&n) || !mul(kv,c->capacity,&cache) ||
        !mul(h,h,&hh) || !mul(kv,h,&kh) || !mul(c->experts,h,&eh) ||
        !mul(rows,c->topk,&routes) || !mul(routes,sizeof(uint64_t),&tmp) ||
        tmp>SIZE_MAX || tmp>limit) return OA_LIMIT;
    if (output_count!=n || cache_count!=cache || route_count!=routes) return OA_INVALID;
    if (!tensor(input,n,limit) || !tensor(w->input_norm,h,limit) ||
        !tensor(w->q,hh,limit) || !tensor(w->k,kh,limit) || !tensor(w->v,kh,limit) ||
        !tensor(w->o,hh,limit) || !tensor(w->q_norm,h,limit) || !tensor(w->k_norm,kv,limit) ||
        !tensor(w->post_norm,h,limit) || !tensor(w->router,eh,limit)) return OA_INVALID;
    if (!bounded(cache,limit)) return OA_LIMIT;
    for (head=0;head<c->kv_heads;++head)
        if (!finite_array(keys+head*c->capacity*d,position*d) ||
            !finite_array(values+head*c->capacity*d,position*d)) return OA_INVALID;
    if (!add(&total,n) || !add(&total,n) || !add(&total,cache) || !add(&total,cache) ||
        !add(&total,h) || !add(&total,h) || !add(&total,h) || !add(&total,kv) ||
        !add(&total,kv) || !add(&total,c->capacity) || !add(&total,c->experts) ||
        !add(&total,routes) || !mul(total,sizeof(float),&bytes) || !add(&bytes,tmp) ||
        bytes>SIZE_MAX || bytes>limit) return OA_LIMIT;
    mem=(float*)malloc((size_t)total*sizeof(float)); if (!mem) return OA_LIMIT;
    ri=(uint64_t*)malloc((size_t)tmp); if (!ri) { free(mem); return OA_LIMIT; }
    x=mem; zn=x+n; ks=zn+n; vs=ks+cache; z=vs+cache; q=z+h; k=q+h;
    v=k+kv; ctx=v+kv; att=ctx+h; prob=att+c->capacity; rw=prob+c->experts;
    memcpy(x,input.data,(size_t)n*sizeof(float));
    memcpy(ks,keys,(size_t)cache*sizeof(float)); memcpy(vs,values,(size_t)cache*sizeof(float));
    for (r=0;r<rows;++r) {
        double maximum=-DBL_MAX,sum=0,selected=0;
        if (!prepare_row(c,w,x+r*h,z,q,k,v,ctx,att,prob,ks,vs,0,position+r)) goto prepare_done;
        memcpy(zn+r*h,z,(size_t)h*sizeof(float));
        for (e=0;e<c->experts;++e) if (prob[e]>maximum) maximum=prob[e];
        for (e=0;e<c->experts;++e) { prob[e]=(float)exp(prob[e]-maximum); sum+=prob[e]; }
        for (a=0;a<c->topk;++a) {
            best=0;
            for (e=1;e<c->experts;++e) if (prob[e]>prob[best]) best=e;
            ri[r*c->topk+a]=best;
            rw[r*c->topk+a]=(float)(prob[best]/sum);
            selected+=rw[r*c->topk+a]; prob[best]=-1;
        }
        if (c->normalize_topk)
            for (a=0;a<c->topk;++a) rw[r*c->topk+a]=(float)(rw[r*c->topk+a]/selected);
    }
    memcpy(residual,x,(size_t)n*sizeof(float)); memcpy(normalized,zn,(size_t)n*sizeof(float));
    memcpy(ids,ri,(size_t)tmp); memcpy(weights,rw,(size_t)routes*sizeof(float));
    for (head=0;head<c->kv_heads;++head) {
        uint64_t base=(head*c->capacity+position)*d;
        memcpy(keys+base,ks+base,(size_t)(rows*d)*sizeof(float));
        memcpy(values+base,vs+base,(size_t)(rows*d)*sizeof(float));
    }
    status=OA_OK;
prepare_done:
    free(ri); free(mem); return status;
}

static float read_scale(const unsigned char *p) {
    uint32_t bits=(uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
    float value; memcpy(&value,&bits,sizeof(value)); return value;
}
static int packed_buffer(const void *p,uint64_t bytes,size_t alignment) {
    return p && bytes<=SIZE_MAX && (uintptr_t)p%alignment==0 &&
           bytes<=UINTPTR_MAX-(uintptr_t)p;
}
static int packed_overlap(const void *a,uint64_t na,const void *b,uint64_t nb) {
    return (uintptr_t)a<(uintptr_t)b+nb && (uintptr_t)b<(uintptr_t)a+na;
}
typedef struct { char pad; float value; } packed_float_alignment;
typedef struct { char pad; uint64_t value; } packed_id_alignment;
static oa_status packed_contract(const void *payload, uint64_t bytes,
    uint64_t h, uint64_t intermediate, uint64_t limit, uint64_t *width) {
    uint64_t sizes=0,scales=0,expected,j;
    const unsigned char *p=(const unsigned char*)payload;
    if (!payload || !h || !intermediate) return OA_INVALID;
    if (!mul(h,intermediate,width) || !mul(*width,3,&sizes) ||
        !add(&scales,intermediate) || !add(&scales,intermediate) || !add(&scales,h) ||
        !mul(scales,4,&expected) || !add(&expected,sizes) || expected>SIZE_MAX || expected>limit)
        return OA_LIMIT;
    if (bytes!=expected) return OA_INVALID;
    for (j=0;j<scales;++j) {
        float value=read_scale(p+sizes+j*4);
        if (!isfinite(value) || value<=0) return OA_INVALID;
    }
    return OA_OK;
}
static int packed_row(const float *x, const void *payload, uint64_t h,
    uint64_t intermediate, uint64_t width, float *activation, float *output) {
    const int8_t *p=(const int8_t*)payload;
    const unsigned char *scales=(const unsigned char*)payload+3*width;
    uint64_t i,j;
    for (i=0;i<intermediate;++i) {
        double g=0,u=0; float gate,up;
        for (j=0;j<h;++j) {
            g+=(double)x[j]*p[i*h+j]; u+=(double)x[j]*p[width+i*h+j];
        }
        g*=read_scale(scales+i*4); u*=read_scale(scales+(intermediate+i)*4);
        if (!isfinite(g) || !isfinite(u) || fabs(g)>FLT_MAX || fabs(u)>FLT_MAX) return 0;
        gate=(float)g; up=(float)u;
        g=gate*sigmoid(gate)*up;
        if (!isfinite(g) || fabs(g)>FLT_MAX) return 0;
        activation[i]=(float)g;
    }
    if (!finite_array(activation,intermediate)) return 0;
    for (j=0;j<h;++j) {
        double value=0;
        for (i=0;i<intermediate;++i) value+=(double)activation[i]*p[2*width+j*intermediate+i];
        value*=read_scale(scales+(2*intermediate+j)*4);
        if (!isfinite(value) || fabs(value)>FLT_MAX) return 0;
        output[j]=(float)value;
    }
    return finite_array(output,h);
}
static double packed_half_scale(const unsigned char *p) {
    unsigned int bits=(unsigned int)p[0]|((unsigned int)p[1]<<8);
    unsigned int exponent=(bits>>10)&31, fraction=bits&1023;
    double value;
    if (exponent==31) return NAN;
    value=exponent ? ldexp(1.0+(double)fraction/1024.0,(int)exponent-15)
                   : ldexp((double)fraction,-24);
    return bits&32768 ? -value : value;
}
static int packed_half_finite(const unsigned char *p) {
    return (p[1]&124)!=124;
}
static oa_status q8_contract(const void *payload,uint64_t bytes,uint64_t h,
    uint64_t intermediate,uint64_t limit,uint64_t *width) {
    uint64_t expected,blocks,j;
    const unsigned char *p=(const unsigned char*)payload;
    if (!payload || !h || !intermediate || h%32 || intermediate%32) return OA_INVALID;
    if (!mul(h,intermediate,width) || !mul(*width/32,3,&blocks) ||
        !mul(blocks,34,&expected) || expected>SIZE_MAX || expected>limit) return OA_LIMIT;
    if (bytes!=expected) return OA_INVALID;
    for (j=0;j<blocks;++j) if (!packed_half_finite(p+j*34)) return OA_INVALID;
    return OA_OK;
}
static double q8_dot(const float *x,const unsigned char *p,uint64_t width) {
    uint64_t block,j; double value=0;
    for (block=0;block<width/32;++block) {
        double sum=0,scale=packed_half_scale(p+block*34);
        for (j=0;j<32;++j) {
            unsigned int raw=p[block*34+2+j];
            int code=raw<128 ? (int)raw : (int)raw-256;
            sum+=(double)x[block*32+j]*code;
        }
        value+=sum*scale;
    }
    return value;
}
static int q8_row(const float *x,const void *payload,uint64_t h,
    uint64_t intermediate,uint64_t width,float *activation,float *output) {
    const unsigned char *p=(const unsigned char*)payload;
    uint64_t i,row_bytes=h/32*34,bank_bytes=width/32*34;
    for (i=0;i<intermediate;++i) {
        double g=q8_dot(x,p+i*row_bytes,h),u=q8_dot(x,p+bank_bytes+i*row_bytes,h),value;
        float gate,up;
        if (!isfinite(g) || !isfinite(u) || fabs(g)>FLT_MAX || fabs(u)>FLT_MAX) return 0;
        gate=(float)g; up=(float)u;
        value=gate*sigmoid(gate)*up;
        if (!isfinite(value) || fabs(value)>FLT_MAX) return 0;
        activation[i]=(float)value;
    }
    if (!finite_array(activation,intermediate)) return 0;
    for (i=0;i<h;++i) {
        double value=q8_dot(activation,p+2*bank_bytes+i*(intermediate/32*34),intermediate);
        if (!isfinite(value) || fabs(value)>FLT_MAX) return 0;
        output[i]=(float)value;
    }
    return finite_array(output,h);
}
static oa_status q4_contract(const void *payload,uint64_t bytes,uint64_t h,
    uint64_t intermediate,uint64_t limit,uint64_t *width) {
    uint64_t expected,blocks,j;
    const unsigned char *p=(const unsigned char*)payload;
    if (!payload || !h || !intermediate || h%32 || intermediate%32) return OA_INVALID;
    if (!mul(h,intermediate,width) || !mul(*width/32,3,&blocks) ||
        !mul(blocks,18,&expected) || expected>SIZE_MAX || expected>limit) return OA_LIMIT;
    if (bytes!=expected) return OA_INVALID;
    for (j=0;j<blocks;++j) if (!packed_half_finite(p+j*18)) return OA_INVALID;
    return OA_OK;
}
static double q4_dot(const float *x,const unsigned char *p,uint64_t width) {
    uint64_t block,j; double value=0;
    for (block=0;block<width/32;++block) {
        const unsigned char *codes=p+block*18+2;
        double sum=0,scale=packed_half_scale(p+block*18);
        for (j=0;j<16;++j) sum+=(double)x[block*32+j]*((int)(codes[j]&15)-8);
        for (j=0;j<16;++j) sum+=(double)x[block*32+16+j]*((int)(codes[j]>>4)-8);
        value+=sum*scale;
    }
    return value;
}
static int q4_row(const float *x,const void *payload,uint64_t h,
    uint64_t intermediate,uint64_t width,float *activation,float *output) {
    const unsigned char *p=(const unsigned char*)payload;
    uint64_t i,row_bytes=h/32*18,bank_bytes=width/32*18;
    for (i=0;i<intermediate;++i) {
        double g=q4_dot(x,p+i*row_bytes,h),u=q4_dot(x,p+bank_bytes+i*row_bytes,h);
        float gate,up;
        if (!isfinite(g) || !isfinite(u) || fabs(g)>FLT_MAX || fabs(u)>FLT_MAX) return 0;
        gate=(float)g; up=(float)u;
        g=gate*sigmoid(gate)*up;
        if (!isfinite(g) || fabs(g)>FLT_MAX) return 0;
        activation[i]=(float)g;
    }
    if (!finite_array(activation,intermediate)) return 0;
    for (i=0;i<h;++i) {
        double value=q4_dot(activation,p+2*bank_bytes+i*(intermediate/32*18),intermediate);
        if (!isfinite(value) || fabs(value)>FLT_MAX) return 0;
        output[i]=(float)value;
    }
    return finite_array(output,h);
}
static uint64_t ggml_extra_block_bytes(oa_packed_format format) {
    switch (format) {
    case OA_PACKED_Q4_1: return 20;
    case OA_PACKED_Q5_0: return 22;
    case OA_PACKED_Q5_1: return 24;
    case OA_PACKED_Q8_1: return 36;
    default: return 0;
    }
}
static oa_status ggml_extra_contract(oa_packed_format format,const void *payload,
    uint64_t bytes,uint64_t h,uint64_t intermediate,uint64_t limit,uint64_t *width) {
    uint64_t size=ggml_extra_block_bytes(format),blocks,expected,j;
    const unsigned char *p=(const unsigned char*)payload;
    if (!size || !payload || !h || !intermediate || h%32 || intermediate%32) return OA_INVALID;
    if (!mul(h,intermediate,width) || !mul(*width/32,3,&blocks) ||
        !mul(blocks,size,&expected) || expected>SIZE_MAX || expected>limit) return OA_LIMIT;
    if (bytes!=expected) return OA_INVALID;
    for (j=0;j<blocks;++j) {
        const unsigned char *block=p+j*size;
        if (!packed_half_finite(block)) return OA_INVALID;
        if (format!=OA_PACKED_Q5_0 && !packed_half_finite(block+2)) return OA_INVALID;
    }
    return OA_OK;
}
static double ggml_extra_dot(oa_packed_format format,const float *x,
    const unsigned char *p,uint64_t width) {
    uint64_t size=ggml_extra_block_bytes(format),block,lane;
    unsigned int high_offset=format==OA_PACKED_Q5_0 ? 2 : (format==OA_PACKED_Q5_1 ? 4 : 0);
    int affine=format==OA_PACKED_Q4_1 || format==OA_PACKED_Q5_1;
    int bias=format==OA_PACKED_Q5_0 ? -16 : 0;
    double value=0;
    for (block=0;block<width/32;++block) {
        const unsigned char *b=p+block*size,*codes;
        double d=packed_half_scale(b),m=0;
        uint32_t high=0;
        if (format==OA_PACKED_Q8_1) {
            for (lane=0;lane<32;++lane) {
                unsigned int raw=b[4+lane];
                int code=raw<128 ? (int)raw : (int)raw-256;
                value+=(double)x[block*32+lane]*(d*code);
            }
            continue;
        }
        if (affine) m=packed_half_scale(b+2);
        if (high_offset) {
            const unsigned char *bits=b+high_offset;
            high=(uint32_t)bits[0] | ((uint32_t)bits[1]<<8) |
                 ((uint32_t)bits[2]<<16) | ((uint32_t)bits[3]<<24);
            codes=bits+4;
        } else codes=b+4;
        /* GGML pairs are split halves, not adjacent even/odd lanes. */
        for (lane=0;lane<16;++lane) {
            int code=(int)(codes[lane]&15)+(int)((high>>lane)&1)*16+bias;
            value+=(double)x[block*32+lane]*(d*code+m);
        }
        for (lane=16;lane<32;++lane) {
            int code=(int)(codes[lane-16]>>4)+(int)((high>>lane)&1)*16+bias;
            value+=(double)x[block*32+lane]*(d*code+m);
        }
    }
    return value;
}
static int ggml_extra_row(oa_packed_format format,const float *x,const void *payload,
    uint64_t h,uint64_t intermediate,uint64_t width,float *activation,float *output) {
    const unsigned char *p=(const unsigned char*)payload;
    uint64_t size=ggml_extra_block_bytes(format),i,row_bytes,bank_bytes,down_stride;
    if (!size) return 0;
    /* Products were admitted by ggml_extra_contract before this traversal. */
    row_bytes=h/32*size; bank_bytes=width/32*size; down_stride=intermediate/32*size;
    for (i=0;i<intermediate;++i) {
        double g=ggml_extra_dot(format,x,p+i*row_bytes,h);
        double u=ggml_extra_dot(format,x,p+bank_bytes+i*row_bytes,h),value;
        float gate,up;
        if (!isfinite(g) || !isfinite(u) || fabs(g)>FLT_MAX || fabs(u)>FLT_MAX) return 0;
        gate=(float)g; up=(float)u;
        value=gate*sigmoid(gate)*up;
        if (!isfinite(value) || fabs(value)>FLT_MAX) return 0;
        activation[i]=(float)value;
    }
    for (i=0;i<h;++i) {
        double value=ggml_extra_dot(format,activation,p+2*bank_bytes+i*down_stride,intermediate);
        if (!isfinite(value) || fabs(value)>FLT_MAX) return 0;
        output[i]=(float)value;
    }
    return finite_array(output,h);
}
static oa_status packed_format_contract(oa_packed_format format,const void *payload,
    uint64_t bytes,uint64_t h,uint64_t intermediate,uint64_t limit,uint64_t *width) {
    if (bytes>SIZE_MAX || bytes>limit) return OA_LIMIT;
    if (!packed_buffer(payload,bytes,1)) return OA_INVALID;
    switch (format) {
    case OA_PACKED_INT8_ROW: return packed_contract(payload,bytes,h,intermediate,limit,width);
    case OA_PACKED_Q8_0: return q8_contract(payload,bytes,h,intermediate,limit,width);
    case OA_PACKED_Q4_0: return q4_contract(payload,bytes,h,intermediate,limit,width);
    case OA_PACKED_Q4_1:
    case OA_PACKED_Q5_0:
    case OA_PACKED_Q5_1:
    case OA_PACKED_Q8_1: return ggml_extra_contract(format,payload,bytes,h,intermediate,limit,width);
    default: return OA_INVALID;
    }
}
static int packed_format_row(oa_packed_format format,const float *x,const void *payload,
    uint64_t h,uint64_t intermediate,uint64_t width,float *activation,float *output) {
    switch (format) {
    case OA_PACKED_INT8_ROW: return packed_row(x,payload,h,intermediate,width,activation,output);
    case OA_PACKED_Q8_0: return q8_row(x,payload,h,intermediate,width,activation,output);
    case OA_PACKED_Q4_0: return q4_row(x,payload,h,intermediate,width,activation,output);
    case OA_PACKED_Q4_1:
    case OA_PACKED_Q5_0:
    case OA_PACKED_Q5_1:
    case OA_PACKED_Q8_1: return ggml_extra_row(format,x,payload,h,intermediate,width,activation,output);
    default: return 0;
    }
}
static oa_status packed_swiglu(oa_tensor input, uint64_t rows,
    uint64_t h, uint64_t intermediate, const void *payload, uint64_t payload_bytes,
    float *output, uint64_t output_count, uint64_t limit,oa_packed_format format) {
    uint64_t n,total,width,r; float *mem;
    oa_status status;
    if (!output || !rows || !h || !intermediate) return OA_INVALID;
    if (!mul(rows,h,&n)) return OA_LIMIT;
    total=n; if (!add(&total,intermediate) || !bounded(total,limit)) return OA_LIMIT;
    if (!packed_buffer(input.data,n*sizeof(float),offsetof(packed_float_alignment,value)) ||
        !packed_buffer(output,n*sizeof(float),offsetof(packed_float_alignment,value))) return OA_INVALID;
    if (output_count!=n || !tensor(input,n,limit)) return OA_INVALID;
    status=packed_format_contract(format,payload,payload_bytes,h,intermediate,limit,&width);
    if (status!=OA_OK) return status;
    if (packed_overlap(output,n*sizeof(float),input.data,n*sizeof(float)) ||
        packed_overlap(output,n*sizeof(float),payload,payload_bytes)) return OA_INVALID;
    mem=(float*)malloc((size_t)total*sizeof(float)); if (!mem) return OA_LIMIT;
    for (r=0;r<rows;++r) if (!packed_format_row(format,input.data+r*h,payload,h,intermediate,width,mem+n,mem+r*h)) {
        free(mem); return OA_INVALID;
    }
    memcpy(output,mem,(size_t)n*sizeof(float)); free(mem); return OA_OK;
}
static oa_status apply_packed(oa_tensor input, uint64_t rows,
    uint64_t h, uint64_t intermediate, const uint64_t *ids, const float *weights,
    uint64_t route_count, uint64_t topk, uint64_t experts, uint64_t expert,
    const void *payload, uint64_t payload_bytes, oa_tensor accumulator,
    float *output, uint64_t output_count, uint64_t limit,oa_packed_format format) {
    uint64_t n,routes,total,width,r,a,b,j; float *mem,*activation,*result;
    int selected=0;
    oa_status status;
    if (!output || !ids || !weights || !rows || !h || !intermediate ||
        !topk || topk>experts || expert>=experts) return OA_INVALID;
    if (!mul(rows,h,&n) || !mul(rows,topk,&routes) || routes>SIZE_MAX/sizeof(uint64_t) ||
        routes>limit/sizeof(uint64_t)) return OA_LIMIT;
    total=n;
    if (!add(&total,intermediate) || !add(&total,h) || !bounded(total,limit)) return OA_LIMIT;
    if (!packed_buffer(input.data,n*sizeof(float),offsetof(packed_float_alignment,value)) ||
        !packed_buffer(accumulator.data,n*sizeof(float),offsetof(packed_float_alignment,value)) ||
        !packed_buffer(output,n*sizeof(float),offsetof(packed_float_alignment,value)) ||
        !packed_buffer(ids,routes*sizeof(uint64_t),offsetof(packed_id_alignment,value)) ||
        !packed_buffer(weights,routes*sizeof(float),offsetof(packed_float_alignment,value))) return OA_INVALID;
    if (route_count!=routes || output_count!=n || !tensor(input,n,limit) ||
        !tensor(accumulator,n,limit)) return OA_INVALID;
    for (r=0;r<rows;++r) for (a=0;a<topk;++a) {
        uint64_t index=r*topk+a;
        if (ids[index]>=experts || !isfinite(weights[index]) || weights[index]<0 || weights[index]>1)
            return OA_INVALID;
        for (b=0;b<a;++b) if (ids[r*topk+b]==ids[index]) return OA_INVALID;
        if (ids[index]==expert) selected=1;
    }
    status=packed_format_contract(format,payload,payload_bytes,h,intermediate,limit,&width);
    if (status!=OA_OK) return status;
    if (packed_overlap(output,n*sizeof(float),input.data,n*sizeof(float)) ||
        packed_overlap(output,n*sizeof(float),accumulator.data,n*sizeof(float)) ||
        packed_overlap(output,n*sizeof(float),payload,payload_bytes) ||
        packed_overlap(output,n*sizeof(float),ids,routes*sizeof(uint64_t)) ||
        packed_overlap(output,n*sizeof(float),weights,routes*sizeof(float))) return OA_INVALID;
    /* Retain full admission/metadata validation even for an unrouted expert.
     * Once those checks succeed, the unchanged accumulator needs no heap stage. */
    if (!selected) {
        memcpy(output,accumulator.data,(size_t)n*sizeof(float)); return OA_OK;
    }
    mem=(float*)malloc((size_t)total*sizeof(float)); if (!mem) return OA_LIMIT;
    activation=mem+n; result=activation+intermediate;
    memcpy(mem,accumulator.data,(size_t)n*sizeof(float));
    for (r=0;r<rows;++r) for (a=0;a<topk;++a) if (ids[r*topk+a]==expert) {
        if (!packed_format_row(format,input.data+r*h,payload,h,intermediate,width,activation,result)) {
            free(mem); return OA_INVALID;
        }
        for (j=0;j<h;++j) mem[r*h+j]+=(float)((double)weights[r*topk+a]*result[j]);
        if (!finite_array(mem+r*h,h)) { free(mem); return OA_INVALID; }
        break;
    }
    memcpy(output,mem,(size_t)n*sizeof(float)); free(mem); return OA_OK;
}
#define OA_PACKED_EXPORT(suffix,mode) \
oa_status OA_CALL oa_executor_swiglu_##suffix(oa_tensor input,uint64_t rows,uint64_t h,uint64_t i, \
    const void *payload,uint64_t bytes,float *output,uint64_t count,uint64_t limit) { \
    return packed_swiglu(input,rows,h,i,payload,bytes,output,count,limit,mode); } \
oa_status OA_CALL oa_executor_apply_##suffix(oa_tensor input,uint64_t rows,uint64_t h,uint64_t i, \
    const uint64_t *ids,const float *weights,uint64_t routes,uint64_t topk,uint64_t experts,uint64_t expert, \
    const void *payload,uint64_t bytes,oa_tensor accumulator,float *output,uint64_t count,uint64_t limit) { \
    return apply_packed(input,rows,h,i,ids,weights,routes,topk,experts,expert,payload,bytes,accumulator,output,count,limit,mode); }
OA_PACKED_EXPORT(int8_row,OA_PACKED_INT8_ROW)
OA_PACKED_EXPORT(q8_0,OA_PACKED_Q8_0)
OA_PACKED_EXPORT(q4_0,OA_PACKED_Q4_0)
OA_PACKED_EXPORT(q4_1,OA_PACKED_Q4_1)
OA_PACKED_EXPORT(q5_0,OA_PACKED_Q5_0)
OA_PACKED_EXPORT(q5_1,OA_PACKED_Q5_1)
OA_PACKED_EXPORT(q8_1,OA_PACKED_Q8_1)
oa_status OA_CALL oa_executor_residual_f32(oa_tensor left, oa_tensor right,
    float *output, uint64_t output_count, uint64_t limit) {
    uint64_t j;
    if (!output || !tensor(left,output_count,limit) || !tensor(right,output_count,limit)) return OA_INVALID;
    for (j=0;j<output_count;++j) {
        double value=(double)left.data[j]+right.data[j];
        if (!isfinite(value) || fabs(value)>FLT_MAX) return OA_INVALID;
    }
    for (j=0;j<output_count;++j) output[j]=(float)((double)left.data[j]+right.data[j]);
    return OA_OK;
}
