/* Original scalar FP32 V4 token-replay engine. Private, length-checked ABI.
 * No model source blocks are copied or mechanically translated here.
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct { const float *data; uint64_t count; } Tensor;
typedef struct {
    int h, a, d, r, q, g, o, i, j, k, e, top, m, u, iters;
    int window, csa, hca, layers, vocab, capacity, activation;
    float eps, hc_eps, limit, route_scale, main_theta, compress_theta;
} Config;
typedef struct {
    int kind, hash;
    Tensor w[33];
    const int64_t *table;
    uint64_t table_count;
} Layer;
typedef struct { float *data; size_t used, count; } Arena;

static float *take(Arena *a, size_t count) {
    float *p;
    if (count > a->count-a->used) return NULL;
    p = a->data+a->used;
    a->used += count;
    return p;
}

static float dot(const float *x, const float *y, int count) {
    int i;
    float sum = 0.0f;
    for (i=0; i<count; ++i) sum += x[i]*y[i];
    return sum;
}

static void linear(const float *x, const float *w, float *out, int rows, int columns) {
    int row;
    for (row=0; row<rows; ++row) out[row] = dot(x, w+(size_t)row*columns, columns);
}

static void norm(float *x, int count, const float *weight, float epsilon) {
    int i;
    float scale = 1.0f/sqrtf(dot(x,x,count)/(float)count+epsilon);
    for (i=0; i<count; ++i) x[i] = (x[i]*scale)*(weight ? weight[i] : 1.0f);
}

static float logistic(float x) {
    float e = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f/(1.0f+e) : e/(1.0f+e);
}

static void probability(float *x, int count) {
    int i;
    float maximum=x[0], sum=0.0f;
    for (i=1; i<count; ++i) if (x[i]>maximum) maximum=x[i];
    for (i=0; i<count; ++i) { x[i]=expf(x[i]-maximum); sum+=x[i]; }
    for (i=0; i<count; ++i) x[i]/=sum;
}

static void rotation(float *x, int width, int rotary, int position, float theta) {
    int pair;
    for (pair=0; pair<rotary; pair+=2) {
        float angle=(float)position/powf(theta,(float)pair/(float)rotary);
        float co=cosf(angle), si=sinf(angle);
        int index=width-rotary+pair;
        float left=x[index], right=x[index+1];
        x[index]=left*co-right*si;
        x[index+1]=left*si+right*co;
    }
}

/* Reject ambiguous cutoff ties instead of claiming Torch's unspecified ordering. */
static int select_entries(const float *scores, int count, int wanted, int *indices) {
    int taken[256]={0}, rank, candidate, best;
    if (count<1 || count>256 || wanted<1 || wanted>count) return 1;
    for (candidate=0; candidate<count; ++candidate)
        if (!isfinite(scores[candidate])) return 3;
    for (rank=0; rank<wanted; ++rank) {
        best=-1;
        for (candidate=0; candidate<count; ++candidate)
            if (!taken[candidate] && (best<0 || scores[candidate]>scores[best])) best=candidate;
        if (best<0 || !isfinite(scores[best])) return 3;
        indices[rank]=best;
        taken[best]=1;
    }
    if (wanted<count) {
        float cutoff=scores[indices[wanted-1]];
        for (candidate=0; candidate<count; ++candidate)
            if (!taken[candidate] && scores[candidate]==cutoff) return 4;
    }
    return 0;
}

API uint64_t oa_deepseek_v4_version(void) { return 2; }

API int oa_deepseek_v4_geometry(const Config *c) {
    if (!c) return 1;
#define CHECK(field, upper) if (c->field<1 || c->field>upper) return 1
    CHECK(h,128); CHECK(a,8); CHECK(d,64); CHECK(r,64); CHECK(q,128);
    CHECK(g,8); CHECK(o,128); CHECK(i,64); CHECK(j,8); CHECK(k,256);
    CHECK(e,32); CHECK(top,8); CHECK(m,256); CHECK(u,4); CHECK(iters,64);
    CHECK(window,256); CHECK(csa,256); CHECK(hca,256); CHECK(layers,8);
    CHECK(vocab,4096); CHECK(capacity,256);
#undef CHECK
    if (c->a%c->g || c->r%2 || c->r>c->d || c->r>c->i || c->top>c->e) return 1;
    if (c->activation!=0 && c->activation!=1) return 1;
    if (!isfinite(c->eps) || !isfinite(c->hc_eps) || !isfinite(c->limit) ||
        !isfinite(c->route_scale) || !isfinite(c->main_theta) || !isfinite(c->compress_theta) ||
        c->eps<=0 || c->hc_eps<=0 || c->limit<=0 || c->route_scale<=0 ||
        c->main_theta<=0 || c->compress_theta<=0) return 1;
    return 0;
}

static int valid(Tensor tensor, uint64_t expected) {
    uint64_t i;
    if (tensor.count!=expected || (expected && !tensor.data)) return 0;
    for (i=0; i<expected; ++i) if (!isfinite(tensor.data[i])) return 0;
    return 1;
}

static int manifest(const Config *c, const Layer *l) {
    uint64_t sizes[33];
    int rate, width, s;
    if (l->kind<0 || l->kind>2 || l->hash<0 || l->hash>1) return 0;
    rate=l->kind==1 ? c->csa : c->hca;
    width=l->kind==1 ? 2*c->d : c->d;
    sizes[0]=sizes[1]=(uint64_t)c->h;
    sizes[2]=(uint64_t)c->q*c->h; sizes[3]=(uint64_t)c->q;
    sizes[4]=(uint64_t)c->a*c->d*c->q; sizes[5]=(uint64_t)c->d*c->h;
    sizes[6]=(uint64_t)c->d; sizes[7]=(uint64_t)c->g*c->o*(c->a*c->d/c->g);
    sizes[8]=(uint64_t)c->h*c->g*c->o; sizes[9]=(uint64_t)c->a;
    sizes[10]=sizes[11]=l->kind ? (uint64_t)width*c->h : 0;
    sizes[12]=l->kind ? (uint64_t)rate*width : 0; sizes[13]=l->kind ? (uint64_t)c->d : 0;
    sizes[14]=sizes[15]=l->kind==1 ? (uint64_t)2*c->i*c->h : 0;
    sizes[16]=l->kind==1 ? (uint64_t)c->csa*2*c->i : 0;
    sizes[17]=l->kind==1 ? (uint64_t)c->i : 0;
    sizes[18]=l->kind==1 ? (uint64_t)c->j*c->i*c->q : 0;
    sizes[19]=l->kind==1 ? (uint64_t)c->j*c->h : 0;
    sizes[20]=sizes[23]=(uint64_t)(c->u+2)*c->u*c->u*c->h;
    sizes[21]=sizes[24]=(uint64_t)(c->u+2)*c->u; sizes[22]=sizes[25]=3;
    sizes[26]=(uint64_t)c->e*c->h; sizes[27]=l->hash ? 0 : (uint64_t)c->e;
    sizes[28]=(uint64_t)c->e*2*c->m*c->h; sizes[29]=(uint64_t)c->e*c->h*c->m;
    sizes[30]=sizes[31]=sizes[32]=(uint64_t)c->m*c->h;
    for (s=0; s<33; ++s) if (!valid(l->w[s],sizes[s])) return 0;
    if (l->hash) {
        uint64_t t;
        if (!l->table || l->table_count!=(uint64_t)c->vocab*c->top) return 0;
        for (t=0; t<l->table_count; ++t) {
            int k;
            if (l->table[t]<0 || l->table[t]>=c->e) return 0;
            for (k=0; k<(int)(t%(uint64_t)c->top); ++k)
                if (l->table[t]==l->table[t-t%(uint64_t)c->top+(uint64_t)k]) return 0;
        }
    } else if (l->table_count || l->table) return 0;
    return 1;
}

/* Mix maps are evaluated from the entire flattened stream vector. */
static void hyper(const Config *c, const Tensor *w, const float *streams,
                  float *collapsed, float *post, float *comb, int head) {
    float flat[512], maps[24], pre[4];
    int j, k, d, iteration, u=c->u, h=c->h;
    memcpy(flat,streams,(size_t)u*h*sizeof(float));
    norm(flat,u*h,NULL,c->eps);
    linear(flat,w[0].data,maps,head ? u : (u+2)*u,u*h);
    for (j=0; j<u; ++j) {
        pre[j]=logistic(maps[j]*w[2].data[0]+w[1].data[j])+c->hc_eps;
        if (!head) post[j]=2.0f*logistic(maps[u+j]*w[2].data[1]+w[1].data[u+j]);
    }
    for (d=0; d<h; ++d) {
        collapsed[d]=0.0f;
        for (j=0; j<u; ++j) collapsed[d]+=pre[j]*streams[j*h+d];
    }
    if (head) return;
    for (j=0; j<u; ++j) {
        for (k=0; k<u; ++k) comb[j*u+k]=maps[2*u+j*u+k]*w[2].data[2]+w[1].data[2*u+j*u+k];
        probability(comb+j*u,u);
        for (k=0; k<u; ++k) comb[j*u+k]+=c->hc_eps;
    }
    for (iteration=0; iteration<c->iters; ++iteration) {
        if (iteration) for (j=0; j<u; ++j) {
            float sum=c->hc_eps;
            for (k=0; k<u; ++k) sum+=comb[j*u+k];
            for (k=0; k<u; ++k) comb[j*u+k]/=sum;
        }
        for (k=0; k<u; ++k) {
            float sum=c->hc_eps;
            for (j=0; j<u; ++j) sum+=comb[j*u+k];
            for (j=0; j<u; ++j) comb[j*u+k]/=sum;
        }
    }
}

static void merge(const Config *c, float *streams, const float *branch,
                  const float *post, const float *comb) {
    float updated[512];
    int j, k, d;
    for (k=0; k<c->u; ++k) for (d=0; d<c->h; ++d) {
        float sum=0.0f;
        for (j=0; j<c->u; ++j) sum+=comb[j*c->u+k]*streams[j*c->h+d];
        updated[k*c->h+d]=sum+post[k]*branch[d];
    }
    memcpy(streams,updated,(size_t)c->u*c->h*sizeof(float));
}

/* Per-channel gated pooling, with a stride-rate two-series CSA overlap. */
static float *pool(const Config *c, Arena *arena, const float *x, int n,
                   const Tensor *w, int rate, int width, int overlap, float theta) {
    int source_width=width*(overlap ? 2 : 1), row, entry, channel, slot;
    float *values=take(arena,(size_t)n*source_width);
    float *gates=take(arena,(size_t)n*source_width);
    float *result=take(arena,(size_t)(n/rate)*width);
    if (!values || !gates || !result) return NULL;
    for (row=0; row<n; ++row) {
        linear(x+(size_t)row*c->h,w[0].data,values+(size_t)row*source_width,source_width,c->h);
        linear(x+(size_t)row*c->h,w[1].data,gates+(size_t)row*source_width,source_width,c->h);
    }
    for (entry=0; entry<n/rate; ++entry) {
        int slots=rate*((overlap && entry) ? 2 : 1);
        for (channel=0; channel<width; ++channel) {
            float scores[512], sum=0.0f;
            for (slot=0; slot<slots; ++slot) {
                int previous=overlap && entry && slot<rate;
                int token=entry*rate+(slot%rate)-(previous ? rate : 0);
                int ch=channel+((overlap && !previous) ? width : 0);
                scores[slot]=gates[(size_t)token*source_width+ch]+w[2].data[(slot%rate)*source_width+ch];
            }
            probability(scores,slots);
            for (slot=0; slot<slots; ++slot) {
                int previous=overlap && entry && slot<rate;
                int token=entry*rate+(slot%rate)-(previous ? rate : 0);
                int ch=channel+((overlap && !previous) ? width : 0);
                sum+=scores[slot]*values[(size_t)token*source_width+ch];
            }
            result[entry*width+channel]=sum;
        }
        norm(result+entry*width,width,w[3].data,c->eps);
        rotation(result+entry*width,width,c->r,entry*rate,theta);
    }
    return result;
}

static int attention(const Config *c, const Layer *l, Arena *arena, const float *x,
                     int n, float *output) {
    int t, head, entry, slot, rate=l->kind==1 ? c->csa : c->hca;
    int width=c->a*c->d;
    float theta=l->kind ? c->compress_theta : c->main_theta;
    float *qr=take(arena,(size_t)n*c->q), *q=take(arena,(size_t)n*width);
    float *kv=take(arena,(size_t)n*c->d), *compressed=NULL, *index=NULL;
    const Tensor *w=l->w;
    if (!qr || !q || !kv) return 2;
    for (t=0; t<n; ++t) {
        linear(x+(size_t)t*c->h,w[2].data,qr+(size_t)t*c->q,c->q,c->h);
        norm(qr+(size_t)t*c->q,c->q,w[3].data,c->eps);
        linear(qr+(size_t)t*c->q,w[4].data,q+(size_t)t*width,width,c->q);
        for (head=0; head<c->a; ++head) {
            float *query=q+(size_t)t*width+head*c->d;
            norm(query,c->d,NULL,c->eps);
            rotation(query,c->d,c->r,t,theta);
        }
        linear(x+(size_t)t*c->h,w[5].data,kv+(size_t)t*c->d,c->d,c->h);
        norm(kv+(size_t)t*c->d,c->d,w[6].data,c->eps);
        rotation(kv+(size_t)t*c->d,c->d,c->r,t,theta);
    }
    if (l->kind) {
        compressed=pool(c,arena,x,n,w+10,rate,c->d,l->kind==1,theta);
        if (!compressed) return 2;
        if (l->kind==1) {
            index=pool(c,arena,x,n,w+14,rate,c->i,1,theta);
            if (!index) return 2;
        }
    }
    for (t=0; t<n; ++t) {
        float index_query[512], index_weights[8], affinities[256];
        float attended[512], grouped[1024];
        int picks[256], ready=l->kind ? (t+1)/rate : 0;
        int chosen=ready, start=t+1-c->window, local, code;
        if (start<0) start=0;
        local=t+1-start;
        if (index && ready) {
            linear(qr+(size_t)t*c->q,w[18].data,index_query,c->j*c->i,c->q);
            linear(x+(size_t)t*c->h,w[19].data,index_weights,c->j,c->h);
            for (head=0; head<c->j; ++head) rotation(index_query+head*c->i,c->i,c->r,t,theta);
            for (entry=0; entry<ready; ++entry) {
                float score=0.0f;
                for (head=0; head<c->j; ++head) {
                    float affinity=dot(index_query+head*c->i,index+entry*c->i,c->i);
                    if (!isfinite(affinity) || !isfinite(index_weights[head])) return 3;
                    score+=fmaxf(affinity,0.0f)/sqrtf((float)c->i)*
                           (index_weights[head]/sqrtf((float)c->j));
                }
                affinities[entry]=score;
            }
            chosen=ready<c->k ? ready : c->k;
            code=select_entries(affinities,ready,chosen,picks);
            if (code) return code;
        } else for (entry=0; entry<ready; ++entry) picks[entry]=entry;
        for (head=0; head<c->a; ++head) {
            float scores[513], *query=q+(size_t)t*width+head*c->d;
            float *values=attended+head*c->d;
            for (slot=0; slot<local+chosen; ++slot) {
                const float *key=slot<local ? kv+(size_t)(start+slot)*c->d :
                                              compressed+(size_t)picks[slot-local]*c->d;
                scores[slot]=dot(query,key,c->d)/sqrtf((float)c->d);
            }
            scores[local+chosen]=w[9].data[head];
            probability(scores,local+chosen+1);
            memset(values,0,(size_t)c->d*sizeof(float));
            for (slot=0; slot<local+chosen; ++slot) {
                int channel;
                const float *key=slot<local ? kv+(size_t)(start+slot)*c->d :
                                              compressed+(size_t)picks[slot-local]*c->d;
                for (channel=0; channel<c->d; ++channel) values[channel]+=scores[slot]*key[channel];
            }
            rotation(values,c->d,c->r,-t,theta);
        }
        for (head=0; head<c->g; ++head)
            linear(attended+head*(width/c->g),w[7].data+(size_t)head*c->o*(width/c->g),
                   grouped+head*c->o,c->o,width/c->g);
        linear(grouped,w[8].data,output+(size_t)t*c->h,c->h,c->g*c->o);
    }
    return 0;
}

static float gated(float gate, float up, float limit, int activation) {
    /* Comparisons preserve NaN rather than fmin/fmax replacing it with a bound. */
    if (gate>limit) gate=limit;
    if (up>limit) up=limit;
    if (up< -limit) up= -limit;
    if (activation) return (gate<0.0f ? 0.0f : gate)*up;
    return (gate*logistic(gate))*up;
}

static int moe(const Config *c, const Layer *l, const float *x, int64_t token, float *out) {
    float affinity[32], selection[32], gu[512], up[256], activation[256], branch[128], sum=0.0f;
    int picks[8], e, rank, j, code;
    const Tensor *w=l->w;
    linear(x,w[26].data,affinity,c->e,c->h);
    for (e=0; e<c->e; ++e) {
        if (!isfinite(affinity[e])) return 3;
    }
    for (e=0; e<c->e; ++e) {
        float z=affinity[e];
        float softplus=z>20.0f ? z : log1pf(expf(z));
        affinity[e]=sqrtf(softplus);
    }
    for (e=0; e<c->e; ++e) {
        selection[e]=affinity[e]+(l->hash ? 0.0f : w[27].data[e]);
    }
    if (l->hash) for (rank=0; rank<c->top; ++rank) picks[rank]=(int)l->table[token*c->top+rank];
    else {
        code=select_entries(selection,c->e,c->top,picks);
        if (code) return code;
    }
    for (rank=0; rank<c->top; ++rank) sum+=affinity[picks[rank]];
    linear(x,w[30].data,gu,c->m,c->h);
    linear(x,w[31].data,up,c->m,c->h);
    for (j=0; j<c->m; ++j) activation[j]=gated(gu[j],up[j],c->limit,c->activation);
    linear(activation,w[32].data,out,c->h,c->m);
    for (rank=0; rank<c->top; ++rank) {
        float weight=affinity[picks[rank]]/(sum+1e-20f)*c->route_scale;
        e=picks[rank];
        linear(x,w[28].data+(size_t)e*2*c->m*c->h,gu,2*c->m,c->h);
        for (j=0; j<c->m; ++j) activation[j]=gated(gu[j],gu[c->m+j],c->limit,c->activation);
        linear(activation,w[29].data+(size_t)e*c->h*c->m,branch,c->h,c->m);
        for (j=0; j<c->h; ++j) out[j]+=branch[j]*weight;
    }
    return 0;
}

/* Same conservative workspace ceiling as the Python contract, in bytes. */
API uint64_t oa_deepseek_v4_workspace(const Config *c, int tokens) {
    uint64_t per;
    if (oa_deepseek_v4_geometry(c) || tokens<1 || tokens>c->capacity) return 0;
    per=(uint64_t)(8*c->u*c->h+16*c->h+8*c->a*c->d+8*c->q+
                   24*c->d+24*c->i+4*c->j*c->i+8*c->u*c->u+
                   12*c->m+8*c->e+4*c->vocab);
    return 4*((uint64_t)tokens*per+65536)+(uint64_t)tokens*32;
}

/* The C operation is stateless: no persistent buffers or weights are modified.
 * On an error, output is unspecified, and the wrapper must not commit history.
 */
API int oa_deepseek_v4_forward(const Config *c, const Tensor *global, const Layer *layers,
                              const int64_t *ids, int n, Tensor output, uint64_t budget) {
    uint64_t global_sizes[6], bytes, offset;
    Arena arena;
    float *streams, *logits;
    int row, layer, site, stream, code=0;
    if (oa_deepseek_v4_geometry(c) || !global || !layers || !ids || n<1 || n>c->capacity ||
        !output.data || output.count!=(uint64_t)n*c->vocab) return 1;
    global_sizes[0]=global_sizes[5]=(uint64_t)c->vocab*c->h;
    global_sizes[1]=(uint64_t)c->h; global_sizes[2]=(uint64_t)c->u*c->u*c->h;
    global_sizes[3]=(uint64_t)c->u; global_sizes[4]=1;
    for (site=0; site<6; ++site) if (!valid(global[site],global_sizes[site])) return 1;
    for (layer=0; layer<c->layers; ++layer) if (!manifest(c,layers+layer)) return 1;
    for (row=0; row<n; ++row) if (ids[row]<0 || ids[row]>=c->vocab) return 1;
    bytes=oa_deepseek_v4_workspace(c,n);
    if (!bytes || bytes>budget) return 2;
    /* The public ceiling also reserves output copies, token logs, metadata and
     * fixed stack scratch. These are not part of the allocated C arena. */
    arena.count=(size_t)(bytes/sizeof(float))-(size_t)n*(4*c->vocab+8)-32768;
    arena.used=0;
    arena.data=(float *)calloc(arena.count,sizeof(float));
    if (!arena.data) return 2;
    streams=take(&arena,(size_t)n*c->u*c->h);
    if (!streams) { free(arena.data); return 2; }
    for (row=0; row<n; ++row) for (stream=0; stream<c->u; ++stream)
        memcpy(streams+((size_t)row*c->u+stream)*c->h,
               global[0].data+(size_t)ids[row]*c->h,(size_t)c->h*sizeof(float));
    offset=(uint64_t)arena.used;
    for (layer=0; layer<c->layers && !code; ++layer) {
        const Layer *l=layers+layer;
        for (site=0; site<2 && !code; ++site) {
            float *x, *post, *comb, *branch;
            arena.used=(size_t)offset;
            x=take(&arena,(size_t)n*c->h); post=take(&arena,(size_t)n*c->u);
            comb=take(&arena,(size_t)n*c->u*c->u); branch=take(&arena,(size_t)n*c->h);
            if (!x || !post || !comb || !branch) { code=2; break; }
            for (row=0; row<n; ++row) {
                hyper(c,l->w+20+site*3,streams+(size_t)row*c->u*c->h,
                      x+(size_t)row*c->h,post+(size_t)row*c->u,comb+(size_t)row*c->u*c->u,0);
                norm(x+(size_t)row*c->h,c->h,l->w[site].data,c->eps);
            }
            if (!site) code=attention(c,l,&arena,x,n,branch);
            else for (row=0; row<n && !code; ++row)
                code=moe(c,l,x+(size_t)row*c->h,ids[row],branch+(size_t)row*c->h);
            if (!code) for (row=0; row<n; ++row)
                merge(c,streams+(size_t)row*c->u*c->h,branch+(size_t)row*c->h,
                      post+(size_t)row*c->u,comb+(size_t)row*c->u*c->u);
        }
    }
    logits=(float *)output.data;
    if (!code) for (row=0; row<n && !code; ++row) {
        float collapsed[128];
        hyper(c,global+2,streams+(size_t)row*c->u*c->h,collapsed,NULL,NULL,1);
        norm(collapsed,c->h,global[1].data,c->eps);
        linear(collapsed,global[5].data,logits+(size_t)row*c->vocab,c->vocab,c->h);
        for (site=0; site<c->vocab; ++site) if (!isfinite(logits[(size_t)row*c->vocab+site])) { code=3; break; }
    }
    free(arena.data);
    return code;
}
