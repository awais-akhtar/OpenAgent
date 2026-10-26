#include "segment_family_models.h"
#include "sha256.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

typedef struct family_model family_model;
struct family_model {
    family_model *next;uint64_t id,resident,used,limit,state_words,work,scratch,refs;
    uint32_t family,width,vocab,context,layers,rows;char *root;char identity[65];
    oa_family_model_descriptor d;oa_family_tensor *weights;
    int32_t *schedule;oa_family_inkling_layer *inkling;
    oa_family_inkling_range inkling_range;
    oa_family_glm_range glm_range;
    oa_family_glm_range_observed glm_observed;
    oa_original_experts *expert_bank;
    oa_family_glm_range_streamed glm_range_streamed;
    oa_family_inkling_streamed inkling_streamed;
    oa_family_inkling_range_streamed inkling_range_streamed;
};
typedef struct {family_model *model;uint32_t begin,end;uint64_t state_words,charge;} family_range;
typedef struct {family_model *model;float *state,*last_routes;uint64_t position,charge,state_words,route_capacity,last_route_count,last_route_position;uint32_t begin,end,last_route_rows,last_route_begin,last_route_end,last_route_experts;char identity[65];} family_session;
typedef struct {family_model *model;uint64_t charge;} family_edge;
static family_model *models;static uint64_t next_model=1;static unsigned model_count,registered;
static atomic_flag family_guard=ATOMIC_FLAG_INIT;
static void lock_models(void){while(atomic_flag_test_and_set_explicit(&family_guard,memory_order_acquire)){} }
static void unlock_models(void){atomic_flag_clear_explicit(&family_guard,memory_order_release);}
static int fail(char *error,size_t bytes,const char *message){if(error&&bytes)snprintf(error,bytes,"%s",message);return -1;}
static int finite_values(const float *p,uint64_t n){uint64_t i;if(!p&&n)return 0;for(i=0;i<n;i++)if(!isfinite(p[i]))return 0;return 1;}
static int cancelled(oa_se_cancel callback,void *data){return callback&&callback(data);}
typedef struct {oa_original_experts *bank;oa_se_cancel cancel;void *cancel_data;} family_expert_request;
static int family_glm_expert(void *owner,int32_t layer,int32_t expert,const float *input,float *output) {
    family_expert_request *request=(family_expert_request *)owner;
    return oa_original_expert_evaluate_cancel(request->bank,layer,expert,input,output,request->cancel,request->cancel_data);
}
static int family_inkling_expert(void *owner,uint64_t layer,uint64_t expert,const float *input,float *output) {
    if(layer>INT32_MAX||expert>INT32_MAX)return 1;
    return family_glm_expert(owner,(int32_t)layer,(int32_t)expert,input,output);
}
static int ink_geometry(const oa_family_inkling_config *c,
                        const oa_family_inkling_layer *l) {
    if (!c || !l || !c->hidden || c->hidden > 1048576 ||
        !c->vocab || c->vocab > 1048576 || !c->output_vocab ||
        c->output_vocab > c->vocab || !c->layers || c->layers > 128 ||
        !c->width || c->width > 1048576 || !c->kernel || c->kernel > 64 ||
        !c->relative || c->relative > 1048576 || !c->capacity ||
        c->capacity > 65536 || c->floor > 1048576 ||
        !isfinite(c->epsilon) || c->epsilon <= 0 ||
        !isfinite(c->multiplier) || c->multiplier <= 0 ||
        !isfinite(c->alpha) || c->alpha < 0 || c->alpha > 100 ||
        !c->moe_width || c->moe_width > 1048576 ||
        !c->experts || c->experts > 65536 || !c->shared ||
        c->shared > 65536 || c->experts + c->shared > 65536 ||
        !c->topk || c->topk > c->experts ||
        !isfinite(c->route_scale) || c->route_scale <= 0) return 0;
    if (!l->heads || l->heads > 65536 || !l->kv_heads ||
        l->kv_heads > 65536 || !l->dim || l->dim > 65536 ||
        l->heads % l->kv_heads || l->heads * l->dim > 65536 ||
        l->kv_heads * l->dim > 65536 || l->heads * c->relative > 65536 ||
        !l->extent || l->extent > 1048576 || l->window > 1048576 ||
        l->window == 1 || l->sparse > 1 ||
        (l->window && l->extent != l->window)) return 0;
    return 1;
}
static int glm_geometry(const int32_t *c,const float *f){unsigned i;for(i=0;i<23;i++)if(c[i]<1)return 0;for(i=0;i<6;i++)if(!isfinite(f[i]))return 0;return c[0]<=16384&&c[1]<=1048576&&c[2]<=128&&c[3]<=8&&c[4]<=64&&c[5]<=65536&&c[6]<=65536&&c[7]<=4096&&c[8]<=32&&c[9]<=c[7]&&c[10]<=256&&c[11]<=8192&&c[12]<=8192&&c[13]<=512&&c[14]<=512&&c[15]<=256&&c[16]<=512&&c[17]<=65536&&c[18]<=32&&c[17]%c[18]==0&&c[19]<=256&&c[20]<=512&&c[21]<=32&&c[22]<=65536&&(c[23]==0||c[23]==1)&&c[24]>=1&&c[24]<=4096&&c[25]>=1&&c[25]<=c[24]&&c[7]%c[24]==0&&c[7]/c[24]>=2&&c[9]<=c[25]*(c[7]/c[24])&&f[0]>0&&f[1]>0&&f[2]>0&&f[3]>0&&(f[5]==0||(f[5]==1&&f[4]<0));}
static uint64_t glm_scratch(const int32_t *c){uint64_t width=(uint64_t)c[19]*c[20],h=c[0],r=c[3];uint64_t kda=8*width+c[20]+c[19],dsa=(uint64_t)c[11]+c[12]+(uint64_t)c[10]*c[13]+(uint64_t)c[10]*(c[13]+c[14])+(uint64_t)c[15]*c[16]+c[15]+2*(uint64_t)c[22]+c[16]+c[18]+(uint64_t)c[10]*c[14],feed=2*(uint64_t)c[7]+h+2*(uint64_t)c[8]*c[6],scratch=kda>dsa?kda:dsa;if(feed>scratch)scratch=feed;if(2*(uint64_t)c[5]>scratch)scratch=2*(uint64_t)c[5];return 4*(3*r*h+2*h+r+r*r+(r+2)*r+scratch);}
static uint64_t ink_scratch(const family_model *m){const oa_family_inkling_config *c=&m->d.inkling_config;uint64_t q=0,v=0,r=0,width=c->width,experts=c->experts+c->shared;unsigned i;if(c->moe_width>width)width=c->moe_width;for(i=0;i<m->layers;i++){const oa_family_inkling_layer *l=&m->inkling[i];if(l->heads*l->dim>q)q=l->heads*l->dim;if(l->kv_heads*l->dim>v)v=l->kv_heads*l->dim;if(l->heads*c->relative>r)r=l->heads*c->relative;}return 4*(5*c->hidden+2*q+2*v+r+2*width+c->capacity+3*experts);}
static void put64(unsigned char *p,uint64_t value){unsigned i;for(i=0;i<8;i++)p[i]=(unsigned char)(value>>(8*i));}
static uint64_t get64(const unsigned char *p){uint64_t value=0;unsigned i;for(i=0;i<8;i++)value|=(uint64_t)p[i]<<(8*i);return value;}
static uint64_t ink_state(const family_model *m,const oa_family_inkling_layer *l){uint64_t v=l->kv_heads*l->dim;return 2*(uint64_t)m->context*v+2*m->d.inkling_config.kernel*(v+m->width);}
static uint64_t glm_state(const family_model *m,unsigned layer,unsigned slot){const int32_t *c=m->d.glm_config;if(m->schedule[3*layer])return slot?(uint64_t)c[19]*c[20]*c[20]:(uint64_t)3*c[19]*c[20]*c[21];return slot?(uint64_t)m->context*2*c[16]:(uint64_t)m->context*c[10]*(c[13]+c[14]);}
static int clone_tensor(family_model *m,oa_family_tensor source,oa_family_tensor *out){uint64_t charge;if(source.count>SIZE_MAX/4||source.count>(m->limit-m->resident)/4||!finite_values(source.data,source.count))return 0;out->data=NULL;out->count=source.count;if(!source.count)return 1;charge=source.count*4;out->data=(float *)malloc((size_t)charge);if(!out->data)return 0;memcpy(out->data,source.data,(size_t)charge);m->resident+=charge;return 1;}
static void free_model(family_model *m){unsigned i,j;if(!m)return;if(m->weights)for(i=0;i<m->layers*42;i++)free(m->weights[i].data);if(m->inkling)for(i=0;i<m->layers;i++)for(j=0;j<26;j++)free(m->inkling[i].weights[j].data);for(i=0;i<4;i++)free(m->d.globals[i].data);free(m->weights);free(m->inkling);free(m->schedule);free(m->root);free(m);}
static family_model *find_model(const char *root,unsigned family){family_model *m;for(m=models;m;m=m->next)if(m->family==family&&root&&!strcmp(root,m->root))return m;return NULL;}
static int acquire(family_model *m,uint64_t charge){int ok=0;lock_models();if(charge<=m->limit-m->used){m->used+=charge;m->refs++;ok=1;}unlock_models();return ok;}
static void release(family_model *m,uint64_t charge){lock_models();m->used-=charge;m->refs--;unlock_models();}
static const char *engine_name(unsigned family){return family==1?"openagent-glm5-native-model":"openagent-inkling-native-model";}
static const char *wire_name(unsigned family){return family==1?"openagent.glm5.final-hidden.f32.v1":"openagent.inkling.final-hidden.f32.v1";}
static const char *model_wire(const family_model *m) {
    if(m->glm_range)return "openagent.glm5.range-streams-mask.f32.v1";
    return m->family==2 && m->inkling_range ? "openagent.inkling.range-residual.f32.v1" : wire_name(m->family);
}
static uint32_t wire_width(const family_model *m) {
    return m->glm_range?(uint32_t)((uint64_t)m->width*m->d.glm_config[3]+m->context):m->width;
}
static int segment_open(unsigned family,void **out,oa_se_segment_caps *caps,
    const oa_se_segment_options *options,char *error,size_t bytes) {
    family_model *m;family_range *range;uint64_t words=0;uint32_t i;
    if(out)*out=NULL;
    if(caps)memset(caps,0,sizeof(*caps));
    if(!out||!caps||!options) return fail(error,bytes,"native family open arguments required");
    lock_models();m=find_model(options->model,family);
    if(m && options->begin<options->end && options->end<=m->layers &&
       ((family==2 && m->inkling_range) || (family==1 && m->glm_range) ||
        (options->begin==0 && options->end==m->layers)) &&
       options->context==m->context && options->memory>=m->resident &&
       options->backend==(UINT64_C(1)<<8)) m->refs++;else m=NULL;
    unlock_models();
    if(!m) return fail(error,bytes,"admitted native family layer range required");
    for(i=(uint32_t)options->begin;i<(uint32_t)options->end;i++)
        words+=family==2?ink_state(m,m->inkling+i):glm_state(m,i,0)+glm_state(m,i,1);
    if(!acquire(m,sizeof(*range))) {release(m,0);return fail(error,bytes,"native range ownership admission exhausted");}
    release(m,0);range=(family_range *)calloc(1,sizeof(*range));
    if(!range) {release(m,sizeof(*range));return fail(error,bytes,"native range allocation failed");}
    range->model=m;range->begin=(uint32_t)options->begin;range->end=(uint32_t)options->end;
    range->state_words=words;range->charge=sizeof(*range);
    memset(caps,0,sizeof(*caps));caps->size=sizeof(*caps);caps->version=1;
    caps->flags=(UINT64_C(1)<<8)|2|4|8;
    snprintf(caps->engine,sizeof(caps->engine),"%s",engine_name(family));
    snprintf(caps->schema,sizeof(caps->schema),"%s",model_wire(m));
    snprintf(caps->numeric,sizeof(caps->numeric),"openagent.native-token-fp32.v1");
    caps->dtype=1;caps->width=wire_width(m);caps->rows=m->rows;caps->context=m->context;caps->layers=m->layers;
    *out=range;return 0;
}
static int glm_open(void **o,oa_se_segment_caps *c,const oa_se_segment_options *p,char *e,size_t n){return segment_open(1,o,c,p,e,n);}
static int ink_open(void **o,oa_se_segment_caps *c,const oa_se_segment_options *p,char *e,size_t n){return segment_open(2,o,c,p,e,n);}
static void segment_destroy(void *owner) {
    family_range *range=(family_range *)owner;
    if(range) {release(range->model,range->charge);free(range);}
}
static int session_create(void *owner,void **out,const oa_se_session_options *options,char *error,size_t bytes) {
    family_range *range=(family_range *)owner;family_model *m;family_session *s;uint64_t charge,route_words=0,route_bytes=0,state_bytes;
    if(out)*out=NULL;
    if(!range||!out||!options) return fail(error,bytes,"native family session arguments required");
    m=range->model;
    if(options->context!=m->context) return fail(error,bytes,"native family session context differs");
    if(range->state_words>UINT64_MAX/4)return fail(error,bytes,"native family session state extent overflows");
    state_bytes=range->state_words*4;
    if(m->family==2 || m->glm_observed) {
        uint64_t layers=range->end-range->begin,width=m->family==2?
            m->d.inkling_config.experts+m->d.inkling_config.shared:(uint64_t)m->d.glm_config[7];
        if(layers&&m->rows>UINT64_MAX/layers)return fail(error,bytes,"Inkling route observation geometry overflows");
        route_words=(uint64_t)m->rows*layers;
        if(width&&route_words>UINT64_MAX/width)return fail(error,bytes,"Inkling route observation geometry overflows");
        route_words*=width;
        if(route_words>SIZE_MAX/sizeof(float)||route_words>UINT64_MAX/sizeof(float))
            return fail(error,bytes,"Inkling route observation allocation exceeds address space");
        route_bytes=route_words*sizeof(float);
    }
    if(state_bytes>UINT64_MAX-m->work||state_bytes+m->work>UINT64_MAX-sizeof(*s)||
       state_bytes+m->work+sizeof(*s)>UINT64_MAX-route_bytes)
        return fail(error,bytes,"native family session aggregate extent overflows");
    charge=state_bytes+m->work+sizeof(*s)+route_bytes;
    if(options->memory<charge||!acquire(m,charge)) return fail(error,bytes,"native family aggregate session admission exhausted");
    s=(family_session *)calloc(1,sizeof(*s));
    if(s) {
        s->state=(float *)calloc((size_t)range->state_words,sizeof(float));
        if(route_words)s->last_routes=(float *)malloc((size_t)route_bytes);
    }
    if(!s||!s->state||(route_words&&!s->last_routes)) {
        if(s){free(s->last_routes);free(s->state);}free(s);release(m,charge);
        return fail(error,bytes,"native range session allocation failed");
    }
    s->model=m;s->charge=charge;s->state_words=range->state_words;s->begin=range->begin;s->end=range->end;
    s->route_capacity=route_words;
    memcpy(s->identity,m->identity,65);
    if(s->begin || s->end!=m->layers) {
        oa_sha256 hash;unsigned char digest[32],bounds[16];unsigned i;
        put64(bounds,s->begin);put64(bounds+8,s->end);
        oa_sha_init(&hash);oa_sha_update(&hash,m->identity,64);oa_sha_update(&hash,bounds,sizeof(bounds));oa_sha_finish(&hash,digest);
        for(i=0;i<32;i++) snprintf(s->identity+2*i,3,"%02x",digest[i]);
    }
    *out=s;return 0;
}
static void session_destroy(void *owner){family_session *s=(family_session *)owner;if(!s)return;release(s->model,s->charge);free(s->last_routes);free(s->state);free(s);}
static int session_run(void *owner,const oa_se_run *request,char *error,size_t bytes) {
    family_session *s=(family_session *)owner;family_model *m;
    float *pending=NULL,*hidden=NULL,*logits=NULL,*routing=NULL,*glm_routes=NULL;int64_t *ids=NULL;
    uint64_t offset=0,i,temporary,routes=0;uint32_t width;int status=0;
    family_expert_request expert_request;
    if(!s||!request)return fail(error,bytes,"native family run arguments required");
    m=s->model;width=wire_width(m);
    expert_request.bank=m->expert_bank;expert_request.cancel=request->cancel;expert_request.cancel_data=request->cancel_data;
    if(!request->rows||request->rows>m->rows||request->position!=s->position||
       request->rows>m->context-s->position||request->token_count!=request->rows||
       !request->tokens||request->input_bytes!=(size_t)request->rows*width*4||
       request->output_bytes!=request->input_bytes||!request->input||!request->output||
       !finite_values((const float *)request->input,(uint64_t)request->rows*width))
        return fail(error,bytes,"native family range geometry/continuation differs");
    for(i=0;i<request->rows;i++)if(request->tokens[i]<0||
        (uint32_t)request->tokens[i]>=m->d.globals[0].count/m->width)
        return fail(error,bytes,"native family token outside vocabulary");
    if(cancelled(request->cancel,request->cancel_data))return fail(error,bytes,"native family request cancelled");
    temporary=s->state_words*4+(uint64_t)request->rows*width*4+(uint64_t)request->rows*8;
    if(m->family==2) {
        routes=(uint64_t)request->rows*(s->end-s->begin)*(m->d.inkling_config.experts+m->d.inkling_config.shared);
        temporary+=routes*4;
        if(!m->inkling_range)temporary+=(uint64_t)request->rows*m->vocab*4;
    }
    else if(m->glm_observed) {
        routes=(uint64_t)request->rows*(s->end-s->begin)*(uint64_t)m->d.glm_config[7];
        temporary+=(routes+2*(uint64_t)m->layers*m->d.glm_config[7])*4;
    }
    if(temporary>m->work||m->scratch>m->work-temporary)
        return fail(error,bytes,"native family range transaction exceeds work admission");
    pending=(float *)malloc((size_t)s->state_words*4);
    hidden=(float *)malloc((size_t)request->rows*width*4);
    ids=(int64_t *)malloc((size_t)request->rows*8);
    if(!pending||!hidden||!ids){status=1;goto done;}
    memcpy(pending,s->state,(size_t)s->state_words*4);
    for(i=0;i<request->rows;i++)ids[i]=request->tokens[i];
    if(m->family==1) {
        oa_family_tensor cache[256],empty={NULL,0};unsigned layer,slot;memset(cache,0,sizeof(cache));
        uint64_t route_extent=(uint64_t)m->layers*m->d.glm_config[7];
        if(m->glm_observed) {
            routing=(float *)malloc((size_t)routes*sizeof(float));
            glm_routes=(float *)malloc((size_t)(2*route_extent)*sizeof(float));
            if(!routing||!glm_routes){status=1;goto done;}
        }
        for(layer=s->begin;layer<s->end;layer++)for(slot=0;slot<2;slot++) {
            cache[2*layer+slot].data=pending+offset;
            cache[2*layer+slot].count=glm_state(m,layer,slot);offset+=cache[2*layer+slot].count;
        }
        for(i=0;i<request->rows;i++) {
            oa_family_tensor input={(float *)request->input+i*width,width},output={hidden+i*width,width};
            if(cancelled(request->cancel,request->cancel_data)){status=2;goto done;}
            if(m->glm_observed) {
                oa_family_tensor observed_logits={glm_routes,route_extent},
                    observed_weights={glm_routes+route_extent,route_extent};
                uint64_t local_extent=(uint64_t)(s->end-s->begin)*m->d.glm_config[7];
                status=m->glm_observed(m->d.glm_config,m->d.glm_floats,m->schedule,m->weights,
                    m->d.globals[0],m->d.globals[2],cache,(int32_t)(s->position+i),ids[i],
                    (int32_t)s->begin,(int32_t)s->end,input,output,
                    m->expert_bank?family_glm_expert:NULL,&expert_request,observed_logits,observed_weights);
                if(!status)memcpy(routing+i*local_extent,
                    observed_weights.data+(uint64_t)s->begin*m->d.glm_config[7],
                    (size_t)local_extent*sizeof(float));
            }
            else if(m->glm_range && m->expert_bank)status=m->glm_range_streamed(m->d.glm_config,m->d.glm_floats,m->schedule,m->weights,
                m->d.globals[0],m->d.globals[2],cache,(int32_t)(s->position+i),ids[i],
                (int32_t)s->begin,(int32_t)s->end,input,output,family_glm_expert,&expert_request);
            else if(m->glm_range)status=m->glm_range(m->d.glm_config,m->d.glm_floats,m->schedule,m->weights,
                m->d.globals[0],m->d.globals[2],cache,(int32_t)(s->position+i),ids[i],
                (int32_t)s->begin,(int32_t)s->end,input,output);
            else status=m->d.glm_step(m->d.glm_config,m->d.glm_floats,m->schedule,m->weights,
                m->d.globals[0],m->d.globals[2],empty,cache,(int32_t)(s->position+i),ids[i],output,empty,input,
                m->expert_bank?family_glm_expert:NULL,&expert_request);
            if(status)goto done;
        }
    } else {
        oa_family_inkling_layer layers[128];
        oa_family_tensor h={hidden,(uint64_t)request->rows*width},l={NULL,0},r,
            input={(float *)request->input,(uint64_t)request->rows*width};
        routing=(float *)malloc((size_t)routes*4);
        if(!routing){status=1;goto done;}
        r.data=routing;r.count=routes;
        memcpy(layers,m->inkling,(size_t)m->layers*sizeof(*layers));
        for(i=s->begin;i<s->end;i++) {
            layers[i].state.data=pending+offset;layers[i].state.count=ink_state(m,layers+i);
            offset+=layers[i].state.count;
        }
        if(m->inkling_range) {
            uint64_t route_width=(s->end-s->begin)*(m->d.inkling_config.experts+m->d.inkling_config.shared);
            for(i=0;i<request->rows;i++) {
                oa_family_tensor row_hidden={hidden+i*width,width},
                    row_routing={routing+i*route_width,route_width},
                    row_input={(float *)request->input+i*width,width};
                if(cancelled(request->cancel,request->cancel_data)){status=2;goto done;}
                if(m->expert_bank)status=m->inkling_range_streamed(&m->d.inkling_config,layers,m->d.globals,
                    ids+i,1,1,1,s->position+i,s->begin,s->end,row_hidden,row_routing,row_input,
                    family_inkling_expert,&expert_request);
                else status=m->inkling_range(&m->d.inkling_config,layers,m->d.globals,
                    ids+i,1,1,1,s->position+i,s->begin,s->end,row_hidden,row_routing,row_input);
                if(status)goto done;
            }
        }
        else {
            logits=(float *)malloc((size_t)request->rows*m->vocab*4);
            if(!logits){status=1;goto done;}
            l.data=logits;l.count=(uint64_t)request->rows*m->vocab;
            if(m->expert_bank)status=m->inkling_streamed(&m->d.inkling_config,layers,m->d.globals,ids,
                request->rows,1,request->rows,s->position,h,l,r,input,family_inkling_expert,&expert_request);
            else status=m->d.inkling_run(&m->d.inkling_config,layers,m->d.globals,ids,
                request->rows,1,request->rows,s->position,h,l,r,input);
        }
        if(status)goto done;
    }
    if(cancelled(request->cancel,request->cancel_data)){status=2;goto done;}
    if(!finite_values(hidden,(uint64_t)request->rows*width)||!finite_values(pending,s->state_words)||
       ((m->family==2||m->glm_observed)&&(!routing||routes>s->route_capacity||!finite_values(routing,routes)))) {status=3;goto done;}
    memcpy(request->output,hidden,(size_t)request->rows*width*4);
    if(m->family==2||m->glm_observed) {
        memcpy(s->last_routes,routing,(size_t)routes*sizeof(float));
        s->last_route_count=routes;s->last_route_rows=request->rows;
        s->last_route_position=request->position;s->last_route_begin=s->begin;s->last_route_end=s->end;
        s->last_route_experts=m->family==2?(uint32_t)(m->d.inkling_config.experts+m->d.inkling_config.shared):
            (uint32_t)m->d.glm_config[7];
    }
    memcpy(s->state,pending,(size_t)s->state_words*4);s->position+=request->rows;
 done:
    if(status&&cancelled(request->cancel,request->cancel_data))status=2;
    free(pending);free(hidden);free(ids);free(logits);free(routing);free(glm_routes);
    return status?fail(error,bytes,status==2?"native family request cancelled":"native family staged range decoder failed"):0;
}
static int snapshot(void *owner,oa_se_write writer,void *data,char *error,size_t bytes){family_session *s=(family_session *)owner;unsigned char header[96],digest[32];oa_sha256 hash;uint64_t offset,total;if(!s||!writer)return fail(error,bytes,"native snapshot writer required");memset(header,0,sizeof(header));memcpy(header,"OACFAM01",8);put64(header+8,s->position);put64(header+16,s->state_words);put64(header+24,s->model->family);memcpy(header+32,s->identity,64);oa_sha_init(&hash);oa_sha_update(&hash,header,sizeof(header));if(writer(data,header,sizeof(header)))return fail(error,bytes,"native snapshot header write failed");total=s->state_words*4;for(offset=0;offset<total;){size_t count=(size_t)(total-offset);if(count>1048576)count=1048576;oa_sha_update(&hash,(unsigned char *)s->state+offset,count);if(writer(data,(unsigned char *)s->state+offset,count))return fail(error,bytes,"native snapshot payload write failed");offset+=count;}oa_sha_finish(&hash,digest);return writer(data,digest,sizeof(digest))?fail(error,bytes,"native snapshot digest write failed"):0;}
static int restore(void *owner,oa_se_read reader,void *data,char *error,size_t bytes){family_session *s=(family_session *)owner;unsigned char header[96],actual[32],expected[32];oa_sha256 hash;float *pending;uint64_t position,total,offset;unsigned difference=0,i;if(!s||!reader||reader(data,header,sizeof(header)))return fail(error,bytes,"native snapshot header read failed");position=get64(header+8);if(memcmp(header,"OACFAM01",8)||get64(header+16)!=s->state_words||get64(header+24)!=s->model->family||memcmp(header+32,s->identity,64)||position>s->model->context)return fail(error,bytes,"native snapshot identity/geometry differs");total=s->state_words*4;if(total>s->model->work)return fail(error,bytes,"native restore work admission exhausted");pending=(float *)malloc((size_t)total);if(!pending)return fail(error,bytes,"native restore allocation failed");oa_sha_init(&hash);oa_sha_update(&hash,header,sizeof(header));for(offset=0;offset<total;){size_t count=(size_t)(total-offset);if(count>1048576)count=1048576;if(reader(data,(unsigned char *)pending+offset,count)){free(pending);return fail(error,bytes,"native snapshot payload read failed");}oa_sha_update(&hash,(unsigned char *)pending+offset,count);offset+=count;}oa_sha_finish(&hash,expected);if(reader(data,actual,sizeof(actual))){free(pending);return fail(error,bytes,"native snapshot digest read failed");}for(i=0;i<32;i++)difference|=actual[i]^expected[i];if(difference||!finite_values(pending,s->state_words)){free(pending);return fail(error,bytes,"native snapshot digest/finite state differs");}memcpy(s->state,pending,(size_t)total);s->position=position;s->last_route_count=0;s->last_route_rows=0;free(pending);return 0;}
static int edge_open(unsigned family,void **out,oa_se_edge_caps *caps,const oa_se_edge_options *options,char *error,size_t bytes){
    family_model *m;family_edge *edge;uint64_t charge;
    if(out)*out=NULL;
    if(caps)memset(caps,0,sizeof(*caps));
    if(!out||!caps||!options||!options->model)
        return fail(error,bytes,"native edge open arguments required");
    lock_models();m=find_model(options->model,family);if(m)m->refs++;unlock_models();
    if(!m)return fail(error,bytes,"admitted native edge model required");
    charge=(uint64_t)m->rows*m->vocab*4+(uint64_t)m->rows*wire_width(m)*4+(uint64_t)m->width*4+sizeof(*edge);
    if(options->backend!=(UINT64_C(1)<<8)||options->memory<m->resident+charge||!acquire(m,charge)){release(m,0);return fail(error,bytes,"native edge aggregate admission exhausted");}
    release(m,0);edge=(family_edge *)malloc(sizeof(*edge));
    if(!edge){release(m,charge);return fail(error,bytes,"native edge allocation failed");}
    edge->model=m;edge->charge=charge;
    caps->size=sizeof(*caps);caps->version=2;caps->flags=(UINT64_C(1)<<8)|4|8;
    snprintf(caps->engine,sizeof(caps->engine),"%s",engine_name(family));
    snprintf(caps->schema,sizeof(caps->schema),"%s",model_wire(m));
    snprintf(caps->numeric,sizeof(caps->numeric),"openagent.native-token-fp32.v1");
    snprintf(caps->tokenizer,sizeof(caps->tokenizer),"external-pinned-tokenizer");
    caps->dtype=1;caps->width=wire_width(m);caps->vocab=m->vocab;caps->rows=m->rows;
    caps->context=m->context;caps->layers=m->layers;caps->bos=caps->eos=-1;
    caps->resident_bytes=m->resident;*out=edge;return 0;
}
static int glm_edge_open(void **o,oa_se_edge_caps *c,const oa_se_edge_options *p,char *e,size_t n){return edge_open(1,o,c,p,e,n);}
static int ink_edge_open(void **o,oa_se_edge_caps *c,const oa_se_edge_options *p,char *e,size_t n){return edge_open(2,o,c,p,e,n);}
static void edge_destroy(void *owner){family_edge *e=(family_edge *)owner;if(e){release(e->model,e->charge);free(e);}}
static int tokenize(void *owner,const char *input,size_t length,int32_t *output,size_t capacity,size_t *written,char *error,size_t bytes){(void)owner;(void)input;(void)length;(void)output;(void)capacity;(void)written;return fail(error,bytes,"native family edge requires external pinned tokenization");}
static int detokenize(void *owner,const int32_t *input,size_t length,char *output,size_t capacity,size_t *written,char *error,size_t bytes){(void)owner;(void)input;(void)length;(void)output;(void)capacity;(void)written;return fail(error,bytes,"native family edge requires external pinned tokenization");}
static int embed(void *owner,const oa_se_embed *request,char *error,size_t bytes) {
    family_edge *e=(family_edge *)owner;family_model *m;
    float *candidate;uint64_t row,d,stream;uint32_t width;
    if(!e||!request)return fail(error,bytes,"native embedding arguments required");
    m=e->model;width=wire_width(m);
    if(!request->rows||request->rows>m->rows||request->token_count!=request->rows||
       request->output_bytes!=(size_t)request->rows*width*4||!request->tokens||!request->output)
        return fail(error,bytes,"native embedding geometry differs");
    candidate=(float *)calloc(1,request->output_bytes);
    if(!candidate)return fail(error,bytes,"native embedding allocation failed");
    for(row=0;row<request->rows;row++) {
        const float *source;float scale=1,sum=0;
        if(request->tokens[row]<0||(uint32_t)request->tokens[row]>=m->d.globals[0].count/m->width||
           cancelled(request->cancel,request->cancel_data)) {
            free(candidate);return fail(error,bytes,"native embedding token/cancellation differs");
        }
        source=m->d.globals[0].data+(uint64_t)request->tokens[row]*m->width;
        if(m->family==2) {
            for(d=0;d<m->width;d++)sum+=source[d]*source[d];
            scale=1/sqrtf(sum/m->width+m->d.inkling_config.epsilon);
        }
        for(d=0;d<m->width;d++)candidate[row*width+d]=source[d]*scale*(m->family==2?m->d.globals[1].data[d]:1);
        if(m->glm_range)for(stream=1;stream<(uint64_t)m->d.glm_config[3];stream++)
            memcpy(candidate+row*width+stream*m->width,candidate+row*width,(size_t)m->width*4);
    }
    if(!finite_values(candidate,(uint64_t)request->rows*width)||cancelled(request->cancel,request->cancel_data)) {
        free(candidate);return fail(error,bytes,"nonfinite/cancelled native embedding");
    }
    memcpy(request->output,candidate,request->output_bytes);free(candidate);return 0;
}
static int glm_final_row(const family_model *m,const float *wire,float *output) {
    uint64_t d,stream,streams=(uint64_t)m->d.glm_config[3];float sum=0,scale;
    for(d=0;d<m->context;d++) {
        float selected=wire[streams*m->width+d];if(selected!=0&&selected!=1)return 1;
    }
    for(d=0;d<m->width;d++) {
        float value=0;for(stream=0;stream<streams;stream++)value+=wire[stream*m->width+d];
        output[d]=value/(float)streams;sum+=output[d]*output[d];
    }
    if(!isfinite(sum))return 1;
    scale=1/sqrtf(sum/(float)m->width+m->d.glm_floats[0]);
    for(d=0;d<m->width;d++)output[d]=output[d]*scale*m->d.globals[2].data[d];
    return finite_values(output,m->width)?0:1;
}
static int head_product(family_edge *e,const float *input,uint32_t rows,float *output,oa_se_cancel callback,void *data) {
    family_model *m=e->model;uint64_t row,token,d;uint32_t width=wire_width(m);float *normalized=NULL;int status=0;
    if(!finite_values(input,(uint64_t)rows*width))return 1;
    if(m->glm_range) {
        normalized=(float *)malloc((size_t)m->width*4);if(!normalized)return 1;
    }
    for(row=0;row<rows;row++) {
        const float *source=input+row*width;
        if(m->glm_range) {
            if(glm_final_row(m,source,normalized)){status=1;break;}
            source=normalized;
        }
        for(token=0;token<m->vocab;token++) {
            float sum=0;
            if((token&255)==0&&cancelled(callback,data)){status=2;break;}
            for(d=0;d<m->width;d++)sum+=(m->family==2?source[d]/m->d.inkling_config.multiplier:source[d])*m->d.globals[3].data[token*m->width+d];
            output[row*m->vocab+token]=sum;
        }
        if(status)break;
    }
    free(normalized);return status?status:(finite_values(output,(uint64_t)rows*m->vocab)?0:1);
}
static int logits(void *owner,const oa_se_logits *request,char *error,size_t bytes){
    family_edge *e=(family_edge *)owner;family_model *m;float *candidate;uint64_t count;int status;
    if(!e||!request)return fail(error,bytes,"native head arguments required");
    m=e->model;count=(uint64_t)request->rows*m->vocab;
    if(!request->rows||request->rows>m->rows||request->input_bytes!=(size_t)request->rows*wire_width(m)*4||request->logits_capacity<count||!request->input||!request->logits)return fail(error,bytes,"native head geometry differs");
    candidate=(float *)malloc((size_t)count*4);if(!candidate)return fail(error,bytes,"native head allocation failed");
    status=head_product(e,(const float *)request->input,request->rows,candidate,request->cancel,request->cancel_data);
    if(!status&&!cancelled(request->cancel,request->cancel_data))memcpy(request->logits,candidate,(size_t)count*4);else status=1;
    free(candidate);return status?fail(error,bytes,"native head failed/cancelled"):0;
}
static int select_tokens(void *owner,const oa_se_select *request,char *error,size_t bytes){
    family_edge *e=(family_edge *)owner;family_model *m;float *candidate;int32_t chosen[4096];float score[4096];uint64_t row,token;int status;
    if(!e||!request)return fail(error,bytes,"native greedy selection arguments required");
    m=e->model;
    if(!request->rows||request->rows>m->rows||request->rows>4096||request->input_bytes!=(size_t)request->rows*wire_width(m)*4||request->token_capacity<request->rows||request->score_capacity<request->rows||!request->tokens||!request->scores||!request->input)return fail(error,bytes,"native greedy selection geometry differs");
    candidate=(float *)malloc((size_t)request->rows*m->vocab*4);if(!candidate)return fail(error,bytes,"native greedy allocation failed");
    status=head_product(e,(const float *)request->input,request->rows,candidate,request->cancel,request->cancel_data);
    if(!status){for(row=0;row<request->rows;row++){chosen[row]=0;score[row]=candidate[row*m->vocab];for(token=1;token<m->vocab;token++)if(candidate[row*m->vocab+token]>score[row]){score[row]=candidate[row*m->vocab+token];chosen[row]=(int32_t)token;}}if(cancelled(request->cancel,request->cancel_data))status=1;else{memcpy(request->tokens,chosen,(size_t)request->rows*4);memcpy(request->scores,score,(size_t)request->rows*4);}}
    free(candidate);return status?fail(error,bytes,"native greedy selection failed/cancelled"):0;
}
static int observe_inkling_routes(void *owner,uint32_t rows,uint64_t position,float *output,
    size_t capacity,size_t *written,uint32_t *begin,uint32_t *end,uint32_t *experts,
    char *error,size_t bytes) {
    family_session *s=(family_session *)owner;
    if(!s||!s->model||(s->model->family!=2&&!s->model->glm_observed)||!written||!begin||!end||!experts||
       !s->last_route_count||s->last_route_rows!=rows||s->last_route_position!=position||
       s->last_route_begin!=s->begin||s->last_route_end!=s->end||
       s->last_route_count>SIZE_MAX||!finite_values(s->last_routes,s->last_route_count))
        return fail(error,bytes,"executed family route observation is unavailable or stale");
    *written=(size_t)s->last_route_count;*begin=s->last_route_begin;
    *end=s->last_route_end;*experts=s->last_route_experts;
    if(!output&&!capacity)return 0;
    if(!output||capacity<s->last_route_count)
        return fail(error,bytes,"Inkling route observation output capacity is too small");
    memcpy(output,s->last_routes,(size_t)s->last_route_count*sizeof(float));
    return 0;
}
static int register_adapters(void) {
    oa_se_segment_adapter segment;
    oa_se_edge_adapter edge;
    unsigned family;
    /* Preserve successful registrations if a later registry allocation fails.
       Retrying must not register an already installed engine a second time. */
    for (family = 1; family <= 2; family++) {
        unsigned segment_bit = 1u << ((family - 1u) * 2u);
        unsigned edge_bit = segment_bit << 1u;
        if (!(registered & segment_bit)) {
            memset(&segment, 0, sizeof(segment));
            segment.size = sizeof(segment);
            segment.version = 1;
            segment.engine = engine_name(family);
            segment.open = family == 1 ? glm_open : ink_open;
            segment.destroy = segment_destroy;
            segment.create = session_create;
            segment.destroy_session = session_destroy;
            segment.run = session_run;
            segment.snapshot = snapshot;
            segment.restore = restore;
            if (coli_segment_adapter_register(&segment)) return 0;
            registered |= segment_bit;
        }
        if (!(registered & edge_bit)) {
            memset(&edge, 0, sizeof(edge));
            edge.size = sizeof(edge);
            edge.version = 2;
            edge.engine = engine_name(family);
            edge.open = family == 1 ? glm_edge_open : ink_edge_open;
            edge.destroy = edge_destroy;
            edge.tokenize = tokenize;
            edge.detokenize = detokenize;
            edge.embed = embed;
            edge.select = select_tokens;
            edge.logits = logits;
            if (coli_edge_adapter_register(&edge)) return 0;
            registered |= edge_bit;
        }
    }
    return registered == 15u;
}
uint32_t oa_segment_family_version(void){return 1;}
size_t oa_segment_family_layout(uint32_t record,uint64_t *output,size_t capacity){
#define FIELD(type,name) ((uint64_t)offsetof(type,name))
    const uint64_t descriptor[]={sizeof(oa_family_model_descriptor),17,FIELD(oa_family_model_descriptor,size),FIELD(oa_family_model_descriptor,version),FIELD(oa_family_model_descriptor,family),FIELD(oa_family_model_descriptor,rows),FIELD(oa_family_model_descriptor,model),FIELD(oa_family_model_descriptor,identity),FIELD(oa_family_model_descriptor,max_resource_bytes),FIELD(oa_family_model_descriptor,max_work_bytes),FIELD(oa_family_model_descriptor,glm_config),FIELD(oa_family_model_descriptor,glm_floats),FIELD(oa_family_model_descriptor,glm_schedule),FIELD(oa_family_model_descriptor,glm_weights),FIELD(oa_family_model_descriptor,glm_step),FIELD(oa_family_model_descriptor,inkling_config),FIELD(oa_family_model_descriptor,inkling_layers),FIELD(oa_family_model_descriptor,inkling_run),FIELD(oa_family_model_descriptor,globals)};
    const uint64_t tensor[]={sizeof(oa_family_tensor),2,FIELD(oa_family_tensor,data),FIELD(oa_family_tensor,count)};
    const uint64_t config[]={sizeof(oa_family_inkling_config),17,FIELD(oa_family_inkling_config,hidden),FIELD(oa_family_inkling_config,vocab),FIELD(oa_family_inkling_config,output_vocab),FIELD(oa_family_inkling_config,layers),FIELD(oa_family_inkling_config,width),FIELD(oa_family_inkling_config,kernel),FIELD(oa_family_inkling_config,relative),FIELD(oa_family_inkling_config,capacity),FIELD(oa_family_inkling_config,floor),FIELD(oa_family_inkling_config,moe_width),FIELD(oa_family_inkling_config,experts),FIELD(oa_family_inkling_config,shared),FIELD(oa_family_inkling_config,topk),FIELD(oa_family_inkling_config,epsilon),FIELD(oa_family_inkling_config,multiplier),FIELD(oa_family_inkling_config,alpha),FIELD(oa_family_inkling_config,route_scale)};
    const uint64_t layer[]={sizeof(oa_family_inkling_layer),8,FIELD(oa_family_inkling_layer,heads),FIELD(oa_family_inkling_layer,kv_heads),FIELD(oa_family_inkling_layer,dim),FIELD(oa_family_inkling_layer,extent),FIELD(oa_family_inkling_layer,window),FIELD(oa_family_inkling_layer,sparse),FIELD(oa_family_inkling_layer,weights),FIELD(oa_family_inkling_layer,state)};
    const uint64_t *values=NULL;size_t count=0;if(record==1){values=descriptor;count=sizeof(descriptor)/8;}else if(record==2){values=tensor;count=sizeof(tensor)/8;}else if(record==3){values=config;count=sizeof(config)/8;}else if(record==4){values=layer;count=sizeof(layer)/8;}if(!values||!output||capacity<count)return 0;memcpy(output,values,count*8);return count;
#undef FIELD
}
int oa_segment_family_initialize(void){int status=0;lock_models();if(registered!=15u && !register_adapters())status=-1;if(!status&&coli_segment_route_observer_register("openagent-inkling-native-model",observe_inkling_routes))status=-1;if(!status&&coli_segment_route_observer_register("openagent-glm5-native-model",observe_inkling_routes))status=-1;unlock_models();return status;}
int oa_segment_family_register(const oa_family_model_descriptor *descriptor,uint64_t *id,char *error,size_t bytes){family_model *m;unsigned i,j;size_t length;uint16_t endian=1;if(id)*id=0;if(!descriptor||descriptor->size!=sizeof(*descriptor)||descriptor->version!=1||!id||descriptor->family<1||descriptor->family>2||!descriptor->model||!descriptor->identity||strlen(descriptor->identity)!=64||!descriptor->rows||descriptor->rows>4096||!descriptor->max_resource_bytes||!descriptor->max_work_bytes||sizeof(float)!=4||FLT_RADIX!=2||FLT_MANT_DIG!=24||FLT_MAX_EXP!=128||*(unsigned char *)&endian!=1)return fail(error,bytes,"explicit native family descriptor required");length=strlen(descriptor->model);if(!length||length>32768)return fail(error,bytes,"native family model root extent differs");m=(family_model *)calloc(1,sizeof(*m));if(!m)return fail(error,bytes,"native family model allocation failed");m->d=*descriptor;memset(m->d.globals,0,sizeof(m->d.globals));m->limit=descriptor->max_resource_bytes;m->work=descriptor->max_work_bytes;m->family=descriptor->family;m->rows=descriptor->rows;memcpy(m->identity,descriptor->identity,65);m->resident=sizeof(*m)+length+1;if(m->resident>m->limit)goto invalid;m->root=(char *)malloc(length+1);if(!m->root)goto invalid;memcpy(m->root,descriptor->model,length+1);if(m->family==1){const int32_t *c=descriptor->glm_config;if(!glm_geometry(c,descriptor->glm_floats)||!descriptor->glm_step||!descriptor->glm_weights||!descriptor->glm_schedule||c[0]<1||c[0]>16384||c[1]<1||c[1]>1048576||c[2]<1||c[2]>128||c[22]<1||c[22]>65536)goto invalid;for(i=0;i<26;i++)if(c[i]<0||c[i]>1048576)goto invalid;m->width=c[0];m->vocab=c[1];m->layers=c[2];m->context=c[22];m->scratch=glm_scratch(c);m->resident+=(uint64_t)m->layers*(42*sizeof(oa_family_tensor)+3*sizeof(int32_t));if(m->resident>m->limit)goto invalid;m->weights=(oa_family_tensor *)calloc((size_t)m->layers*42,sizeof(*m->weights));m->schedule=(int32_t *)malloc((size_t)m->layers*3*sizeof(int32_t));if(!m->weights||!m->schedule)goto invalid;memcpy(m->schedule,descriptor->glm_schedule,(size_t)m->layers*3*sizeof(int32_t));for(i=0;i<m->layers*3;i++)if(m->schedule[i]<0||m->schedule[i]>1)goto invalid;for(i=0;i<m->layers*42;i++)if(!clone_tensor(m,descriptor->glm_weights[i],m->weights+i))goto invalid;for(i=0;i<m->layers;i++)m->state_words+=glm_state(m,i,0)+glm_state(m,i,1);}else{const oa_family_inkling_config *c=&descriptor->inkling_config;if(!descriptor->inkling_run||!descriptor->inkling_layers||!ink_geometry(c,descriptor->inkling_layers)||!c->hidden||c->hidden>1048576||!c->output_vocab||c->output_vocab>1048576||!c->layers||c->layers>128||!c->capacity||c->capacity>65536||!c->kernel||c->kernel>64||!c->experts||c->experts>65536||!c->shared||c->shared>65536||!isfinite(c->epsilon)||c->epsilon<=0||!isfinite(c->multiplier)||c->multiplier<=0)goto invalid;m->width=(uint32_t)c->hidden;m->vocab=(uint32_t)c->output_vocab;m->layers=(uint32_t)c->layers;m->context=(uint32_t)c->capacity;m->resident+=(uint64_t)m->layers*sizeof(oa_family_inkling_layer);if(m->resident>m->limit)goto invalid;m->inkling=(oa_family_inkling_layer *)calloc(m->layers,sizeof(*m->inkling));if(!m->inkling)goto invalid;for(i=0;i<m->layers;i++){m->inkling[i]=descriptor->inkling_layers[i];memset(m->inkling[i].weights,0,sizeof(m->inkling[i].weights));m->inkling[i].state.data=NULL;m->inkling[i].state.count=0;if(!ink_geometry(c,&m->inkling[i]))goto invalid;for(j=0;j<26;j++)if(!clone_tensor(m,descriptor->inkling_layers[i].weights[j],&m->inkling[i].weights[j]))goto invalid;m->state_words+=ink_state(m,&m->inkling[i]);}}if(m->family==2)m->scratch=ink_scratch(m);if(m->state_words>SIZE_MAX/4||m->work<m->state_words*4||m->resident>=m->limit||m->work>m->limit-m->resident)goto invalid;for(i=0;i<4;i++)if(!clone_tensor(m,descriptor->globals[i],&m->d.globals[i]))goto invalid;if(m->d.globals[0].count!=(m->family==2?m->d.inkling_config.vocab:(uint64_t)m->vocab)*m->width||m->d.globals[2].count!=m->width||m->d.globals[3].count!=(m->family==2?m->d.inkling_config.vocab:(uint64_t)m->vocab)*m->width||(m->family==2&&m->d.globals[1].count!=m->width)||m->work>m->limit-m->resident)goto invalid;m->used=m->resident;lock_models();if(model_count>=128||next_model==UINT64_MAX||find_model(m->root,m->family)){unlock_models();goto invalid;}if(registered!=15u && !register_adapters()){unlock_models();goto invalid;}m->id=next_model++;m->next=models;models=m;model_count++;*id=m->id;unlock_models();return 0;
invalid:free_model(m);return fail(error,bytes,"native family model inventory/resource admission failed");}
int oa_segment_family_release(uint64_t id,char *error,size_t bytes) {
    family_model **link,*m;
    lock_models();for(link=&models;*link&&(*link)->id!=id;link=&(*link)->next){}
    m=*link;
    if(!m||m->refs){unlock_models();return fail(error,bytes,"native family model missing or still owns live handles");}
    if(m->expert_bank&&oa_original_experts_release(m->expert_bank)) {
        unlock_models();return fail(error,bytes,"native original expert ownership release failed");
    }
    m->expert_bank=NULL;*link=m->next;model_count--;unlock_models();free_model(m);return 0;
}
int oa_segment_family_enable_inkling_ranges(uint64_t id,oa_family_inkling_range run,char *error,size_t bytes) {
    family_model *m;int ok=0;
    if(!run)return fail(error,bytes,"native Inkling range function required");
    lock_models();
    for(m=models;m&&m->id!=id;m=m->next){}
    if(m&&m->family==2&&!m->refs&&!m->inkling_range&&(!m->expert_bank||m->inkling_range_streamed)) {m->inkling_range=run;ok=1;}
    unlock_models();
    return ok?0:fail(error,bytes,"idle registered Inkling model required for range extension");
}
int oa_segment_family_enable_glm_ranges(uint64_t id,oa_family_glm_range run,char *error,size_t bytes) {
    family_model *m;int ok=0;
    if(!run)return fail(error,bytes,"native GLM range function required");
    lock_models();for(m=models;m&&m->id!=id;m=m->next){}
    if(m&&m->family==1&&!m->refs&&!m->glm_range&&(!m->expert_bank||m->glm_range_streamed)) {m->glm_range=run;ok=1;}
    unlock_models();
    return ok?0:fail(error,bytes,"idle registered GLM model required for range extension");
}
int oa_segment_family_enable_glm_route_observer(uint64_t id,oa_family_glm_range_observed run,char *error,size_t bytes) {
    family_model *m;int ok=0;
    if(!run)return fail(error,bytes,"native GLM range observer required");
    lock_models();for(m=models;m&&m->id!=id;m=m->next){}
    if(m&&m->family==1&&!m->refs&&m->glm_range&&!m->glm_observed){m->glm_observed=run;ok=1;}
    unlock_models();
    return ok?0:fail(error,bytes,"idle GLM range model required for route observer");
}
int oa_segment_family_glm_final_hidden(uint64_t id,const float *input,uint32_t rows,
    uint64_t input_count,float *output,uint64_t output_count,char *error,size_t bytes) {
    family_model *m;uint32_t row,width;int status=0;
    lock_models();for(m=models;m&&m->id!=id;m=m->next){}
    if(m&&m->glm_range)m->refs++;else m=NULL;
    unlock_models();
    if(!m)return fail(error,bytes,"registered GLM stream model required");
    width=wire_width(m);
    if(!rows||rows>m->rows||input_count!=(uint64_t)rows*width||
       output_count!=(uint64_t)rows*m->width||!input||!output||!finite_values(input,input_count))status=1;
    else for(row=0;row<rows;row++)if(glm_final_row(m,input+(uint64_t)row*width,output+(uint64_t)row*m->width)) {status=1;break;}
    release(m,0);
    return status?fail(error,bytes,"native GLM final hidden geometry/values differ"):0;
}
int oa_segment_family_attach_original_experts(uint64_t id,oa_original_experts *bank,
    oa_family_glm_range_streamed glm,oa_family_inkling_streamed inkling,
    oa_family_inkling_range_streamed inkling_range,char *error,size_t bytes) {
    family_model *m;uint32_t layer,expert;uint64_t width,count;int status=0;
    if(!bank||oa_original_experts_retain(bank))return fail(error,bytes,"owned native original expert bank required");
    lock_models();for(m=models;m&&m->id!=id;m=m->next){}
    if(!m||m->refs||m->expert_bank||(m->family==1&&m->glm_range&&!glm)||
       (m->family==2&&(!inkling||(m->inkling_range&&!inkling_range))))status=1;
    if(!status) {
        count=m->family==1?(uint64_t)m->d.glm_config[7]:m->d.inkling_config.experts;
        width=m->family==1?(uint64_t)m->d.glm_config[6]:m->d.inkling_config.moe_width;
        for(layer=0;layer<m->layers&&!status;layer++)
            if(m->family==1?m->schedule[3*layer+1]:m->inkling[layer].sparse)
                for(expert=0;expert<count;expert++)if(!oa_original_experts_supports(bank,layer,expert,m->width,width)){status=1;break;}
    }
    if(!status) {
        m->expert_bank=bank;m->glm_range_streamed=glm;m->inkling_streamed=inkling;m->inkling_range_streamed=inkling_range;
    }
    unlock_models();
    if(status)oa_original_experts_release(bank);
    return status?fail(error,bytes,"native original experts differ from the idle model inventory"):0;
}
