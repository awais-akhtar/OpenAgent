#include "qwen4_public_session.h"
#include "qwen4_public_internal.h"
#include "sha256.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef _MSC_VER
#define Q4_THREAD_LOCAL __declspec(thread)
#else
#define Q4_THREAD_LOCAL _Thread_local
#endif
static Q4_THREAD_LOCAL char q4_observation_identity[65];
static Q4_THREAD_LOCAL oa_q4_routing_observer q4_observer;
static Q4_THREAD_LOCAL void *q4_observer_context;
static Q4_THREAD_LOCAL uint64_t q4_observer_generation;
int oa_qwen4_public_observation_begin(const char *identity,oa_q4_routing_observer observer,void *context){
    size_t i;
    if(!identity||!observer||q4_observer||strlen(identity)!=64)return -1;
    for(i=0;i<64;i++)if(!((identity[i]>='0'&&identity[i]<='9')||(identity[i]>='a'&&identity[i]<='f')))return -1;
    memcpy(q4_observation_identity,identity,65);q4_observer_context=context;q4_observer=observer;
    q4_observer_generation++;return 0;
}
int oa_qwen4_public_observation_end(const char *identity,oa_q4_routing_observer observer,void *context){
    if(!identity||!q4_observer||observer!=q4_observer||context!=q4_observer_context||strcmp(identity,q4_observation_identity))return -1;
    q4_observer=NULL;q4_observer_context=NULL;memset(q4_observation_identity,0,65);
    q4_observer_generation++;return 0;
}
typedef struct {oa_q4_tensor cache[3],history;int64_t *previous;uint64_t previous_count;} q4_state;
struct oa_q4_public_session {
    oa_q4_public_model *model;atomic_flag busy;
    uint32_t begin,end;uint64_t position,charge,state_bytes;
    q4_state *state;uint8_t *mask;int64_t *positions;
};
static int q4_error(char *e,size_t n,const char *s){if(e&&n)snprintf(e,n,"%s",s);return -1;}
static void q4_lock(oa_q4_public_model *m){while(atomic_flag_test_and_set(&m->guard)){} }
static void q4_unlock(oa_q4_public_model *m){atomic_flag_clear(&m->guard);}
static int q4_add(uint64_t *used,uint64_t count,unsigned width,uint64_t limit){
    if(!used||!width||*used>limit||count>SIZE_MAX/width||count>UINT64_MAX/width||
        count*width>limit-*used)return 0;
    *used+=count*width;return 1;
}
static uint64_t q4_count(const oa_q4_public_model *m,uint32_t layer,unsigned slot,uint64_t position){
    const oa_q4_geometry *c=&m->config.text;
    if(!position)return 0;
    if(!m->layers[layer].kind){
        if(slot==0)return (2*c->key_heads*c->key_dim+c->value_heads*c->value_dim)*c->kernel;
        return slot==1?c->value_heads*c->key_dim*c->value_dim:0;
    }
    return slot<2?c->kv*position*c->dim:position*m->config.index_dim;
}
static uint64_t q4_history(const oa_q4_public_model *m,uint32_t layer,uint64_t position){
    return position&&m->layers[layer].ple_enabled?
        m->config.streams*m->config.text.hidden*(m->config.ple_kernel-1)*m->config.ple_ngram:0;
}
static void q4_free(q4_state *state,uint32_t layers){uint32_t i,j;if(!state)return;
    for(i=0;i<layers;i++){for(j=0;j<3;j++)free(state[i].cache[j].data);free(state[i].history.data);free(state[i].previous);}free(state);
}
static int q4_extent(oa_q4_public_session *s,uint64_t position,uint64_t *bytes){
    oa_q4_public_model *m=s->model;uint32_t i,j;*bytes=0;
    if(position>m->config.text.capacity)return 0;
    for(i=s->begin;i<s->end;i++){
        for(j=0;j<3;j++)if(!q4_add(bytes,q4_count(m,i,j,position),4,m->work))return 0;
        if(!q4_add(bytes,q4_history(m,i,position),4,m->work))return 0;
        if(position&&m->layers[i].ple_enabled&&!q4_history(m,i,position)&&!q4_add(bytes,1,4,m->work))return 0;
        if(position&&m->layers[i].ple_enabled&&!q4_add(bytes,m->config.ple_ngram-1,8,m->work))return 0;
    }
    return q4_add(bytes,position,1,m->work)&&q4_add(bytes,position*(m->config.mrope_enabled?3:1),8,m->work);
}
static q4_state *q4_allocate(oa_q4_public_session *s,uint64_t position){
    oa_q4_public_model *m=s->model;q4_state *state;uint32_t i,j;uint64_t count;
    state=(q4_state *)calloc((size_t)m->config.layers,sizeof(*state));if(!state)return NULL;
    for(i=s->begin;i<s->end;i++){
        for(j=0;j<3;j++){
            count=q4_count(m,i,j,position);state[i].cache[j].count=count;
            if(count&&!(state[i].cache[j].data=(float *)calloc((size_t)count,4)))goto invalid;
        }
        count=q4_history(m,i,position);state[i].history.count=count;
        if(position&&m->layers[i].ple_enabled&&!(state[i].history.data=(float *)calloc((size_t)(count?count:1),4)))goto invalid;
        if(position&&m->layers[i].ple_enabled){
            state[i].previous_count=m->config.ple_ngram-1;
            state[i].previous=(int64_t *)malloc((size_t)state[i].previous_count*8);if(!state[i].previous)goto invalid;
            for(j=0;j<state[i].previous_count;j++)state[i].previous[j]=(int64_t)m->config.eos;
        }
    }
    return state;
invalid:q4_free(state,(uint32_t)m->config.layers);return NULL;
}
int oa_qwen4_public_session_create(oa_q4_public_model *m,uint32_t begin,uint32_t end,uint64_t maximum,
    oa_q4_public_session **out,char *error,size_t bytes){
    oa_q4_public_session *s;uint64_t charge;
    /* A failed constructor must never leave a stale caller-owned handle behind. */
    if(out)*out=NULL;
    if(!m||!out||begin>=end||end>m->config.layers||oa_qwen4_public_model_retain(m))return q4_error(error,bytes,"admitted Qwen4 layer range required");
    charge=sizeof(*s)+m->config.layers*sizeof(q4_state)+m->work;
    q4_lock(m);if(charge>maximum||charge>m->limit-m->used){q4_unlock(m);oa_qwen4_public_model_release(m);return q4_error(error,bytes,"Qwen4 session aggregate admission exhausted");}
    m->used+=charge;q4_unlock(m);
    s=(oa_q4_public_session *)calloc(1,sizeof(*s));
    if(s){s->model=m;s->begin=begin;s->end=end;s->charge=charge;atomic_flag_clear(&s->busy);s->state=q4_allocate(s,0);}
    if(!s||!s->state){free(s);q4_lock(m);m->used-=charge;q4_unlock(m);oa_qwen4_public_model_release(m);return q4_error(error,bytes,"Qwen4 session allocation failed");}
    *out=s;return 0;
}
int oa_qwen4_public_session_close(oa_q4_public_session *s){
    oa_q4_public_model *m;if(!s||atomic_flag_test_and_set(&s->busy))return -1;m=s->model;
    q4_free(s->state,(uint32_t)m->config.layers);free(s->mask);free(s->positions);
    q4_lock(m);m->used-=s->charge;q4_unlock(m);free(s);oa_qwen4_public_model_release(m);return 0;
}
int oa_qwen4_public_session_run(oa_q4_public_session *s,const oa_se_run *r,char *error,size_t error_bytes){
    oa_q4_public_model *m;oa_q4_compose_layer *layers=NULL;oa_q4_compose_trace *traces=NULL;
    q4_state *pending=NULL;uint8_t *mask=NULL;int64_t *positions=NULL;float *output=NULL;
    uint64_t width,stride,end,extent,charge,rows,i,j,k,observer_generation=0;
    oa_q4_routing_observer observer=NULL;void *observer_context=NULL;
    oa_q4_compose_config config;oa_q4_cancel_guard cancel_state;int status=-1;
    if(!s||!r||atomic_flag_test_and_set(&s->busy))return q4_error(error,error_bytes,"Qwen4 session unavailable");
    m=s->model;rows=r->rows;end=s->position+rows;width=m->config.streams*m->config.text.hidden;stride=width+5;
    cancel_state=(oa_q4_cancel_guard){r->cancel,r->cancel_data,0};
    if(!rows||rows>m->rows||r->position!=s->position||end>m->config.text.capacity||!r->input||!r->output||
        r->input_bytes!=rows*stride*4||r->output_bytes<rows*stride*4||!q4_extent(s,end,&extent))goto finish;
    charge=s->state_bytes;
    if(!q4_add(&charge,extent,1,m->work)||!q4_add(&charge,rows*stride,4,m->work)||
        !q4_add(&charge,m->config.layers,sizeof(*layers)+sizeof(*traces)+sizeof(q4_state),m->work))goto finish;
    for(i=s->begin;i<s->end;i++){
        if(!q4_add(&charge,rows*(width+m->config.text.experts+m->config.text.topk),4,m->work)||
            !q4_add(&charge,rows*m->config.text.topk,8,m->work))goto finish;
        if(m->layers[i].kind&&(!q4_add(&charge,rows*m->config.text.heads*end,4,m->work)||!q4_add(&charge,rows*end,1,m->work)))goto finish;
    }
    if(charge>=m->work)goto finish;
    pending=q4_allocate(s,end);layers=(oa_q4_compose_layer *)malloc((size_t)m->config.layers*sizeof(*layers));
    traces=(oa_q4_compose_trace *)calloc((size_t)m->config.layers,sizeof(*traces));
    mask=(uint8_t *)calloc((size_t)end,1);positions=(int64_t *)calloc((size_t)end*(m->config.mrope_enabled?3:1),8);
    output=(float *)malloc((size_t)rows*stride*4);
    if(!pending||!layers||!traces||!mask||!positions||!output)goto finish;
    memcpy(layers,m->layers,(size_t)m->config.layers*sizeof(*layers));
    if(s->position){memcpy(mask,s->mask,(size_t)s->position);for(j=0;j<(m->config.mrope_enabled?3:1);j++)
        memcpy(positions+j*end,s->positions+j*s->position,(size_t)s->position*8);}
    for(i=s->begin;i<s->end;i++){
        for(j=0;j<3;j++){
            layers[i].cache[j]=pending[i].cache[j];
            if(!s->position||!s->state[i].cache[j].count)continue;
            if(m->layers[i].kind&&j<2)for(k=0;k<m->config.text.kv;k++)
                memcpy(pending[i].cache[j].data+k*end*m->config.text.dim,
                    s->state[i].cache[j].data+k*s->position*m->config.text.dim,(size_t)s->position*m->config.text.dim*4);
            else memcpy(pending[i].cache[j].data,s->state[i].cache[j].data,(size_t)s->state[i].cache[j].count*4);
        }
        layers[i].ple_history=pending[i].history;layers[i].previous=pending[i].previous;layers[i].previous_count=pending[i].previous_count;
        if(s->position&&pending[i].previous){memcpy(pending[i].previous,s->state[i].previous,(size_t)pending[i].previous_count*8);
            if(pending[i].history.count)memcpy(pending[i].history.data,s->state[i].history.data,(size_t)pending[i].history.count*4);}
        traces[i].incoming_hidden=(oa_q4_tensor){(float *)malloc((size_t)rows*width*4),rows*width};
        traces[i].router=(oa_q4_tensor){(float *)malloc((size_t)rows*m->config.text.experts*4),rows*m->config.text.experts};
        traces[i].scores=(oa_q4_tensor){(float *)malloc((size_t)rows*m->config.text.topk*4),rows*m->config.text.topk};
        traces[i].selected_count=rows*m->config.text.topk;traces[i].selected=(int64_t *)malloc((size_t)traces[i].selected_count*8);
        if(!traces[i].incoming_hidden.data||!traces[i].router.data||!traces[i].scores.data||!traces[i].selected)goto finish;
        if(m->layers[i].kind){traces[i].attention=(oa_q4_tensor){(float *)malloc((size_t)rows*m->config.text.heads*end*4),rows*m->config.text.heads*end};
            traces[i].chosen_count=rows*end;traces[i].chosen=(uint8_t *)malloc((size_t)rows*end);
            if(!traces[i].attention.data||!traces[i].chosen)goto finish;}
    }
    config=m->config;config.max_work_bytes=m->work-charge;
    status=m->wire(&config,layers,traces,rows,s->position,s->begin,s->end,
        (oa_q4_tensor){(float *)r->input,rows*stride},(oa_q4_tensor){output,rows*stride},mask,end,
        positions,end*(config.mrope_enabled?3:1),m->experts?oa_original_expert_evaluate_counted64:NULL,m->experts,
        m->ple?oa_original_rows_lookup:NULL,NULL,
        r->cancel?oa_q4_checked_cancel:NULL,r->cancel?&cancel_state:NULL);
    /* The kernel selects the row callback only for layers carrying an owned
       native table context; resident PLE layers retain their ordinary table. */
    if(status||oa_q4_checked_cancel(&cancel_state)) {status=-1;goto finish;}
    observer=q4_observer;observer_context=q4_observer_context;
    if(observer&&!strcmp(q4_observation_identity,m->identity)){
        observer_generation=q4_observer_generation;
        for(i=s->begin;i<s->end;i++){
            if(oa_q4_checked_cancel(&cancel_state)||observer(observer_context,(uint32_t)i,rows,
                m->config.text.experts,m->config.text.topk,traces[i].router.data,traces[i].selected,traces[i].scores.data)||
                q4_observer_generation!=observer_generation||q4_observer!=observer||q4_observer_context!=observer_context||
                strcmp(q4_observation_identity,m->identity)){
                status=-1;goto finish;
            }
        }
    }
    if(oa_q4_checked_cancel(&cancel_state)){status=-1;goto finish;}
    q4_free(s->state,(uint32_t)m->config.layers);free(s->mask);free(s->positions);
    s->state=pending;pending=NULL;s->mask=mask;mask=NULL;s->positions=positions;positions=NULL;
    s->position=end;s->state_bytes=extent;memcpy(r->output,output,(size_t)rows*stride*4);status=0;
finish:
    if(traces)for(i=s->begin;i<s->end;i++){free(traces[i].incoming_hidden.data);free(traces[i].router.data);free(traces[i].scores.data);
        free(traces[i].selected);free(traces[i].attention.data);free(traces[i].chosen);}
    free(traces);free(layers);q4_free(pending,(uint32_t)m->config.layers);free(mask);free(positions);free(output);
    atomic_flag_clear(&s->busy);return status?q4_error(error,error_bytes,"Qwen4 native append failed; continuation unchanged"):0;
}

static void q4_put64(unsigned char *p,uint64_t value){unsigned i;for(i=0;i<8;i++)p[i]=(unsigned char)(value>>(8*i));}
static uint64_t q4_get64(const unsigned char *p){uint64_t value=0;unsigned i;for(i=0;i<8;i++)value|=(uint64_t)p[i]<<(8*i);return value;}
static uint64_t q4_payload(oa_q4_public_session *s,uint64_t position){
    uint64_t result=position*(1+8*(s->model->config.mrope_enabled?3:1));uint32_t i,j;
    for(i=s->begin;i<s->end;i++){
        for(j=0;j<3;j++)result+=4*q4_count(s->model,i,j,position);
        result+=4*q4_history(s->model,i,position);
        if(position&&s->model->layers[i].ple_enabled)result+=8*(s->model->config.ple_ngram-1);
    }
    return result;
}
static int q4_block(void *pointer,uint64_t bytes,oa_sha256 *hash,oa_se_write writer,oa_se_read reader,void *context){
    uint64_t offset=0;if(bytes&&!pointer)return -1;
    while(offset<bytes){size_t count=(size_t)(bytes-offset);if(count>1048576)count=1048576;
        if(reader&&reader(context,(unsigned char *)pointer+offset,count))return -1;
        oa_sha_update(hash,(unsigned char *)pointer+offset,count);
        if(writer&&writer(context,(unsigned char *)pointer+offset,count))return -1;offset+=count;}
    return 0;
}
static int q4_stream(oa_q4_public_session *s,q4_state *state,uint8_t *mask,int64_t *positions,uint64_t position,
    oa_sha256 *hash,oa_se_write writer,oa_se_read reader,void *context){
    uint32_t i,j;
    for(i=s->begin;i<s->end;i++){
        for(j=0;j<3;j++)if(q4_block(state[i].cache[j].data,4*q4_count(s->model,i,j,position),hash,writer,reader,context))return -1;
        if(q4_block(state[i].history.data,4*q4_history(s->model,i,position),hash,writer,reader,context))return -1;
        if(position&&s->model->layers[i].ple_enabled&&q4_block(state[i].previous,
            8*(s->model->config.ple_ngram-1),hash,writer,reader,context))return -1;
    }
    return q4_block(mask,position,hash,writer,reader,context)||
        q4_block(positions,position*8*(s->model->config.mrope_enabled?3:1),hash,writer,reader,context)?-1:0;
}
static int q4_finite_state(oa_q4_public_session *s,q4_state *state,uint8_t *mask,int64_t *positions,uint64_t position){
    uint64_t i,j,k;oa_q4_public_model *m=s->model;
    for(i=s->begin;i<s->end;i++){
        for(j=0;j<3;j++)for(k=0;k<state[i].cache[j].count;k++)if(!isfinite(state[i].cache[j].data[k]))return 0;
        for(k=0;k<state[i].history.count;k++)if(!isfinite(state[i].history.data[k]))return 0;
        for(k=0;k<state[i].previous_count;k++)if(state[i].previous[k]<0||(uint64_t)state[i].previous[k]>=m->config.vocabulary)return 0;
    }
    for(i=0;i<position;i++)if(mask[i]>1)return 0;
    for(i=0;i<position*(m->config.mrope_enabled?3:1);i++)if(positions[i]<0||(uint64_t)positions[i]>=m->config.text.capacity)return 0;
    return 1;
}
int oa_qwen4_public_session_snapshot(oa_q4_public_session *s,oa_se_write writer,void *context,char *error,size_t bytes){
    unsigned char header[128],digest[32];oa_sha256 hash;int status=-1;
    if(!s||!writer||atomic_flag_test_and_set(&s->busy))return q4_error(error,bytes,"Qwen4 snapshot owner unavailable");
    if(!q4_finite_state(s,s->state,s->mask,s->positions,s->position))goto finish;
    memset(header,0,sizeof(header));memcpy(header,"OAQ4PUB1",8);q4_put64(header+8,s->position);
    q4_put64(header+16,q4_payload(s,s->position));q4_put64(header+24,s->begin);q4_put64(header+32,s->end);
    memcpy(header+40,s->model->identity,64);q4_put64(header+104,s->model->config.text.capacity);
    q4_put64(header+112,s->model->config.streams*s->model->config.text.hidden+5);
    q4_put64(header+120,s->model->config.mrope_enabled);
    oa_sha_init(&hash);oa_sha_update(&hash,header,sizeof(header));
    if(writer(context,header,sizeof(header))||q4_stream(s,s->state,s->mask,s->positions,s->position,&hash,writer,NULL,context))goto finish;
    oa_sha_finish(&hash,digest);if(writer(context,digest,sizeof(digest)))goto finish;status=0;
finish:atomic_flag_clear(&s->busy);return status?q4_error(error,bytes,"Qwen4 complete snapshot write failed"):0;
}
int oa_qwen4_public_session_restore(oa_q4_public_session *s,oa_se_read reader,void *context,char *error,size_t bytes){
    unsigned char header[128],actual[32],expected[32];oa_sha256 hash;q4_state *state=NULL;
    uint8_t *mask=NULL;int64_t *positions=NULL;uint64_t position=0,extent,charge;unsigned i,difference=0;int status=-1;
    if(!s||!reader||atomic_flag_test_and_set(&s->busy))return q4_error(error,bytes,"Qwen4 restoration owner unavailable");
    if(reader(context,header,sizeof(header)))goto finish;position=q4_get64(header+8);
    if(memcmp(header,"OAQ4PUB1",8)||position>s->model->config.text.capacity||
        q4_get64(header+24)!=s->begin||q4_get64(header+32)!=s->end||memcmp(header+40,s->model->identity,64)||
        q4_get64(header+104)!=s->model->config.text.capacity||q4_get64(header+112)!=s->model->config.streams*s->model->config.text.hidden+5||
        q4_get64(header+120)!=s->model->config.mrope_enabled||!q4_extent(s,position,&extent)||
        q4_get64(header+16)!=q4_payload(s,position))goto finish;
    charge=s->state_bytes;if(!q4_add(&charge,extent,1,s->model->work)||
        !q4_add(&charge,s->model->config.layers,sizeof(q4_state),s->model->work))goto finish;
    state=q4_allocate(s,position);if(!state)goto finish;
    if(position){mask=(uint8_t *)malloc((size_t)position);positions=(int64_t *)malloc((size_t)position*(s->model->config.mrope_enabled?3:1)*8);
        if(!mask||!positions)goto finish;}
    oa_sha_init(&hash);oa_sha_update(&hash,header,sizeof(header));
    if(q4_stream(s,state,mask,positions,position,&hash,NULL,reader,context)||reader(context,actual,sizeof(actual)))goto finish;
    oa_sha_finish(&hash,expected);for(i=0;i<32;i++)difference|=actual[i]^expected[i];
    if(difference||!q4_finite_state(s,state,mask,positions,position))goto finish;
    q4_free(s->state,(uint32_t)s->model->config.layers);free(s->mask);free(s->positions);
    s->state=state;state=NULL;s->mask=mask;mask=NULL;s->positions=positions;positions=NULL;
    s->position=position;s->state_bytes=extent;status=0;
finish:q4_free(state,(uint32_t)s->model->config.layers);free(mask);free(positions);
    atomic_flag_clear(&s->busy);return status?q4_error(error,bytes,"Qwen4 complete restoration failed; continuation unchanged"):0;
}
