#include "qwen4_public_session.h"
#include "qwen4_public_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
typedef struct q4_entry {struct q4_entry *next;uint64_t id;oa_q4_public_model *model;} q4_entry;
typedef struct {oa_q4_public_model *model;uint32_t begin,end;uint64_t charge;} q4_range;
typedef struct {oa_q4_public_model *model;uint64_t charge;} q4_edge;
static q4_entry *entries;static uint64_t next_id=1;static unsigned entry_count,installed;
static atomic_flag registry_guard=ATOMIC_FLAG_INIT;
static const char *q4_engine="openagent-qwen4-native-model";
static const char *q4_schema="openagent.qwen4.hyper-token-mask-mrope.f32.v1";
static void registry_lock(void){while(atomic_flag_test_and_set(&registry_guard)){} }
static void registry_unlock(void){atomic_flag_clear(&registry_guard);}
static int error(char *e,size_t n,const char *s){if(e&&n)snprintf(e,n,"%s",s);return -1;}
static uint64_t width(const oa_q4_public_model *m){return m->config.streams*m->config.text.hidden+5;}
static oa_q4_public_model *find(const char *root){q4_entry *e;oa_q4_public_model *m=NULL;
    registry_lock();for(e=entries;e;e=e->next)if(root&&!strcmp(root,e->model->root)){
        if(!oa_qwen4_public_model_retain(e->model))m=e->model;break;}registry_unlock();return m;}
static int reserve(oa_q4_public_model *m,uint64_t bytes){int result=0;
    while(atomic_flag_test_and_set(&m->guard)){} if(bytes<=m->limit-m->used){m->used+=bytes;result=1;}atomic_flag_clear(&m->guard);return result;}
static void unreserve(oa_q4_public_model *m,uint64_t bytes){while(atomic_flag_test_and_set(&m->guard)){}m->used-=bytes;atomic_flag_clear(&m->guard);}
static int open_range(void **out,oa_se_segment_caps *caps,const oa_se_segment_options *p,char *e,size_t n){
    oa_q4_public_model *m;q4_range *r;
    if(!out||!caps||!p||p->backend!=(UINT64_C(1)<<8))return error(e,n,"Qwen4 native CPU range options required");
    m=find(p->model);if(!m)return error(e,n,"registered Qwen4 native model required");
    if(p->begin>=p->end||p->end>m->config.layers||p->context!=m->config.text.capacity||p->memory<m->resident||!reserve(m,sizeof(*r))){
        oa_qwen4_public_model_release(m);return error(e,n,"Qwen4 range geometry or aggregate admission differs");}
    r=(q4_range *)malloc(sizeof(*r));if(!r){unreserve(m,sizeof(*r));oa_qwen4_public_model_release(m);return error(e,n,"Qwen4 range allocation failed");}
    r->model=m;r->begin=p->begin;r->end=p->end;r->charge=sizeof(*r);
    memset(caps,0,sizeof(*caps));caps->size=sizeof(*caps);caps->version=1;caps->flags=(UINT64_C(1)<<8)|2|4|8;
    snprintf(caps->engine,sizeof(caps->engine),"%s",q4_engine);snprintf(caps->schema,sizeof(caps->schema),"%s",q4_schema);
    snprintf(caps->numeric,sizeof(caps->numeric),"openagent.native-token-fp32.v1");caps->dtype=1;
    caps->width=(uint32_t)width(m);caps->rows=m->rows;caps->context=(uint32_t)m->config.text.capacity;caps->layers=(uint32_t)m->config.layers;
    *out=r;return 0;
}
static void destroy_range(void *owner){q4_range *r=(q4_range *)owner;if(r){unreserve(r->model,r->charge);oa_qwen4_public_model_release(r->model);free(r);}}
static int create_session(void *owner,void **out,const oa_se_session_options *p,char *e,size_t n){
    q4_range *r=(q4_range *)owner;if(!r||!out||!p||p->context!=r->model->config.text.capacity)return error(e,n,"matching Qwen4 session context required");
    return oa_qwen4_public_session_create(r->model,r->begin,r->end,p->memory,(oa_q4_public_session **)out,e,n);
}
static void destroy_session(void *owner){if(owner)oa_qwen4_public_session_close((oa_q4_public_session *)owner);}
static int run(void *s,const oa_se_run *r,char *e,size_t n){return oa_qwen4_public_session_run((oa_q4_public_session *)s,r,e,n);}
static int snapshot(void *s,oa_se_write w,void *p,char *e,size_t n){return oa_qwen4_public_session_snapshot((oa_q4_public_session *)s,w,p,e,n);}
static int restore(void *s,oa_se_read r,void *p,char *e,size_t n){return oa_qwen4_public_session_restore((oa_q4_public_session *)s,r,p,e,n);}
static int open_edge(void **out,oa_se_edge_caps *caps,const oa_se_edge_options *p,char *e,size_t n){
    oa_q4_public_model *m;q4_edge *edge;uint64_t charge;
    if(!out||!caps||!p||p->backend!=(UINT64_C(1)<<8))return error(e,n,"Qwen4 native CPU Edge options required");
    m=find(p->model);if(!m)return error(e,n,"registered Qwen4 Edge model required");charge=sizeof(*edge)+m->work;
    if(p->memory<m->resident||charge>p->memory-m->resident||!reserve(m,charge)){
        oa_qwen4_public_model_release(m);return error(e,n,"Qwen4 Edge aggregate admission exhausted");}
    edge=(q4_edge *)malloc(sizeof(*edge));if(!edge){unreserve(m,charge);oa_qwen4_public_model_release(m);return error(e,n,"Qwen4 Edge allocation failed");}
    edge->model=m;edge->charge=charge;memset(caps,0,sizeof(*caps));caps->size=sizeof(*caps);caps->version=2;
    caps->flags=(UINT64_C(1)<<8)|4|8;caps->dtype=1;caps->width=(uint32_t)width(m);caps->vocab=(uint32_t)m->config.vocabulary;
    caps->rows=m->rows;caps->context=(uint32_t)m->config.text.capacity;caps->layers=(uint32_t)m->config.layers;caps->bos=caps->eos=-1;
    caps->resident_bytes=m->resident;snprintf(caps->engine,sizeof(caps->engine),"%s",q4_engine);
    snprintf(caps->schema,sizeof(caps->schema),"%s",q4_schema);snprintf(caps->numeric,sizeof(caps->numeric),"openagent.native-token-fp32.v1");
    snprintf(caps->tokenizer,sizeof(caps->tokenizer),"external-pinned-tokenizer");*out=edge;return 0;
}
static void destroy_edge(void *owner){q4_edge *e=(q4_edge *)owner;if(e){unreserve(e->model,e->charge);oa_qwen4_public_model_release(e->model);free(e);}}
static int tokenize(void *p,const char *a,size_t b,int32_t *c,size_t d,size_t *f,char *e,size_t n){
    (void)p;(void)a;(void)b;(void)c;(void)d;(void)f;return error(e,n,"Qwen4 Edge requires the external pinned publisher tokenizer");}
static int detokenize(void *p,const int32_t *a,size_t b,char *c,size_t d,size_t *f,char *e,size_t n){
    (void)p;(void)a;(void)b;(void)c;(void)d;(void)f;return error(e,n,"Qwen4 Edge requires the external pinned publisher tokenizer");}
static int embed(void *owner,const oa_se_embed *r,char *e,size_t n){
    q4_edge *edge=(q4_edge *)owner;oa_q4_public_model *m;float *pending;
    oa_q4_cancel_guard cancel_state;
    uint64_t row,stream,h,w,words;
    if(!edge)return error(e,n,"Qwen4 Edge owner required");
    m=edge->model;h=m->config.text.hidden;w=width(m);
    if(!r||!r->rows||r->rows>m->rows||r->rows>m->config.text.capacity||r->token_count!=r->rows||!r->tokens||!r->output||r->output_bytes<(uint64_t)r->rows*w*4)
        return error(e,n,"complete Qwen4 lexical input rows required");
    cancel_state=(oa_q4_cancel_guard){r->cancel,r->cancel_data,0};
    words=(uint64_t)r->rows*w;if(words>m->work/4)return error(e,n,"Qwen4 embedding work admission exhausted");
    pending=(float *)malloc((size_t)words*4);if(!pending)return error(e,n,"Qwen4 embedding allocation failed");
    for(row=0;row<r->rows;row++){
        int32_t token=r->tokens[row];if(token<0||(uint64_t)token>=m->config.vocabulary||oa_q4_checked_cancel(&cancel_state)){free(pending);return error(e,n,cancel_state.invalid?"invalid Qwen4 cancellation callback":"Qwen4 embedding token/cancellation differs");}
        for(stream=0;stream<m->config.streams;stream++)memcpy(pending+row*w+stream*h,m->embedding.data+(uint64_t)token*h,(size_t)h*4);
        pending[row*w+w-5]=(float)token;pending[row*w+w-4]=1;
        pending[row*w+w-3]=pending[row*w+w-2]=pending[row*w+w-1]=(float)row;
    }
    if(oa_q4_checked_cancel(&cancel_state)){free(pending);return error(e,n,cancel_state.invalid?"invalid Qwen4 cancellation callback":"Qwen4 embedding cancelled");}
    memcpy(r->output,pending,(size_t)words*4);free(pending);return 0;
}
static int project(q4_edge *edge,const void *input,size_t input_bytes,uint32_t rows,float *output,uint64_t retained,
    oa_se_cancel cancel,void *context,char *e,size_t n){
    oa_q4_public_model *m=edge->model;oa_q4_compose_config c=m->config;uint64_t words=(uint64_t)rows*c.vocabulary;int result;
    if(!rows||rows>m->rows||!input||input_bytes!=(uint64_t)rows*width(m)*4||words>m->work/4||
        retained>=m->work-words*4)return error(e,n,"Qwen4 Edge wire/work geometry differs");
    c.max_work_bytes=m->work-words*4-retained;if(!c.max_work_bytes)return error(e,n,"Qwen4 head work admission exhausted");
    result=m->head(&c,rows,(oa_q4_tensor){(float *)input,(uint64_t)rows*width(m)},(oa_q4_tensor){output,words},cancel,context);
    return result?error(e,n,"Qwen4 native final mixer/head failed"):0;
}
static int logits(void *owner,const oa_se_logits *r,char *e,size_t n){
    q4_edge *edge=(q4_edge *)owner;float *pending;uint64_t words;int result;oa_q4_cancel_guard cancel_state;
    if(!edge||!r||!r->rows||r->rows>edge->model->rows||!r->logits)return error(e,n,"Qwen4 bounded logits destination required");
    words=(uint64_t)r->rows*edge->model->config.vocabulary;
    cancel_state=(oa_q4_cancel_guard){r->cancel,r->cancel_data,0};
    if(r->logits_capacity<words||words>edge->model->work/4)return error(e,n,"Qwen4 logits destination admission exhausted");
    pending=(float *)malloc((size_t)words*4);if(!pending)return error(e,n,"Qwen4 logits staging allocation failed");
    result=project(edge,r->input,r->input_bytes,r->rows,pending,0,
        r->cancel?oa_q4_checked_cancel:NULL,r->cancel?&cancel_state:NULL,e,n);
    if(cancel_state.invalid)result=error(e,n,"invalid Qwen4 cancellation callback");
    if(!result)memcpy(r->logits,pending,(size_t)words*4);free(pending);return result;
}
static int select_tokens(void *owner,const oa_se_select *r,char *e,size_t n){
    q4_edge *edge=(q4_edge *)owner;float *pending,*scores;uint64_t words,row,token;int32_t *tokens;int result;oa_q4_cancel_guard cancel_state;
    if(!edge||!r||!r->rows||r->rows>edge->model->rows||r->rows>4096||!r->tokens||!r->scores||r->token_capacity<r->rows||r->score_capacity<r->rows)
        return error(e,n,"Qwen4 bounded token selection destination required");
    words=(uint64_t)r->rows*edge->model->config.vocabulary;if(words>edge->model->work/4||
        (uint64_t)r->rows*8>=edge->model->work-words*4)return error(e,n,"Qwen4 selection staging exceeds admission");
    cancel_state=(oa_q4_cancel_guard){r->cancel,r->cancel_data,0};
    pending=(float *)malloc((size_t)words*4);scores=(float *)malloc((size_t)r->rows*4);tokens=(int32_t *)malloc((size_t)r->rows*4);
    if(!pending||!scores||!tokens){free(pending);free(scores);free(tokens);return error(e,n,"Qwen4 selection staging allocation failed");}
    result=project(edge,r->input,r->input_bytes,r->rows,pending,(uint64_t)r->rows*8,
        r->cancel?oa_q4_checked_cancel:NULL,r->cancel?&cancel_state:NULL,e,n);
    if(cancel_state.invalid)result=error(e,n,"invalid Qwen4 cancellation callback");
    if(!result){for(row=0;row<r->rows;row++){
        if(oa_q4_checked_cancel(&cancel_state)){result=error(e,n,cancel_state.invalid?"invalid Qwen4 cancellation callback":"Qwen4 selection cancelled");break;}
        tokens[row]=0;scores[row]=pending[row*edge->model->config.vocabulary];
        for(token=1;token<edge->model->config.vocabulary;token++)if(pending[row*edge->model->config.vocabulary+token]>scores[row]){
            tokens[row]=(int32_t)token;scores[row]=pending[row*edge->model->config.vocabulary+token];}}
        if(result||oa_q4_checked_cancel(&cancel_state))result=error(e,n,cancel_state.invalid?"invalid Qwen4 cancellation callback":"Qwen4 selection cancelled");
        else{memcpy(r->tokens,tokens,(size_t)r->rows*4);memcpy(r->scores,scores,(size_t)r->rows*4);}}
    free(pending);free(scores);free(tokens);return result;
}
int oa_qwen4_public_initialize(void){
    oa_se_segment_adapter s;oa_se_edge_adapter e;int result=0;registry_lock();
    if(!(installed&1)){memset(&s,0,sizeof(s));s.size=sizeof(s);s.version=1;s.engine=q4_engine;s.open=open_range;
        s.destroy=destroy_range;s.create=create_session;s.destroy_session=destroy_session;s.run=run;s.snapshot=snapshot;s.restore=restore;
        if(coli_segment_adapter_register(&s)){result=-1;goto finish;}installed|=1;}
    if(!(installed&2)){memset(&e,0,sizeof(e));e.size=sizeof(e);e.version=2;e.engine=q4_engine;e.open=open_edge;e.destroy=destroy_edge;
        e.tokenize=tokenize;e.detokenize=detokenize;e.embed=embed;e.logits=logits;e.select=select_tokens;
        if(coli_edge_adapter_register(&e)){result=-1;goto finish;}installed|=2;}
finish:registry_unlock();return result;
}
int oa_qwen4_public_register(const oa_q4_public_descriptor *d,uint64_t *id,char *e,size_t n){
    oa_q4_public_model *m;q4_entry *entry,*cursor;
    if(!id||oa_qwen4_public_initialize())return error(e,n,"Qwen4 adapter registry admission failed");
    if(oa_qwen4_public_model_create(d,&m,e,n))return -1;
    if(!reserve(m,sizeof(*entry))){oa_qwen4_public_model_close(m);return error(e,n,"Qwen4 registry aggregate admission exhausted");}
    entry=(q4_entry *)malloc(sizeof(*entry));if(!entry){oa_qwen4_public_model_close(m);return error(e,n,"Qwen4 registry allocation failed");}
    registry_lock();for(cursor=entries;cursor;cursor=cursor->next)if(!strcmp(cursor->model->root,m->root))break;
    if(cursor||entry_count>=128||next_id==UINT64_MAX){registry_unlock();free(entry);oa_qwen4_public_model_close(m);return error(e,n,"unique bounded Qwen4 model registration required");}
    entry->model=m;entry->id=next_id++;entry->next=entries;entries=entry;entry_count++;*id=entry->id;registry_unlock();return 0;
}
int oa_qwen4_public_unregister(uint64_t id,char *e,size_t n){
    q4_entry **cursor,*entry;registry_lock();cursor=&entries;while(*cursor&&(*cursor)->id!=id)cursor=&(*cursor)->next;
    entry=*cursor;if(!entry){registry_unlock();return error(e,n,"unknown Qwen4 model registration");}
    if(oa_qwen4_public_model_close(entry->model)){registry_unlock();return error(e,n,"Qwen4 model still owns public handles/sessions");}
    *cursor=entry->next;entry_count--;free(entry);registry_unlock();return 0;
}
int oa_qwen4_public_hidden(uint64_t id,uint32_t rows,const float *input,uint64_t input_count,
    float *output,uint64_t output_count,oa_q4_cancel cancel,void *context,char *e,size_t n){
    q4_entry *entry;oa_q4_public_model *m=NULL;oa_q4_compose_config c;float *pending;uint64_t words,i;oa_q4_cancel_guard cancel_state;int result;
    registry_lock();for(entry=entries;entry;entry=entry->next)if(entry->id==id){
        if(!oa_qwen4_public_model_retain(entry->model))m=entry->model;break;}registry_unlock();
    if(!m)return error(e,n,"registered Qwen4 hidden owner required");
    words=(uint64_t)rows*m->config.text.hidden;
    if(!rows||rows>m->rows||!input||!output||input_count!=(uint64_t)rows*width(m)||output_count!=words||
        words>=m->work/4||!reserve(m,m->work)){
        oa_qwen4_public_model_release(m);return error(e,n,"Qwen4 hidden geometry or aggregate admission exhausted");}
    for(i=0;i<input_count;i++)if(!isfinite(input[i])){
        unreserve(m,m->work);oa_qwen4_public_model_release(m);return error(e,n,"Qwen4 hidden input must be finite");}
    pending=(float *)malloc((size_t)words*4);
    if(!pending){unreserve(m,m->work);oa_qwen4_public_model_release(m);return error(e,n,"Qwen4 hidden staging allocation failed");}
    c=m->config;c.max_work_bytes=m->work-words*4;
    cancel_state=(oa_q4_cancel_guard){cancel,context,0};
    result=m->hidden(&c,rows,(oa_q4_tensor){(float *)input,input_count},(oa_q4_tensor){pending,words},
                     cancel?oa_q4_checked_cancel:NULL,cancel?&cancel_state:NULL);
    if(!result&&cancel_state.invalid)result=4;
    if(!result)for(i=0;i<words;i++)if(!isfinite(pending[i])){result=4;break;}
    if(!result)memcpy(output,pending,(size_t)words*4);
    free(pending);unreserve(m,m->work);oa_qwen4_public_model_release(m);
    return result?error(e,n,"Qwen4 native hidden projection failed"):0;
}
