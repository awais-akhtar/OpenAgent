#include "qwen4_public_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
static int q4_fail(char *error,size_t bytes,const char *message){if(error&&bytes)snprintf(error,bytes,"%s",message);return -1;}
static int q4_charge(oa_q4_public_model *m,uint64_t bytes){if(bytes>m->limit-m->resident)return 0;m->resident+=bytes;return 1;}
static int q4_clone(oa_q4_public_model *m,oa_q4_tensor source,oa_q4_tensor *output){
    uint64_t i;output->data=NULL;output->count=0;
    if(!source.count)return source.data==NULL;
    if(!source.data||source.count>SIZE_MAX/4||!q4_charge(m,source.count*4))return 0;
    for(i=0;i<source.count;i++)if(!isfinite(source.data[i]))return 0;
    output->data=(float *)malloc((size_t)source.count*4);if(!output->data)return 0;
    memcpy(output->data,source.data,(size_t)source.count*4);output->count=source.count;return 1;
}
static int q4_ints(oa_q4_public_model *m,const int64_t *source,uint64_t count,const int64_t **output){
    int64_t *copy;*output=NULL;if(!source||!count||count>SIZE_MAX/8||!q4_charge(m,count*8))return 0;
    copy=(int64_t *)malloc((size_t)count*8);if(!copy)return 0;memcpy(copy,source,(size_t)count*8);*output=copy;return 1;
}
static int q4_geometry(const oa_q4_compose_config *c){
    const oa_q4_geometry *g=&c->text;
    return g->hidden&&g->hidden<=4096&&g->heads&&g->heads<=128&&g->kv&&g->kv<=128&&
        g->dim&&g->dim<=4096&&g->key_heads&&g->key_heads<=128&&g->value_heads&&g->value_heads<=128&&
        g->key_dim&&g->key_dim<=4096&&g->value_dim&&g->value_dim<=4096&&g->kernel&&g->kernel<=64&&
        g->experts&&g->experts<=4096&&g->topk&&g->topk<=g->experts&&g->intermediate&&g->intermediate<=65536&&
        g->shared&&g->shared<=65536&&g->heads%g->kv==0&&g->value_heads%g->key_heads==0&&
        g->rotary&&g->rotary<=g->dim&&g->rotary%2==0&&isfinite(g->epsilon)&&g->epsilon>0&&
        isfinite(g->theta)&&g->theta>0&&c->rank&&c->rank<=65536&&c->index_heads<=128&&c->index_dim<=4096&&
        c->index_budget<=65536&&c->index_ratio<=65536&&c->ple_embedding<=65536&&c->ple_kernel<=64&&
        c->ple_ngram<=64&&c->ple_lanes<=128&&c->mrope_enabled<=1&&c->sigmoid_gate<=1&&
        c->normalize_topk<=1&&c->finalize<=1&&isfinite(c->rope_scale)&&c->rope_scale>0;
}
static int q4_hyper_inventory(const oa_q4_compose_config *c,const oa_q4_tensor *w,int inject){
    uint64_t width=c->streams*c->text.hidden;
    return w[0].data&&w[0].count==width&&w[1].data&&w[1].count==c->rank*width&&
        w[2].data&&w[2].count==width*c->rank&&
        (!inject||(w[3].data&&w[3].count==c->streams*width));
}
static int q4_tensor_extent(oa_q4_tensor value,uint64_t count){return value.data&&value.count==count;}
static int q4_layer_inventory(const oa_q4_compose_config *c,const oa_q4_compose_layer *l,int streamed){
    const oa_q4_geometry *g=&c->text;uint64_t counts[9],i,kw=g->key_heads*g->key_dim,
        vw=g->value_heads*g->value_dim,aw=g->heads*g->dim,kvw=g->kv*g->dim;
    if(l->kind){
        if(!c->index_heads||!c->index_dim||g->rotary>c->index_dim||!c->index_ratio||
            c->index_ratio>g->capacity||!c->index_budget||c->index_budget>g->capacity||
            c->index_budget%c->index_ratio||c->rope_scale<1)return 0;
        counts[0]=2*aw*g->hidden;counts[1]=counts[2]=kvw*g->hidden;counts[3]=g->hidden*aw;
        counts[4]=counts[5]=g->dim;counts[6]=(c->index_heads+1)*c->index_dim*g->hidden;
        counts[7]=counts[8]=c->index_dim;
    }else{
        counts[0]=(2*kw+vw)*g->hidden;counts[1]=vw*g->hidden;
        counts[2]=counts[3]=g->value_heads*g->hidden;counts[4]=(2*kw+vw)*g->kernel;
        counts[5]=g->value_dim;counts[6]=counts[7]=g->value_heads;counts[8]=g->hidden*vw;
    }
    for(i=0;i<9;i++)if(!q4_tensor_extent(l->attention[i],counts[i]))return 0;
    counts[0]=g->experts*g->hidden;counts[1]=g->experts*2*g->intermediate*g->hidden;
    counts[2]=g->experts*g->hidden*g->intermediate;counts[3]=counts[4]=g->shared*g->hidden;
    counts[5]=g->hidden*g->shared;counts[6]=g->hidden;
    for(i=0;i<7;i++){
        if(streamed&&(i==1||i==2)){if(l->mlp[i].data||l->mlp[i].count)return 0;}
        else if(!q4_tensor_extent(l->mlp[i],counts[i]))return 0;
    }
    return 1;
}
static void q4_destroy(oa_q4_public_model *m){
    uint64_t i,j;if(!m)return;
    if(m->layers)for(i=0;i<m->config.layers;i++){
        oa_q4_compose_layer *l=m->layers+i;
        for(j=0;j<4;j++){free(l->attention_hyper[j].data);free(l->mlp_hyper[j].data);}
        for(j=0;j<9;j++)free(l->attention[j].data);
        for(j=0;j<7;j++){free(l->mlp[j].data);free(l->ple_weights[j].data);}
        free((void *)l->multipliers);free((void *)l->sizes);free((void *)l->offsets);
    }
    if(m->ple)for(i=0;i<m->config.layers;i++)if(m->ple[i])oa_original_rows_release(m->ple[i]);
    if(m->experts)oa_original_experts_release(m->experts);
    for(i=0;i<3;i++)free(m->config.final_hyper[i].data);
    free(m->config.head.data);free(m->embedding.data);free(m->layers);free(m->ple);free(m->root);free(m);
}
int oa_qwen4_public_model_retain(oa_q4_public_model *m){
    uint_fast64_t refs;if(!m)return -1;refs=atomic_load(&m->refs);
    do{if(!refs||refs==UINT64_MAX)return -1;}while(!atomic_compare_exchange_weak(&m->refs,&refs,refs+1));return 0;
}
int oa_qwen4_public_model_release(oa_q4_public_model *m){
    uint_fast64_t refs;if(!m)return -1;refs=atomic_load(&m->refs);
    do{if(!refs)return -1;}while(!atomic_compare_exchange_weak(&m->refs,&refs,refs-1));if(refs==1)q4_destroy(m);return 0;
}
int oa_qwen4_public_model_close(oa_q4_public_model *m){
    uint_fast64_t expected=1;if(!m)return -1;
    if(!atomic_compare_exchange_strong(&m->refs,&expected,0))return -1;
    q4_destroy(m);return 0;
}
int oa_qwen4_public_model_create(const oa_q4_public_descriptor *d,oa_q4_public_model **out,char *error,size_t bytes){
    oa_q4_public_model *m;uint64_t i,j,width,rows,columns;size_t length;uint16_t endian=1;
    /* Clear a provided output even when any descriptor admission check fails. */
    if(out)*out=NULL;
    if(sizeof(float)!=4||FLT_RADIX!=2||FLT_MANT_DIG!=24||FLT_MAX_EXP!=128||*(unsigned char *)&endian!=1||
        !d||d->size!=sizeof(*d)||d->version!=2||d->reserved||!out||!d->root||!d->identity||
        strlen(d->identity)!=64||!d->rows||d->rows>4096||!d->wire||!d->head||!d->hidden||!d->layers||
        !d->max_work_bytes||d->external_store_bytes>=d->max_resource_bytes||
        d->config.size!=sizeof(d->config)||d->config.version!=1||!q4_geometry(&d->config)||!d->config.layers||d->config.layers>4096||
        !d->config.text.hidden||d->config.text.hidden>4096||d->config.streams<2||d->config.streams>64||
        !d->config.text.capacity||d->config.text.capacity>65536||!d->config.vocabulary||d->config.vocabulary>1048576)
        return q4_fail(error,bytes,"explicit bounded Qwen4 public model descriptor required");
    length=strlen(d->root);if(!length||length>32768)return q4_fail(error,bytes,"Qwen4 model root extent differs");
    for(i=0;i<64;i++)if(!((d->identity[i]>='0'&&d->identity[i]<='9')||(d->identity[i]>='a'&&d->identity[i]<='f')))
        return q4_fail(error,bytes,"Qwen4 model identity must be lowercase SHA-256");
    width=d->config.streams*d->config.text.hidden;
    m=(oa_q4_public_model *)calloc(1,sizeof(*m));if(!m)return q4_fail(error,bytes,"Qwen4 public model allocation failed");
    atomic_init(&m->refs,1);atomic_flag_clear(&m->guard);m->config=d->config;
    memset(m->config.final_hyper,0,sizeof(m->config.final_hyper));memset(&m->config.head,0,sizeof(m->config.head));
    m->limit=d->max_resource_bytes-d->external_store_bytes;m->work=d->max_work_bytes;m->rows=d->rows;
    m->wire=d->wire;m->head=d->head;m->hidden=d->hidden;memcpy(m->identity,d->identity,65);
    if(!q4_charge(m,sizeof(*m)+length+1+m->config.layers*(sizeof(*m->layers)+sizeof(*m->ple))))goto invalid;
    m->root=(char *)malloc(length+1);m->layers=(oa_q4_compose_layer *)calloc((size_t)m->config.layers,sizeof(*m->layers));
    m->ple=(oa_original_rows **)calloc((size_t)m->config.layers,sizeof(*m->ple));
    if(!m->root||!m->layers||!m->ple)goto invalid;memcpy(m->root,d->root,length+1);
    if(d->experts){if(!d->external_store_bytes||oa_original_experts_retain(d->experts))goto invalid;m->experts=d->experts;}
    for(i=0;i<m->config.layers;i++){
        const oa_q4_compose_layer *source=d->layers+i;oa_q4_compose_layer *target=m->layers+i;
        if(source->kind>1||source->ple_enabled>1||source->previous||source->previous_count||source->ple_history.count||source->ple_history.data)goto invalid;
        for(j=0;j<3;j++)if(source->cache[j].data||source->cache[j].count)goto invalid;
        target->kind=source->kind;target->ple_enabled=source->ple_enabled;
        if(!q4_hyper_inventory(&m->config,source->attention_hyper,1)||
            !q4_hyper_inventory(&m->config,source->mlp_hyper,1)||
            !q4_layer_inventory(&m->config,source,m->experts!=NULL))goto invalid;
        if(!source->ple_enabled&&d->ple&&d->ple[i])goto invalid;
        for(j=0;j<4;j++)if(!q4_clone(m,source->attention_hyper[j],target->attention_hyper+j)||!q4_clone(m,source->mlp_hyper[j],target->mlp_hyper+j))goto invalid;
        for(j=0;j<9;j++)if(!q4_clone(m,source->attention[j],target->attention+j))goto invalid;
        for(j=0;j<7;j++)if(!q4_clone(m,source->mlp[j],target->mlp+j))goto invalid;
        if(m->experts&&(source->mlp[1].count||source->mlp[1].data||source->mlp[2].count||source->mlp[2].data))goto invalid;
        if(m->experts)for(j=0;j<m->config.text.experts;j++)if(!oa_original_experts_supports(m->experts,(uint32_t)i,(uint32_t)j,m->config.text.hidden,m->config.text.intermediate))goto invalid;
        if(source->ple_enabled){
            if(m->config.eos>=m->config.vocabulary||!m->config.ple_kernel||m->config.ple_kernel>64||
                !m->config.ple_embedding||m->config.ple_ngram<2||m->config.ple_ngram>64||!m->config.ple_lanes||m->config.ple_lanes>128||
                source->multiplier_count!=m->config.ple_ngram||source->lane_count!=(m->config.ple_ngram-1)*m->config.ple_lanes)goto invalid;
            target->multiplier_count=source->multiplier_count;target->lane_count=source->lane_count;
            if(!q4_ints(m,source->multipliers,target->multiplier_count,&target->multipliers)||
                !q4_ints(m,source->sizes,target->lane_count,&target->sizes)||!q4_ints(m,source->offsets,target->lane_count,&target->offsets))goto invalid;
            for(j=0;j<7;j++)if(!q4_clone(m,source->ple_weights[j],target->ple_weights+j))goto invalid;
            if(d->ple&&d->ple[i]){
                if(source->ple_weights[0].count||source->ple_weights[0].data)goto invalid;
                if(!d->external_store_bytes||oa_original_rows_retain(d->ple[i]))goto invalid;m->ple[i]=d->ple[i];
                if(oa_original_rows_shape(m->ple[i],&rows,&columns)||rows!=source->ple_table_rows||!columns||
                    m->config.ple_embedding!=columns*target->lane_count)goto invalid;
                target->ple_table_rows=rows;target->ple_user=m->ple[i];
            }
            if(m->config.ple_embedding%target->lane_count)goto invalid;
            columns=m->config.ple_embedding/target->lane_count;
            if(!target->ple_user){
                if(!target->ple_weights[0].data||!columns||target->ple_weights[0].count%columns)goto invalid;
                target->ple_table_rows=target->ple_weights[0].count/columns;
            }
            if(!target->ple_table_rows||!q4_tensor_extent(target->ple_weights[1],width*m->config.ple_embedding)||
                !q4_tensor_extent(target->ple_weights[2],m->config.text.hidden*m->config.ple_embedding)||
                !q4_tensor_extent(target->ple_weights[3],width)||!q4_tensor_extent(target->ple_weights[4],width)||
                !q4_tensor_extent(target->ple_weights[5],width)||!q4_tensor_extent(target->ple_weights[6],width*m->config.ple_kernel))goto invalid;
            for(j=0;j<target->multiplier_count;j++)if(target->multipliers[j]<=0)goto invalid;
            for(j=0;j<target->lane_count;j++)if(target->sizes[j]<=0||target->offsets[j]<0||
                (uint64_t)target->offsets[j]>=target->ple_table_rows||
                (uint64_t)target->sizes[j]>target->ple_table_rows-(uint64_t)target->offsets[j])goto invalid;
        }
    }
    if(!q4_hyper_inventory(&m->config,d->config.final_hyper,0))goto invalid;
    for(i=0;i<3;i++)if(!q4_clone(m,d->config.final_hyper[i],m->config.final_hyper+i))goto invalid;
    if(!q4_clone(m,d->config.head,&m->config.head)||!q4_clone(m,d->embedding,&m->embedding)||
        m->embedding.count!=m->config.vocabulary*m->config.text.hidden||m->config.head.count!=m->embedding.count||
        m->config.final_hyper[0].count!=width||m->work>m->limit-m->resident)goto invalid;
    m->used=m->resident;*out=m;return 0;
invalid:q4_destroy(m);return q4_fail(error,bytes,"Qwen4 public model inventory, ownership or admission differs");
}
size_t oa_qwen4_public_model_layout(uint64_t *output,size_t capacity){
#define Q4_OFFSET(name) ((uint64_t)offsetof(oa_q4_public_descriptor,name))
    const uint64_t values[]={sizeof(oa_q4_public_descriptor),17,Q4_OFFSET(size),Q4_OFFSET(version),Q4_OFFSET(rows),
        Q4_OFFSET(reserved),Q4_OFFSET(root),Q4_OFFSET(identity),Q4_OFFSET(max_resource_bytes),Q4_OFFSET(max_work_bytes),
        Q4_OFFSET(external_store_bytes),Q4_OFFSET(config),Q4_OFFSET(layers),Q4_OFFSET(embedding),Q4_OFFSET(wire),Q4_OFFSET(head),
        Q4_OFFSET(experts),Q4_OFFSET(ple),Q4_OFFSET(hidden)};
#undef Q4_OFFSET
    size_t n=sizeof(values)/sizeof(values[0]);if(output&&capacity>=n)memcpy(output,values,sizeof(values));return n;
}
