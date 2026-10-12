/* Included by native_qwen4.c: complete native hyper-channel composition.
 * All cache/history buffers are transaction staging owned by the caller.
 * A nonzero return invalidates every staging buffer; none may be published.
 * This is an additive counted-buffer ABI, not a Python execution callback. */
#define OA_Q4_TENSOR Tensor
#define OA_Q4_GEOMETRY Config
#include "native/qwen4_composition_abi.h"
#undef OA_Q4_TENSOR
#undef OA_Q4_GEOMETRY
typedef oa_q4_compose_layer Q4ComposeLayer;
typedef oa_q4_compose_config Q4ComposeConfig;
typedef oa_q4_compose_trace Q4ComposeTrace;
typedef oa_q4_cancel Q4ComposeCancel;

API uint64_t oa_qwen4_compose_version(void) {return 1;}
API size_t oa_qwen4_compose_layout(uint32_t record,uint64_t *out,size_t capacity) {
    uint64_t values[40];size_t n=0;
#define Q4_FIELD(type,field) values[n++]=(uint64_t)offsetof(type,field)
    if(record==1) {
        values[n++]=sizeof(Q4ComposeConfig);values[n++]=25;
        Q4_FIELD(Q4ComposeConfig,size);Q4_FIELD(Q4ComposeConfig,version);Q4_FIELD(Q4ComposeConfig,text);
        Q4_FIELD(Q4ComposeConfig,layers);Q4_FIELD(Q4ComposeConfig,streams);Q4_FIELD(Q4ComposeConfig,rank);
        Q4_FIELD(Q4ComposeConfig,index_heads);Q4_FIELD(Q4ComposeConfig,index_dim);Q4_FIELD(Q4ComposeConfig,index_budget);
        Q4_FIELD(Q4ComposeConfig,index_ratio);Q4_FIELD(Q4ComposeConfig,ple_embedding);Q4_FIELD(Q4ComposeConfig,ple_kernel);
        Q4_FIELD(Q4ComposeConfig,ple_ngram);Q4_FIELD(Q4ComposeConfig,ple_lanes);Q4_FIELD(Q4ComposeConfig,eos);
        Q4_FIELD(Q4ComposeConfig,vocabulary);Q4_FIELD(Q4ComposeConfig,max_work_bytes);Q4_FIELD(Q4ComposeConfig,rope_scale);
        Q4_FIELD(Q4ComposeConfig,mrope_sections);Q4_FIELD(Q4ComposeConfig,mrope_enabled);Q4_FIELD(Q4ComposeConfig,sigmoid_gate);
        Q4_FIELD(Q4ComposeConfig,normalize_topk);Q4_FIELD(Q4ComposeConfig,finalize);
        Q4_FIELD(Q4ComposeConfig,final_hyper);Q4_FIELD(Q4ComposeConfig,head);
    } else if(record==2) {
        values[n++]=sizeof(Q4ComposeLayer);values[n++]=18;
        Q4_FIELD(Q4ComposeLayer,kind);Q4_FIELD(Q4ComposeLayer,ple_enabled);
        Q4_FIELD(Q4ComposeLayer,attention_hyper);Q4_FIELD(Q4ComposeLayer,mlp_hyper);
        Q4_FIELD(Q4ComposeLayer,attention);Q4_FIELD(Q4ComposeLayer,mlp);Q4_FIELD(Q4ComposeLayer,cache);
        Q4_FIELD(Q4ComposeLayer,ple_weights);Q4_FIELD(Q4ComposeLayer,ple_history);
        Q4_FIELD(Q4ComposeLayer,multipliers);Q4_FIELD(Q4ComposeLayer,sizes);Q4_FIELD(Q4ComposeLayer,offsets);
        Q4_FIELD(Q4ComposeLayer,previous);Q4_FIELD(Q4ComposeLayer,multiplier_count);
        Q4_FIELD(Q4ComposeLayer,lane_count);Q4_FIELD(Q4ComposeLayer,previous_count);Q4_FIELD(Q4ComposeLayer,ple_table_rows);
        Q4_FIELD(Q4ComposeLayer,ple_user);
    } else if(record==3) {
        values[n++]=sizeof(Q4ComposeTrace);values[n++]=8;
        Q4_FIELD(Q4ComposeTrace,router);Q4_FIELD(Q4ComposeTrace,scores);Q4_FIELD(Q4ComposeTrace,attention);
        Q4_FIELD(Q4ComposeTrace,incoming_hidden);
        Q4_FIELD(Q4ComposeTrace,selected);Q4_FIELD(Q4ComposeTrace,chosen);
        Q4_FIELD(Q4ComposeTrace,selected_count);Q4_FIELD(Q4ComposeTrace,chosen_count);
    } else return 0;
#undef Q4_FIELD
    if(out&&capacity>=n)memcpy(out,values,n*sizeof(uint64_t));
    return n;
}

static int q4_hyper_run(const Q4ComposeConfig *c,const Tensor *w,uint64_t rows,
        Tensor x,Tensor mixed,Tensor gate,int inject) {
    uint64_t width=c->streams*c->text.hidden;
    if(!valid(w[0],width)||!valid(w[1],c->rank*width)||!valid(w[2],width*c->rank)||
        (inject&&!valid(w[3],c->streams*width)))return 1;
    return oa_qwen4_hyper(rows,c->streams,c->text.hidden,c->rank,c->text.epsilon,
        x.data,w[0].data,w[1].data,w[2].data,inject?w[3].data:NULL,mixed.data,inject?gate.data:NULL);
}

API int oa_qwen4_compose(const Q4ComposeConfig *c,Q4ComposeLayer *layers,Q4ComposeTrace *traces,
        uint64_t batch,uint64_t tokens,uint64_t start,uint64_t begin,uint64_t end,
        const int64_t *ids,const uint8_t *full_mask,const int64_t *full_positions,
        Tensor incoming,Tensor hidden,Tensor logits,ExpertFFN experts,void *expert_user,
        PLELookup ple_lookup,void *ple_user,Q4ComposeCancel cancelled,void *cancel_user) {
    uint64_t rows,width,occupied,i,units,helper,channels,vw;
    float *memory;uint8_t *mask;Tensor x,next,mixed,value,gate;
    int result=0;
    if(!c||c->size!=sizeof(*c)||c->version!=1||c->finalize>1||!geometry(&c->text)||!layers||!traces||
        !batch||batch>128||!tokens||tokens>65536||start>c->text.capacity||tokens>c->text.capacity-start||
        !c->layers||c->layers>4096||begin>=end||end>c->layers||c->streams<2||c->streams>64||
        !c->rank||c->rank>65536||!c->max_work_bytes||!c->vocabulary||c->vocabulary>1048576||
        c->mrope_enabled>1||c->sigmoid_gate>1||c->normalize_topk>1||
        c->index_heads>128||c->index_dim>4096||c->index_budget>65536||c->index_ratio>65536||
        c->ple_embedding>65536||c->ple_kernel>64||c->ple_ngram>64||c->ple_lanes>128||
        !isfinite(c->rope_scale)||c->rope_scale<=0||!ids||!full_mask||!full_positions)return 1;
    rows=batch*tokens;width=c->streams*c->text.hidden;occupied=start+tokens;
    if(rows>1048576||rows>67108864/width||!valid(incoming,rows*width))return 1;
    if(c->finalize&&end!=c->layers)return 1;
    if(!hidden.data||hidden.count!=rows*(c->finalize?c->text.hidden:width))return 1;
    if(logits.data&&(!c->finalize||logits.count!=rows*c->vocabulary))return 1;
    if(!logits.data&&logits.count)return 1;
    for(i=0;i<rows;i++)if(ids[i]<0||(uint64_t)ids[i]>=c->vocabulary)return 1;
    for(i=0;i<batch*occupied;i++)if(full_mask[i]>1)return 1;
    for(i=0;i<(c->mrope_enabled?3:1)*batch*occupied;i++)
        if(full_positions[i]<0||(uint64_t)full_positions[i]>=c->text.capacity)return 1;
    vw=c->text.value_heads*c->text.value_dim;
    channels=2*c->text.key_heads*c->text.key_dim+vw;
    units=rows*(2*width+2*c->text.hidden+c->streams);
    helper=2*channels+3*vw+2*c->text.value_heads+2*c->text.key_dim+c->text.value_dim+c->text.hidden;
    helper+=c->text.experts+2*c->text.intermediate+2*c->text.shared+c->text.hidden+1;
    helper+=4*width+4*c->rank+3*rows*(c->text.hidden+c->streams);
    helper+=3*c->text.heads*c->text.dim+
        ((c->index_heads+1)*c->index_dim>2*c->text.kv*c->text.dim?
            (c->index_heads+1)*c->index_dim:2*c->text.kv*c->text.dim)+2*c->index_dim+9*occupied;
    helper+=c->ple_embedding+4*width+c->text.hidden+2*c->ple_ngram;
    if(units>SIZE_MAX/sizeof(float)||helper>SIZE_MAX/sizeof(float)||
        units>c->max_work_bytes/4||helper>(c->max_work_bytes-4*units)/4||
        rows>c->max_work_bytes-4*(units+helper))return 2;
    memory=allocate(units);mask=(uint8_t *)malloc((size_t)rows);
    if(!memory||!mask){free(memory);free(mask);return 2;}
    x=(Tensor){memory,rows*width};next=(Tensor){x.data+x.count,x.count};
    mixed=(Tensor){next.data+next.count,rows*c->text.hidden};value=(Tensor){mixed.data+mixed.count,mixed.count};
    gate=(Tensor){value.data+value.count,rows*c->streams};
    memcpy(x.data,incoming.data,(size_t)x.count*sizeof(float));
    for(i=0;i<batch;i++)memcpy(mask+i*tokens,full_mask+i*occupied+start,(size_t)tokens);
    for(i=begin;i<end;i++) {
        Q4ComposeLayer *layer=layers+i;Q4ComposeTrace *trace=traces+i;
        if(cancelled&&cancelled(cancel_user)){result=4;goto finish;}
        if(layer->kind>1||layer->ple_enabled>1){result=1;goto finish;}
        if(!trace->incoming_hidden.data||trace->incoming_hidden.count!=x.count){result=1;goto finish;}
        memcpy(trace->incoming_hidden.data,x.data,(size_t)x.count*sizeof(float));
        if(layer->ple_enabled) {
            if(c->ple_ngram<2||c->ple_ngram>64||!c->ple_lanes||c->ple_lanes>128||
                layer->multiplier_count!=c->ple_ngram||layer->lane_count!=(c->ple_ngram-1)*c->ple_lanes||
                layer->previous_count!=batch*(c->ple_ngram-1)){result=1;goto finish;}
            result=qwen4_ple_run(batch,tokens,c->streams,c->text.hidden,c->ple_embedding,c->ple_kernel,
                c->ple_ngram,c->ple_lanes,c->eos,c->text.epsilon,layer->ple_weights,
                layer->multipliers,layer->sizes,layer->offsets,ids,layer->previous,mask,
                x,layer->ple_history,next,(layer->ple_user||ple_user)?ple_lookup:NULL,
                layer->ple_user?layer->ple_user:ple_user,layer->ple_table_rows);
            if(result)goto finish;
            result=oa_qwen4_add(x,next,next);if(result)goto finish;
            {Tensor swap=x;x=next;next=swap;}
        }
        result=q4_hyper_run(c,layer->attention_hyper,rows,x,mixed,gate,1);if(result)goto finish;
        if(!layer->kind)result=oa_qwen4_linear(&c->text,layer->attention,batch,tokens,mask,
            c->sigmoid_gate,mixed,layer->cache[0],layer->cache[1],value);
        else {
            if(trace->chosen_count!=rows*occupied||!trace->chosen){result=1;goto finish;}
            result=qwen4_qsa_run(&c->text,layer->attention,batch,tokens,start,c->index_heads,c->index_dim,
                c->index_budget,c->index_ratio,full_mask,full_positions,mixed,
                layer->cache[0],layer->cache[1],layer->cache[2],value,trace->attention,trace->chosen,
                c->rope_scale,c->mrope_enabled?c->mrope_sections:NULL);
        }
        if(result)goto finish;
        result=oa_qwen4_inject(rows,c->streams,c->text.hidden,x,value,gate,next);if(result)goto finish;
        {Tensor swap=x;x=next;next=swap;}
        result=q4_hyper_run(c,layer->mlp_hyper,rows,x,mixed,gate,1);if(result)goto finish;
        if(trace->selected_count!=rows*c->text.topk||!trace->selected){result=1;goto finish;}
        result=moe_run(&c->text,layer->mlp,rows,c->normalize_topk,mixed,value,trace->router,
            trace->selected,trace->scores,experts,expert_user,i);if(result)goto finish;
        result=oa_qwen4_inject(rows,c->streams,c->text.hidden,x,value,gate,next);if(result)goto finish;
        {Tensor swap=x;x=next;next=swap;}
    }
    if(c->finalize) {
        result=q4_hyper_run(c,c->final_hyper,rows,x,mixed,gate,0);if(result)goto finish;
        if(logits.data){result=oa_qwen4_head(rows,c->text.hidden,c->vocabulary,mixed,c->head,logits);if(result)goto finish;}
        x=mixed;
    }
    if(cancelled&&cancelled(cancel_user)){result=4;goto finish;}
    if(!valid(x,hidden.count)){result=3;goto finish;}
    memcpy(hidden.data,x.data,(size_t)hidden.count*sizeof(float));
finish:
    free(memory);free(mask);return result;
}

/* Segment wire rows are [streams*hidden, token ID, occupied-mask, T, H, W].
 * Integer side lanes use exact FP32 representations: admitted vocabulary and
 * context bounds are below 2**24. They are never rounded into admissibility.
 * Complete prefix banks and layer state supplied here are transaction staging.
 * Segment output remains raw hyper-channel residuals, including the last range;
 * final mixing and vocabulary projection belong to the native Edge entrypoint.
 */
API uint64_t oa_qwen4_compose_wire_version(void){return 1;}
API int oa_qwen4_compose_wire(const Q4ComposeConfig *c,Q4ComposeLayer *layers,Q4ComposeTrace *traces,
    uint64_t rows,uint64_t start,uint64_t begin,uint64_t end,Tensor wire,Tensor output,
    uint8_t *full_mask,uint64_t mask_count,int64_t *positions,uint64_t position_count,
    ExpertFFN experts,void *expert_user,PLELookup lookup,void *lookup_user,
    Q4ComposeCancel cancel,void *cancel_user) {
    uint64_t width,stride,row,axis,occupied,allocation;float *memory;
    int64_t *ids;Tensor incoming,hidden;Q4ComposeConfig config;int result;
    if(!c||c->size!=sizeof(*c)||c->version!=1||!geometry(&c->text)||c->streams<2||c->streams>64||
        !rows||rows>4096||start>c->text.capacity||rows>c->text.capacity-start||
        !full_mask||!positions||c->mrope_enabled>1||!c->vocabulary||c->vocabulary>1048576)return 1;
    width=c->streams*c->text.hidden;stride=width+5;occupied=start+rows;
    if(!valid(wire,rows*stride)||!output.data||output.count!=wire.count||mask_count!=occupied||
        position_count!=(c->mrope_enabled?3:1)*occupied)return 1;
    allocation=rows*(8+8*width)+sizeof(config);
    if(allocation>=c->max_work_bytes||allocation>SIZE_MAX)return 2;
    memory=allocate(2*rows*width);ids=(int64_t *)malloc((size_t)rows*sizeof(int64_t));
    if(!memory||!ids){free(memory);free(ids);return 2;}
    incoming=(Tensor){memory,rows*width};hidden=(Tensor){memory+incoming.count,incoming.count};
    for(row=0;row<rows;row++) {
        const float *source=wire.data+row*stride;
        float token=source[width],mask=source[width+1];
        if(token<0||token>=(float)c->vocabulary||floorf(token)!=token||(mask!=0&&mask!=1)){
            free(memory);free(ids);return 1;}
        ids[row]=(int64_t)token;full_mask[start+row]=(uint8_t)mask;
        for(axis=0;axis<3;axis++) {
            float value=source[width+2+axis];
            if(value<0||value>=(float)c->text.capacity||floorf(value)!=value||
                (!c->mrope_enabled&&value!=source[width+2])){free(memory);free(ids);return 1;}
            if(c->mrope_enabled||!axis)positions[axis*occupied+start+row]=(int64_t)value;
        }
        memcpy(incoming.data+row*width,source,(size_t)width*sizeof(float));
    }
    config=*c;config.finalize=0;config.max_work_bytes-=allocation;
    result=oa_qwen4_compose(&config,layers,traces,1,rows,start,begin,end,ids,full_mask,positions,
        incoming,hidden,(Tensor){NULL,0},experts,expert_user,lookup,lookup_user,cancel,cancel_user);
    if(!result&&cancel&&cancel(cancel_user))result=4;
    if(!result)for(row=0;row<rows;row++) {
        memcpy(output.data+row*stride,hidden.data+row*width,(size_t)width*sizeof(float));
        memcpy(output.data+row*stride+width,wire.data+row*stride+width,5*sizeof(float));
    }
    free(memory);free(ids);return result;
}

static int q4_compose_edge_wire(const Q4ComposeConfig *c,uint64_t rows,Tensor wire,Tensor logits,
    Q4ComposeCancel cancel,void *cancel_user,int hidden_only) {
    uint64_t width,row,units,helper;float *memory;Tensor raw,mixed,output,unused={NULL,0};int result;
    if(!c||c->size!=sizeof(*c)||c->version!=1||!geometry(&c->text)||c->streams<2||c->streams>64||
        !c->rank||c->rank>65536||!rows||rows>4096||!c->vocabulary||c->vocabulary>1048576)return 1;
    width=c->streams*c->text.hidden;
    if(!valid(wire,rows*(width+5))||!logits.data||logits.count!=rows*(hidden_only?c->text.hidden:c->vocabulary))return 1;
    for(row=0;row<rows;row++){
        const float *side=wire.data+row*(width+5)+width;uint64_t axis;
        if(side[0]<0||side[0]>=(float)c->vocabulary||floorf(side[0])!=side[0]||(side[1]!=0&&side[1]!=1))return 1;
        for(axis=2;axis<5;axis++)if(side[axis]<0||side[axis]>=(float)c->text.capacity||
            floorf(side[axis])!=side[axis]||(!c->mrope_enabled&&side[axis]!=side[2]))return 1;
    }
    units=rows*(width+c->text.hidden+(hidden_only?c->text.hidden:c->vocabulary));
    helper=4*(width+c->rank)+4*rows*(c->text.hidden+c->streams);
    if(units>SIZE_MAX/4||units>c->max_work_bytes/4||helper>c->max_work_bytes-4*units)return 2;
    memory=allocate(units);if(!memory)return 2;
    raw=(Tensor){memory,rows*width};mixed=(Tensor){memory+raw.count,rows*c->text.hidden};
    output=(Tensor){mixed.data+mixed.count,logits.count};
    for(row=0;row<rows;row++)memcpy(raw.data+row*width,wire.data+row*(width+5),(size_t)width*sizeof(float));
    if(cancel&&cancel(cancel_user)){free(memory);return 4;}
    result=q4_hyper_run(c,c->final_hyper,rows,raw,mixed,unused,0);
    if(!result){
        if(hidden_only)memcpy(output.data,mixed.data,(size_t)mixed.count*sizeof(float));
        else result=oa_qwen4_head(rows,c->text.hidden,c->vocabulary,mixed,c->head,output);
    }
    if(!result&&cancel&&cancel(cancel_user))result=4;
    if(!result)memcpy(logits.data,output.data,(size_t)logits.count*sizeof(float));
    free(memory);return result;
}
API int oa_qwen4_compose_head_wire(const Q4ComposeConfig *c,uint64_t rows,Tensor wire,Tensor logits,
    Q4ComposeCancel cancel,void *cancel_user){return q4_compose_edge_wire(c,rows,wire,logits,cancel,cancel_user,0);}
API int oa_qwen4_compose_hidden_wire(const Q4ComposeConfig *c,uint64_t rows,Tensor wire,Tensor hidden,
    Q4ComposeCancel cancel,void *cancel_user){return q4_compose_edge_wire(c,rows,wire,hidden,cancel,cancel_user,1);}
