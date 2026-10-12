/* Shared in-house scalar primitives; Qwen4 has its own layer composition. */
#include "native_qwen35.c"
#include "native_qwen4_hyper.c"
#include <stddef.h>

API uint64_t oa_qwen4_version(void) { return 1; }
API int oa_qwen4_geometry(const Config *c) { return geometry(c)?0:1; }

API int oa_qwen4_linear(const Config *c, const Tensor *w, uint64_t batch,
        uint64_t tokens, const uint8_t *mask, int sigmoid_gate, Tensor x, Tensor history,
        Tensor state, Tensor output) {
    uint64_t width, kw, vw, row, head, j, n, channel, counts[10];
    float *memory, *projection, *z, *beta, *decay, *read, *qn, *kn;
    if (!geometry(c) || !w || !batch || batch>128 || !tokens || tokens>c->capacity || !mask || (sigmoid_gate!=0&&sigmoid_gate!=1)) return 1;
    kw=c->key_heads*c->key_dim;vw=c->value_heads*c->value_dim;width=2*kw+vw;
    counts[0]=width*c->hidden;counts[1]=vw*c->hidden;
    counts[2]=counts[3]=c->value_heads*c->hidden;counts[4]=width*c->kernel;
    counts[5]=c->value_dim;counts[6]=counts[7]=c->value_heads;
    counts[8]=c->hidden*vw;counts[9]=0;
    for(j=0;j<9;j++) if(!valid(w[j],counts[j])) return 1;
    if(!valid(x,batch*tokens*c->hidden) || !valid(history,batch*width*c->kernel) ||
        !valid(state,batch*c->value_heads*c->key_dim*c->value_dim) ||
        !output.data || output.count!=x.count) return 1;
    memory=allocate(2*width+3*vw+2*c->value_heads+2*c->key_dim+c->value_dim+c->hidden);
    if(!memory) return 2;
    projection=memory;z=projection+width;beta=z+vw;
    decay=beta+c->value_heads;read=decay+c->value_heads;
    qn=read+c->value_dim;kn=qn+c->key_dim;
    for(row=0;row<batch*tokens;row++) {
        uint64_t b=row/tokens;
        float *input=kn+c->key_dim, *core=input+c->hidden;
        if(mask[row]>1) {free(memory);return 1;}
        for(j=0;j<c->hidden;j++) input[j]=mask[row]?x.data[row*c->hidden+j]:0;
        matvec(input,w[0].data,width,c->hidden,projection);
        matvec(input,w[1].data,vw,c->hidden,z);
        matvec(input,w[2].data,c->value_heads,c->hidden,beta);
        matvec(input,w[3].data,c->value_heads,c->hidden,decay);
        for(channel=0;channel<width;channel++) {
            double sum=0;
            float *past=history.data+(b*width+channel)*c->kernel;
            memmove(past,past+1,(size_t)(c->kernel-1)*sizeof(float));
            past[c->kernel-1]=projection[channel];
            for(j=0;j<c->kernel;j++) sum+=(double)past[j]*w[4].data[channel*c->kernel+j];
            projection[channel]=silu((float)sum);
        }
        for(head=0;head<c->value_heads;head++) {
            uint64_t kh=head/(c->value_heads/c->key_heads);
            float *q=projection+kh*c->key_dim,*k=projection+kw+kh*c->key_dim;
            float *v=projection+2*kw+head*c->value_dim;
            float *s=state.data+(b*c->value_heads+head)*c->key_dim*c->value_dim;
            double qs=1e-6,ks=1e-6;
            float d=(float)exp(-exp((double)w[7].data[head])*softplus((double)decay[head]+w[6].data[head]));
            float rate=sigmoid(beta[head]);
            for(j=0;j<c->key_dim;j++) {qs+=(double)q[j]*q[j];ks+=(double)k[j]*k[j];}
            for(j=0;j<c->key_dim;j++) {qn[j]=(float)(q[j]/sqrt(qs)/sqrt((double)c->key_dim));kn[j]=(float)(k[j]/sqrt(ks));}
            for(j=0;j<c->key_dim*c->value_dim;j++) s[j]*=d;
            for(j=0;j<c->value_dim;j++) {
                double predicted=0,answer=0;
                float error;
                for(n=0;n<c->key_dim;n++) predicted+=(double)kn[n]*s[n*c->value_dim+j];
                error=(v[j]-(float)predicted)*rate;
                for(n=0;n<c->key_dim;n++) {s[n*c->value_dim+j]+=kn[n]*error;answer+=(double)qn[n]*s[n*c->value_dim+j];}
                read[j]=(float)answer;
            }
            norm(read,w[5].data,c->value_dim,c->epsilon,0,read);
            for(j=0;j<c->value_dim;j++) core[head*c->value_dim+j]=read[j]*(sigmoid_gate?sigmoid(z[head*c->value_dim+j]):silu(z[head*c->value_dim+j]));
        }
        matvec(core,w[8].data,c->hidden,vw,output.data+row*c->hidden);
    }
    free(memory);
    return valid(output,output.count)&&valid(history,history.count)&&valid(state,state.count)?0:3;
}

typedef int (*ExpertFFN)(void *,uint64_t,uint64_t,const float *,uint64_t,float *,uint64_t);
static int moe_run(const Config *c,const Tensor *w,uint64_t rows,int normalize,
        Tensor x,Tensor output,Tensor router,int64_t *indices,Tensor weights,
        ExpertFFN selected_expert,void *expert_user,uint64_t global_layer) {
    uint64_t i,e,j,rank,counts[7];
    float *memory,*prob,*expert,*shared,*temporary;
    if(!geometry(c)||!w||!rows||rows>1048576||!indices||(normalize!=0&&normalize!=1)) return 1;
    counts[0]=c->experts*c->hidden;counts[1]=c->experts*2*c->intermediate*c->hidden;
    counts[2]=c->experts*c->hidden*c->intermediate;
    counts[3]=counts[4]=c->shared*c->hidden;counts[5]=c->hidden*c->shared;counts[6]=c->hidden;
    for(j=0;j<7;j++) {
        if(selected_expert&&(j==1||j==2)) {
            if(w[j].data||w[j].count) return 1;
        } else if(!valid(w[j],counts[j])) return 1;
    }
    if(!valid(x,rows*c->hidden)||!output.data||output.count!=x.count||
        !router.data||router.count!=rows*c->experts||!weights.data||weights.count!=rows*c->topk) return 1;
    memory=allocate(c->experts+2*c->intermediate+2*c->shared+c->hidden+1);
    if(!memory) return 2;
    prob=memory;expert=prob+c->experts;shared=expert+2*c->intermediate;temporary=shared+2*c->shared;
    memset(output.data,0,(size_t)output.count*sizeof(float));
    for(i=0;i<rows;i++) {
        float *input=x.data+i*c->hidden,*logits=router.data+i*c->experts;
        double maximum=-DBL_MAX,total=0,selected_sum=0;
        matvec(input,w[0].data,c->experts,c->hidden,logits);
        for(e=0;e<c->experts;e++) if(logits[e]>maximum) maximum=logits[e];
        for(e=0;e<c->experts;e++) {prob[e]=(float)exp((double)logits[e]-maximum);total+=prob[e];}
        for(e=0;e<c->experts;e++) prob[e]=(float)(prob[e]/total);
        for(rank=0;rank<c->topk;rank++) {
            uint64_t best=0;float score=-1;
            for(e=0;e<c->experts;e++) {
                int used=0;
                for(j=0;j<rank;j++) if((uint64_t)indices[i*c->topk+j]==e) used=1;
                if(!used&&prob[e]>score) {best=e;score=prob[e];}
            }
            indices[i*c->topk+rank]=(int64_t)best;weights.data[i*c->topk+rank]=score;selected_sum+=score;
        }
        for(rank=0;rank<c->topk;rank++) {
            float scale=weights.data[i*c->topk+rank];
            if(normalize) scale=(float)(scale/selected_sum);
            weights.data[i*c->topk+rank]=scale;e=(uint64_t)indices[i*c->topk+rank];
            if(selected_expert) {
                if(selected_expert(expert_user,global_layer,e,input,c->hidden,temporary,c->hidden)) {
                    free(memory);return 4;
                }
                for(j=0;j<c->hidden;j++) if(!isfinite(temporary[j])) {free(memory);return 3;}
            } else {
                matvec(input,w[1].data+e*2*c->intermediate*c->hidden,2*c->intermediate,c->hidden,expert);
                for(j=0;j<c->intermediate;j++) expert[j]=silu(expert[j])*expert[c->intermediate+j];
                matvec(expert,w[2].data+e*c->hidden*c->intermediate,c->hidden,c->intermediate,temporary);
            }
            for(j=0;j<c->hidden;j++) output.data[i*c->hidden+j]+=scale*temporary[j];
        }
        matvec(input,w[3].data,c->shared,c->hidden,shared);
        matvec(input,w[4].data,c->shared,c->hidden,shared+c->shared);
        for(j=0;j<c->shared;j++) shared[j]=silu(shared[j])*shared[c->shared+j];
        matvec(shared,w[5].data,c->hidden,c->shared,temporary);
        matvec(input,w[6].data,1,c->hidden,temporary+c->hidden);
        for(j=0;j<c->hidden;j++) output.data[i*c->hidden+j]+=temporary[j]*sigmoid(temporary[c->hidden]);
    }
    free(memory);return valid(output,output.count)&&valid(router,router.count)&&valid(weights,weights.count)?0:3;
}

API int oa_qwen4_moe(const Config *c,const Tensor *w,uint64_t rows,int normalize,
        Tensor x,Tensor output,Tensor router,int64_t *indices,Tensor weights) {
    return moe_run(c,w,rows,normalize,x,output,router,indices,weights,NULL,NULL,0);
}
API uint64_t oa_qwen4_streamed_version(void) {return 1;}
API int oa_qwen4_moe_streamed(const Config *c,const Tensor *w,uint64_t rows,int normalize,
        Tensor x,Tensor output,Tensor router,int64_t *indices,Tensor weights,
        ExpertFFN selected_expert,void *expert_user,uint64_t global_layer) {
    if(!selected_expert) return 1;
    return moe_run(c,w,rows,normalize,x,output,router,indices,weights,selected_expert,expert_user,global_layer);
}

API int oa_qwen4_inject(uint64_t rows,uint64_t heads,uint64_t hidden,
        Tensor residual,Tensor value,Tensor gates,Tensor output) {
    uint64_t row,g,j;
    if(!rows||rows>1048576||heads<2||heads>64||!hidden||hidden>4096||
        !valid(residual,rows*heads*hidden)||!valid(value,rows*hidden)||
        !valid(gates,rows*heads)||!output.data||output.count!=residual.count) return 1;
    for(row=0;row<rows;row++) for(g=0;g<heads;g++) for(j=0;j<hidden;j++)
        output.data[(row*heads+g)*hidden+j]=residual.data[(row*heads+g)*hidden+j]+value.data[row*hidden+j]*gates.data[row*heads+g];
    return valid(output,output.count)?0:3;
}
API int oa_qwen4_head(uint64_t rows,uint64_t hidden,uint64_t vocab,Tensor x,Tensor weight,Tensor output) {
    uint64_t row;
    if(!rows||rows>1048576||!hidden||hidden>4096||!vocab||vocab>1048576||
        !valid(x,rows*hidden)||!valid(weight,vocab*hidden)||!output.data||output.count!=rows*vocab) return 1;
    for(row=0;row<rows;row++) matvec(x.data+row*hidden,weight.data,vocab,hidden,output.data+row*vocab);
    return valid(output,output.count)?0:3;
}

API int oa_qwen4_add(Tensor x,Tensor value,Tensor output) {
    uint64_t i;
    if(!x.count||!valid(x,x.count)||!valid(value,x.count)||!output.data||output.count!=x.count) return 1;
    for(i=0;i<x.count;i++) output.data[i]=x.data[i]+value.data[i];
    return valid(output,output.count)?0:3;
}

static void qwen4_rotate(float *x,uint64_t n,int64_t position,double theta,double scale) {
    uint64_t j;
    if(scale==1.0) {rotate(x,n,position,theta);return;}
    for(j=0;j<n/2;j++) {
        double angle=((double)position/scale)*pow(theta,-(double)(2*j)/(double)n);
        float a=x[j],b=x[j+n/2];
        x[j]=(float)(a*cos(angle)-b*sin(angle));
        x[j+n/2]=(float)(a*sin(angle)+b*cos(angle));
    }
}

/* Position banks are [axis,batch,context]. Publisher interleaving leaves T
   in every lane except the admitted H and W strides. */
static void qwen4_rotate_at(float *x,uint64_t n,const int64_t *positions,
    uint64_t index,uint64_t bank,const int32_t *sections,double theta,double scale) {
    uint64_t j;
    if(!sections) {qwen4_rotate(x,n,positions[index],theta,scale);return;}
    for(j=0;j<n/2;j++) {
        uint64_t axis=0;
        double angle; float a=x[j],b=x[j+n/2];
        if(j%3==1 && j<(uint64_t)sections[1]*3) axis=1;
        if(j%3==2 && j<(uint64_t)sections[2]*3) axis=2;
        angle=((double)positions[axis*bank+index]/scale)*pow(theta,-(double)(2*j)/(double)n);
        x[j]=(float)(a*cos(angle)-b*sin(angle));
        x[j+n/2]=(float)(a*sin(angle)+b*cos(angle));
    }
}

static int qwen4_qsa_run(const Config *c,const Tensor *w,uint64_t batch,uint64_t tokens,
    uint64_t start,uint64_t ih,uint64_t id,uint64_t budget,uint64_t ratio,
    const uint8_t *mask,const int64_t *positions,Tensor x,Tensor keys,Tensor values,
    Tensor raw,Tensor output,Tensor attention,uint8_t *chosen,double scale,const int32_t *sections) {
    uint64_t end,aw,kvw,b,t,j,k,g,head,counts[9],scratch,blocks,ip_width;
    float *memory,*qg,*ip,*pooled,*ik,*context,*scores,*prob;
    uint64_t *visible,*starts,*selected;
    if(!isfinite(scale)||scale<1||!geometry(c)||!w||!batch||batch>128||!tokens||tokens>c->capacity||start>c->capacity-tokens||
        !ih||ih>128||!id||id>4096||c->rotary>id||!ratio||ratio>c->capacity||
        !budget||budget>c->capacity||budget%ratio||!mask||!positions||!chosen) return 1;
    end=start+tokens;aw=c->heads*c->dim;kvw=c->kv*c->dim;
    counts[0]=2*aw*c->hidden;counts[1]=counts[2]=kvw*c->hidden;counts[3]=c->hidden*aw;
    counts[4]=counts[5]=c->dim;counts[6]=(ih+1)*id*c->hidden;counts[7]=counts[8]=id;
    for(j=0;j<9;j++) if(!valid(w[j],counts[j])) return 1;
    if(!valid(x,batch*tokens*c->hidden)||!valid(keys,batch*c->kv*end*c->dim)||
        !valid(values,keys.count)||!valid(raw,batch*end*id)||!output.data||output.count!=x.count||
        !attention.data||attention.count!=batch*c->heads*tokens*end) return 1;
    if(sections && (sections[0]<0||sections[1]<0||sections[2]<0||
        (uint64_t)sections[0]>c->rotary/2||(uint64_t)sections[1]>c->rotary/2||
        (uint64_t)sections[2]>c->rotary/2||
        (uint64_t)sections[0]+(uint64_t)sections[1]+(uint64_t)sections[2]!=c->rotary/2)) return 1;
    for(j=0;j<batch*end;j++) if(mask[j]>1) return 1;
    for(j=0;j<batch*end*(sections?3:1);j++) if(positions[j]<0||(uint64_t)positions[j]>=c->capacity) return 1;
    ip_width=(ih+1)*id>2*kvw?(ih+1)*id:2*kvw;
    scratch=3*aw+ip_width+2*id+3*end;
    memory=allocate(scratch);visible=(uint64_t *)calloc((size_t)(3*end),sizeof(uint64_t));
    if(!memory||!visible) {free(memory);free(visible);return 2;}
    qg=memory;ip=qg+2*aw;pooled=ip+ip_width;ik=pooled+id;context=ik+id;
    scores=context+aw;prob=scores+end;starts=visible+end;selected=starts+end;
    memset(chosen,0,(size_t)(batch*tokens*end));
    for(b=0;b<batch;b++) for(t=0;t<tokens;t++) {
        uint64_t row=b*tokens+t,slot=start+t,count=0,nchosen=0,block;
        float *input=x.data+row*c->hidden,*query=qg;
        matvec(input,w[6].data,(ih+1)*id,c->hidden,ip);
        memcpy(raw.data+(b*end+slot)*id,ip+ih*id,(size_t)id*sizeof(float));
        for(head=0;head<ih;head++) {norm(ip+head*id,w[7].data,id,c->epsilon,1,ip+head*id);qwen4_rotate_at(ip+head*id,c->rotary,positions,b*end+slot,batch*end,sections,c->theta,scale);}
        for(j=0;j<=slot;j++) if(mask[b*end+j]) visible[count++]=j;
        if(!count) {free(memory);free(visible);return 1;}
        blocks=count/ratio;
        for(block=0;block<blocks;block++) {
            double score=0;
            starts[block]=visible[block*ratio];
            for(j=0;j<id;j++) {
                double sum=0;
                for(k=0;k<ratio;k++) sum+=raw.data[(b*end+visible[block*ratio+k])*id+j];
                pooled[j]=(float)(sum/(double)ratio);
            }
            norm(pooled,w[8].data,id,c->epsilon,1,ik);
            qwen4_rotate_at(ik,c->rotary,positions,b*end+starts[block],batch*end,sections,c->theta,scale);
            for(head=0;head<ih;head++) {
                double dot=0;
                for(j=0;j<id;j++) dot+=(double)ip[head*id+j]*ik[j];
                if(dot>0) score+=dot;
            }
            scores[block]=(float)(score/sqrt((double)id));
        }
        for(g=0;g<budget/ratio&&g<blocks;g++) {
            uint64_t best=0;float largest=-1;
            for(block=0;block<blocks;block++) if(scores[block]>largest) {largest=scores[block];best=block;}
            scores[best]=-1;
            for(j=0;j<ratio;j++) selected[nchosen++]=visible[best*ratio+j];
        }
        for(j=blocks*ratio;j<count;j++) selected[nchosen++]=visible[j];
        for(j=0;j<nchosen;j++) chosen[row*end+selected[j]]=1;
        matvec(input,w[0].data,2*aw,c->hidden,qg);
        for(head=0;head<c->heads;head++) {
            norm(qg+head*2*c->dim,w[4].data,c->dim,c->epsilon,1,qg+head*2*c->dim);
            qwen4_rotate_at(qg+head*2*c->dim,c->rotary,positions,b*end+slot,batch*end,sections,c->theta,scale);
        }
        matvec(input,w[1].data,kvw,c->hidden,ip);
        matvec(input,w[2].data,kvw,c->hidden,ip+kvw);
        for(head=0;head<c->kv;head++) {
            norm(ip+head*c->dim,w[5].data,c->dim,c->epsilon,1,ip+head*c->dim);
            qwen4_rotate_at(ip+head*c->dim,c->rotary,positions,b*end+slot,batch*end,sections,c->theta,scale);
            memcpy(keys.data+((b*c->kv+head)*end+slot)*c->dim,ip+head*c->dim,(size_t)c->dim*sizeof(float));
            memcpy(values.data+((b*c->kv+head)*end+slot)*c->dim,ip+kvw+head*c->dim,(size_t)c->dim*sizeof(float));
        }
        for(head=0;head<c->heads;head++) {
            uint64_t kh=head/(c->heads/c->kv);
            double maximum=-DBL_MAX,total=0;
            query=qg+head*2*c->dim;
            for(j=0;j<nchosen;j++) {
                double dot=0;
                for(k=0;k<c->dim;k++) dot+=(double)query[k]*keys.data[((b*c->kv+kh)*end+selected[j])*c->dim+k];
                prob[j]=(float)(dot/sqrt((double)c->dim));if(prob[j]>maximum) maximum=prob[j];
            }
            for(j=0;j<nchosen;j++) {prob[j]=(float)exp((double)prob[j]-maximum);total+=prob[j];}
            for(j=0;j<end;j++) attention.data[((b*c->heads+head)*tokens+t)*end+j]=0;
            for(j=0;j<nchosen;j++) {
                prob[j]=(float)(prob[j]/total);
                attention.data[((b*c->heads+head)*tokens+t)*end+selected[j]]=prob[j];
            }
            for(k=0;k<c->dim;k++) {
                double sum=0;
                for(j=0;j<nchosen;j++) sum+=(double)prob[j]*values.data[((b*c->kv+kh)*end+selected[j])*c->dim+k];
                context[head*c->dim+k]=(float)sum*sigmoid(query[c->dim+k]);
            }
        }
        matvec(context,w[3].data,c->hidden,aw,output.data+row*c->hidden);
    }
    free(memory);free(visible);
    return valid(output,output.count)&&valid(keys,keys.count)&&valid(values,values.count)&&valid(raw,raw.count)&&valid(attention,attention.count)?0:3;
}

API int oa_qwen4_qsa(const Config *c,const Tensor *w,uint64_t batch,uint64_t tokens,
    uint64_t start,uint64_t ih,uint64_t id,uint64_t budget,uint64_t ratio,
    const uint8_t *mask,const int64_t *positions,Tensor x,Tensor keys,Tensor values,
    Tensor raw,Tensor output,Tensor attention,uint8_t *chosen) {
    return qwen4_qsa_run(c,w,batch,tokens,start,ih,id,budget,ratio,mask,positions,
        x,keys,values,raw,output,attention,chosen,1.0,NULL);
}

API uint64_t oa_qwen4_linear_rope_version(void) {return 1;}
API int oa_qwen4_qsa_scaled(const Config *c,const Tensor *w,uint64_t batch,uint64_t tokens,
    uint64_t start,uint64_t ih,uint64_t id,uint64_t budget,uint64_t ratio,
    const uint8_t *mask,const int64_t *positions,Tensor x,Tensor keys,Tensor values,
    Tensor raw,Tensor output,Tensor attention,uint8_t *chosen,double scale) {
    return qwen4_qsa_run(c,w,batch,tokens,start,ih,id,budget,ratio,mask,positions,
        x,keys,values,raw,output,attention,chosen,scale,NULL);
}

API uint64_t oa_qwen4_mrope_version(void) {return 1;}
API int oa_qwen4_qsa_mrope(const Config *c,const Tensor *w,uint64_t batch,uint64_t tokens,
    uint64_t start,uint64_t ih,uint64_t id,uint64_t budget,uint64_t ratio,
    const uint8_t *mask,const int64_t *positions,Tensor x,Tensor keys,Tensor values,
    Tensor raw,Tensor output,Tensor attention,uint8_t *chosen,double scale,const int32_t *sections) {
    if(!sections) return 1;
    return qwen4_qsa_run(c,w,batch,tokens,start,ih,id,budget,ratio,mask,positions,
        x,keys,values,raw,output,attention,chosen,scale,sections);
}

typedef int (*PLELookup)(void *,uint64_t,uint64_t,float *);
static int qwen4_ple_run(uint64_t batch,uint64_t tokens,uint64_t heads,uint64_t hidden,
    uint64_t embedding,uint64_t kernel,uint64_t ngram,uint64_t lanes_per_order,
    uint64_t eos,double epsilon,const Tensor *w,const int64_t *multipliers,
    const int64_t *sizes,const int64_t *offsets,const int64_t *ids,int64_t *previous,
    const uint8_t *mask,Tensor x,Tensor history,Tensor output,PLELookup lookup,void *lookup_user,uint64_t table_rows) {
    uint64_t width,lanes,part,span,rows,row,g,j,lag,lane,embedding_rows;
    float *memory,*features,*key,*value,*query,*gated,*normalized;
    int64_t *tokens_window;
    if(!batch||batch>128||!tokens||tokens>65536||heads<2||heads>64||!hidden||hidden>4096||
        ngram<2||ngram>64||!lanes_per_order||lanes_per_order>128||!embedding||embedding>65536||
        !kernel||kernel>64||!w||!multipliers||!sizes||!offsets||!ids||!previous||!mask||
        !isfinite(epsilon)||epsilon<=0||eos>INT64_MAX) return 1;
    width=heads*hidden;lanes=(ngram-1)*lanes_per_order;span=(kernel-1)*ngram;rows=batch*tokens;
    if(embedding%lanes) return 1;
    part=embedding/lanes;embedding_rows=lookup?table_rows:w[0].count/part;
    if(!embedding_rows||(lookup?(w[0].count||w[0].data):(w[0].count%part||!valid(w[0],w[0].count)))||!valid(w[1],width*embedding)||
        !valid(w[2],hidden*embedding)||!valid(w[3],width)||!valid(w[4],width)||
        !valid(w[5],width)||!valid(w[6],width*kernel)||!valid(x,rows*width)||
        !valid(history,batch*width*span)||!output.data||output.count!=x.count) return 1;
    for(lane=0;lane<lanes;lane++) if(sizes[lane]<=0||offsets[lane]<0||
        (uint64_t)offsets[lane]>=embedding_rows||(uint64_t)sizes[lane]>embedding_rows-(uint64_t)offsets[lane]) return 1;
    for(lag=0;lag<ngram;lag++) if(multipliers[lag]<=0) return 1;
    for(j=0;j<rows;j++) if(ids[j]<0||mask[j]>1) return 1;
    for(j=0;j<batch*(ngram-1);j++) if(previous[j]<0) return 1;
    memory=allocate(embedding+4*width+hidden);
    tokens_window=(int64_t *)calloc((size_t)ngram,sizeof(int64_t));
    if(!memory||!tokens_window) {free(memory);free(tokens_window);return 2;}
    features=memory;key=features+embedding;value=key+width;query=value+hidden;
    gated=query+width;normalized=gated+width;
    for(row=0;row<rows;row++) {
        uint64_t b=row/tokens,hash=0;
        for(j=0;j<ngram-1;j++) tokens_window[j]=previous[b*(ngram-1)+j];
        tokens_window[ngram-1]=mask[row]?ids[row]:(int64_t)eos;
        for(lag=0;lag<ngram;lag++) {
            uint64_t source=ngram-1-lag,token=(uint64_t)tokens_window[source];
            for(j=source+1;j<ngram;j++) if((uint64_t)tokens_window[j]==eos) {
                /* Current EOS belongs to the preceding segment, not a new one. */
                if(j<ngram-1) token=eos;
            }
            if(token>INT64_MAX/(uint64_t)multipliers[lag]) {free(memory);free(tokens_window);return 1;}
            hash^=token*(uint64_t)multipliers[lag];
            if(lag) for(lane=(lag-1)*lanes_per_order;lane<lag*lanes_per_order;lane++) {
                uint64_t index=hash%(uint64_t)sizes[lane]+(uint64_t)offsets[lane];
                if(lookup) {
                    if(lookup(lookup_user,index,part,features+lane*part)) {free(memory);free(tokens_window);return 4;}
                } else memcpy(features+lane*part,w[0].data+index*part,(size_t)part*sizeof(float));
            }
        }
        memcpy(previous+b*(ngram-1),tokens_window+1,(size_t)(ngram-1)*sizeof(int64_t));
        matvec(features,w[1].data,width,embedding,key);
        matvec(features,w[2].data,hidden,embedding,value);
        for(g=0;g<heads;g++) {
            double score=0;
            float gate;
            norm(key+g*hidden,w[3].data+g*hidden,hidden,epsilon,1,key+g*hidden);
            norm(x.data+row*width+g*hidden,w[4].data+g*hidden,hidden,epsilon,1,query+g*hidden);
            for(j=0;j<hidden;j++) score+=(double)key[g*hidden+j]*query[g*hidden+j];
            score/=sqrt((double)hidden);
            gate=score==0?0:(float)(sqrt(fmax(fabs(score),1e-6))*(score>0?1:-1));
            gate=sigmoid(gate);
            for(j=0;j<hidden;j++) gated[g*hidden+j]=gate*value[j];
            norm(gated+g*hidden,w[5].data+g*hidden,hidden,epsilon,1,normalized+g*hidden);
        }
        for(g=0;g<width;g++) {
            double sum=0;
            float *past=history.data+(b*width+g)*span;
            if(!mask[row]) gated[g]=normalized[g]=0;
            for(j=0;j<kernel;j++) {
                float token=j*ngram<span?past[j*ngram]:normalized[g];
                sum+=(double)token*w[6].data[g*kernel+j];
            }
            output.data[row*width+g]=gated[g]+silu((float)sum);
            if(span) {memmove(past,past+1,(size_t)(span-1)*sizeof(float));past[span-1]=normalized[g];}
        }
    }
    free(memory);free(tokens_window);
    return valid(output,output.count)&&valid(history,history.count)?0:3;
}

API int oa_qwen4_ple(uint64_t batch,uint64_t tokens,uint64_t heads,uint64_t hidden,
    uint64_t embedding,uint64_t kernel,uint64_t ngram,uint64_t lanes_per_order,
    uint64_t eos,double epsilon,const Tensor *w,const int64_t *multipliers,
    const int64_t *sizes,const int64_t *offsets,const int64_t *ids,int64_t *previous,
    const uint8_t *mask,Tensor x,Tensor history,Tensor output) {
    return qwen4_ple_run(batch,tokens,heads,hidden,embedding,kernel,ngram,lanes_per_order,
        eos,epsilon,w,multipliers,sizes,offsets,ids,previous,mask,x,history,output,NULL,NULL,0);
}
API uint64_t oa_qwen4_streamed_ple_version(void) {return 1;}
API int oa_qwen4_ple_streamed(uint64_t batch,uint64_t tokens,uint64_t heads,uint64_t hidden,
    uint64_t embedding,uint64_t kernel,uint64_t ngram,uint64_t lanes_per_order,
    uint64_t eos,double epsilon,const Tensor *w,const int64_t *multipliers,
    const int64_t *sizes,const int64_t *offsets,const int64_t *ids,int64_t *previous,
    const uint8_t *mask,Tensor x,Tensor history,Tensor output,PLELookup lookup,void *user,uint64_t rows) {
    if(!lookup) return 1;
    return qwen4_ple_run(batch,tokens,heads,hidden,embedding,kernel,ngram,lanes_per_order,
        eos,epsilon,w,multipliers,sizes,offsets,ids,previous,mask,x,history,output,lookup,user,rows);
}

#include "native_qwen4_composition.c"
