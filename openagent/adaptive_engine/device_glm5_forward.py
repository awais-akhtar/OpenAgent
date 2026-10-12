"""GLM5-next device hyperconnections, KDA, shared DSA and expert composition."""
import math
from .device_publisher_projections import device_projection
from .device_qwen35_layers import _cancel

def glm5_device_forward(torch,ids,weights,c,slots,*,previous,start,capacity,max_work_bytes,
                        inputs_embeds=None,cancel=None,expert_provider=None,capture_routing=False):
    batch,tokens=ids.shape;h,r=c.hidden_size,c.hc_mult
    if not batch or not tokens or ids.dtype!=torch.int64 or start<0 or start+tokens>capacity:
        raise ValueError('bounded GLM device token transaction required')
    if bool((ids<0).any()) or bool((ids>=c.vocab_size).any()):
        raise ValueError('GLM token ID outside vocabulary')
    if bool(start)!=bool(previous) or (previous and len(previous)!=c.num_hidden_layers):
        raise ValueError('complete GLM physical continuation required')
    if type(capture_routing) is not bool:raise ValueError('boolean GLM routing observation policy required')
    routing_bytes=batch*tokens*c.num_hidden_layers*c.n_routed_experts*8 if capture_routing else 0
    if routing_bytes+batch*tokens*h*8+capacity*(c.num_attention_heads+c.index_head_dim)*16+32*max(h,c.intermediate_size,c.moe_intermediate_size)>max_work_bytes:
        raise MemoryError('GLM device layer transaction exceeds working admission')
    def rms(value,weight=None):
        result=value*torch.rsqrt(value.square().mean(dim=-1,keepdim=True)+c.rms_norm_eps)
        return result if weight is None else result*weight
    def project(value,weight):
        return device_projection(torch,value.reshape(1,-1),weight,cancel=cancel)[0]
    def choose(scores,count):
        if not bool(torch.isfinite(scores).all()): raise ArithmeticError('nonfinite GLM routing/index scores')
        order=torch.argsort(scores,descending=True,stable=True)
        if 0<count<scores.numel() and bool(scores[order[count-1]]==scores[order[count]]):
            raise ValueError('ambiguous GLM selection boundary')
        return order[:count]
    def collapse(streams,a,base):
        projected=project(rms(streams.reshape(-1)),a[base])
        pre=(projected[:r]*a[base+2][0]+a[base+1][:r]).sigmoid()+c.hc_eps
        post=2*(projected[r:2*r]*a[base+2][1]+a[base+1][r:2*r]).sigmoid()
        matrix=(projected[2*r:]*a[base+2][2]+a[base+1][2*r:]).reshape(r,r).softmax(dim=-1)+c.hc_eps
        for iteration in range(c.hc_sinkhorn_iters):
            if iteration: matrix=matrix/(matrix.sum(dim=-1,keepdim=True)+c.hc_eps)
            matrix=matrix/(matrix.sum(dim=0,keepdim=True)+c.hc_eps)
        return (streams*pre[:,None]).sum(dim=0),post,matrix
    def expand(streams,value,post,matrix):
        return post[:,None]*value+matrix.T@streams
    def mlp(x,gate,up,down):
        g=project(x,gate).clamp_max(c.swiglu_limit)
        u=project(x,up).clamp(-c.swiglu_limit,c.swiglu_limit)
        return project(torch.nn.functional.silu(g)*u,down)
    def shape(kind):
        if kind=='linear_attention':
            return ((3*c.linear_num_heads*c.linear_head_dim,c.linear_conv_kernel_dim),
                    (c.linear_num_heads,c.linear_head_dim,c.linear_head_dim))
        return ((capacity,c.num_attention_heads,c.qk_nope_head_dim+c.v_head_dim),(capacity,2*c.index_head_dim))
    if inputs_embeds is not None and (inputs_embeds.shape!=(batch,tokens,h) or inputs_embeds.dtype!=torch.float32 or inputs_embeds.device!=ids.device):
        raise ValueError('matching GLM device media embeddings required')
    _cancel(cancel)
    with torch.inference_mode():
        pending=tuple(bank.clone() for bank in previous) if previous else tuple(torch.zeros(
            (batch,sum(math.prod(value) for value in shape(kind))),dtype=torch.float32,device=ids.device) for kind in c.layer_types)
        hidden=torch.empty((batch,tokens,h),device=ids.device,dtype=torch.float32)
        routing_shape=(batch,tokens,c.num_hidden_layers,c.n_routed_experts)
        router_logits=torch.zeros(routing_shape,device=ids.device,dtype=torch.float32) if capture_routing else None
        routing_weights=torch.zeros(routing_shape,device=ids.device,dtype=torch.float32) if capture_routing else None
        for row in range(batch):
            for step in range(tokens):
                _cancel(cancel);pos=start+step;selected=None
                embedding=weights['embed_tokens.weight'][ids[row,step]] if inputs_embeds is None else inputs_embeds[row,step]
                streams=embedding.expand(r,h).clone()
                for layer,kind in enumerate(c.layer_types):
                    _cancel(cancel);a=[None if name is None else weights[name] for name in slots[layer]]
                    first_shape,second_shape=shape(kind);extent=math.prod(first_shape)
                    first=pending[layer][row,:extent].reshape(first_shape);second=pending[layer][row,extent:].reshape(second_shape)
                    x,post,matrix=collapse(streams,a,0);x=rms(x,a[6])
                    if kind=='linear_attention':
                        n,d=c.linear_num_heads,c.linear_head_dim;width=n*d
                        qkv=torch.cat(tuple(project(x,a[index]) for index in (8,9,10)))
                        if c.linear_conv_kernel_dim>1: first[:,:-1]=first[:,1:].clone()
                        first[:,-1]=qkv;qkv=torch.nn.functional.silu((first*a[11][:,0]).sum(dim=-1))
                        q,k,v=qkv.reshape(3,n,d).unbind(0)
                        q=q*torch.rsqrt(q.square().sum(dim=-1,keepdim=True)+1e-6)/math.sqrt(d)
                        k=k*torch.rsqrt(k.square().sum(dim=-1,keepdim=True)+1e-6)
                        forget=project(project(x,a[12]),a[13]).reshape(n,d)+a[14].reshape(n,d)
                        rate=a[15].exp()[:,None]
                        logdec=(c.linear_lower_bound*(rate*forget).sigmoid() if c.linear_lower_bound is not None
                                else -rate*torch.nn.functional.softplus(forget))
                        second*=logdec.exp()[...,None]
                        error=(v-torch.einsum('hi,hij->hj',k,second))*project(x,a[16]).sigmoid()[:,None]
                        second+=k[...,None]*error[:,None,:]
                        core=rms(torch.einsum('hi,hij->hj',q,second),a[19])
                        gate=project(project(x,a[17]),a[18]).reshape(n,d).sigmoid()
                        value=project((core*gate).reshape(width),a[20])
                    else:
                        n,d,v=c.num_attention_heads,c.qk_nope_head_dim,c.v_head_dim
                        qr=rms(project(x,a[8]),a[9]);query=project(qr,a[10]).reshape(n,d)
                        latent=rms(project(x,a[11]),a[12]);first[pos]=project(latent,a[13]).reshape(n,d+v)
                        if c.indexer_types[layer]=='full':
                            iq=project(qr,a[15]).reshape(c.index_n_heads,c.index_head_dim)
                            key=project(x,a[16]);key=(key-key.mean())*torch.rsqrt((key-key.mean()).square().mean()+1e-6)*a[17]+a[18]
                            second[pos,:c.index_head_dim]=key;second[pos,c.index_head_dim:]=project(x,a[21])
                            iw=project(x,a[19]);blocks=(pos+1)//c.index_kpool
                            selected=torch.zeros(pos+1,dtype=torch.bool,device=ids.device)
                            if blocks:
                                data=second[:blocks*c.index_kpool].reshape(blocks,c.index_kpool,2*c.index_head_dim)
                                gates=(data[...,c.index_head_dim:]+a[20]).softmax(dim=1)
                                pooled=(data[...,:c.index_head_dim]*gates).sum(dim=1)
                                scores=((iq@pooled.T)/math.sqrt(c.index_head_dim)).clamp_min(0)
                                scores=(scores*iw[:,None]).sum(dim=0)/math.sqrt(c.index_n_heads)
                                picked=choose(scores,min(blocks,c.index_topk//c.index_kpool))
                                selected[:blocks*c.index_kpool].reshape(blocks,c.index_kpool)[picked]=True
                            selected[blocks*c.index_kpool:]=True
                        if selected is None: raise ValueError('GLM shared DSA has no preceding full indexer mask')
                        indices=torch.nonzero(selected,as_tuple=False).flatten()
                        core=torch.empty((n,v),device=ids.device,dtype=torch.float32)
                        for head in range(n):
                            _cancel(cancel);probability=((first[indices,head,:d]@query[head])/math.sqrt(d)).softmax(dim=0)
                            core[head]=probability@first[indices,head,d:]
                        value=project(core.reshape(-1),a[14])
                    intermediate=expand(streams,value,post,matrix)
                    x,post,matrix=collapse(intermediate,a,3);x=rms(x,a[7])
                    if c.mlp_layer_types[layer]=='dense':
                        value=mlp(x,a[24],a[25],a[26])
                    else:
                        raw_scores=project(x,a[24]);scores=raw_scores.sigmoid();choice=scores+a[25]
                        if capture_routing:router_logits[row,step,layer]=raw_scores
                        grouped=choice.reshape(c.n_group,-1)
                        group_scores=grouped.topk(2,dim=-1).values.sum(dim=-1)
                        groups=choose(group_scores,c.topk_group);eligible=torch.zeros(c.n_group,dtype=torch.bool,device=ids.device);eligible[groups]=True
                        choice=choice.masked_fill(~eligible.repeat_interleave(c.n_routed_experts//c.n_group),-torch.finfo(torch.float32).max)
                        picked=choose(choice,c.num_experts_per_tok)
                        factor=c.routed_scaling_factor/(scores[picked].sum()+1e-20) if c.norm_topk_prob else c.routed_scaling_factor
                        if capture_routing:routing_weights[row,step,layer,picked]=scores[picked]*factor
                        value=mlp(x,a[28],a[29],a[30])
                        for expert_tensor in picked.sort().values:
                            _cancel(cancel);expert=int(expert_tensor)
                            if expert_provider is None:
                                gate,up=a[26][expert].chunk(2,dim=0)
                                product=mlp(x,gate,up,a[27][expert])
                            else:
                                product=expert_provider.execute(layer,expert,x.reshape(1,-1),torch=torch,device=str(ids.device),cancel=cancel)[0]
                            value+=product*scores[expert]*factor
                    streams=expand(intermediate,value,post,matrix)
                hidden[row,step]=rms(streams.mean(dim=0),weights['norm.weight'])
        _cancel(cancel)
        if any(not bool(torch.isfinite(value).all()) for value in (hidden,*pending)):
            raise ArithmeticError('nonfinite GLM device transaction')
        if capture_routing:
            if not bool(torch.isfinite(router_logits).all()) or not bool(torch.isfinite(routing_weights).all()):
                raise ArithmeticError('nonfinite GLM device routing observation')
            return hidden,pending,router_logits,routing_weights
        return hidden,pending

__all__=['glm5_device_forward']
