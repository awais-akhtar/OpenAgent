"""Inkling device token composition with caller-owned pending cache banks."""
import math
from .independent_inkling import LAYER_NAMES,MOE_NAMES
from .device_publisher_projections import device_projection
from .device_qwen35_layers import _cancel

def inkling_device_forward(torch,ids,weights,c,geometry,*,previous,start,max_work_bytes,
                           inputs_embeds=None,cancel=None,expert_provider=None,raw_hidden=False,position_origin=0):
    batch,tokens=ids.shape;h=c['hidden_size'];capacity=c['max_position_embeddings'];kernel=c['conv_kernel_size']
    if ids.dtype!=torch.int64 or not batch or not tokens or start<0 or start+tokens>capacity:
        raise ValueError('bounded Inkling device token transaction required')
    if type(previous) is not tuple or len(previous) not in (0,len(geometry)):
        raise ValueError('complete Inkling device cache inventory required')
    if bool((ids<0).any()) or bool((ids>=c['vocab_size']).any()):
        raise ValueError('Inkling token ID outside admitted vocabulary')
    if bool(start)!=bool(previous):
        raise ValueError('Inkling cache position and state disagree')
    peak=batch*tokens*(h+len(geometry)*(c['n_routed_experts']+c['n_shared_experts']))*8
    peak+=16*max(h,c['intermediate_size'],c['moe_intermediate_size'],capacity)
    if type(max_work_bytes) is not int or peak>max_work_bytes:
        raise MemoryError('Inkling device activation transaction exceeds admission')
    if inputs_embeds is not None and (inputs_embeds.shape!=(batch,tokens,h)
            or inputs_embeds.dtype!=torch.float32 or inputs_embeds.device!=ids.device):
        raise ValueError('matching Inkling device media embeddings required')
    def rms(value,weight):
        return value*torch.rsqrt(value.square().mean(dim=-1,keepdim=True)+c['rms_norm_eps'])*weight
    def project(value,weight):
        return device_projection(torch,value.reshape(1,-1),weight,cancel=cancel)[0]
    def fir(value,weight,history):
        if kernel>1:
            history[:-1]=history[1:].clone()
        history[-1]=value
        return value+(history*weight[:,0,:].T).sum(dim=0)
    def views(state,g):
        v=g.kv_width
        sizes=(capacity*v,capacity*v,kernel*v,kernel*v,kernel*h,kernel*h)
        shapes=((capacity,g.kv_heads,g.dim),(capacity,g.kv_heads,g.dim),(kernel,v),(kernel,v),(kernel,h),(kernel,h))
        values=[];offset=0
        for extent,shape in zip(sizes,shapes):
            values.append(state[offset:offset+extent].reshape(shape));offset+=extent
        return values
    _cancel(cancel)
    with torch.inference_mode():
        pending=tuple(bank.clone() for bank in previous) if previous else tuple(
            torch.zeros((batch,2*capacity*g.kv_width+2*kernel*(g.kv_width+h)),device=ids.device,dtype=torch.float32)
            for g in geometry)
        hidden=torch.empty((batch,tokens,h),device=ids.device,dtype=torch.float32)
        routing=torch.zeros((batch,tokens,len(geometry),c['n_routed_experts']+c['n_shared_experts']),device=ids.device,dtype=torch.float32)
        for row in range(batch):
            for step in range(tokens):
                _cancel(cancel);pos=start+step
                x=(rms(weights['model.embed_tokens.weight'][ids[row,step]],weights['model.embed_norm.weight'])
                    if inputs_embeds is None else inputs_embeds[row,step])
                for layer,g in enumerate(geometry):
                    _cancel(cancel);prefix=f'model.layers.{layer}.'
                    a=[weights.get(prefix+name) for name in LAYER_NAMES]
                    keys,values,kc,vc,ac,mc=views(pending[layer][row],g)
                    normalized=rms(x,a[0])
                    q=rms(project(normalized,a[2]).reshape(g.heads,g.dim),a[9])
                    keys[pos]=rms(fir(project(normalized,a[3]),a[7],kc).reshape(g.kv_heads,g.dim),a[10])
                    values[pos]=fir(project(normalized,a[4]),a[8],vc).reshape(g.kv_heads,g.dim)
                    relative=project(normalized,a[5]).reshape(g.heads,c['d_rel'])
                    floor=c['log_scaling_n_floor'];tau=1.0
                    if not g.window and floor is not None:
                        tau+=c['log_scaling_alpha']*math.log(max((pos+position_origin+1)/floor,1.0))
                    q=q*tau;first=max(0,pos-g.window+1) if g.window else 0
                    distances=pos-torch.arange(first,pos+1,device=ids.device)
                    valid=distances<g.extent;context=torch.empty((g.heads,g.dim),device=ids.device,dtype=torch.float32)
                    for head in range(g.heads):
                        _cancel(cancel);kh=head//(g.heads//g.kv_heads)
                        scores=(keys[first:pos+1,kh]@q[head])/g.dim
                        profile=project(relative[head],a[11].T)
                        scores[valid]+=profile[distances[valid]]*tau
                        context[head]=scores.softmax(dim=0)@values[first:pos+1,kh]
                    x=x+fir(project(context.reshape(-1),a[6]),a[12],ac)
                    normalized=rms(x,a[1])
                    if c['mlp_layer_types'][layer]=='dense':
                        gate=project(normalized,a[14]);up=project(normalized,a[15])
                        feedforward=project(torch.nn.functional.silu(gate)*up,a[16])*a[17][0]
                    else:
                        moe=[weights.get(prefix+name) for name in MOE_NAMES]
                        e,s,top=c['n_routed_experts'],c['n_shared_experts'],c['num_experts_per_tok']
                        logits=project(normalized,moe[0]);choice=logits[:e].sigmoid()+moe[2]
                        order=torch.argsort(choice,descending=True,stable=True)
                        if top<e and bool(choice[order[top-1]]==choice[order[top]]):
                            raise ValueError('ambiguous Inkling device top-k boundary')
                        chosen=torch.cat((order[:top],torch.arange(e,e+s,device=ids.device)))
                        probability=torch.nn.functional.logsigmoid(logits[chosen]).softmax(dim=0)*c['route_scale']*moe[1][0]
                        gammas=torch.zeros_like(logits);gammas[chosen]=probability
                        routing[row,step,layer]=gammas;feedforward=torch.zeros_like(normalized)
                        for expert_tensor in order[:top].sort().values:
                            _cancel(cancel);expert=int(expert_tensor)
                            if expert_provider is None:
                                gate,up=project(normalized,moe[3][expert]).chunk(2)
                                product=project(torch.nn.functional.silu(gate)*up,moe[4][expert])
                            else:
                                product=expert_provider.execute(layer,expert,normalized.reshape(1,-1),torch=torch,device=str(ids.device),cancel=cancel)[0]
                            feedforward+=product*gammas[expert]
                        for expert in range(s):
                            _cancel(cancel);gate=project(normalized,moe[5][expert]);up=project(normalized,moe[6][expert])
                            feedforward+=project(torch.nn.functional.silu(gate)*up*gammas[e+expert],moe[7][expert])
                    x=x+fir(feedforward,a[13],mc)
                hidden[row,step]=x if raw_hidden else rms(x,weights['model.norm.weight'])
        _cancel(cancel)
        if any(not bool(torch.isfinite(value).all()) for value in (hidden,routing,*pending)):
            raise ArithmeticError('nonfinite Inkling device transaction')
        return hidden,routing,pending

__all__=['inkling_device_forward']
