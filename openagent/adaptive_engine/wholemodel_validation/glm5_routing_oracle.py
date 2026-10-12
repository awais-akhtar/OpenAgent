"""Observe unchanged pinned publisher router outputs with temporary hooks."""
import re
import threading

import numpy as np

from ..independent_glm5_next import Glm5NextTextContract


_LOCK=threading.RLock()
_ROUTER_CLASS='transformers.models.glm5_next.modeling_glm5_next.Glm5NextTextTopkRouter'
_ROUTER_PATH=re.compile(r'(?:(?:model\.)?language_model\.|model\.)layers\.(\d+)\.mlp\.gate')


class Glm5RoutingOracleForward:
    def __init__(self,config,*,max_observation_bytes):
        self.contract=Glm5NextTextContract.parse(config)
        if type(max_observation_bytes) is not int or max_observation_bytes<=0:
            raise ValueError('bounded publisher routing observation admission required')
        self.max_bytes=max_observation_bytes

    def __call__(self,model,kwargs):
        with _LOCK:
            c=self.contract;ids=kwargs.get('input_ids')
            if ids is None or ids.ndim!=2 or not all(ids.shape):
                raise ValueError('explicit publisher routing input token geometry required')
            batch,tokens=ids.shape;experts=c.n_routed_experts
            extent=batch*tokens*c.num_hidden_layers*experts
            if extent*24>self.max_bytes:
                raise MemoryError('complete publisher router observation exceeds admission')
            sparse={i for i,kind in enumerate(c.mlp_layer_types) if kind=='sparse'}
            routers={}
            for path,module in model.named_modules():
                identity=type(module).__module__+'.'+type(module).__qualname__
                if identity!=_ROUTER_CLASS:continue
                match=_ROUTER_PATH.fullmatch(path)
                if match is None:raise ValueError('publisher router outside the declared decoder layout')
                layer=int(match.group(1))
                if layer in routers:raise ValueError('duplicate publisher routing layer')
                routers[layer]=module
            if set(routers)!=sparse:
                raise ValueError('complete pinned publisher router inventory required')
            raw=np.zeros((batch,tokens,c.num_hidden_layers,experts),np.float32)
            weights=np.zeros_like(raw);observed=set();handles=[]
            def hook(layer):
                def capture(module,inputs,output):
                    if layer in observed:raise ValueError('publisher router executed twice in one token observation')
                    if not isinstance(output,tuple) or len(output)!=3:
                        raise ValueError('pinned publisher router output contract differs')
                    logits,selected,indices=output
                    rows=batch*tokens
                    if tuple(logits.shape)!=(rows,experts) or tuple(selected.shape)!=(rows,c.num_experts_per_tok) or indices.shape!=selected.shape:
                        raise ValueError('publisher router output geometry differs')
                    if str(logits.dtype)!='torch.float32' or str(selected.dtype)!='torch.float32' or str(indices.dtype)!='torch.int64':
                        raise ValueError('publisher router observation precision differs')
                    scores=logits.detach().cpu().numpy()
                    chosen=indices.detach().cpu().numpy()
                    gammas=selected.detach().cpu().numpy()
                    if not np.isfinite(scores).all() or not np.isfinite(gammas).all() or np.any(chosen<0) or np.any(chosen>=experts):
                        raise ValueError('invalid publisher router observation')
                    if any(np.unique(row).size!=c.num_experts_per_tok for row in chosen):
                        raise ValueError('publisher router selected duplicate experts')
                    raw[:,:,layer,:]=scores.reshape(batch,tokens,experts)
                    dense=weights[:,:,layer,:].reshape(rows,experts)
                    # Scatter the publisher's returned weights; do not rerun its
                    # group selection, sigmoid, normalization or scaling.
                    dense[np.arange(rows)[:,None],chosen]=gammas
                    observed.add(layer)
                return capture
            try:
                for layer,module in routers.items():handles.append(module.register_forward_hook(hook(layer)))
                output=model(**kwargs)
                if observed!=sparse:raise ValueError('publisher forward omitted declared routing layers')
                output._openagent_routing_observations={'router_logits':raw,'routing_weights':weights}
                return output
            finally:
                for handle in reversed(handles):handle.remove()


def make_forward_call(*,options):
    return Glm5RoutingOracleForward(options['config'],max_observation_bytes=options['max_observation_bytes'])

__all__=['Glm5RoutingOracleForward','make_forward_call']
