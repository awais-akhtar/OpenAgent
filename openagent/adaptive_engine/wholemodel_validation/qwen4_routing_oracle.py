"""Observe explicitly pinned publisher routing outputs without recomputing routes."""
import threading

import numpy as np

from .qwen4_publisher_caches import contract

_LOCK = threading.RLock()


class Qwen4RoutingOracleForward:
    contract_factory=staticmethod(contract)
    def __init__(self, options):
        self.max_bytes = options['max_observation_bytes']
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise ValueError('bounded Qwen4 oracle observation admission required')
        self.config = self.contract_factory(options['config'], self.max_bytes)
        self.paths = tuple(options['router_paths'])
        self.classes = tuple(options['router_classes'])
        self.fields = tuple(options['output_indices'])
        if (len(self.paths)!=self.config.num_hidden_layers or
                len(set(self.paths))!=len(self.paths) or len(self.classes)!=len(self.paths) or
                any(type(v) is not str or not v for v in self.paths+self.classes) or
                len(self.fields)!=3 or len(set(self.fields))!=3 or
                any(type(v) is not int or v < 0 for v in self.fields)):
            raise ValueError('complete pinned router paths, classes and returned-field indices required')

    def __call__(self, model, kwargs):
        with _LOCK:
            ids = kwargs.get('input_ids')
            if ids is None or ids.ndim!=2 or not all(ids.shape):
                raise ValueError('explicit Qwen4 oracle token input required')
            batch, tokens = ids.shape
            c = self.config
            extent = batch*tokens*c.num_hidden_layers*c.num_experts
            if batch > 128 or extent*32 > self.max_bytes:
                raise MemoryError('complete Qwen4 oracle routing exceeds admission')
            modules = dict(model.named_modules())
            routers = []
            for path, expected in zip(self.paths,self.classes):
                module = modules.get(path)
                if module is None or type(module).__module__+'.'+type(module).__qualname__!=expected:
                    raise ValueError('pinned Qwen4 publisher router inventory differs: '+path)
                routers.append(module)
            raw = np.empty((batch,tokens,c.num_hidden_layers,c.num_experts),np.float32)
            weights = np.zeros_like(raw)
            seen = set()
            handles = []
            def hook(layer):
                def capture(module, inputs, output):
                    if layer in seen or not isinstance(output,(tuple,list)) or len(output)<=max(self.fields):
                        raise ValueError('pinned Qwen4 returned router contract differs')
                    logits, indices, scales = (output[index] for index in self.fields)
                    rows = batch*tokens
                    if (tuple(logits.shape)!=(rows,c.num_experts) or
                            tuple(indices.shape)!=(rows,c.num_experts_per_tok) or scales.shape!=indices.shape or
                            str(logits.dtype)!='torch.float32' or str(scales.dtype)!='torch.float32' or
                            str(indices.dtype)!='torch.int64'):
                        raise ValueError('pinned Qwen4 routing output geometry or precision differs')
                    values, chosen, gammas = (v.detach().cpu().numpy() for v in (logits,indices,scales))
                    if (not np.isfinite(values).all() or not np.isfinite(gammas).all() or
                            np.any(chosen<0) or np.any(chosen>=c.num_experts) or
                            any(np.unique(row).size!=c.num_experts_per_tok for row in chosen)):
                        raise ValueError('invalid observed publisher Qwen4 routing')
                    raw[:,:,layer,:] = values.reshape(batch,tokens,c.num_experts)
                    dense = np.zeros((rows,c.num_experts),np.float32)
                    np.put_along_axis(dense,chosen,gammas,axis=-1)
                    weights[:,:,layer,:] = dense.reshape(batch,tokens,c.num_experts)
                    seen.add(layer)
                return capture
            try:
                for layer,module in enumerate(routers):
                    handles.append(module.register_forward_hook(hook(layer)))
                output = model(**kwargs)
                if len(seen)!=c.num_hidden_layers:
                    raise ValueError('publisher forward omitted Qwen4 routing layers')
                output._openagent_routing_observations = {'router_logits':raw,'routing_weights':weights}
                return output
            finally:
                for handle in reversed(handles):
                    handle.remove()


def make_forward_call(*, options):
    return Qwen4RoutingOracleForward(options)
