"""Selected native Qwen3.5 layers and separately owned embedding/head."""
import math
import numpy as np

from .independent_qwen35 import IndependentQwen35TextDecoder
from .native_qwen35 import NativeQwen35TextDecoder,tensor


def qwen35_tensor_inventory(config):
    contract = IndependentQwen35TextDecoder.__new__(IndependentQwen35TextDecoder)
    IndependentQwen35TextDecoder.__init__(contract,config,{},_inventory_only=True)
    return dict(contract.tensor_shapes)


def _load(directory,*,begin,end,edge,library,max_weight_bytes,max_bytes,read_limits,cancel,expert_provider=None,context_tokens=None):
    from .native_checkpoint import read_native_text_checkpoint,_finish_decoder
    if type(max_bytes) is not int or max_bytes<=0:
        raise ValueError('explicit Qwen3.5 ownership memory budget required')
    def select(config,inventory):
        shapes = qwen35_tensor_inventory(config)
        physical=config['max_position_embeddings'] if context_tokens is None else context_tokens
        if type(physical) is not int or not 1<=physical<=min(config['max_position_embeddings'],65536):
            raise ValueError('bounded native Qwen3.5 physical context required before payload conversion')
        if config.get('tie_word_embeddings',False) and 'lm_head.weight' not in inventory:
            shapes.pop('lm_head.weight')
        if set(shapes)!=set(inventory):
            raise ValueError('exact complete Qwen3.5 source inventory required')
        for name,shape in shapes.items():
            if tuple(inventory[name]['shape'])!=tuple(shape):
                raise ValueError('Qwen3.5 source shape mismatch: '+name)
        if edge:
            names = tuple(sorted(name for name in shapes if not name.startswith('model.layers.')))
        else:
            if type(begin) is not int or type(end) is not int or not 0<=begin<end<=config['num_hidden_layers']:
                raise ValueError('valid Qwen3.5 selected layer interval required')
            prefixes = tuple(f'model.layers.{index}.' for index in range(begin,end))
            names = tuple(sorted(name for name in shapes if name.startswith(prefixes)))
            if expert_provider is not None:
                names = tuple(name for name in names if not name.endswith(
                    ('mlp.experts.gate_up_proj','mlp.experts.down_proj')))
        size = 4*sum(math.prod(shapes[name]) for name in names)
        if edge and config.get('tie_word_embeddings',False) and 'lm_head.weight' not in shapes:
            size += 4*math.prod(shapes['model.embed_tokens.weight'])
        if 3*size+getattr(expert_provider,'max_bytes',0)>max_bytes:
            raise MemoryError('Qwen3.5 selected checkpoint conversion exceeds admission')
        return names
    config,weights = read_native_text_checkpoint(directory,model_type='qwen3_5_moe_text',
        max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,tensor_selection=select)
    if edge:
        model=NativeQwen35Edge(config,weights,library=library,max_bytes=max_bytes,context_tokens=context_tokens)
    else:
        model = NativeQwen35Range(config,weights,begin=begin,end=end,library=library,max_bytes=max_bytes,
                                  expert_provider=expert_provider,context_tokens=context_tokens)
    return _finish_decoder(model,cancel)


class NativeQwen35Range(NativeQwen35TextDecoder):
    implementation = 'c-qwen35-selected-hidden-range'

    def __init__(self,config,tensors,*,begin,end,library,max_bytes,expert_provider=None,context_tokens=None):
        super().__init__(config,tensors,library=library,max_bytes=max_bytes,layer_range=(begin,end),
                         expert_provider=expert_provider,context_tokens=context_tokens)

    @classmethod
    def from_directory(cls,directory,*,begin,end,library,max_weight_bytes,max_bytes,read_limits=None,cancel=None,
                       expert_provider=None,context_tokens=None):
        return _load(directory,begin=begin,end=end,edge=False,library=library,max_weight_bytes=max_weight_bytes,
                     max_bytes=max_bytes,read_limits=read_limits,cancel=cancel,expert_provider=expert_provider,context_tokens=context_tokens)

    def forward(self,hidden,*,attention_mask=None,position_ids=None,cancelled=None):
        with self._lock:
            self._check_open()
            values = np.asarray(hidden)
            if values.dtype!=np.float32 or values.ndim!=3 or not all(values.shape) or values.shape[-1]!=self.config.hidden_size:
                raise ValueError('FP32 [batch,tokens,hidden] range input required')
            ids = np.zeros(values.shape[:2],np.int64)
            return self._execute(ids,attention_mask,position_ids,values,cancelled=cancelled)


class NativeQwen35Edge(NativeQwen35TextDecoder):
    implementation = 'c-qwen35-native-edge'

    def __init__(self,config,tensors,*,library,max_bytes,context_tokens=None):
        super().__init__(config,tensors,library=library,max_bytes=max_bytes,_edge_only=True,context_tokens=context_tokens)

    @classmethod
    def from_directory(cls,directory,*,library,max_weight_bytes,max_bytes,read_limits=None,cancel=None,context_tokens=None):
        return _load(directory,begin=None,end=None,edge=True,library=library,max_weight_bytes=max_weight_bytes,
                     max_bytes=max_bytes,read_limits=read_limits,cancel=cancel,context_tokens=context_tokens)

    @property
    def resident_weight_bytes(self):
        return sum(value.nbytes for value in self.weights.values())

    def embeddings(self,token_ids):
        with self._lock:
            self._check_open()
            ids = np.asarray(token_ids)
            c = self.config
            if ids.ndim not in (1,2) or not ids.size or ids.size>65536 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=c.vocab_size):
                raise ValueError('bounded Qwen3.5 edge token IDs required')
            self._check_bytes(self.resident_weight_bytes+ids.size*(16+4*c.hidden_size))
            ids = np.ascontiguousarray(ids,dtype=np.int64)
            output = np.empty((*ids.shape,c.hidden_size),np.float32)
            self._check(self._embed(tensor(self.weights['model.embed_tokens.weight']),c.vocab_size,c.hidden_size,
                                   ids.ctypes.data,ids.size,tensor(output)))
            return output

    def final_head(self,hidden):
        with self._lock:
            self._check_open()
            values = np.asarray(hidden)
            c = self.config
            if values.ndim not in (2,3) or values.dtype!=np.float32 or values.shape[-1]!=c.hidden_size or not np.isfinite(values).all():
                raise ValueError('finite FP32 Qwen3.5 edge hidden rows required')
            rows = values.size//c.hidden_size
            if not 0<rows<=65536:
                raise ValueError('Qwen3.5 edge rows exceed admission')
            self._check_bytes(self.resident_weight_bytes+rows*(12*c.hidden_size+4*c.vocab_size))
            incoming = np.ascontiguousarray(values)
            normalized = np.empty_like(incoming)
            output = np.empty((*values.shape[:-1],c.vocab_size),np.float32)
            self._check(self._head(tensor(incoming),tensor(self.weights['model.norm.weight']),
                tensor(self.weights['lm_head.weight']),c.hidden_size,c.vocab_size,c.rms_norm_eps,
                tensor(normalized),tensor(output)))
            return output


__all__ = ['NativeQwen35Range','NativeQwen35Edge','qwen35_tensor_inventory']
