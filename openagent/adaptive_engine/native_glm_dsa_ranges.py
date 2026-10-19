"""Selected native DSA ranges and edges with complete original inventory admission."""
from copy import deepcopy
import ctypes as C
import hashlib
import json
import math
import threading
from types import MappingProxyType

import numpy as np

from .independent_glm_dsa import IndependentGlmDsaDecoder
from .native_glm_dsa import NativeGlmDsaDecoder, Tensor, tensor, integers

GLOBALS = ('model.embed_tokens.weight','model.norm.weight','lm_head.weight')


def admit_inventory(config, records):
    owner = IndependentGlmDsaDecoder(config, {}, _inventory_only=True)
    shapes = dict(owner.tensor_shapes)
    if owner.config.tie_word_embeddings and 'lm_head.weight' not in records:
        shapes.pop('lm_head.weight')
    if set(records) != set(shapes) or any(tuple(records[name]['shape']) != shape or
            records[name].get('dtype') not in ('F32','F16','BF16') for name,shape in shapes.items()):
        raise ValueError('complete exact original DSA tensor inventory required')
    return owner.config, shapes


def partitions(config, stride):
    c = IndependentGlmDsaDecoder(config, {}, _inventory_only=True).config
    if type(stride) is not int or not 1 <= stride <= c.num_hidden_layers:
        raise ValueError('bounded DSA layers-per-range required')
    result, begin = [], 0
    while begin < c.num_hidden_layers:
        end = min(c.num_hidden_layers,begin+stride)
        while end < c.num_hidden_layers and c.indexer_types[end] != 'full':
            end += 1
        result.append((begin,end))
        begin = end
    return tuple(result)


class NativeGlmDsaRange(NativeGlmDsaDecoder):
    implementation = 'compiled-c-glm-dsa-selected-hidden-range'

    def __init__(self, config, tensors, *, begin, end, library, max_bytes, context_tokens=None,
                 expert_provider=None):
        if expert_provider is not None:
            raise ValueError(
                'GLM-DSA packed experts are restricted until source-content identity is bound')
        source = deepcopy(dict(config))
        inventory = IndependentGlmDsaDecoder(source, {}, _inventory_only=True)
        c = inventory.config
        if (type(begin) is not int or type(end) is not int or not 0 <= begin < end <= c.num_hidden_layers
                or c.indexer_types[begin] != 'full'):
            raise ValueError('DSA range must begin at a full indexer and own a valid layer interval')
        if context_tokens is None:
            context_tokens = c.max_position_embeddings
        if type(context_tokens) is not int or not 0 < context_tokens <= c.max_position_embeddings:
            raise ValueError('bounded DSA range context required')
        selected = {name:shape for name,shape in inventory.tensor_shapes.items()
                    if name.startswith('model.layers.') and begin <= int(name.split('.')[2]) < end}
        if expert_provider is not None:
            from .glm_dsa_streamed_experts import config_identity
            if expert_provider.config_digest != config_identity(source):
                raise ValueError('streamed DSA range configuration differs from expert manifest')
            selected = {name:shape for name,shape in selected.items() if not name.endswith(
                ('mlp.experts.gate_up_proj','mlp.experts.down_proj'))}
        if set(tensors) != set(selected):
            raise ValueError('exact selected original DSA layer names required')
        local = deepcopy(source)
        local['num_hidden_layers'] = end-begin
        local['max_position_embeddings'] = context_tokens
        for schedule in ('mlp_layer_types','indexer_types','layer_types'):
            local[schedule] = source[schedule][begin:end]
        renamed = {'.'.join(('model','layers',str(int(name.split('.')[2])-begin),*name.split('.')[3:])):value
                   for name,value in tensors.items()}
        super().__init__(local,renamed,library=library,max_bytes=max_bytes,
                         retain_diagnostics=False,_hidden_only=True,
                         expert_provider=expert_provider,_expert_layer_offset=begin)
        self.begin,self.end,self.capacity = begin,end,context_tokens
        self.source_config = source
        self._source_identity = hashlib.sha256(json.dumps(source,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).digest()
        self.source_tensors = MappingProxyType({name:self.weights[
            '.'.join(('model','layers',str(int(name.split('.')[2])-begin),*name.split('.')[3:]))]
            for name in selected})
        self.identity = hashlib.sha256((self.identity+json.dumps(source,sort_keys=True,
            separators=(',',':'),allow_nan=False)+f':range:{begin}:{end}').encode()).hexdigest()

    def _check_open(self):
        super()._check_open()
        expected = getattr(self,'_source_identity',None)
        if expected is not None and hashlib.sha256(json.dumps(self.source_config,sort_keys=True,
                separators=(',',':'),allow_nan=False).encode()).digest()!=expected:
            raise ValueError('DSA range source configuration changed after admission')

    @classmethod
    def from_directory(cls, directory, *, begin, end, library, max_weight_bytes,
                       max_bytes, context_tokens=None, read_limits=None, cancel=None, expert_provider=None):
        if expert_provider is not None:
            raise ValueError(
                'GLM-DSA packed experts are restricted until source-content identity is bound')
        from .native_checkpoint import read_native_text_checkpoint, _finish_decoder
        def select(config,records):
            c,shapes = admit_inventory(config,records)
            if (type(begin) is not int or type(end) is not int or not 0 <= begin < end <= c.num_hidden_layers
                    or c.indexer_types[begin]!='full'):
                raise ValueError('DSA selected range requires a full-indexer boundary')
            names = tuple(sorted(name for name in shapes if name.startswith('model.layers.')
                and begin <= int(name.split('.')[2]) < end and
                (expert_provider is None or not name.endswith(('mlp.experts.gate_up_proj','mlp.experts.down_proj')))))
            if 3*sum(math.prod(shapes[name])*4 for name in names) > max_bytes:
                raise MemoryError('selected DSA source conversion and ownership exceed admission')
            return names
        config,weights = read_native_text_checkpoint(directory,model_type='glm_moe_dsa',
            max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,
            tensor_selection=select,source_backend='public-range')
        return _finish_decoder(cls(config,weights,begin=begin,end=end,library=library,
            max_bytes=max_bytes,context_tokens=context_tokens,expert_provider=expert_provider),cancel)

    def close(self):
        super().close()
        self.source_tensors = MappingProxyType({})


class NativeGlmDsaEdge:
    def __init__(self,config,tensors,*,library,max_bytes):
        if not isinstance(library,C.CDLL) or type(max_bytes) is not int or max_bytes<=0:
            raise ValueError('borrowed native DSA library and positive edge budget required')
        self.config = IndependentGlmDsaDecoder(config,{},_inventory_only=True).config
        self._configuration_identity = hashlib.sha256(json.dumps(vars(self.config),
            sort_keys=True,allow_nan=False).encode()).digest()
        self.library,self.max_bytes = library,max_bytes
        self._lock,self._closed = threading.RLock(),False
        supplied = dict(tensors)
        c = self.config
        if c.tie_word_embeddings:
            if 'lm_head.weight' in supplied and not np.array_equal(supplied['lm_head.weight'],supplied.get(GLOBALS[0])):
                raise ValueError('inconsistent DSA tied edge head')
            supplied['lm_head.weight'] = supplied.get(GLOBALS[0])
        shapes = dict(zip(GLOBALS,((c.vocab_size,c.hidden_size),(c.hidden_size,),(c.vocab_size,c.hidden_size))))
        self.resident_weight_bytes = sum(math.prod(shape)*4 for shape in shapes.values())
        if set(supplied)!=set(shapes) or 2*self.resident_weight_bytes>max_bytes:
            raise ValueError('exact admitted DSA edge globals required')
        self.weights = {}
        digest = hashlib.sha256(json.dumps(vars(c),sort_keys=True,allow_nan=False).encode())
        for name,shape in shapes.items():
            value = np.asarray(supplied[name])
            if value.dtype!=np.float32 or value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('invalid DSA edge global: '+name)
            owned = np.frombuffer(value.tobytes(order='C'),np.float32).reshape(shape)
            self.weights[name] = owned
            digest.update(name.encode());digest.update(memoryview(owned).cast('B'))
        self.identity = digest.hexdigest()
        version=library.oa_glm_dsa_version
        version.argtypes,version.restype=[],C.c_uint64
        if version()!=2:raise ValueError('native DSA ABI version 2 required')
        library.oa_glm_dsa_embed.argtypes=[Tensor,C.c_uint64,C.c_uint64,C.POINTER(C.c_int64),C.c_uint64,Tensor]
        library.oa_glm_dsa_embed.restype=C.c_int
        library.oa_glm_dsa_head.argtypes=[Tensor,Tensor,Tensor,C.c_uint64,C.c_uint64,C.c_double,Tensor,Tensor]
        library.oa_glm_dsa_head.restype=C.c_int

    @classmethod
    def from_directory(cls,directory,*,library,max_weight_bytes,max_bytes,read_limits=None,cancel=None):
        from .native_checkpoint import read_native_text_checkpoint,_finish_decoder
        def select(config,records):
            _,shapes=admit_inventory(config,records)
            names=tuple(name for name in GLOBALS if name in shapes)
            if 3*sum(math.prod(shapes[name])*4 for name in names)>max_bytes:
                raise MemoryError('DSA edge conversion and ownership exceed admission')
            return names
        config,weights=read_native_text_checkpoint(directory,model_type='glm_moe_dsa',
            max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,
            tensor_selection=select,source_backend='public-range')
        return _finish_decoder(cls(config,weights,library=library,max_bytes=max_bytes),cancel)

    def _check(self,extra):
        if self._closed:raise RuntimeError('native DSA edge closed')
        if hashlib.sha256(json.dumps(vars(self.config),sort_keys=True,allow_nan=False).encode()).digest()!=self._configuration_identity:
            raise ValueError('native DSA edge configuration changed after admission')
        if self.resident_weight_bytes+extra>self.max_bytes:
            raise MemoryError('native DSA edge scratch exceeds admission')

    def embeddings(self,ids):
        with self._lock:
            values=np.asarray(ids);c=self.config
            if (values.ndim not in (1,2) or values.dtype.kind not in 'iu' or not 0<values.size<=65536
                    or np.any(values<0) or np.any(values>=c.vocab_size)):
                raise ValueError('bounded in-vocabulary DSA edge token rows required')
            self._check(values.size*(c.hidden_size*4+16))
            values=np.ascontiguousarray(values,np.int64)
            output=np.empty((*values.shape,c.hidden_size),np.float32)
            NativeGlmDsaDecoder._result(self.library.oa_glm_dsa_embed(tensor(self.weights[GLOBALS[0]]),
                c.vocab_size,c.hidden_size,integers(values),values.size,tensor(output)))
            return output

    def final_head(self,hidden):
        with self._lock:
            values=np.asarray(hidden);c=self.config
            if (values.dtype!=np.float32 or values.ndim not in (1,2,3) or values.shape[-1]!=c.hidden_size
                    or not 0<values.size//c.hidden_size<=65536 or not np.isfinite(values).all()):
                raise ValueError('bounded finite DSA edge hidden rows required')
            self._check(values.size*12+(values.size//c.hidden_size)*c.vocab_size*4)
            incoming=np.ascontiguousarray(values);normalized=np.empty_like(incoming)
            output=np.empty((*values.shape[:-1],c.vocab_size),np.float32)
            NativeGlmDsaDecoder._result(self.library.oa_glm_dsa_head(tensor(incoming),
                tensor(self.weights[GLOBALS[1]]),tensor(self.weights[GLOBALS[2]]),c.hidden_size,c.vocab_size,
                c.rms_norm_eps,tensor(normalized),tensor(output)))
            if not np.isfinite(output).all():raise FloatingPointError('nonfinite native DSA edge logits')
            return output

    def close(self):
        with self._lock:
            self.weights.clear();self._closed=True


__all__=['NativeGlmDsaRange','NativeGlmDsaEdge','partitions','admit_inventory']
