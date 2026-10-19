"""Owned native Mixtral embedding/final norm/head for split execution."""
import ctypes as C
import hashlib
import json
import math
import threading

import numpy as np

from .native_mixtral_decoder import configuration, GLOBAL_NAMES, Tensor, tensor


class NativeMixtralEdge:
    @classmethod
    def from_directory(cls, directory, *, library, max_weight_bytes, max_bytes,
                       tokenizer=None, read_limits=None, cancel=None):
        from .mixtral_inventory import admit_mixtral_inventory
        from .native_checkpoint import read_native_text_checkpoint, _finish_decoder
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('explicit edge memory budget required')
        def select(config, inventory):
            settings,_,_,globals_ = admit_mixtral_inventory(config,inventory)
            h,v = settings['hidden_size'],settings['vocab_size']
            owned_bytes = 4*(2*v*h+h)
            if 3*owned_bytes > max_bytes:
                raise MemoryError('Mixtral edge source conversion and ownership exceed admission')
            return tuple(sorted(globals_))
        config,weights = read_native_text_checkpoint(directory,model_type='mixtral',
            max_weight_bytes=max_weight_bytes,read_limits=read_limits,cancel=cancel,
            tensor_selection=select,source_backend='public-range')
        model = cls(config,weights,library=library,max_bytes=max_bytes,tokenizer=tokenizer)
        return _finish_decoder(model,cancel)

    def __init__(self, config, tensors, *, library, max_bytes, tokenizer=None):
        self.config,_ = configuration(config,None)
        if not isinstance(library,C.CDLL) or type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('borrowed native library and explicit edge memory budget required')
        self.library,self.max_bytes,self.tokenizer = library,max_bytes,tokenizer
        self._lock,self._closed = threading.RLock(),False
        c = self.config
        h,v = c['hidden_size'],c['vocab_size']
        supplied = dict(tensors)
        if c['tie_word_embeddings']:
            if 'lm_head.weight' in supplied and not np.array_equal(supplied['lm_head.weight'],supplied.get('model.embed_tokens.weight')):
                raise ValueError('inconsistent Mixtral tied edge weights')
            supplied['lm_head.weight'] = supplied.get('model.embed_tokens.weight')
        shapes = dict(zip(GLOBAL_NAMES,((v,h),(h,),(v,h))))
        if set(supplied) != set(shapes):
            raise ValueError('exact Mixtral edge inventory required')
        self._weight_bytes = 4*sum(math.prod(shape) for shape in shapes.values())
        if 2*self._weight_bytes > max_bytes:
            raise MemoryError('Mixtral edge weights exceed ownership budget')
        self.weights = {}
        digest = hashlib.sha256(json.dumps(dict(c),sort_keys=True,separators=(',',':')).encode())
        for name,shape in shapes.items():
            value = np.asarray(supplied[name])
            if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError('invalid Mixtral edge tensor: '+name)
            owned = np.frombuffer(value.tobytes(order='C'),dtype=np.float32).reshape(shape)
            self.weights[name] = owned
            digest.update(name.encode())
            digest.update(memoryview(owned).cast('B'))
        self.identity = 'oa-mixtral-edge-v1:'+digest.hexdigest()
        version = library.oa_mixtral_edge_version
        version.argtypes,version.restype = [],C.c_uint64
        if version() != 1:
            raise ValueError('Mixtral edge ABI version 1 required')
        self._embed = library.oa_mixtral_edge_embed
        self._embed.argtypes = [C.c_uint64,C.c_uint64,Tensor,C.POINTER(C.c_int64),C.c_uint64,Tensor]
        self._embed.restype = C.c_int
        self._logits = library.oa_mixtral_edge_logits
        self._logits.argtypes = [C.c_uint64,C.c_uint64,C.c_float,Tensor,Tensor,Tensor,C.c_uint64,Tensor,Tensor]
        self._logits.restype = C.c_int

    def _check(self, scratch):
        if self._closed:
            raise RuntimeError('Mixtral edge is closed')
        if self._weight_bytes+scratch > self.max_bytes:
            raise MemoryError('Mixtral edge operation exceeds admission')

    def embed(self, token_ids):
        with self._lock:
            ids = np.asarray(token_ids)
            if ids.ndim not in (1,2) or ids.dtype.kind not in 'iu' or not 0 < ids.size <= 65536:
                raise ValueError('nonempty integer Mixtral edge tokens required')
            h,v = self.config['hidden_size'],self.config['vocab_size']
            if np.any(ids < 0) or np.any(ids >= v):
                raise ValueError('Mixtral edge token outside vocabulary')
            self._check(ids.size*(h*4+16))
            ids = np.ascontiguousarray(ids,dtype=np.int64)
            output = np.empty((*ids.shape,h),np.float32)
            status = self._embed(h,v,tensor(self.weights['model.embed_tokens.weight']),
                ids.ctypes.data_as(C.POINTER(C.c_int64)),ids.size,tensor(output))
            if status:
                raise RuntimeError('native Mixtral edge embedding failed: '+str(status))
            return output

    def logits(self, hidden):
        with self._lock:
            values = np.asarray(hidden)
            h,v = self.config['hidden_size'],self.config['vocab_size']
            if values.ndim not in (1,2,3) or values.dtype != np.float32 or values.shape[-1] != h or not np.isfinite(values).all():
                raise ValueError('finite FP32 Mixtral edge hidden rows required')
            count = values.size//h
            if not 0 < count <= 65536:
                raise ValueError('Mixtral edge row count exceeds admission')
            self._check(count*(h*12+v*4))
            incoming = np.ascontiguousarray(values)
            normalized = np.empty_like(incoming)
            output = np.empty((*values.shape[:-1],v),np.float32)
            status = self._logits(h,v,self.config['rms_norm_eps'],tensor(self.weights['model.norm.weight']),
                tensor(self.weights['lm_head.weight']),tensor(incoming),count,tensor(normalized),tensor(output))
            if status:
                raise RuntimeError('native Mixtral edge final logits failed: '+str(status))
            return output

    def tokenize(self, text):
        with self._lock:
            self._check(0)
            if self.tokenizer is None or type(text) is not str or len(text.encode('utf-8')) > 1024**2:
                raise ValueError('borrowed tokenizer and bounded text required')
            tokens = self.tokenizer.encode(text,add_special_tokens=False).ids
            if len(tokens) > 65536 or any(type(t) is not int or not 0 <= t < self.config['vocab_size'] for t in tokens):
                raise ValueError('tokenizer produced invalid Mixtral edge tokens')
            return tuple(tokens)

    def detokenize(self, tokens):
        with self._lock:
            self._check(0)
            ids = tuple(tokens)
            if self.tokenizer is None or len(ids) > 65536 or any(type(t) is not int or not 0 <= t < self.config['vocab_size'] for t in ids):
                raise ValueError('borrowed tokenizer and valid bounded tokens required')
            return self.tokenizer.decode(list(ids),skip_special_tokens=True)

    def close(self):
        with self._lock:
            self._closed = True
            self.weights.clear()
            self.tokenizer = None
