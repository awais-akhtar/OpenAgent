"""Bounded exports of executed K3 routes and arithmetic boundaries.

This recorder observes operands/results already used by an admitted owner.
It never recomputes a route, quantizes a fixture or grants exact admission.
"""
from types import MappingProxyType
import numpy as np


def observation_schema(config,*,routing,projection_names):
    result={}
    if routing:
        for layer in range(config.dense_prefix,len(config.layer_types)):
            result.update((f'routing.layer.{layer}.{name}',layer) for name in ('logits','ids','weights'))
    for name in projection_names:
        fields=name.split('.')
        layer=int(fields[3]) if len(fields)>4 and fields[:3]==['language_model','model','layers'] else 'global'
        if layer!='global' and not 0<=layer<len(config.layer_types):raise ValueError('observed projection layer outside schedule')
        result.update(('quantization.'+name+'.'+stage,layer) for stage in ('input','output'))
    return result


class K3ObservationRecorder:
    def __init__(self,config,*,max_bytes,routing,projection_names):
        if type(max_bytes) is not int or max_bytes<=0 or type(routing) is not bool:
            raise ValueError('positive observation budget and explicit routing policy required')
        if (type(projection_names) is not tuple or len(projection_names)>65536 or
                any(type(name) is not str or not 0<len(name)<=1024 for name in projection_names) or
                len(set(projection_names))!=len(projection_names)):
            raise ValueError('bounded unique explicitly selected projection boundary names required')
        baseline=1024+sum(256+len(name)*4 for name in projection_names)+256*len(config.layer_types)
        if baseline>max_bytes:raise MemoryError('K3 observation names and layer inventory exceed admission')
        self.config=config;self.maximum=max_bytes;self.routing=routing
        self.projection_names=frozenset(projection_names)
        self._values={};self._counts={};self._bytes=baseline;self._failed=False

    def _record(self,name,value):
        if self._failed:raise RuntimeError('failed K3 observation recorder')
        if name in self._values:raise ValueError('duplicate executed observation name')
        shape=tuple(value.shape);dtype=str(value.dtype)
        if len(shape)>8 or any(type(size) is not int or size<0 for size in shape):
            raise ValueError('bounded executed observation geometry required')
        elements=int(np.prod(shape,dtype=object))
        charge=elements*32+len(name)*4+256
        if charge>self.maximum-self._bytes:
            self._failed=True;raise MemoryError('complete executed K3 observations exceed admission')
        if hasattr(value,'detach'):
            value=value.detach().cpu()
            if dtype=='torch.bfloat16':value=value.float()
            value=value.numpy()
        array=np.asarray(value)
        if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
            self._failed=True;raise ValueError('finite actual numeric observation required')
        detached=np.frombuffer(array.tobytes(order='C'),dtype=array.dtype).reshape(shape)
        self._values[name]=detached;self._bytes+=charge

    def route(self,prefix,logits,indices,weights):
        if not self.routing:return
        layer=int(prefix.split('.')[3]);c=self.config
        if not c.dense_prefix<=layer<len(c.layer_types):raise ValueError('executed routed layer required')
        logits,indices,weights=map(np.asarray,(logits,indices,weights))
        if logits.ndim==1:logits,indices,weights=logits[None],indices[None],weights[None]
        if (logits.ndim!=2 or logits.shape[1]!=c.experts or indices.shape!=(len(logits),c.selected) or
                weights.shape!=indices.shape or indices.dtype.kind not in 'iu' or np.any(indices<0) or
                np.any(indices>=c.experts) or any(np.unique(row).size!=c.selected for row in indices)):
            raise ValueError('complete actual router logits and selected route inventory required')
        indices=indices.astype(np.int64,copy=False)
        key=f'routing.layer.{layer}';call=self._counts.get(key,0);self._counts[key]=call+1
        for suffix,value in (('logits',logits),('ids',indices),('weights',weights)):
            self._record(f'{key}.call.{call}.{suffix}',value)

    def projection(self,name,stage,value):
        if name not in self.projection_names:return
        if stage not in ('input','output'):raise ValueError('actual arithmetic boundary stage required')
        if value.ndim==1:value=value[None]
        key='quantization.'+name
        if stage=='input':
            call=self._counts.get(key,0);self._counts[key]=call+1
        else:
            call=self._counts.get(key,0)-1
            if call<0:raise ValueError('projection result has no observed executed input')
        self._record(f'{key}.call.{call}.{stage}',value)

    def export(self,*,rows=None):
        if self._failed:raise RuntimeError('failed observation recorder cannot publish evidence')
        if self.routing:
            expected={f'routing.layer.{layer}' for layer in range(self.config.dense_prefix,len(self.config.layer_types))}
            if not expected<=set(self._counts):raise ValueError('executed route observations omit required layers')
        for name in self.projection_names:
            key='quantization.'+name;count=self._counts.get(key,0)
            if (not count or
                    any(f'{key}.call.{call}.input' not in self._values or
                        f'{key}.call.{call}.output' not in self._values
                        for call in range(count)) or
                    any(record.startswith(key+'.call.') and
                        record.endswith('.output') and
                        record.rsplit('.call.',1)[1].split('.',1)[0].isdigit() and
                        int(record.rsplit('.call.',1)[1].split('.',1)[0]) >= count
                        for record in self._values)):
                raise ValueError('selected arithmetic boundary was not completely observed')
        result={}
        for name in sorted(self.projection_names):
            key='quantization.'+name;count=self._counts[key]
            for stage in ('input','output'):
                values=[self._values[f'{key}.call.{call}.{stage}'] for call in range(count)]
                joined=np.concatenate(values,axis=0)
                result[key+'.'+stage]=np.frombuffer(joined.tobytes(),dtype=joined.dtype).reshape(joined.shape)
        if self.routing:
            for key in sorted(expected):
                count=self._counts[key]
                for suffix in ('logits','ids','weights'):
                    values=[self._values[f'{key}.call.{call}.{suffix}'] for call in range(count)]
                    value=np.concatenate(values,axis=0)
                    if rows is not None and (type(rows) is not int or rows<=0 or len(value)!=rows):
                        raise ValueError('executed routing rows differ from the occupied call extent')
                    result[key+'.'+suffix]=np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)
        return MappingProxyType(result)


__all__=['K3ObservationRecorder','observation_schema']
