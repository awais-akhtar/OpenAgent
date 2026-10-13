"""Publisher source-to-runtime operations with no implicit name fallbacks."""
from dataclasses import dataclass
import re
import math
import numpy as np
from .publisher_tensor_reader import TensorSlice

@dataclass(frozen=True)
class WeightBinding:
    destination: str
    sources: tuple
    shape: tuple
    operation: str='identity'

def qwen4_ple_bindings(snapshot,config):
    result={};parts=config.get('split_ngram_parts',512)
    if type(parts) is not int or parts<=0: raise ValueError('positive publisher PLE shard count required')
    layer_ids=config.get('ple_layer_ids',())
    if (type(layer_ids) not in (tuple,list) or
            any(type(value) is not int or value <= 0 for value in layer_ids) or
            len(set(layer_ids)) != len(layer_ids)):
        raise ValueError('unique positive publisher PLE layer inventory required')
    layer_count = config.get('num_hidden_layers', config.get('num_layers'))
    if layer_count is not None and (type(layer_count) is not int or layer_count <= 0):
        raise ValueError('positive publisher layer count required for PLE admission')
    if layer_count is not None and any(value > layer_count for value in layer_ids):
        raise ValueError('publisher PLE layer inventory exceeds model depth')
    for one_based in layer_ids:
        layer=one_based-1
        candidates=(f'model.layers.{layer}.ple.ple_embedding.ngram_embedding.',
                    f'model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.')
        selected=[]
        for prefix in candidates:
            full=prefix+'weight'
            shards=[(int(match.group(1)),name) for name in snapshot.tensors
                if (match:=re.fullmatch(re.escape(prefix)+r'shard_(\d+)\.weight',name))]
            if full in snapshot.tensors:
                if shards: raise ValueError('ambiguous resident and sharded PLE source')
                selected.append((full,))
            elif shards:
                shards.sort()
                if [index for index,_ in shards]!=list(range(parts)):
                    raise ValueError('publisher PLE logical shard ordinal inventory differs')
                selected.append(tuple(name for _,name in shards))
        if len(selected)!=1: raise ValueError('one exact publisher PLE source namespace required')
        result[layer]=selected[0]
    return result

def expert_bindings(snapshot,*,prefix,expert_count,hidden_size,intermediate_size):
    """Map split gate/up/down publishers to stacked decoder expert tensors."""
    e,h,m=expert_count,hidden_size,intermediate_size
    if any(type(v) is not int or v<=0 for v in (e,h,m)): raise ValueError('positive expert mapping geometry required')
    gate_up=prefix+'gate_up_proj';down=prefix+'down_proj'
    if gate_up in snapshot.tensors or down in snapshot.tensors:
        if not {gate_up,down}<=snapshot.tensors.keys(): raise ValueError('incomplete fused publisher expert pair')
        bindings=(WeightBinding(gate_up,(TensorSlice(gate_up,(e,2*m,h)),),(e,2*m,h)),
                  WeightBinding(down,(TensorSlice(down,(e,h,m)),),(e,h,m)))
    else:
        gates=tuple(TensorSlice(f'{prefix}{i}.gate_proj.weight',(m,h)) for i in range(e))
        ups=tuple(TensorSlice(f'{prefix}{i}.up_proj.weight',(m,h)) for i in range(e))
        downs=tuple(TensorSlice(f'{prefix}{i}.down_proj.weight',(h,m)) for i in range(e))
        bindings=(WeightBinding(gate_up,tuple(value for pair in zip(gates,ups) for value in pair),(e,2*m,h),'gate-up-stack'),
                  WeightBinding(down,downs,(e,h,m),'expert-stack'))
    for binding in bindings:
        for source in binding.sources:
            record=snapshot.tensors.get(source.source)
            if record is None or record.shape!=source.shape: raise ValueError('publisher expert shape or inventory differs: '+source.source)
    return bindings

def materialize(reader,bindings,*,max_bytes,cancelled=None):
    bindings=tuple(bindings)
    reader._check()
    destinations=set()
    for binding in bindings:
        if (type(binding) is not WeightBinding or type(binding.destination) is not str
                or not binding.destination or type(binding.shape) is not tuple
                or any(type(value) is not int or value<=0 for value in binding.shape)
                or type(binding.sources) is not tuple or not binding.sources):
            raise ValueError('explicit complete publisher weight binding required')
        if binding.destination in destinations:
            raise ValueError('duplicate mapped publisher destination')
        destinations.add(binding.destination)
        for source in binding.sources:
            if (type(source) is not TensorSlice or type(source.source) is not str
                    or type(source.shape) is not tuple
                    or any(type(value) is not int or value<=0 for value in source.shape)
                    or type(source.element_offset) is not int or source.element_offset<0):
                raise ValueError('explicit bounded publisher source slice required')
            record=reader.snapshot.tensors.get(source.source)
            if (record is None or record.dtype not in ('F32','F16','BF16','F8_E4M3','F8_E8M0')
                    or source.element_offset+math.prod(source.shape)>math.prod(record.shape)):
                raise ValueError('publisher source slice exceeds admitted inventory: '+source.source)
        shape=binding.shape
        if binding.operation=='identity':
            valid=len(binding.sources)==1 and math.prod(binding.sources[0].shape)==math.prod(shape)
        elif binding.operation=='expert-stack':
            valid=(len(shape)==3 and len(binding.sources)==shape[0]
                and all(source.shape==shape[1:] for source in binding.sources))
        elif binding.operation=='gate-up-stack':
            valid=(len(shape)==3 and shape[1]%2==0 and len(binding.sources)==2*shape[0]
                and all(source.shape==(shape[1]//2,shape[2]) for source in binding.sources))
        elif binding.operation=='concatenate':
            valid=sum(math.prod(source.shape) for source in binding.sources)==math.prod(shape)
        else:
            valid=False
        if not valid:
            raise ValueError('publisher mapping operation or source geometry differs: '+binding.destination)
    reader.admit_fp8_inventory()
    extent=sum(math.prod(value.shape)*4 for value in bindings)
    if type(max_bytes) is not int or 2*extent+reader.max_bytes>max_bytes:
        raise MemoryError('publisher mapped resident inventory exceeds aggregate admission')
    result={}
    def fill(output,source):
        flat=output.reshape(-1);capacity=reader.max_bytes//128
        if capacity<1: raise MemoryError('one publisher conversion element exceeds admission')
        for begin in range(0,flat.size,capacity):
            if cancelled is not None and cancelled(): raise InterruptedError('publisher mapping cancelled')
            count=min(capacity,flat.size-begin)
            flat[begin:begin+count]=reader.read(TensorSlice(source.source,(count,),source.element_offset+begin))
    for binding in bindings:
        if cancelled is not None and cancelled(): raise InterruptedError('publisher mapping cancelled')
        output=np.empty(binding.shape,np.float32)
        if binding.operation=='identity' and len(binding.sources)==1:
            fill(output,binding.sources[0])
        elif binding.operation=='expert-stack':
            if len(binding.sources)!=binding.shape[0]: raise ValueError('expert stack source count differs')
            for index,source in enumerate(binding.sources):
                if cancelled is not None and cancelled(): raise InterruptedError('publisher mapping cancelled')
                fill(output[index],source)
        elif binding.operation=='gate-up-stack':
            if len(binding.sources)!=2*binding.shape[0]: raise ValueError('gate/up expert source count differs')
            half=binding.shape[1]//2
            for index in range(binding.shape[0]):
                if cancelled is not None and cancelled(): raise InterruptedError('publisher mapping cancelled')
                fill(output[index,:half],binding.sources[index*2])
                fill(output[index,half:],binding.sources[index*2+1])
        elif binding.operation=='concatenate':
            flat=output.reshape(-1);offset=0
            for source in binding.sources:
                extent=math.prod(source.shape)
                if offset+extent>flat.size: raise ValueError('concatenated publisher source exceeds destination')
                fill(flat[offset:offset+extent],source);offset+=extent
            if offset!=flat.size: raise ValueError('concatenated publisher sources do not cover destination')
        else: raise ValueError('unsupported publisher weight operation')
        output.flags.writeable=False;result[binding.destination]=output
    return result

__all__=['WeightBinding','qwen4_ple_bindings','expert_bindings','materialize']
