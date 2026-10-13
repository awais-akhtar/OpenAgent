"""Bounded original publisher tensor conversion and streamed matrix rows."""
from dataclasses import dataclass
from types import MappingProxyType
import math
import numpy as np
from .publisher_checkpoint.storage import Snapshot

@dataclass(frozen=True)
class TensorSlice:
    source: str
    shape: tuple
    element_offset: int=0

class PublisherTensorReader:
    def __init__(self,snapshot,*,max_work_bytes,block_shape=None,scale_names=None):
        if type(snapshot) is not Snapshot or type(max_work_bytes) is not int or max_work_bytes<=0:
            raise ValueError('immutable checkpoint and positive conversion budget required')
        if block_shape is not None and (type(block_shape) not in (list,tuple) or len(block_shape)!=2 or any(type(v) is not int or v<=0 for v in block_shape)):
            raise ValueError('explicit positive FP8 weight block shape required')
        self.snapshot=snapshot;self.max_bytes=max_work_bytes
        self.block_shape=None if block_shape is None else tuple(block_shape)
        if scale_names is None:scale_names={}
        if type(scale_names) is not dict or any(type(source) is not str or type(scale) is not str
                or source not in snapshot.tensors or scale not in snapshot.tensors
                for source,scale in scale_names.items()):
            raise ValueError('explicit existing source-to-scale tensor mapping required')
        self.scale_names=MappingProxyType(dict(scale_names))
        self._closed=False

    def _check(self):
        if self._closed: raise RuntimeError('publisher tensor reader is closed')

    def admit_fp8_inventory(self):
        """Admit all original FP8 scale headers before any selected payload read."""
        self._check()
        for name,record in self.snapshot.tensors.items():
            if record.dtype!='F8_E4M3':continue
            if self.block_shape is None or len(record.shape) not in (2,3):
                raise ValueError('complete FP8 publisher inventory requires explicit matrix block geometry')
            rows,columns=record.shape[-2:];br,bc=self.block_shape
            scale_name=self.scale_names.get(name,
                name[:-7]+'.weight_scale_inv' if name.endswith('.weight') else name+'_scale_inv')
            scale=self.snapshot.tensors.get(scale_name)
            shape=record.shape[:-2]+((rows+br-1)//br,(columns+bc-1)//bc)
            if scale is None or scale.shape!=shape or scale.dtype not in ('F32','F16','BF16','F8_E8M0'):
                raise ValueError('complete FP8 publisher scale inventory differs: '+name)
        self.snapshot.assert_unchanged()

    @staticmethod
    def _decode(raw,dtype):
        if dtype=='BF16': return (np.frombuffer(raw,'<u2').astype(np.uint32)<<np.uint32(16)).view('<f4')
        if dtype in ('F32','F16'): return np.frombuffer(raw,'<f4' if dtype=='F32' else '<f2').astype(np.float32)
        if dtype=='F8_E8M0':
            bits=np.frombuffer(raw,np.uint8).astype(np.uint32)
            if np.any(bits==255): raise ValueError('nonfinite E8M0 scale')
            return np.where(bits==0,np.float32(2**-127),(bits<<np.uint32(23)).view(np.float32)).astype(np.float32)
        if dtype=='F8_E4M3':
            bits=np.frombuffer(raw,np.uint8);sign=np.where(bits&128,-1.0,1.0)
            exponent=((bits>>3)&15).astype(np.int32);fraction=(bits&7).astype(np.float32)
            if np.any((exponent==15)&(fraction==7)): raise ValueError('nonfinite E4M3 checkpoint value')
            magnitude=np.where(exponent==0,np.ldexp(fraction,-9),np.ldexp(1+fraction/8,exponent-7))
            return (sign*magnitude).astype(np.float32)
        raise ValueError('unadmitted publisher tensor encoding: '+dtype)

    def read(self,binding):
        self._check()
        if (type(binding) is not TensorSlice or type(binding.source) is not str or
                binding.source not in self.snapshot.tensors or type(binding.shape) is not tuple or
                any(type(v) is not int or v<=0 for v in binding.shape) or
                type(binding.element_offset) is not int or binding.element_offset<0):
            raise ValueError('positive explicit contiguous tensor slice required')
        record=self.snapshot.tensors[binding.source];count=math.prod(binding.shape)
        if binding.element_offset+count>math.prod(record.shape): raise ValueError('slice exceeds original tensor')
        if record.dtype=='F8_E8M0' and binding.source not in self.scale_names.values():
            raise ValueError('F8_E8M0 is admitted only for an explicit FP8 scale tensor')
        width={'F32':4,'F16':2,'BF16':2,'F8_E4M3':1,'F8_E8M0':1}.get(record.dtype)
        if width is None: raise ValueError('unadmitted publisher source encoding')
        if count*128>self.max_bytes: raise MemoryError('publisher tensor conversion exceeds work admission')
        raw=self.snapshot.read_range(binding.source,binding.element_offset*width,count*width)
        value=self._decode(raw,record.dtype)
        if record.dtype=='F8_E4M3':
            if self.block_shape is None or len(record.shape) not in (2,3):
                raise ValueError('FP8 matrix requires explicit publisher block geometry')
            columns=record.shape[-1];rows=record.shape[-2];br,bc=self.block_shape
            scale_name=binding.source[:-7]+'.weight_scale_inv' if binding.source.endswith('.weight') else binding.source+'_scale_inv'
            scale_name=self.scale_names.get(binding.source,scale_name)
            scale=self.snapshot.tensors.get(scale_name)
            scale_shape=record.shape[:-2]+((rows+br-1)//br,(columns+bc-1)//bc)
            if scale is None or scale.shape!=scale_shape or scale.dtype not in ('F32','F16','BF16','F8_E8M0'):
                raise ValueError('FP8 publisher scale inventory differs')
            # Read only the scales touched by this contiguous selection.
            flat=np.arange(binding.element_offset,binding.element_offset+count,dtype=np.int64)
            matrix=flat//(rows*columns);local=flat%(rows*columns)
            indices=matrix*math.prod(scale_shape[-2:])+(local//columns//br)*scale_shape[-1]+local%columns//bc
            scale_width={'F32':4,'F16':2,'BF16':2,'F8_E8M0':1}[scale.dtype]
            for index in np.unique(indices):
                raw_scale=self.snapshot.read_range(scale_name,int(index)*scale_width,scale_width)
                factor=self._decode(raw_scale,scale.dtype)[0]
                if not np.isfinite(factor) or factor<=0: raise ValueError('invalid publisher FP8 block scale')
                value[indices==index]*=factor
        if not np.isfinite(value).all(): raise ValueError('nonfinite publisher tensor conversion')
        return np.frombuffer(value.tobytes(),np.float32).reshape(binding.shape)

    def _admit_matrix_slice(self, binding):
        """Keep streamed projection slices aligned to the source matrix.

        A flat byte range can be numerically readable while still changing
        row geometry at a packed-format boundary.  Projection consumers must
        therefore bind the declared columns and start on a complete source
        row; scale lookup and packing order then remain unambiguous.
        """
        if type(binding) is not TensorSlice or len(binding.shape) != 2:
            raise ValueError('two-dimensional projection slice required')
        record = self.snapshot.tensors.get(binding.source)
        if record is None or len(record.shape) != 2:
            raise ValueError('projection source must be a two-dimensional tensor')
        rows, columns = record.shape
        if binding.shape[1] != columns or binding.shape[0] <= 0:
            raise ValueError('projection slice columns differ from source geometry')
        if binding.element_offset % columns or binding.element_offset + math.prod(binding.shape) > rows * columns:
            raise ValueError('projection slice must cover complete aligned source rows')
        return rows, columns

    def linear(self,ops,binding,inputs,*,bias=None,row_block=128,cancelled=None):
        """Native products over decoded row blocks; no resident matrix copy."""
        self._check()
        self._admit_matrix_slice(binding)
        if type(row_block) is not int or row_block<=0:
            raise ValueError('positive row block required')
        x=np.asarray(inputs)
        rows,columns=binding.shape
        if x.dtype!=np.float32 or x.ndim<2 or x.shape[-1]!=columns:
            raise ValueError('FP32 inputs must match streamed matrix width')
        if (x.size+math.prod(x.shape[:-1])*rows)*8+min(row_block,rows)*columns*128>self.max_bytes:
            raise MemoryError('streamed native projection exceeds admission')
        if bias is not None and (bias.shape!=(rows,) or bias.dtype!=np.float32):
            raise ValueError('projection bias differs from output rows')
        output=np.empty(x.shape[:-1]+(rows,),np.float32)
        for begin in range(0,rows,row_block):
            if cancelled is not None and cancelled(): raise InterruptedError('streamed publisher projection cancelled')
            end=min(rows,begin+row_block)
            weights=self.read(TensorSlice(binding.source,(end-begin,columns),binding.element_offset+begin*columns))
            output[...,begin:end]=ops.linear(x,weights,None if bias is None else bias[begin:end])
        return output

    def close(self): self._closed=True

__all__=['TensorSlice','PublisherTensorReader']
