"""Explicit bounded host weight decoding for selected device expert staging."""
import importlib
import struct
import numpy as np

_MODULES = {'int4-row':'packed_expert','int4-grouped-g64':'grouped_int4_expert',
    'q4_0':'q4_0_expert','q4_1':'q4_1_expert','q5_0':'q5_0_expert','q5_1':'q5_1_expert',
    'q8_0':'q8_0_expert','q8_1':'q8_1_expert'}
FORMATS = frozenset(_MODULES)|{'float32','float16','int8-row','mxfp4-e2m1-g32-ue8m0',
    'fp8-e4m3-b128-f32','e8-lattice-caller-v1','iq3-format6-98-v1','int2-row','int3-grouped-g64'}


def decode_device_expert(payload,*,hidden,intermediate,dtype,max_bytes,lattice_contract=None,iq3_grid=None):
    if (not isinstance(payload, (bytes, bytearray, memoryview)) or
            dtype not in FORMATS or
            any(type(value) is not int or value<=0 for value in (hidden,intermediate,max_bytes))):
        raise ValueError('explicit supported device expert format/geometry/budget required')
    payload = memoryview(payload).cast('B')
    shapes = ((intermediate,hidden),(intermediate,hidden),(hidden,intermediate))
    if 24*hidden*intermediate+len(payload)>max_bytes:
        raise MemoryError('device expert host decode exceeds admission')
    if dtype in ('float32','float16','int8-row'):
        from .expert_kernels import _decode_payload
        expected = 3*hidden*intermediate*(4 if dtype=='float32' else 2)
        if dtype=='int8-row':
            expected = 3*hidden*intermediate+4*(2*intermediate+hidden)
        if len(payload)!=expected:
            raise ValueError('device expert payload extent mismatch')
        return _decode_payload(payload,hidden,intermediate,dtype,expected)
    if dtype in _MODULES:
        module = importlib.import_module('.'+_MODULES[dtype],__package__)
        matrices = module.decode_expert(payload,hidden,intermediate)
    elif dtype in ('int2-row','int3-grouped-g64'):
        from .lowbit_expert import parameters,payload_size
        bits,group = parameters(dtype)
        if len(payload)!=payload_size(hidden,intermediate,dtype=dtype):
            raise ValueError('low-bit selected expert extent mismatch')
        strides = [(cols*bits+7)//8 for _,cols in shapes]
        code_sizes = [rows*stride for (rows,_),stride in zip(shapes,strides)]
        scale_cursor = sum(code_sizes)
        code_cursor = 0
        matrices = []
        for (rows,cols),stride,code_size in zip(shapes,strides,code_sizes):
            group_width = group or cols
            groups = (cols+group_width-1)//group_width
            scales = np.frombuffer(payload,dtype='<f4',count=rows*groups,offset=scale_cursor).reshape(rows,groups)
            if not np.isfinite(scales).all():
                raise ValueError('nonfinite selected low-bit row/group scales')
            value = np.empty((rows,cols),np.float32)
            unused = (cols*bits)%8
            for row in range(rows):
                offset = code_cursor+row*stride
                if unused and int(payload[offset+stride-1])>>unused:
                    raise ValueError('nonzero selected low-bit row padding')
                for column in range(cols):
                    address,shift = divmod(column*bits,8)
                    code = int(payload[offset+address])>>shift
                    if shift+bits>8:
                        code |= int(payload[offset+address+1])<<(8-shift)
                    code &= (1<<bits)-1
                    signed = code if code<(1<<(bits-1)) else code-(1<<bits)
                    value[row,column] = signed*scales[row,column//group_width]
            matrices.append(value)
            code_cursor += code_size
            scale_cursor += rows*groups*4
    elif dtype=='iq3-format6-98-v1':
        from .iq3_lattice import IQ3Grid,payload_size,decode_iq3
        if type(iq3_grid) is not IQ3Grid:
            raise ValueError('exact persisted IQ3 grid required for device decoding')
        sizes = [payload_size(*shape) for shape in shapes]
        if sum(sizes)!=len(payload):
            raise ValueError('IQ3 selected expert extent mismatch')
        matrices,cursor = [],0
        for (rows,cols),size in zip(shapes,sizes):
            matrices.append(decode_iq3(payload[cursor:cursor+size],rows=rows,cols=cols,
                                      grid=iq3_grid,max_bytes=max_bytes))
            cursor += size
    elif dtype=='e8-lattice-caller-v1':
        from .e8_lattice_expert import LatticeContract,payload_size
        if type(lattice_contract) is not LatticeContract or len(payload)!=payload_size(hidden,intermediate):
            raise ValueError('exact E8 contract/payload required for device decoding')
        matrices,cursor = [],0
        for rows,cols in shapes:
            value = np.empty((rows,cols),np.float32)
            for row in range(rows):
                for start in range(0,cols,8):
                    index,scale = struct.unpack_from('<Hf',payload,cursor);cursor += 6
                    if index>=len(lattice_contract.codebook) or not np.isfinite(scale) or scale<0:
                        raise ValueError('invalid E8 selected expert record')
                    value[row,start:start+8] = (lattice_contract.codebook[index]*scale)@lattice_contract.transform
            matrices.append(value)
    elif dtype=='mxfp4-e2m1-g32-ue8m0':
        from .mxfp4_expert import payload_size,decode_code,decode_scale
        if len(payload)!=payload_size(hidden,intermediate):
            raise ValueError('MXFP4 selected expert extent mismatch')
        matrices,cursor = [],0
        for rows,cols in shapes:
            value = np.empty((rows,cols),np.float32)
            for row in range(rows):
                for start in range(0,cols,32):
                    scale = decode_scale(int(payload[cursor]))
                    for lane in range(32):
                        code = (int(payload[cursor+1+lane//2])>>(4*(lane%2)))&15
                        if start+lane<cols:
                            value[row,start+lane] = decode_code(code)*scale
                        elif code:
                            raise ValueError('nonzero MXFP4 selected expert padding')
                    cursor += 17
            matrices.append(value)
    else:
        from .fp8_block_expert import payload_size,matrix_payload_size,decode_code
        if len(payload)!=payload_size(hidden,intermediate):
            raise ValueError('FP8 selected expert extent mismatch')
        # Validate the complete persisted representation before allocating any
        # decoded matrix.  Besides making admission atomic with respect to bad
        # late matrices, this checks each block scale once (rather than once
        # per weight) and enforces the canonical zero-scale/all-zero-tile rule.
        cursor = 0
        for rows,cols in shapes:
            groups = (cols+127)//128
            scale_rows = (rows+127)//128
            scale_base = cursor+rows*cols
            for block_row in range(scale_rows):
                row_end = min(rows,(block_row+1)*128)
                for block_col in range(groups):
                    scale = struct.unpack_from('<f',payload,
                        scale_base+4*(block_row*groups+block_col))[0]
                    if not np.isfinite(scale) or scale<0:
                        raise ValueError('invalid FP8 selected expert scale')
                    if scale == 0:
                        col_start = block_col*128
                        col_end = min(cols,col_start+128)
                        for row in range(block_row*128,row_end):
                            start = cursor+row*cols+col_start
                            if any(int(code)&127 for code in payload[start:start+col_end-col_start]):
                                raise ValueError('nonzero FP8 code with zero selected expert scale')
            for index in range(rows*cols):
                if not np.isfinite(decode_code(int(payload[cursor+index]))):
                    raise ValueError('nonfinite FP8 selected expert code')
            cursor += matrix_payload_size(rows,cols)
        matrices,cursor = [],0
        for rows,cols in shapes:
            value = np.empty((rows,cols),np.float32)
            groups = (cols+127)//128
            for row in range(rows):
                for column in range(cols):
                    offset = cursor+rows*cols+4*((row//128)*groups+column//128)
                    scale = struct.unpack_from('<f',payload,offset)[0]
                    code = int(payload[cursor+row*cols+column])
                    value[row,column] = decode_code(code)*scale
            cursor += matrix_payload_size(rows,cols)
            matrices.append(value)
    if len(matrices)!=3 or any(value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all()
                              for value,shape in zip(matrices,shapes)):
        raise ValueError('invalid device expert decoded matrices')
    return (*matrices,np.float32(1),np.float32(1),np.float32(1))


__all__ = ['FORMATS','decode_device_expert']
