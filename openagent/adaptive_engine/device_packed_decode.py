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
        if dtype=='int8-row':
            # A finite positive row scale can still make an admitted signed
            # INT8 code non-representable in the FP32 weight domain. Check
            # each row's actual maximum code before staging any device data;
            # using the global 128 bound would reject valid rows needlessly.
            width = hidden*intermediate
            codes = np.frombuffer(payload,dtype=np.int8,count=3*width)
            scales = np.frombuffer(payload,dtype='<f4',count=2*intermediate+hidden,
                                   offset=3*width)
            cursor = 0
            scale_cursor = 0
            fp32_max = float(np.finfo(np.float32).max)
            for rows,cols in shapes:
                matrix = codes[cursor:cursor+rows*cols].reshape(rows,cols)
                row_scales = scales[scale_cursor:scale_cursor+rows]
                for row,scale in zip(matrix,row_scales):
                    code_max = max(abs(int(row.min())),abs(int(row.max())))
                    if code_max*float(scale)>fp32_max:
                        raise ValueError('INT8 selected expert scale overflows decoded FP32 weights')
                cursor += rows*cols
                scale_cursor += rows
        return _decode_payload(payload,hidden,intermediate,dtype,expected)
    if dtype in _MODULES:
        # The Q4 decoders allocate a complete projection before walking its
        # blocks.  Validate all persisted metadata first so a malformed late
        # block cannot leave earlier decoded projections resident while the
        # selected-device admission fails.
        if dtype in ('q4_0','q4_1'):
            block_bytes = 18 if dtype=='q4_0' else 20
            expected = sum(rows*cols//32*block_bytes for rows,cols in shapes)
            if len(payload)!=expected:
                raise ValueError('Q4 selected expert extent mismatch')
            cursor = 0
            for rows,cols in shapes:
                for _ in range(rows*cols//32):
                    first = struct.unpack_from('<e',payload,cursor)[0]
                    second = (struct.unpack_from('<e',payload,cursor+2)[0]
                              if dtype=='q4_1' else 0.0)
                    if not np.isfinite(first) or not np.isfinite(second):
                        # Q4_0's format permits signed finite deltas and Q4_1
                        # permits signed finite delta/minimum metadata. Reject
                        # only non-finite records here, before the decoder
                        # allocates any projection matrices.
                        raise ValueError('invalid Q4 selected expert metadata')
                    cursor += block_bytes
        elif dtype == 'q8_1':
            # Q8_1's general full decoder validates block metadata while
            # filling each newly allocated projection. Preflight the complete
            # persisted payload here so a corrupt late block cannot cause
            # partial device-admission decode allocations.
            from .q8_1_expert import BLOCK_BYTES, payload_size
            if len(payload) != payload_size(hidden, intermediate):
                raise ValueError('Q8_1 selected expert extent mismatch')
            cursor = 0
            for rows, cols in shapes:
                for _ in range(rows * cols // 32):
                    delta, total = struct.unpack_from('<ee', payload, cursor)
                    if not np.isfinite(delta) or not np.isfinite(total):
                        raise ValueError('nonfinite Q8_1 selected expert metadata')
                    cursor += BLOCK_BYTES
        elif dtype == 'int4-grouped-g64':
            # The general grouped decoder detects FP32 reconstruction overflow
            # after it has allocated and filled a complete projection.  Scan
            # this selected-device representation first so corrupt scales fail
            # before any decoded matrix is resident.  The largest absolute
            # two's-complement nibble is 8, so this bound guarantees every
            # possible code*scale product remains finite in FP32.
            from .grouped_int4_expert import _layout
            layout, _, expected = _layout(hidden, intermediate, 64)
            if len(payload) != expected:
                raise ValueError('grouped INT4 selected expert extent mismatch')
            max_scale = np.float32(np.finfo(np.float32).max / 8.0)
            for rows, cols, _, scale_offset in layout:
                groups = (cols + 63) // 64
                count = rows * groups
                scales = np.frombuffer(payload, dtype='<f4', count=count,
                                       offset=scale_offset)
                if (not np.isfinite(scales).all() or
                        np.any(np.abs(scales) > max_scale)):
                    raise ValueError('invalid grouped INT4 selected expert scale')
        module = importlib.import_module('.'+_MODULES[dtype],__package__)
        matrices = module.decode_expert(payload,hidden,intermediate)
    elif dtype in ('int2-row','int3-grouped-g64'):
        from .lowbit_expert import parameters,payload_size
        bits,group = parameters(dtype)
        if len(payload)!=payload_size(hidden,intermediate,dtype=dtype):
            raise ValueError('low-bit selected expert extent mismatch')
        strides = [(cols*bits+7)//8 for _,cols in shapes]
        code_sizes = [rows*stride for (rows,_),stride in zip(shapes,strides)]
        code_cursor = 0
        scale_cursor = sum(code_sizes)
        # Validate scales and row padding for every projection before creating
        # any decoded matrix. A late malformed down-projection must not leave
        # the already-decoded gate/up matrices resident during device admission.
        max_code = 1 << (bits - 1)
        for (rows,cols),stride,code_size in zip(shapes,strides,code_sizes):
            group_width = group or cols
            groups = (cols+group_width-1)//group_width
            scale_count = rows*groups
            scales = np.frombuffer(payload,dtype='<f4',count=scale_count,offset=scale_cursor)
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError('invalid selected low-bit row/group scales')
            # The largest signed code magnitude is 2**(bits-1). Check the
            # actual rounded FP32 product before allocating decoded matrices.
            with np.errstate(over='ignore',invalid='ignore'):
                worst_case = (scales.astype(np.float64)*max_code).astype(np.float32)
            if not np.isfinite(worst_case).all():
                raise ValueError('selected low-bit scale overflows decoded FP32 weights')
            unused = (cols*bits)%8
            if unused:
                for row in range(rows):
                    offset = code_cursor+row*stride
                    if int(payload[offset+stride-1])>>unused:
                        raise ValueError('nonzero selected low-bit row padding')
            code_cursor += code_size
            scale_cursor += scale_count*4
        code_cursor = 0
        scale_cursor = sum(code_sizes)
        matrices = []
        for (rows,cols),stride,code_size in zip(shapes,strides,code_sizes):
            group_width = group or cols
            groups = (cols+group_width-1)//group_width
            scales = np.frombuffer(payload,dtype='<f4',count=rows*groups,offset=scale_cursor).reshape(rows,groups)
            value = np.empty((rows,cols),np.float32)
            for row in range(rows):
                offset = code_cursor+row*stride
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
        # Validate every record before allocating decoded projections. A bad
        # late codebook index/scale must not leave earlier matrices resident;
        # finite operands can also overflow when multiplied in FP32.
        cursor = 0
        codebook = lattice_contract.codebook
        for rows,cols in shapes:
            for _ in range(rows*(cols//8)):
                index,scale = struct.unpack_from('<Hf',payload,cursor);cursor += 6
                if index>=len(codebook) or not np.isfinite(scale):
                    raise ValueError('invalid E8 selected expert record')
                with np.errstate(over='ignore',invalid='ignore'):
                    decoded = np.asarray(codebook[index],dtype=np.float32)*np.float32(scale)
                if not np.isfinite(decoded).all():
                    raise ValueError('nonfinite E8 selected expert value')
        matrices,cursor = [],0
        for rows,cols in shapes:
            value = np.empty((rows,cols),np.float32)
            for row in range(rows):
                for start in range(0,cols,8):
                    index,scale = struct.unpack_from('<Hf',payload,cursor);cursor += 6
                    # Keep the persisted codebook-basis vector here. The
                    # selected-device consumer applies R to the activation
                    # before its matmul; rotating both sides would apply the
                    # orthogonal transform twice and change the result.
                    value[row,start:start+8] = lattice_contract.codebook[index]*scale
            matrices.append(value)
    elif dtype=='mxfp4-e2m1-g32-ue8m0':
        from .mxfp4_expert import payload_size,decode_code,decode_scale
        if len(payload)!=payload_size(hidden,intermediate):
            raise ValueError('MXFP4 selected expert extent mismatch')
        # Validate the complete encoded representation before allocating any
        # decoded projection.  The persisted scale byte 255 denotes NaN, and
        # a finite UE8M0 scale can still overflow FP32 after multiplication by
        # an E2M1 code.  Also require zero codes in the final partial 32-value
        # block, matching the packer's canonical padding.
        cursor = 0
        for rows,cols in shapes:
            groups = (cols+31)//32
            for row in range(rows):
                for group in range(groups):
                    base = cursor+(row*groups+group)*17
                    scale_code = int(payload[base])
                    scale = decode_scale(scale_code)
                    if not np.isfinite(scale):
                        raise ValueError('invalid MXFP4 selected expert scale')
                    valid = min(32,cols-group*32)
                    for lane in range(32):
                        packed = int(payload[base+1+lane//2])
                        code = (packed>>(4*(lane%2)))&15
                        if lane >= valid:
                            if code:
                                raise ValueError('nonzero MXFP4 selected expert padding')
                        else:
                            with np.errstate(over='ignore',invalid='ignore'):
                                decoded = np.float32(decode_code(code)*scale)
                            if not np.isfinite(decoded):
                                raise ValueError('nonfinite MXFP4 selected expert value')
            cursor += rows*groups*17
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
                    col_start = block_col*128
                    col_end = min(cols,col_start+128)
                    max_code_value = 0.0
                    if scale == 0:
                        for row in range(block_row*128,row_end):
                            start = cursor+row*cols+col_start
                            if any(int(code)&127 for code in payload[start:start+col_end-col_start]):
                                raise ValueError('nonzero FP8 code with zero selected expert scale')
                    else:
                        # Finite operands can still overflow while the decoder
                        # materializes a projection. Scan this exact tile's
                        # finite codes against its own scale before allocating
                        # any selected-device matrix; a whole-format max-scale
                        # bound would reject valid tiles unnecessarily.
                        for row in range(block_row*128,row_end):
                            start = cursor+row*cols+col_start
                            for code in payload[start:start+col_end-col_start]:
                                value = abs(decode_code(int(code)))
                                if not np.isfinite(value):
                                    raise ValueError('nonfinite FP8 selected expert code')
                                max_code_value = max(max_code_value,value)
                        if max_code_value*scale > float(np.finfo(np.float32).max):
                            raise ValueError('FP8 selected expert scale overflows decoded FP32 weights')
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
