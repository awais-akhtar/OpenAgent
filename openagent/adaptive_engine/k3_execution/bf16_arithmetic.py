"""Explicit PyTorch BF16/FP32 boundaries and bounded K3 projections."""
import hashlib
import json
import math
import threading
from contextlib import contextmanager
import numpy as np
from .precision import check_precision


class K3BF16Arithmetic:
    """Bound explicit tensor buffers; do not cache an unbounded weight copy.

    Operations return FP32 host arrays containing BF16-rounded values where
    the model boundary is BF16. Backend matmul and softmax kernels still need
    numerical comparison on the chosen device and installed Torch version.
    """

    def __init__(self, *, device, torch_version, max_bytes, packed_scale_policy='bf16-before-multiply',
                 packed_decoder=None, attention_backend=None, projection_backend=None,
                 attention_mode='eager'):
        import torch
        if str(torch.__version__) != torch_version:
            raise ValueError('installed Torch differs from explicit arithmetic pin')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('positive arithmetic buffer budget required')
        if packed_scale_policy not in ('bf16-before-multiply', 'fp32-before-cast'):
            raise ValueError('explicit supported packed scale policy required')
        self.packed_scale_policy = packed_scale_policy
        if attention_mode not in ('eager','paged-online-fp32'):
            raise ValueError('explicit K3 attention reduction policy required')
        if attention_mode!='eager' and attention_backend is not None:
            raise ValueError('paged attention owns its reduction; a fused attention backend cannot be combined')
        self.attention_mode=attention_mode
        if packed_decoder is not None:
            from .compressed_tensors import K3CompressedTensorsDecoder
            if type(packed_decoder) is not K3CompressedTensorsDecoder:
                raise ValueError('explicit pinned compressed-tensors decoder required')
        self._packed_decoder = packed_decoder
        if attention_backend is not None:
            from .flash_attention import K3FlashAttention
            if type(attention_backend) is not K3FlashAttention:
                raise ValueError('explicit pinned K3 FlashAttention backend required')
        self._attention_backend = attention_backend
        from .device_policy import admit_device
        target = admit_device(torch, device, allow_cpu=True)
        if target.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA arithmetic requested but unavailable')
            target = torch.device('cuda', torch.cuda.current_device() if target.index is None else target.index)
            if target.index >= torch.cuda.device_count():
                raise ValueError('selected CUDA backend does not admit BF16 arithmetic')
            with torch.cuda.device(target):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError('selected CUDA backend does not admit BF16 arithmetic')
        self.torch, self.device, self.max_bytes = torch, target, max_bytes
        self._observations=threading.local()
        self._projection_backend = projection_backend
        self._projection_identity = None
        if projection_backend is not None:
            from ..tensor_parallel import TensorParallelProjections
            from .publisher_projections import K3PublisherProjections
            if type(projection_backend) not in (TensorParallelProjections,K3PublisherProjections):
                raise ValueError('explicit distributed or original K3 projection owner required')
            if projection_backend.device != target:
                raise ValueError('distributed and K3 arithmetic devices differ')
            self._projection_identity = projection_backend.identity
        if attention_backend is not None and target.type != 'cuda':
            raise ValueError('FlashAttention policy requires CUDA arithmetic')
        check_precision(torch, target)
        self.identity = hashlib.sha256(json.dumps([torch_version, str(target),
            'k3-bf16-fp32-boundaries-v3-moonvit', packed_scale_policy,
            None if packed_decoder is None else packed_decoder.identity,
            None if attention_backend is None else attention_backend.identity,
            self._projection_identity,attention_mode]).encode()).hexdigest()

    def named_matrix(self, name, weights, values, *, router=False, check_cancelled=None):
        """Dispatch admitted projection names through an optional borrowed bank.

        Nonselected weights retain the local arithmetic path. Router banks must
        use FP32; other banks use BF16 inputs/weights and model-boundary rounding.
        The executor still owns its ordinary inventory for attention/state math.
        """
        previous=getattr(self._observations,'name',None)
        prior_input=getattr(self._observations,'input_recorded',False)
        self._observations.name=name
        self._observations.input_recorded=False
        try:
            observer=getattr(self._observations,'observer',None)
            if (observer is not None and name in observer.projection_names and
                    self._projection_backend is not None and self._projection_backend.handles(name) and
                    not getattr(self._projection_backend,'boundary_observation_supported',False)):
                raise ValueError('borrowed-bank boundary observations require its actual operand hook')
            result=self._named_matrix(name,weights,values,router=router,check_cancelled=check_cancelled)
            if observer is not None and name in observer.projection_names:
                observer.projection(name,'output',result)
            return result
        finally:
            self._observations.name=previous
            self._observations.input_recorded=prior_input

    @contextmanager
    def observation_scope(self,observer):
        if getattr(self._observations,'observer',None) is not None:
            raise RuntimeError('nested arithmetic observation scope')
        self._observations.observer=observer
        try:yield
        finally:self._observations.observer=None

    def _named_matrix(self,name,weights,values,*,router=False,check_cancelled=None):
        backend = self._projection_backend
        if check_cancelled is not None:
            if not callable(check_cancelled):raise ValueError('projection cancellation callback required')
            check_cancelled()
        if backend is not None:
            if backend.identity != self._projection_identity:
                raise ValueError('distributed bank changed after K3 arithmetic admission')
            if backend.handles(name):
                if backend.shape(name) != tuple(weights.shape):
                    raise ValueError('distributed bank differs from K3 matrix geometry')
                expected = 'float32' if router else 'bfloat16'
                if backend.dtype_name != expected:
                    raise ValueError('distributed projection precision differs from K3 boundary')
                result = backend.linear(name, values)
                if check_cancelled is not None:check_cancelled()
                return result if router else self.cast(result)
        if np.asarray(values).ndim == 1:
            return self.matrix(weights, values, router=router,check_cancelled=check_cancelled)
        return self.matrix_batch(weights, values, router=router,check_cancelled=check_cancelled)

    def _admit(self, *arrays, extra=0):
        check_precision(self.torch, self.device)
        values = tuple(np.asarray(a) for a in arrays)
        if sum(a.size*12 for a in values)+extra > self.max_bytes:
            raise MemoryError('BF16 arithmetic buffers exceed admission')
        if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in values):
            raise ValueError('finite FP32 host buffers required')
        return values

    def _tensor(self, value):
        return self.torch.tensor(value, device=self.device, dtype=self.torch.bfloat16)

    def _host(self, value):
        if not bool(self.torch.isfinite(value).all()):
            raise ValueError('nonfinite BF16 arithmetic result')
        return value.float().cpu().numpy().copy()

    def cast(self, value):
        value, = self._admit(value)
        with self.torch.inference_mode():
            return self._host(self._tensor(value))

    def matrix(self, weights, value, *, router=False,check_cancelled=None):
        weights,value=np.asarray(weights),np.asarray(value)
        if weights.ndim != 2 or value.shape != (weights.shape[1],):
            raise ValueError('matrix/vector geometry mismatch')
        return self.matrix_batch(weights,value[None],router=router,check_cancelled=check_cancelled)[0]

    def packed_matrix(self, packed, scales, value, *, rows, cols, cancel=None):
        """Decode bounded low-nibble-first E2M1/E8M0 row tiles before BF16 GEMV.

        This is a decompressed BF16 policy, not a fused quantized hardware
        kernel or a claim of bitwise agreement with a publisher accelerator.
        Column reductions are not split or reassociated by this scheduler.
        """
        if (type(rows) is not int or type(cols) is not int or rows <= 0
                or cols <= 0 or cols % 32):
            raise ValueError('positive packed geometry with group32 columns required')
        for array, shape in ((packed, (rows, cols//2)), (scales, (rows, cols//32))):
            if (type(array) is not np.ndarray or array.dtype != np.uint8
                    or array.shape != shape or not array.flags.c_contiguous):
                raise ValueError('contiguous exact MXFP4 packed/scale arrays required')
        value, = self._admit(value)
        if (value.ndim not in (1, 2) or value.shape[-1] != cols
                or value.ndim == 2 and not value.shape[0]):
            raise ValueError('packed projection vector geometry mismatch')
        if cancel is not None and not callable(getattr(cancel, 'is_set', None)):
            raise ValueError('cancellation must provide is_set')
        batch = 1 if value.ndim == 1 else value.shape[0]
        fixed = value.size*12+batch*rows*8+1024
        row_charge = cols*64+batch*64
        tile_rows = min(rows, (self.max_bytes-fixed)//row_charge)
        if tile_rows <= 0:
            raise MemoryError('one MXFP4 row plus output cannot fit arithmetic admission')
        def check():
            if cancel is not None and cancel.is_set():
                from ..execution_service import ExecutionCancelled
                raise ExecutionCancelled('packed BF16 projection cancelled')
        check()
        result = np.empty((rows,) if value.ndim == 1 else (batch, rows), np.float32)
        torch = self.torch
        with torch.inference_mode():
            vector = self._tensor(value)
            lookup = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.,
                                   -0., -.5, -1., -1.5, -2., -3., -4., -6.),
                                  device=self.device, dtype=torch.float32)
            for start in range(0, rows, tile_rows):
                check()
                stop = min(rows, start+tile_rows)
                if np.any(scales[start:stop] == 255):
                    raise ValueError('nonfinite E8M0 scale code')
                raw = torch.tensor(packed[start:stop], device=self.device, dtype=torch.uint8)
                if self._packed_decoder is not None:
                    scale_codes = torch.tensor(scales[start:stop], device=self.device, dtype=torch.uint8)
                    decoded = self._packed_decoder.decode(raw, scale_codes, rows=stop-start, cols=cols)
                    projected = torch.nn.functional.linear(vector, decoded)
                    result[..., start:stop] = self._host(projected)
                    check()
                    continue
                codes = torch.empty((stop-start, cols), device=self.device, dtype=torch.int64)
                codes[:, 0::2] = (raw & 15).to(torch.int64)
                codes[:, 1::2] = (raw >> 4).to(torch.int64)
                exponent = torch.tensor(scales[start:stop], device=self.device,
                                        dtype=torch.int32)-127
                if self.packed_scale_policy == 'bf16-before-multiply':
                    scale = torch.pow(2.0, exponent.to(torch.bfloat16))
                    if not bool(torch.isfinite(scale).all()):
                        raise ValueError('MXFP4 scale exceeds BF16 range')
                    decoded = lookup[codes].to(torch.bfloat16)*scale.repeat_interleave(32, dim=1)
                else:
                    weights = torch.ldexp(lookup[codes], exponent.repeat_interleave(32, dim=1))
                    if not bool(torch.isfinite(weights).all()):
                        raise ValueError('decoded MXFP4 weight exceeds FP32 range')
                    decoded = weights.to(torch.bfloat16)
                if not bool(torch.isfinite(decoded).all()):
                    raise ValueError('decoded MXFP4 weight exceeds BF16 range')
                projected = torch.nn.functional.linear(vector, decoded)
                result[..., start:stop] = self._host(projected)
                check()
        return result

    def matrix_batch(self, weights, values, *, router=False,check_cancelled=None):
        """Stream output-row tiles; each dot retains its complete input axis.

        Host checkpoint ownership belongs to the executor. This owner admits
        the live input, complete output and only one device weight tile, rather
        than constructing another full resident BF16 copy of a dense matrix.
        """
        weights,values=np.asarray(weights),np.asarray(values)
        if (weights.ndim != 2 or values.ndim != 2 or not values.shape[0]
                or not all(weights.shape) or values.shape[1] != weights.shape[1]
                or weights.dtype!=np.float32 or values.dtype!=np.float32):
            raise ValueError('batch projection geometry mismatch')
        if type(router) is not bool or (check_cancelled is not None and not callable(check_cancelled)):
            raise ValueError('explicit routing and cancellation policies required')
        def check():
            if check_cancelled is not None:check_cancelled()
        check();check_precision(self.torch,self.device)
        batch,outputs,columns=values.shape[0],weights.shape[0],weights.shape[1]
        fixed=values.size*16+batch*outputs*8+4096
        row_charge=columns*16+batch*16
        rows=min(outputs,(self.max_bytes-fixed)//row_charge)
        if rows<=0:raise MemoryError('one dense projection row and complete output exceed admission')
        if not np.isfinite(values).all():raise ValueError('finite dense projection input required')
        result=np.empty((batch,outputs),np.float32)
        with self.torch.inference_mode():
            incoming=self._tensor(values)
            if router:incoming=incoming.float()
            observer=getattr(self._observations,'observer',None)
            name=getattr(self._observations,'name',None)
            if (observer is not None and name in observer.projection_names and
                    not getattr(self._observations,'input_recorded',False)):
                observer.projection(name,'input',incoming)
                self._observations.input_recorded=True
            for begin in range(0,outputs,rows):
                check();end=min(outputs,begin+rows)
                tile=weights[begin:end]
                if not np.isfinite(tile).all():raise ValueError('finite dense projection weight rows required')
                matrix=self._tensor(tile)
                if router:matrix=matrix.float()
                projected=self.torch.nn.functional.linear(incoming,matrix)
                result[:,begin:end]=self._host(projected)
                del matrix,projected
        check();return result

    def norm(self, value, weights, epsilon):
        value, weights = self._admit(value, weights)
        if value.shape != weights.shape or value.ndim != 1 or not value.size:
            raise ValueError('normalization geometry mismatch')
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('positive norm epsilon required')
        with self.torch.inference_mode():
            x = self._tensor(value)
            fp = x.float()
            inverse = self.torch.rsqrt(fp.square().mean()+epsilon)
            return self._host((fp*inverse).to(x.dtype)*self._tensor(weights))

    def residual(self, values, gain, query, epsilon):
        values, gain, query = self._admit(values, gain, query)
        if values.ndim != 2 or not all(values.shape) or gain.shape != values.shape[1:] or query.shape != gain.shape:
            raise ValueError('residual geometry mismatch')
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('positive residual epsilon required')
        with self.torch.inference_mode():
            rows = self._tensor(values).float()
            normalized = rows*self.torch.rsqrt(rows.square().mean(-1, keepdim=True)+epsilon)
            scorer = self._tensor(gain).float()*self._tensor(query).float()
            scores = (normalized*scorer).sum(-1)
            mixture = self.torch.matmul(scores.softmax(-1), rows)
            return self._host(mixture.to(self.torch.bfloat16))

    def situ(self, gate, up, beta, linear_beta):
        gate, up = self._admit(gate, up)
        if gate.shape != up.shape or not gate.size:
            raise ValueError('Situ geometry mismatch')
        if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in (beta, linear_beta)):
            raise ValueError('positive Situ coefficients required')
        with self.torch.inference_mode():
            g, u = self._tensor(gate).float(), self._tensor(up).float()
            bounded_gate = beta*self.torch.tanh(g/beta)*g.sigmoid()
            bounded_up = linear_beta*self.torch.tanh(u/linear_beta)
            return self._host((bounded_gate*bounded_up).to(self.torch.bfloat16))

    def add(self, left, right):
        left, right = self._admit(left, right)
        if left.shape != right.shape:
            raise ValueError('residual addition geometry mismatch')
        with self.torch.inference_mode():
            return self._host(self._tensor(left)+self._tensor(right))

    def gate(self, value):
        value, = self._admit(value)
        with self.torch.inference_mode():
            return self._host(self._tensor(value).sigmoid())

    def router_sigmoid(self, value):
        value, = self._admit(value)
        with self.torch.inference_mode():
            logits = self.torch.tensor(value, device=self.device, dtype=self.torch.float32)
            return self._host(logits.sigmoid())

    def rotate(self,rows,positions,*,rotary,theta):
        rows, = self._admit(rows)
        positions=np.asarray(positions)
        if (rows.ndim!=2 or type(rotary) is not int or rotary<=0 or rotary%2
                or rotary>rows.shape[1] or positions.shape!=(rows.shape[0],)
                or positions.dtype.kind not in 'iu' or not math.isfinite(theta) or theta<=0):
            raise ValueError('matching rotary rows, integer positions and positive theta required')
        torch=self.torch
        with torch.inference_mode():
            x=self._tensor(rows).float()
            phase=torch.tensor(positions.astype(np.float32),device=self.device)[:,None]*theta**(
                -torch.arange(0,rotary,2,device=self.device,dtype=torch.float32)/rotary)
            even,odd=x[:,:rotary:2].clone(),x[:,1:rotary:2].clone()
            x[:,:rotary:2]=even*phase.cos()-odd*phase.sin()
            x[:,1:rotary:2]=even*phase.sin()+odd*phase.cos()
            return self._host(x.to(torch.bfloat16))

    def multiply(self, left, right):
        left, right = self._admit(left, right)
        if left.shape != right.shape:
            raise ValueError('gated multiplication geometry mismatch')
        with self.torch.inference_mode():
            return self._host(self._tensor(left)*self._tensor(right))

    def attention(self, query, keys, values):
        query, keys, values = self._admit(query, keys, values)
        if (query.ndim != 2 or keys.ndim != 3 or values.ndim != 3
                or keys.shape[1:] != query.shape or keys.shape[:2] != values.shape[:2]
                or not keys.shape[0]):
            raise ValueError('expanded attention geometry mismatch')
        self._admit(query, keys, values, extra=16*keys.shape[0]*query.shape[0])
        with self.torch.inference_mode():
            if self._attention_backend is not None:
                padded_bytes = 8*keys.shape[0]*query.size
                self._admit(query, keys, values, extra=padded_bytes)
                return self._host(self._attention_backend.attention(
                    self._tensor(query), self._tensor(keys), self._tensor(values)))
            q = self._tensor(query)[None, :, None, :]
            k = self._tensor(keys).permute(1, 0, 2)[None]
            v = self._tensor(values).permute(1, 0, 2)[None]
            scores = self.torch.einsum('bhqd,bhkd->bhqk', q, k)*query.shape[1]**(-0.5)
            probabilities = scores.float().softmax(-1).to(self.torch.bfloat16)
            output = self.torch.einsum('bhqk,bhkd->bhqd', probabilities, v)
            return self._host(output[0, :, 0, :])

    def status(self):
        return {'implementation': 'torch-k3-bf16-boundaries', 'identity': self.identity,
                'distributed_projection_bank': self._projection_identity,
                'packed_scale_policy': self.packed_scale_policy,
                'packed_decoder': None if self._packed_decoder is None else self._packed_decoder.identity,
                'attention_backend': 'eager' if self._attention_backend is None else self._attention_backend.identity,
                'attention_reduction_policy':self.attention_mode,
                'device': str(self.device), 'certified': False}

    def attention_pages(self,query,pages,*,check_cancelled=None):
        """Two-pass bounded attention over a repeatable occupied-cache source.

        Scores retain BF16 model boundaries. The global normalizer and page
        output reduction use FP32, a separately identified policy rather than
        an assertion of equality to a single resident accelerator GEMM.
        """
        if self.attention_mode!='paged-online-fp32' or not callable(pages):
            raise ValueError('explicit paged reduction and repeatable cache source required')
        if check_cancelled is not None and not callable(check_cancelled):
            raise ValueError('attention cancellation callback required')
        query,=self._admit(query)
        if query.ndim!=2 or not all(query.shape):raise ValueError('nonempty head/channel query required')
        torch=self.torch;heads,dimension=query.shape
        def check():
            if check_cancelled is not None:check_cancelled()
        check();lengths=[];digests=[];value_width=None
        def page_digest(keys,values):
            digest=hashlib.sha256()
            for array in (keys,values):
                array=np.ascontiguousarray(array)
                digest.update(repr((array.shape,array.dtype.str)).encode())
                digest.update(memoryview(array).cast('B'))
            return digest.digest()
        with torch.inference_mode():
            q=self._tensor(query).unsqueeze(1)
            peak=torch.full((heads,1),-float('inf'),device=self.device,dtype=torch.float32)
            total=torch.zeros((heads,1),device=self.device,dtype=torch.float32)
            def score_page(keys,values):
                keys,values=self._admit(keys,values,extra=heads*len(keys)*32+query.size*16+
                                       heads*np.asarray(values).shape[-1]*24+len(lengths)*256)
                if (keys.ndim!=3 or values.ndim!=3 or not keys.shape[0]
                        or keys.shape[1:]!=query.shape or values.shape[:2]!=keys.shape[:2]):
                    raise ValueError('matching occupied expanded attention pages required')
                score=(torch.matmul(q,self._tensor(keys).permute(1,2,0))[:,0]*dimension**-.5).float()
                if not bool(torch.isfinite(score).all()):raise ValueError('nonfinite paged attention scores')
                return score,values
            for keys,values in pages():
                check();score,values=score_page(keys,values)
                if value_width is None:value_width=values.shape[-1]
                if values.shape[-1]!=value_width:raise ValueError('cache value width changed between pages')
                if (len(lengths)+1)*256+query.size*16+heads*value_width*24>self.max_bytes:
                    raise MemoryError('paged attention inventory exceeds arithmetic admission')
                lengths.append(len(keys))
                digests.append(page_digest(keys,values))
                # Rescale both accumulated statistics when a page raises max.
                new_peak=torch.maximum(peak,score.max(-1,keepdim=True).values)
                total=total*torch.exp(peak-new_peak)+torch.exp(score-new_peak).sum(-1,keepdim=True)
                peak=new_peak
                del score,keys,values
            if not lengths or value_width is None or not bool(torch.isfinite(total).all()) or bool((total<=0).any()):
                raise ValueError('complete nonempty finite occupied attention source required')
            self._admit(query,extra=heads*value_width*24+heads*32)
            output=torch.zeros((heads,value_width),device=self.device,dtype=torch.float32)
            observed=0
            for index,(keys,values) in enumerate(pages()):
                check()
                if index>=len(lengths) or len(keys)!=lengths[index] or values.shape[-1]!=value_width:
                    raise ValueError('cache page inventory changed between attention passes')
                score,values=score_page(keys,values)
                if page_digest(keys,values)!=digests[index]:
                    raise ValueError('cache page content changed between attention passes')
                probability=(torch.exp(score-peak)/total).to(torch.bfloat16)
                output+=torch.matmul(probability.unsqueeze(1),self._tensor(values).permute(1,0,2))[:,0].float()
                observed+=1
                del score,probability,keys,values
            if observed!=len(lengths):raise ValueError('second attention pass omitted occupied cache pages')
            result=self._host(output.to(torch.bfloat16))
        check();return result

    def attention_sequence(self, query, keys, values):
        query, keys, values = self._admit(query, keys, values)
        if (query.ndim != 3 or keys.ndim != 3 or values.ndim != 3
                or keys.shape[1:] != query.shape[1:] or keys.shape[:2] != values.shape[:2]
                or not query.shape[0] or keys.shape[0] < query.shape[0]):
            raise ValueError('occupied-prefix attention batch geometry mismatch')
        if self._attention_backend is None:
            raise ValueError('batched K3 MLA requires explicit FlashAttention backend')
        self._admit(query, keys, values, extra=8*keys.size+8*query.shape[0]*values.shape[1]*values.shape[2])
        with self.torch.inference_mode():
            return self._host(self._attention_backend.attention_sequence(
                self._tensor(query), self._tensor(keys), self._tensor(values)))
