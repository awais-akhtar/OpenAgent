"""Explicit packed device kernel for borrowed tensor-parallel row shards."""
import hashlib
from pathlib import Path
import threading
from .device_arithmetic import DevicePublisherV41Arithmetic


class DevicePublisherPackedProjection:
    _tensor = DevicePublisherV41Arithmetic._tensor
    _host = DevicePublisherV41Arithmetic._host
    _admit = DevicePublisherV41Arithmetic._admit
    packed_matvec = DevicePublisherV41Arithmetic.packed_matvec
    _packed_matvec_open = DevicePublisherV41Arithmetic._packed_matvec_open

    def __init__(self,*,device,torch_version,max_bytes,encoding):
        import torch
        from ..k3_execution.device_policy import admit_device
        if (str(torch.__version__)!=torch_version or type(max_bytes) is not int or max_bytes<=0
                or encoding not in ('fp8-e4m3-g32x32','v41-fp4-g32')):
            raise ValueError('explicit pinned device packed projection policy required')
        self.torch,self.device,self.max_bytes = torch,admit_device(torch,device),max_bytes
        self.encoding,self._closed,self._lock = encoding,False,threading.RLock()
        self.numerical_policy='device-fp32-original-packed-row-gather-v1:'+encoding
        source=hashlib.sha256(Path(__file__).with_name('device_arithmetic.py').read_bytes()).hexdigest()
        # Rank-local ordinals differ in a multi-GPU process group. The shared
        # kernel contract identifies its backend; each bank owns local placement.
        self.identity=hashlib.sha256(repr((source,torch_version,self.device.type,
            getattr(torch.version,'hip',None),max_bytes,self.numerical_policy)).encode()).hexdigest()

    def _ensure_open(self):
        if self._closed:raise RuntimeError('device packed projection kernel closed')

    def tile_rows(self,columns):
        count=(self.max_bytes-48*columns)//(64*columns+32)
        if self.encoding=='fp8-e4m3-g32x32':count=count//32*32
        if count<=0:raise MemoryError('device packed kernel cannot admit one aligned row tile')
        return count

    def matvec(self,packed,scales,x,*,rows,cols):
        with self._lock:
            self._ensure_open()
            if (type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0):
                raise ValueError('packed projection geometry exceeds the admitted kernel contract')
            return self.packed_matvec(packed,scales,x,rows=rows,cols=cols,encoding=self.encoding)

    def close(self):
        with self._lock:self._closed=True
