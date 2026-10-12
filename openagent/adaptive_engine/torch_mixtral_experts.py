"""Selected packed Mixtral expert products on the decoder's explicit device."""
import hashlib
import json
import threading
from contextlib import contextmanager
import numpy as np

from .expert_runtime import SparseExpertManifest
from .expert_store import DiskExpertStore
from .device_execution_policy import admit_fp32


class TorchMixtralExperts:
    """Borrow a verified local store; stage one selected expert at a time.

    Decoder hidden rows and expert products stay on device. Packed weights and
    row scales decode on the host within an explicit working-set reservation.
    Formats are admitted by the shared packed decoder, including explicit
    persisted lattice contracts. Provider storage remains caller-owned.
    """
    architecture = 'mixtral'

    def _admit_layer_ids(self,ids):
        return ids==tuple(str(index) for index in range(len(ids)))

    def _layer_slot(self,layer):
        if type(layer) is not int or not 0<=layer<len(self.layers):
            raise ValueError('valid original expert layer required')
        return layer

    def __init__(self,manifest,store,*,max_work_bytes,lattice_contract=None,iq3_grid=None):
        if type(manifest) is not SparseExpertManifest or manifest.architecture!=self.architecture or type(store) is not DiskExpertStore:
            raise ValueError('exact Mixtral manifest and borrowed disk store required')
        manifest.validate()
        if type(max_work_bytes) is not int or max_work_bytes<=0:
            raise ValueError('explicit device expert working-set budget required')
        self.manifest,self.store,self.work_bytes = manifest,store,max_work_bytes
        self.lattice_contract,self.iq3_grid = lattice_contract,iq3_grid
        from .device_packed_decode import FORMATS
        self.layers = tuple(manifest.layers)
        if not self.layers or not self._admit_layer_ids(tuple(layer.id for layer in self.layers)):
            raise ValueError('complete ordered original Mixtral layer inventory required')
        first = self.layers[0]
        self.hidden_size,self.intermediate_size,self.selected_experts = first.hidden_size,first.intermediate_size,first.experts_used
        self.expert_count = len(first.regions)
        for layer in self.layers:
            if (layer.dtype not in FORMATS or
                    (layer.hidden_size,layer.intermediate_size,layer.experts_used)!=(self.hidden_size,self.intermediate_size,self.selected_experts) or
                    set(layer.regions)!={str(index) for index in range(self.expert_count)}):
                raise ValueError('exact admitted device expert format/geometry required')
            if layer.dtype in ('e8-lattice-caller-v1','iq3-format6-98-v1'):
                contract = lattice_contract if layer.dtype=='e8-lattice-caller-v1' else iq3_grid
                if contract is None or layer.contract!=contract.metadata():
                    raise ValueError('exact persisted device lattice contract required')
            for expert,region in layer.regions.items():
                key = layer.id+'/'+expert
                if not region.sha256 or store.regions.get(key)!=region:
                    raise ValueError('hashed device expert/store inventory mismatch')
        if 36*self.hidden_size*self.intermediate_size+16*(self.hidden_size+self.intermediate_size)>max_work_bytes:
            raise MemoryError('selected host/device expert weights exceed admission')
        self.max_bytes = store.capacity_bytes+max_work_bytes
        self._lock = threading.RLock()
        self._busy = False
        self._identity = self.identity

    @contextmanager
    def _operation(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('selected device expert callback reentry')
            self._busy=True
            try:
                yield
            finally:
                self._busy=False

    @property
    def identity(self):
        document = dict(manifest=self.manifest.to_dict(),capacity=self.store.capacity_bytes,
            regions=[(key,region.path,region.offset,region.size,region.sha256,region.storage)
                     for key,region in sorted(self.store.regions.items())],work=self.work_bytes,
            policy=self.architecture+'-selected-device-fp32-iq3-rotation-v2')
        return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def execute(self,layer,expert,value,*,torch,device,cancel=None):
        admit_fp32(torch,device)
        from .device_packed_decode import decode_device_expert
        from .device_qwen35_layers import _cancel
        slot=self._layer_slot(layer)
        if type(expert) is not int or not 0<=expert<self.expert_count:
            raise ValueError('valid original Mixtral layer/expert required')
        if value.ndim!=2 or value.shape[-1]!=self.hidden_size or value.dtype!=torch.float32 or str(value.device)!=device:
            raise ValueError('matching FP32 device expert hidden rows required')
        h,m = self.hidden_size,self.intermediate_size
        entry = self.layers[slot]
        row_work=value.shape[0]*(16*h+24*m)
        if entry.dtype=='iq3-format6-98-v1':
            row_work=value.shape[0]*64*(h+m)+min(max(h,m),32768)*8+4096
        if 36*h*m+row_work>self.work_bytes:
            raise MemoryError('selected expert device rows exceed work admission')
        with self._operation():
            if self.identity!=self._identity:
                raise ValueError('device expert provider changed after admission')
            _cancel(cancel)
            with self.store.acquire(entry.id+'/'+str(expert)) as payload:
                decoded = decode_device_expert(payload,hidden=h,intermediate=m,dtype=entry.dtype,
                    max_bytes=self.work_bytes,lattice_contract=self.lattice_contract,iq3_grid=self.iq3_grid)
                weights = tuple(torch.tensor(np.array(item,dtype=np.float32,copy=True),device=device)
                                for item in decoded)
                del decoded
            gate,up,down,gs,us,ds = weights
            if entry.dtype=='iq3-format6-98-v1':
                from .device_iq3_rotation import rotate_device_iq3
                rotation_budget=self.work_bytes-36*h*m
                rotated=rotate_device_iq3(torch,value,max_bytes=rotation_budget,cancel=cancel)
                activation=torch.nn.functional.silu((rotated@gate.T)*gs)*((rotated@up.T)*us)
                del rotated
                activation=rotate_device_iq3(torch,activation,max_bytes=rotation_budget,cancel=cancel)
                result=(activation@down.T)*ds
            else:
                result = ((torch.nn.functional.silu((value@gate.T)*gs)*((value@up.T)*us))@down.T)*ds
            _cancel(cancel)
            if not bool(torch.isfinite(result).all().item()):
                raise ArithmeticError('nonfinite selected device expert output')
            return result


__all__ = ['TorchMixtralExperts']
