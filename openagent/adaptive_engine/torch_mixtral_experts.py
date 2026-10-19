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
    Formats are admitted by the shared packed decoder only when this consumer
    has the matching activation transform, including the E8 and IQ3 contracts.
    This implementation is inherited by the Qwen3.5 and Qwen4 device expert
    providers, so format restrictions here apply to all three architectures.
    Provider storage remains caller-owned.
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
        # This shared provider can decode more persisted formats than any one
        # family has admitted. Enforce the family/device capability row here
        # as well as in the CLI so direct provider/decoder construction cannot
        # widen support merely by choosing a format present in FORMATS.
        from .source_scope import require_format_execution_consumer
        source_family = {
            'mixtral':'mixtral', 'qwen3_5_moe_text':'qwen35',
            'qwen4_exp_text':'qwen4', 'glm_moe_dsa':'glm-dsa',
        }.get(self.architecture)
        if source_family is None:
            raise ValueError('selected-device expert family has no source capability policy')
        from .e8_lattice_expert import LatticeContract
        for layer in manifest.layers:
            require_format_execution_consumer(
                family=source_family, backend='device',
                persisted_format=layer.dtype)
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
            if layer.dtype == 'e8-lattice-caller-v1':
                if (type(lattice_contract) is not LatticeContract or
                        layer.contract != lattice_contract.metadata() or
                        layer.hidden_size % 8 or layer.intermediate_size % 8):
                    raise ValueError('exact persisted E8 contract and aligned family geometry required')
            if layer.dtype == 'iq3-format6-98-v1':
                if iq3_grid is None or layer.contract != iq3_grid.metadata():
                    raise ValueError('exact persisted device lattice contract required')
                # Each of the three matrices is stored as independent
                # format-6 256-weight blocks.  Reject incompatible model
                # geometry at admission instead of waiting until a selected
                # expert is staged and decoded on the first request.
                if layer.hidden_size % 256 or layer.intermediate_size % 256:
                    raise ValueError(
                        'persisted IQ3 device experts require hidden and intermediate widths aligned to 256')
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
            policy=self.architecture+'-selected-device-fp32-e8-rotation-v1-iq3-rotation-256-aligned-v5')
        return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def execute(self,layer,expert,value,*,torch,device,cancel=None):
        admit_fp32(torch,device)
        from .device_packed_decode import decode_device_expert
        from .device_qwen35_layers import _cancel
        slot=self._layer_slot(layer)
        if type(expert) is not int or not 0<=expert<self.expert_count:
            raise ValueError('valid original Mixtral layer/expert required')
        if (value.ndim!=2 or value.shape[-1]!=self.hidden_size or value.dtype!=torch.float32
                or str(value.device)!=device or not bool(torch.isfinite(value).all().item())):
            raise ValueError('matching FP32 device expert hidden rows required')
        h,m = self.hidden_size,self.intermediate_size
        entry = self.layers[slot]
        row_work=value.shape[0]*(16*h+24*m)
        if entry.dtype=='e8-lattice-caller-v1':
            row_work=value.shape[0]*(24*h+32*m)+8192
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
                # Device admission can involve three large host->device
                # copies.  Make cancellation observable between each copy and
                # release every completed temporary if a later copy fails;
                # do not leave a partially staged expert looking usable.
                staged=[]
                try:
                    for item in decoded:
                        _cancel(cancel)
                        staged.append(torch.tensor(
                            np.array(item,dtype=np.float32,copy=True),device=device))
                    _cancel(cancel)
                    weights=tuple(staged)
                except BaseException:
                    staged.clear()
                    # Release decoded host arrays without deleting the local
                    # binding; this also keeps the post-try cleanup path clear
                    # to static flow analysis.
                    decoded = ()
                    raise
                del staged,decoded
            gate,up,down,gs,us,ds = weights
            if entry.dtype=='e8-lattice-caller-v1':
                from .device_e8_rotation import rotate_device_e8
                rotation_budget=self.work_bytes-36*h*m
                rotated=rotate_device_e8(torch,value,transform=self.lattice_contract.transform,
                    max_bytes=rotation_budget,cancel=cancel)
                activation=torch.nn.functional.silu((rotated@gate.T)*gs)*((rotated@up.T)*us)
                del rotated
                activation=rotate_device_e8(torch,activation,transform=self.lattice_contract.transform,
                    max_bytes=rotation_budget,cancel=cancel)
                result=(activation@down.T)*ds
                del activation
            elif entry.dtype=='iq3-format6-98-v1':
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
