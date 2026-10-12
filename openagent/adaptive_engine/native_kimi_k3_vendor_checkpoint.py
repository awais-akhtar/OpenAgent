"""Local, payload-free publisher K3 inspection. No decoder is admitted here."""
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path

from .native_checkpoint import _json, _open_local, _pairs, _stamp


VENDOR_REVISION = 'f831ab66814297da540d832a5235f8e904f29d06'
VENDOR_CONFIG_SHA256 = '9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213'


class UnsupportedContract(str, Enum):
    MULTIMODAL_WRAPPER = 'multimodal_wrapper'
    QUANTIZATION = 'quantization'
    GEOMETRY = 'geometry'
    DENSE_PREFIX = 'dense_prefix'
    TENSOR_LAYOUT = 'tensor_layout'
    FULL_HYBRID_ORACLE = 'full_hybrid_oracle'


@dataclass(frozen=True)
class StructuralGap:
    contract: UnsupportedContract
    path: str
    detail: str


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    shard: str
    dtype: str
    shape: tuple
    offsets: tuple


@dataclass(frozen=True)
class VendorCheckpointInspection:
    config_sha256: str
    pinned_config_matches: bool
    layer_types: tuple
    tensors: tuple
    payload_bytes: int
    gaps: tuple

    @property
    def published_checkpoint_supported(self):
        return False

    @property
    def admitted(self):
        return False


def _schedule(text):
    count = text.get('num_hidden_layers')
    linear = text.get('linear_attn_config')
    if type(count) is not int or not 1 <= count <= 4096 or type(linear) is not dict:
        raise ValueError('explicit bounded hybrid layer configuration required')
    groups = []
    for key in ('kda_layers', 'full_attn_layers'):
        values = linear.get(key)
        if (type(values) is not list or any(type(i) is not int or not 1 <= i <= count
                                          for i in values) or len(set(values)) != len(values)):
            raise ValueError('invalid one-based layer inventory: ' + key)
        groups.append(set(values))
    kda, mla = groups
    if kda & mla or kda | mla != set(range(1, count + 1)) or count not in mla:
        raise ValueError('hybrid schedule must partition layers and end in MLA')
    return tuple('kda' if i in kda else 'mla' for i in range(1, count + 1))


def _gaps(config, text):
    result = []

    def add(kind, path, detail):
        result.append(StructuralGap(kind, path, detail))

    add(UnsupportedContract.MULTIMODAL_WRAPPER, 'model_type',
        'kimi_k3 vision/projector wrapper is not a native hybrid text fixture; no implicit unwrap')
    for path, document in (('', config), ('text_config.', text)):
        if document.get('quantization_config') is not None:
            add(UnsupportedContract.QUANTIZATION, path + 'quantization_config',
                'compressed weights require an exact packing/scale/dequantization contract; '
                'BF16/F32 conversion does not decode MXFP4')
    limits = {'hidden_size': 64, 'num_hidden_layers': 8, 'num_attention_heads': 4,
              'num_experts': 64, 'routed_expert_hidden_size': 64,
              'moe_intermediate_size': 128, 'intermediate_size': 128,
              'q_lora_rank': 64, 'kv_lora_rank': 64, 'qk_nope_head_dim': 16,
              'qk_rope_head_dim': 16, 'v_head_dim': 16, 'attn_res_block_size': 8,
              'vocab_size': 256}
    for key, upper in limits.items():
        value = text.get(key)
        if type(value) is int and value > upper:
            add(UnsupportedContract.GEOMETRY, 'text_config.' + key,
                f'publisher value {value} exceeds fixture bound {upper}')
    linear = text['linear_attn_config']
    for key, upper in (('num_heads', 8), ('head_dim', 32), ('short_conv_kernel_size', 8)):
        value = linear.get(key)
        if type(value) is int and value > upper:
            add(UnsupportedContract.GEOMETRY, 'text_config.linear_attn_config.' + key,
                f'publisher value {value} exceeds KDA fixture bound {upper}')
    if text.get('first_k_dense_replace', 0) != 0:
        add(UnsupportedContract.DENSE_PREFIX, 'text_config.first_k_dense_replace',
              'publisher dense-prefix layers require explicit tensor mapping; '
              'optional fixture dense layers are not automatic publisher admission')
    add(UnsupportedContract.TENSOR_LAYOUT, 'tensor_inventory',
        'vendor language_model/self_attn/block_sparse_moe names, per-expert w1/w2/w3 '
        'weight_packed and weight_scale, combined shared MLP and '
        'rank-2 residual queries are not fixture attention/moe stacked tensors; '
        'exact whole-model mapping and numerical equivalence are unproved')
    add(UnsupportedContract.FULL_HYBRID_ORACLE, 'linear_attn_config',
        'unchanged full hybrid CPU logits/cache oracle not established; dense-only parity is insufficient')
    return tuple(result)


def inspect_native_kimi_k3_vendor_checkpoint(directory, *, max_metadata_bytes=64 * 1024**2):
    """Validate local config/index/header structure, never read tensor payloads.

    Malformed metadata raises ValueError. Valid but unsupported structures return
    typed gaps. Matching the pinned config is provenance evidence, not admission.
    No network, remote code, libraries, numpy arrays or inference are involved.
    """
    if type(max_metadata_bytes) is not int or max_metadata_bytes <= 0:
        raise ValueError('positive metadata budget required')
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('checkpoint must be a local directory')
    with ExitStack() as stack:
        handles, inventory = [], {}
        remaining = max_metadata_bytes

        def opened(path):
            source, identity = _open_local(root, path, stack)
            handles.append((source, identity))
            return source, identity

        def debit(size):
            nonlocal remaining
            remaining -= size
            if remaining < 0:
                raise ValueError('checkpoint exceeds metadata budget')

        source, identity = opened(root / 'config.json')
        debit(identity[2])
        config = _json(source, identity)
        source.seek(0)
        digest = hashlib.sha256(source.read(identity[2])).hexdigest()
        if type(config) is not dict or config.get('model_type') != 'kimi_k3':
            raise ValueError('exact publisher kimi_k3 wrapper required; fixtures remain separate')
        text = config.get('text_config')
        if type(text) is not dict or text.get('model_type') != 'kimi_linear':
            raise ValueError('explicit publisher text_config required')
        schedule = _schedule(text)
        payload = 0
        shards = sorted(root.glob('*.safetensors'))
        if not shards:
            raise ValueError('checkpoint has no SafeTensors shards')
        if len(shards) > 4096:
            raise ValueError('checkpoint exceeds shard count bound')
        widths = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2,
                  'I32': 4, 'U32': 4, 'I64': 8, 'U64': 8,
                  'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8,
                  'F8_E4M3': 1, 'F8_E5M2': 1, 'F8_E8M0': 1}
        for shard in shards:
            source, identity = opened(shard)
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError('missing shard header length')
            length = int.from_bytes(prefix, 'little')
            if not 1 < length <= min(16 * 1024**2, identity[2] - 8):
                raise ValueError('invalid shard header length')
            debit(length + 8)
            header = json.loads(source.read(length), object_pairs_hook=_pairs,
                                parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            if type(header) is not dict:
                raise ValueError('shard header must be an object')
            ranges = []
            extent = identity[2] - 8 - length
            for name, tensor in header.items():
                if name == '__metadata__':
                    if type(tensor) is not dict or any(type(v) is not str for v in tensor.values()):
                        raise ValueError('invalid SafeTensors metadata')
                    continue
                if not name or name in inventory or type(tensor) is not dict:
                    raise ValueError('duplicate or invalid tensor: ' + name)
                dtype, shape, offsets = (tensor.get(k) for k in ('dtype', 'shape', 'data_offsets'))
                if (type(dtype) is not str or dtype not in widths or type(shape) is not list
                        or len(shape) > 32 or any(type(n) is not int or not 0 <= n <= 2**63-1 for n in shape)
                        or type(offsets) is not list or len(offsets) != 2
                        or any(type(n) is not int for n in offsets)):
                    raise ValueError('invalid tensor metadata: ' + name)
                start, end = offsets
                if not 0 <= start <= end <= extent or end-start != math.prod(shape)*widths[dtype]:
                    raise ValueError('invalid tensor byte range: ' + name)
                ranges.append((start, end))
                inventory[name] = TensorMetadata(name, shard.name, dtype, tuple(shape), tuple(offsets))
            position = 0
            for start, end in sorted(ranges):
                if start != position:
                    raise ValueError('overlapping or noncontiguous shard payloads')
                position = end
            if position != extent:
                raise ValueError('unaccounted shard payload bytes')
            payload += extent
        if not inventory:
            raise ValueError('checkpoint has no tensors')
        path = root / 'model.safetensors.index.json'
        if path.exists():
            source, identity = opened(path)
            debit(identity[2])
            index = _json(source, identity)
            mapping = index.get('weight_map') if type(index) is dict else None
            if type(mapping) is not dict or set(mapping) != set(inventory):
                raise ValueError('shard index does not match tensor inventory')
            if any(mapping[n] != t.shard for n, t in inventory.items()):
                raise ValueError('shard index contains mismatched or nonlocal paths')
        if any(_stamp(s) != stamp for s, stamp in handles):
            raise ValueError('checkpoint file changed during inspection')
        gaps = _gaps(config, text)
        packed = tuple(t for t in inventory.values()
                       if t.name.endswith(('.weight_packed', '.weight_scale')))
        if packed:
            gaps += (StructuralGap(UnsupportedContract.QUANTIZATION, 'tensor_inventory',
                                   f'{len(packed)} packed/scale tensors observed; metadata is not '
                                   'a proof of FP4 nibble order or E8M0 scale semantics'),)
        return VendorCheckpointInspection(digest, digest == VENDOR_CONFIG_SHA256,
                                          schedule, tuple(inventory[n] for n in sorted(inventory)),
                                          payload, gaps)
