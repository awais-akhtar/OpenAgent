"""SafeTensors loading for the explicit dense CED equation format."""
import ctypes as C
import _ctypes
from dataclasses import asdict, fields
import math
from pathlib import Path

from .native_checkpoint import _read_native_checkpoint
from .native_deepseek_v41_ced import (
    DeepSeekV41CEDConfig, NativeDeepSeekV41CED, ced_tensor_shapes,
)

MODEL_TYPE = 'openagent_v41_dense_ced_fixture'


def ced_fixture_document(config):
    if type(config) is not DeepSeekV41CEDConfig:
        raise ValueError('explicit CED configuration required')
    value = asdict(config)
    value['modes'] = list(config.modes)
    return {'model_type': MODEL_TYPE, 'fixture_version': 1, 'ced': value}


def ced_fixture_config(document):
    if type(document) is not dict or set(document) != {'model_type', 'fixture_version', 'ced'}:
        raise ValueError('exact CED fixture envelope required')
    if document['model_type'] != MODEL_TYPE:
        raise ValueError('published checkpoints are not dense CED fixtures')
    if type(document['fixture_version']) is not int or document['fixture_version'] != 1:
        raise ValueError('unsupported CED fixture version')
    value = document['ced']
    if type(value) is not dict or set(value) != {f.name for f in fields(DeepSeekV41CEDConfig)}:
        raise ValueError('complete explicit CED geometry required')
    if type(value['modes']) is not list:
        raise ValueError('explicit CED mode list required')
    return DeepSeekV41CEDConfig(**{**value, 'modes': tuple(value['modes'])})


def _release_library(handle):
    release = _ctypes.FreeLibrary if hasattr(_ctypes, 'FreeLibrary') else _ctypes.dlclose
    release(handle)


def _preflight(path):
    library = C.CDLL(str(path))
    try:
        version = library.oa_v41_ced_version
        version.argtypes, version.restype = [], C.c_uint32
        if version() != 1:
            raise ValueError('unsupported CED ABI')
        getattr(library, 'oa_v41_ced_forward')
    except AttributeError as error:
        raise ValueError('missing native CED symbol') from error
    finally:
        _release_library(library._handle)


def load_native_ced_fixture(directory, *, library, max_weight_bytes,
                            max_bytes=64 * 1024**2):
    """Load local FP32 fixture shards after ABI and metadata admission."""
    if type(max_bytes) is not int or not 0 < max_bytes <= 2**63 - 1:
        raise ValueError('invalid CED memory budget')
    path = Path(library).resolve(strict=True)
    if not path.is_file():
        raise ValueError('native CED library must be a file')
    _preflight(path)
    admitted = []

    def admit(document, headers, resident):
        config = ced_fixture_config(document)
        shapes = ced_tensor_shapes(config)
        if set(headers) != set(shapes):
            raise ValueError('exact CED tensor inventory required')
        for name, shape in shapes.items():
            header = headers[name]
            if header['dtype'] != 'F32' or tuple(header['shape']) != shape:
                raise ValueError('invalid CED tensor metadata: ' + name)
        expected = sum(math.prod(s) * 4 for s in shapes.values())
        if resident != expected:
            raise ValueError('CED resident tensor extent mismatch')
        if 3 * expected + 1024**2 > max_bytes:
            raise ValueError('CED weight memory budget exceeded')
        # Admit one token of native staging before payload reads.
        heap = 4 * ((4 * config.layers + 1) * config.hidden + config.vocabulary)
        trace = 4 * config.layers * (2 + config.topk)
        required = 2 * expected + heap + 8 * config.vocabulary + 8 + 2 * trace + 1024**2
        if required > max_bytes:
            raise ValueError('CED transaction memory budget exceeded')
        admitted.append(config)

    _, weights = _read_native_checkpoint(
        directory, model_type=MODEL_TYPE, max_weight_bytes=max_weight_bytes,
        metadata_admission=admit,
    )
    return NativeDeepSeekV41CED(admitted[0], weights, library=path, max_bytes=max_bytes)
