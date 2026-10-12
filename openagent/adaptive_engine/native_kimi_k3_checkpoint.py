"""Local loading for the explicit OpenAgent K3 hybrid equation-fixture format.

This format does not accept or reinterpret publisher K3 checkpoints.
"""

from dataclasses import fields
import math
from pathlib import Path
import ctypes as C
import _ctypes

from .native_checkpoint import _read_native_checkpoint
from .native_kimi_k3_decoder import KimiK3DenseMLAConfig
from .native_kimi_k3_kda import KimiK3KDAConfig
from .native_kimi_k3_moe import KimiK3MoEConfig
from .native_kimi_k3_hybrid import (
    KimiK3HybridConfig,
    NativeKimiK3Hybrid,
    hybrid_tensor_shapes,
)


FIXTURE_MODEL_TYPE = "openagent_kimi_k3_hybrid_fixture"


def _release_library(handle):
    release = (
        _ctypes.FreeLibrary if hasattr(_ctypes, "FreeLibrary") else _ctypes.dlclose
    )
    release(handle)


def _preflight_library(path):
    library = C.CDLL(str(path))
    try:
        for name, result in (
            ("oa_k3_blocks_version", C.c_int),
            ("oa_k3_kda_projection_version", C.c_int),
            ("oa_kimi_kda_version", C.c_uint64),
        ):
            version = getattr(library, name)
            version.argtypes, version.restype = [], result
            if version() != 1:
                raise ValueError("unsupported native hybrid component ABI: " + name)
        for name in (
            "oa_k3_blocks_norm",
            "oa_k3_blocks_add",
            "oa_k3_blocks_matrix",
            "oa_k3_blocks_mla",
            "oa_kimi_k3_attnres",
            "oa_kimi_k3_moe",
            "oa_k3_kda_project",
            "oa_k3_kda_output",
            "oa_kimi_kda",
        ):
            getattr(library, name)
    except AttributeError as error:
        raise ValueError("missing native hybrid component symbol") from error
    finally:
        _release_library(library._handle)


def _component(document, component):
    if type(document) is not dict or set(document) != {
        f.name for f in fields(component)
    }:
        raise ValueError("complete explicit component configuration required")
    return component(**document)


def hybrid_fixture_config(document):
    """Validate the versioned local format without a vendor-family alias."""
    keys = {"model_type", "fixture_version", "mla", "kda", "moe", "layer_types"}
    if (
        type(document) is not dict
        or not keys <= set(document)
        or set(document) - keys - {"dense_layers", "dense_intermediate"}
    ):
        raise ValueError("exact hybrid fixture configuration required")
    if document["model_type"] != FIXTURE_MODEL_TYPE:
        raise ValueError("publisher checkpoints are not hybrid equation fixtures")
    if type(document["fixture_version"]) is not int or document["fixture_version"] != 1:
        raise ValueError("unsupported hybrid fixture version")
    if type(document["layer_types"]) is not list:
        raise ValueError("explicit layer_types list required")
    if type(document.get("dense_layers", [])) is not list:
        raise ValueError("explicit dense_layers list required")
    return KimiK3HybridConfig(
        mla=_component(document["mla"], KimiK3DenseMLAConfig),
        kda=_component(document["kda"], KimiK3KDAConfig),
        moe=_component(document["moe"], KimiK3MoEConfig),
        layer_types=tuple(document["layer_types"]),
        dense_layers=tuple(document.get("dense_layers", [])),
        dense_intermediate=document.get("dense_intermediate"),
    )


def hybrid_fixture_document(config):
    """Describe an explicit fixture for storage alongside local SafeTensors."""
    if type(config) is not KimiK3HybridConfig:
        raise ValueError("explicit hybrid config required")
    document = {
        "model_type": FIXTURE_MODEL_TYPE,
        "fixture_version": 1,
        **{
            name: {
                f.name: getattr(getattr(config, name), f.name)
                for f in fields(getattr(config, name))
            }
            for name in ("mla", "kda", "moe")
        },
        "layer_types": list(config.layer_types),
    }
    if config.dense_layers:
        document["dense_layers"] = list(config.dense_layers)
    if config.dense_intermediate is not None:
        document["dense_intermediate"] = config.dense_intermediate
    return document


def load_native_kimi_k3_fixture(
    directory, *, library, max_weight_bytes, max_bytes=128 * 1024 * 1024
):
    """Load admitted FP32 fixture shards; no downloads or numerical fallback."""
    if type(max_bytes) is not int or not 0 < max_bytes <= 2**63 - 1:
        raise ValueError("invalid hybrid memory budget")
    library = Path(library).resolve(strict=True)
    if not library.is_file():
        raise ValueError("library must be an explicit local file")
    _preflight_library(library)
    admitted = []

    def admit(document, headers, resident):
        config = hybrid_fixture_config(document)
        if config.dense_layers:
            dense_library = C.CDLL(str(library))
            try:
                getattr(dense_library, "oa_k3_dense")
            except AttributeError as error:
                raise ValueError("missing native dense SwiGLU symbol") from error
            finally:
                _release_library(dense_library._handle)
        shapes = hybrid_tensor_shapes(config)
        if set(headers) != set(shapes):
            raise ValueError("exact hybrid tensor inventory required")
        for name, shape in shapes.items():
            header = headers[name]
            if header["dtype"] != "F32" or tuple(header["shape"]) != shape:
                raise ValueError("invalid hybrid FP32 tensor metadata: " + name)
        c, k = config.mla, config.kda
        cache = sum(
            4 * c.capacity * c.heads * (c.nope + c.rope + c.value)
            if kind == "mla"
            else 4
            * (
                k.heads * k.key * k.value
                + (k.kernel - 1) * k.heads * (2 * k.key + k.value)
            )
            for kind in config.layer_types
        )
        expected = sum(math.prod(shape) * 4 for shape in shapes.values())
        if resident != expected:
            raise ValueError("hybrid resident tensor extent mismatch")
        required = 5 * resident + 5 * cache + 4 * (c.vocab + 16 * c.hidden) + 1024**2
        if required > max_bytes:
            raise ValueError("hybrid transaction memory budget exceeded")
        admitted.append(config)

    _, weights = _read_native_checkpoint(
        directory,
        model_type=FIXTURE_MODEL_TYPE,
        max_weight_bytes=max_weight_bytes,
        metadata_admission=admit,
    )
    return NativeKimiK3Hybrid(
        admitted[0], weights, library=library, max_bytes=max_bytes
    )
