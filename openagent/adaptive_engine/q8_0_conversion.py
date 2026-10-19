"""Public matrix-to-Q8_0 container converter; not a GGUF/checkpoint importer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .expert_runtime import ExpertLayer, SparseExpertManifest
from .expert_store import ExpertRegion
from .q8_0_expert import DTYPE, pack_expert, payload_size, _positive


def convert_experts(
    experts: Mapping[str, tuple], output: str | Path, *,
    hidden_size: int, intermediate_size: int, model_id: str,
    architecture: str, fingerprint: str, layer_id: str = "block.0",
    experts_used: int = 1, block_count: int = 64,
) -> SparseExpertManifest:
    """Publish one layer from named (gate, up, down) matrix triples.

    Dimensions describe each triple's [i,h], [i,h], [h,i] shapes. This opt-in
    converter returns a validated manifest whose region paths are relative to
    output.parent; call manifest.write separately if a JSON file is wanted.
    SHA256 covers exact expert bytes. Regions are consecutive, unpadded.
    Only one packed expert is held at a time (in addition to caller matrices).
    The container is published after all validation, without overwriting an
    existing path. Publication uses a same-directory hard link; filesystems
    without hard-link support raise OSError without publishing output.
    """
    size = payload_size(hidden_size, intermediate_size)
    _positive(block_count, "block_count")
    _positive(experts_used, "experts_used")
    if not isinstance(experts, Mapping) or not experts:
        raise ValueError("experts must be a non-empty mapping of matrix triples")
    if any(not isinstance(key, str) or not key or "/" in key for key in experts):
        raise ValueError("expert ids must be non-empty strings without '/'")
    destination = Path(output).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        regions = {}
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".q8_0-", delete=False) as stream:
            temporary = Path(stream.name)
            for key, matrices in experts.items():
                if not isinstance(matrices, (tuple, list)) or len(matrices) != 3:
                    raise ValueError("each expert must contain gate, up and down matrices")
                # Packing validates all values; explicit dimensions also bind shape.
                shapes = ((intermediate_size, hidden_size), (intermediate_size, hidden_size),
                          (hidden_size, intermediate_size))
                if any(np.shape(matrix) != shape for matrix, shape in zip(matrices, shapes)):
                    raise ValueError("expert matrices do not match declared dimensions")
                payload = pack_expert(*matrices, block_count=block_count)
                offset = stream.tell()
                stream.write(payload)
                regions[key] = ExpertRegion(destination.name, offset, size, hashlib.sha256(payload).hexdigest())
                del payload
            manifest = SparseExpertManifest(
                model_id=model_id, architecture=architecture, fingerprint=fingerprint,
                layers=(ExpertLayer(layer_id, hidden_size, intermediate_size, experts_used,
                                    DTYPE, "swiglu", regions),),
            )
            manifest.validate()
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except BaseException as primary:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as cleanup:
                primary.add_note("Q8_0 conversion temporary cleanup pending: " + repr(cleanup))
        raise
    else:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        return manifest
