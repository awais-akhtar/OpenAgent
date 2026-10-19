"""Exclusive Q5_1 container publication; no shared manifest edits."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import numpy as np

from .expert_store import ExpertRegion
from .q5_0_expert import _positive
from .q5_1_expert import DTYPE, _shapes, pack_expert, payload_size


@dataclass(frozen=True)
class Q5_1ConversionResult:
    path: Path
    hidden_size: int
    intermediate_size: int
    regions: Mapping[str, ExpertRegion]
    dtype: str = DTYPE
    activation: str = "swiglu"


def convert_experts(experts, output, *, hidden_size, intermediate_size, block_count=64):
    """Return immutable region metadata; refuse overwrite and partial publication."""
    dimensions = _shapes(hidden_size, intermediate_size)
    _positive(block_count, "block_count")
    if not isinstance(experts, Mapping) or not experts:
        raise ValueError("experts must be a non-empty mapping")
    for key in experts:
        if not isinstance(key, str) or not key or "/" in key or "\\" in key:
            raise ValueError("expert keys must be non-empty without path separators")
    destination = Path(output).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=".q5_1-")
    temporary = Path(name)
    regions = {}
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = None
        with stream:
            for key, projections in experts.items():
                if not isinstance(projections, (tuple, list)) or len(projections) != 3:
                    raise ValueError("each expert must provide gate, up, down")
                if any(np.shape(matrix) != shape for matrix, shape in zip(projections, dimensions)):
                    raise ValueError("declared dimensions do not match expert projections")
                encoded = pack_expert(*projections, block_count=block_count)
                regions[key] = ExpertRegion(destination.name, stream.tell(), payload_size(hidden_size, intermediate_size),
                                           hashlib.sha256(encoded).hexdigest())
                stream.write(encoded)
                del encoded
            stream.flush()
            os.fsync(stream.fileno())
        metadata = Q5_1ConversionResult(destination, hidden_size, intermediate_size, MappingProxyType(regions))
        os.link(temporary, destination)
    except BaseException as primary:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as cleanup:
                primary.add_note("Q5_1 conversion descriptor cleanup pending: " + repr(cleanup))
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup:
            primary.add_note("Q5_1 conversion temporary cleanup pending: " + repr(cleanup))
        raise
    else:
        temporary.unlink(missing_ok=True)
        return metadata
