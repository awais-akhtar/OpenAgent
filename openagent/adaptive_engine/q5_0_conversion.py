"""Q5_0 container conversion independent of the parent's manifest registry."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import numpy as np

from .expert_store import ExpertRegion
from .q5_0_expert import DTYPE, _positive, _shapes, pack_expert, payload_size


@dataclass(frozen=True)
class Q5_0ConversionResult:
    path: Path
    hidden_size: int
    intermediate_size: int
    regions: Mapping[str, ExpertRegion]
    dtype: str = DTYPE
    activation: str = "swiglu"


def convert_experts(experts, output, *, hidden_size, intermediate_size, block_count=64):
    """Publish exact hashed expert ranges using an exclusive hard link.

    Holds one packed expert at a time; model and layer identity are supplied
    by the parent when adapting these regions into its shared manifest.
    """
    shapes = _shapes(hidden_size, intermediate_size)
    _positive(block_count, "block_count")
    if not isinstance(experts, Mapping) or len(experts) == 0:
        raise ValueError("experts must be a non-empty mapping")
    for key in experts:
        if not isinstance(key, str) or not key or any(char in key for char in "/\\"):
            raise ValueError("expert keys must be non-empty without path separators")
    destination = Path(output).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".q5_0-", dir=destination.parent)
    temporary = Path(name)
    regions = {}
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = None
        with stream:
            for key, projections in experts.items():
                if not isinstance(projections, (tuple, list)) or len(projections) != 3:
                    raise ValueError("each expert must provide gate, up, down")
                if tuple(np.shape(matrix) for matrix in projections) != shapes:
                    raise ValueError("declared expert dimensions do not match projections")
                blob = pack_expert(*projections, block_count=block_count)
                regions[key] = ExpertRegion(destination.name, stream.tell(),
                                           payload_size(hidden_size, intermediate_size),
                                           hashlib.sha256(blob).hexdigest())
                stream.write(blob)
                del blob
            stream.flush()
            os.fsync(stream.fileno())
        result = Q5_0ConversionResult(destination, hidden_size, intermediate_size,
                                     MappingProxyType(regions))
        os.link(temporary, destination)
    except BaseException as primary:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as cleanup:
                primary.add_note("Q5_0 conversion descriptor cleanup pending: " + repr(cleanup))
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup:
            primary.add_note("Q5_0 conversion temporary cleanup pending: " + repr(cleanup))
        raise
    else:
        # Keep cleanup failure visible after successful publication; on the
        # error path above it is recorded without replacing the primary cause.
        temporary.unlink(missing_ok=True)
        return result
