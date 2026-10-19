"""Independent grouped INT4 container publication with explicit group size."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

import numpy as np

from .expert_store import ExpertRegion
from .grouped_int4_expert import DEFAULT_GROUP_SIZE, DTYPE, _shapes, pack_expert, payload_size
from .q5_0_expert import _positive


@dataclass(frozen=True)
class GroupedInt4ConversionResult:
    path: Path
    hidden_size: int
    intermediate_size: int
    group_size: int
    regions: Mapping[str, ExpertRegion]
    dtype: str = DTYPE
    activation: str = "swiglu"


def convert_experts(experts, output, *, hidden_size, intermediate_size, group_size=DEFAULT_GROUP_SIZE):
    shapes = _shapes(hidden_size, intermediate_size)
    _positive(group_size, "group_size")
    size = payload_size(hidden_size, intermediate_size, group_size=group_size)
    if not isinstance(experts, Mapping) or not experts:
        raise ValueError("experts must be a non-empty mapping")
    if any(not isinstance(key, str) or not key or "/" in key or "\\" in key for key in experts):
        raise ValueError("expert keys must be non-empty without path separators")
    destination = Path(output).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".grouped-int4-", dir=destination.parent)
    temporary = Path(name)
    regions = {}
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = None
        with stream:
            for key, matrices in experts.items():
                if not isinstance(matrices, (tuple, list)) or len(matrices) != 3:
                    raise ValueError("each expert must provide gate, up, down")
                if any(np.shape(matrix) != shape for matrix, shape in zip(matrices, shapes)):
                    raise ValueError("declared dimensions do not match expert projections")
                blob = pack_expert(*matrices, group_size=group_size)
                regions[key] = ExpertRegion(destination.name, stream.tell(), size, hashlib.sha256(blob).hexdigest())
                stream.write(blob)
                del blob
            stream.flush()
            os.fsync(stream.fileno())
        result = GroupedInt4ConversionResult(destination, hidden_size, intermediate_size,
                                             group_size, MappingProxyType(regions))
        os.link(temporary, destination)
    except BaseException as primary:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as cleanup:
                primary.add_note("grouped INT4 conversion descriptor cleanup pending: " + repr(cleanup))
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup:
            primary.add_note("grouped INT4 conversion temporary cleanup pending: " + repr(cleanup))
        raise
    else:
        temporary.unlink(missing_ok=True)
        return result
