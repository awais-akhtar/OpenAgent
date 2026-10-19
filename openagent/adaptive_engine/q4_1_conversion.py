"""Independent matrix converter; no GGUF reader or shared manifest mutation."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from .expert_store import ExpertRegion
from .q4_1_expert import DTYPE, _positive, _shapes, pack_expert, payload_size


@dataclass(frozen=True)
class Q4_1ConversionResult:
    """Container metadata for the parent to adapt to its shared manifest."""

    path: Path
    hidden_size: int
    intermediate_size: int
    regions: Mapping[str, ExpertRegion]
    dtype: str = DTYPE
    activation: str = "swiglu"


def convert_experts(experts, output, *, hidden_size, intermediate_size, block_count=64):
    """Publish consecutive hashed experts without overwriting an existing file.

    Uses a same-directory temporary and exclusive hard-link publication. Failure
    removes the temporary and leaves no partial destination. Caller supplies
    model/layer identity separately when adapting to SparseExpertManifest.
    """
    shapes = _shapes(hidden_size, intermediate_size)
    size = payload_size(hidden_size, intermediate_size)
    _positive(block_count, "block_count")
    if not isinstance(experts, Mapping) or not experts:
        raise ValueError("experts must be a non-empty mapping")
    for key in experts:
        if not isinstance(key, str) or not key or "/" in key or "\\" in key:
            raise ValueError("expert ids must be non-empty without path separators")
    target = Path(output).absolute()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    regions = {}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".q4_1-", delete=False) as stream:
            temporary = Path(stream.name)
            for key, triple in experts.items():
                if not isinstance(triple, (tuple, list)) or len(triple) != 3:
                    raise ValueError("each expert requires gate, up, down")
                if any(np.shape(a) != shape for a, shape in zip(triple, shapes)):
                    raise ValueError("expert projection shape mismatch")
                blob = pack_expert(*triple, block_count=block_count)
                regions[key] = ExpertRegion(target.name, stream.tell(), size,
                                           hashlib.sha256(blob).hexdigest())
                stream.write(blob)
                del blob
            stream.flush()
            os.fsync(stream.fileno())
        result = Q4_1ConversionResult(target, hidden_size, intermediate_size, regions)
        os.link(temporary, target)
    except BaseException as primary:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as cleanup:
                primary.add_note("Q4_1 conversion temporary cleanup pending: " + repr(cleanup))
        raise
    else:
        # Preserve the conversion/publication failure as primary. If
        # publication succeeded, however, a failed temporary unlink is still
        # an observable cleanup failure rather than a silent artifact leak.
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        return result
