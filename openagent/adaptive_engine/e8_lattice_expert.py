"""Caller-configured E8 vector quantization; not an upstream IQ3 byte ABI.

Lattice definition: coordinates all integer or all half-integer, even sum.
Each eight-weight block stores a little-endian uint16 index and FP32 scale.
An immutable caller codebook and orthogonal eight-lane transform define its
execution identity. Transformed weights consume correspondingly rotated input.
"""

import hashlib
import itertools
import math
import struct
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from .expert_kernels import (
    ExpertPayloadSource, KernelBackendInfo, KernelCapabilities, KernelEvidence,
    _finite_array, _finite_matrix, _lease, _validate_routes,
)
from .lowbit_expert import _read_f32
from .q5_0_expert import _detach_borrowed_frames
from .q8_0_expert import _positive

DTYPE = "e8-lattice-caller-v1"
_RECORD = struct.Struct("<Hf")


def root_codebook():
    """Return zero and the 240 length-squared-two roots in a stable order."""
    roots = [np.zeros(8, np.float32)]
    for left, right in itertools.combinations(range(8), 2):
        for a, b in itertools.product((-1., 1.), repeat=2):
            point = np.zeros(8, np.float32)
            point[left], point[right] = a, b
            roots.append(point)
    for signs in itertools.product((-1., 1.), repeat=8):
        if sum(sign < 0 for sign in signs) % 2 == 0:
            roots.append(np.asarray(signs, np.float32) * .5)
    return np.asarray(roots, np.float32)


def hadamard_transform():
    """Normalized Sylvester order-eight transform, independently constructed."""
    matrix = np.ones((1, 1), np.float64)
    for _ in range(3):
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return (matrix / math.sqrt(8)).astype(np.float32)


@dataclass(frozen=True, init=False)
class LatticeContract:
    codebook: np.ndarray
    transform: np.ndarray
    identity: str

    def __init__(self, codebook, transform):
        book = np.asarray(codebook)
        rotation = np.asarray(transform)
        if book.ndim != 2 or book.shape[1] != 8 or not 1 <= book.shape[0] <= 65536 or book.dtype.kind not in "iuf":
            raise ValueError("codebook must contain 1..65536 real eight-coordinate points")
        if rotation.shape != (8, 8) or rotation.dtype.kind not in "iuf":
            raise ValueError("transform must be a real 8x8 matrix")
        with np.errstate(over="ignore", invalid="ignore"):
            book = book.astype("<f4")
            rotation = rotation.astype("<f4")
        if not np.isfinite(book).all() or not np.isfinite(rotation).all():
            raise ValueError("contract values must be finite float32")
        doubled = book.astype(np.float64) * 2
        if (np.abs(doubled) > 2 ** 24).any() or not np.equal(doubled, np.rint(doubled)).all():
            raise ValueError("codebook coordinates must be bounded integers or half-integers")
        parity = np.remainder(doubled.astype(np.int64), 2)
        if not np.equal(parity, parity[:, :1]).all() or not np.equal(np.remainder(doubled.sum(axis=1), 4), 0).all():
            raise ValueError("codebook points must lie on the even E8 lattice")
        if len(np.unique(book, axis=0)) != len(book):
            raise ValueError("codebook points must be unique")
        orthogonal = rotation.astype(np.float64) @ rotation.astype(np.float64).T
        if not np.allclose(orthogonal, np.eye(8), rtol=0, atol=1e-6):
            raise ValueError("activation transform must be orthogonal")
        # Immutable byte owners prevent callers changing a hashed execution table.
        object.__setattr__(self, "codebook", np.frombuffer(book.tobytes(), dtype="<f4").reshape(book.shape))
        object.__setattr__(self, "transform", np.frombuffer(rotation.tobytes(), dtype="<f4").reshape(8, 8))
        digest = hashlib.sha256(DTYPE.encode("ascii"))
        digest.update(struct.pack("<I", len(book)))
        digest.update(self.codebook.tobytes())
        digest.update(self.transform.tobytes())
        object.__setattr__(self, "identity", digest.hexdigest())

    def metadata(self):
        return {"dtype": DTYPE, "contract_sha256": self.identity,
                "codebook_rows": len(self.codebook), "vector_width": 8,
                "record": "uint16-le-index,float32-le-scale", "activation_transform": "orthogonal-8x8"}

    def write(self, path):
        """Atomically persist the exact caller codebook and transform."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (b"OAE8CT01" + struct.pack("<I", len(self.codebook))
                   + bytes.fromhex(self.identity) + self.codebook.tobytes()
                   + self.transform.tobytes())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def read(cls, path, *, expected_identity):
        """Read a bounded artifact and require the manifest's contract digest."""
        if not _digest(expected_identity):
            raise ValueError("expected_identity must be a lowercase SHA256 digest")
        limit = 44 + 65536 * 8 * 4 + 8 * 8 * 4
        with Path(path).open("rb") as stream:
            payload = stream.read(limit + 1)
        if len(payload) < 44 or len(payload) > limit or payload[:8] != b"OAE8CT01":
            raise ValueError("invalid lattice contract artifact")
        rows = struct.unpack_from("<I", payload, 8)[0]
        if not 1 <= rows <= 65536 or len(payload) != 44 + rows * 32 + 256:
            raise ValueError("invalid lattice contract extent")
        if payload[12:44].hex() != expected_identity:
            raise ValueError("lattice contract artifact identity differs from manifest")
        book = np.frombuffer(payload, dtype="<f4", count=rows * 8, offset=44).reshape(rows, 8)
        rotation = np.frombuffer(payload, dtype="<f4", count=64, offset=44 + rows * 32).reshape(8, 8)
        contract = cls(book, rotation)
        if contract.identity != expected_identity:
            raise ValueError("lattice contract artifact checksum mismatch")
        return contract


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_contract_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("E8 expert layers require explicit contract metadata")
    rows = metadata.get("codebook_rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= 65536:
        raise ValueError("invalid E8 codebook row count")
    expected = {"dtype": DTYPE, "contract_sha256": metadata.get("contract_sha256"),
                "codebook_rows": rows, "vector_width": 8,
                "record": "uint16-le-index,float32-le-scale", "activation_transform": "orthogonal-8x8"}
    if metadata != expected or not _digest(expected["contract_sha256"]):
        raise ValueError("invalid E8 lattice contract metadata")


def _shapes(hidden, intermediate):
    _positive(hidden, "hidden_size")
    _positive(intermediate, "intermediate_size")
    if hidden % 8 or intermediate % 8:
        raise ValueError("E8 row widths must be multiples of eight")
    return ((intermediate, hidden), (intermediate, hidden), (hidden, intermediate))


def payload_size(hidden_size, intermediate_size):
    _shapes(hidden_size, intermediate_size)
    return 3 * hidden_size * intermediate_size // 8 * 6


def pack_matrix(matrix, contract):
    if not isinstance(contract, LatticeContract):
        raise TypeError("an explicit LatticeContract is required")
    weights = np.asarray(matrix)
    if weights.ndim != 2 or weights.dtype.kind not in "iuf":
        raise ValueError("weights must be real matrices")
    rows, columns = weights.shape
    _positive(rows, "rows")
    _positive(columns, "columns")
    if columns % 8:
        raise ValueError("E8 row widths must be multiples of eight")
    result = bytearray(rows * columns // 8 * 6)
    book = contract.codebook.astype(np.float64)
    norms = np.sum(book * book, axis=1)
    cursor = 0
    for row in weights:
        for start in range(0, columns, 8):
            values = row[start:start + 8].astype(np.float64)
            if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
                raise ValueError("weights must be finite float32")
            target = values @ contract.transform.astype(np.float64).T
            dots = book @ target
            scales = np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0)
            scales = np.maximum(scales, 0)
            errors = np.sum((book * scales[:, None] - target) ** 2, axis=1)
            index = int(np.argmin(errors))
            with np.errstate(over="ignore"):
                scale = np.float32(scales[index])
                decoded = (book[index] * float(scale)).astype(np.float32)
            if not np.isfinite(scale) or not np.isfinite(decoded).all():
                raise ValueError("quantized E8 weights exceed float32 range")
            _RECORD.pack_into(result, cursor, index, float(scale))
            cursor += 6
    return bytes(result)


def pack_expert(gate, up, down, contract):
    matrices = [np.asarray(matrix) for matrix in (gate, up, down)]
    if matrices[0].ndim != 2:
        raise ValueError("gate must be a matrix")
    intermediate, hidden = matrices[0].shape
    if any(matrix.shape != shape for matrix, shape in zip(matrices, _shapes(hidden, intermediate))):
        raise ValueError("expert projection shape mismatch")
    return b"".join(pack_matrix(matrix, contract) for matrix in matrices)


def _rotated_dot(raw, offset, vector, width, book, rotation):
    total = np.float32(0)
    for group in range(width // 8):
        position = offset + group * 6
        index = int(raw[position]) | int(raw[position + 1]) << 8
        scale = _read_f32(raw, position + 2)
        for lane in range(8):
            rotated = np.float32(0)
            for column in range(8):
                rotated = np.float32(rotated + np.float32(rotation[lane, column] * vector[group * 8 + column]))
            weight = np.float32(book[index, lane] * scale)
            total = np.float32(total + np.float32(weight * rotated))
    return total


def _kernel(raw, vector, hidden, intermediate, coefficient, activation, output, book, rotation):
    for position in range(0, raw.size, 6):
        index = int(raw[position]) | int(raw[position + 1]) << 8
        if index >= book.shape[0] or not math.isfinite(_read_f32(raw, position + 2)):
            return 1
        scale = _read_f32(raw, position + 2)
        for lane in range(8):
            if not math.isfinite(np.float32(book[index, lane] * scale)):
                return 2
    area = hidden * intermediate // 8 * 6
    stride = hidden // 8 * 6
    for row in range(intermediate):
        gate = _rotated_dot(raw, row * stride, vector, hidden, book, rotation)
        up = _rotated_dot(raw, area + row * stride, vector, hidden, book, rotation)
        if not math.isfinite(gate) or not math.isfinite(up):
            return 2
        exponential = math.exp(-abs(float(gate)))
        sigmoid = np.float32((1.0 if gate >= 0 else exponential) / (1.0 + exponential))
        activation[row] = np.float32(np.float32(gate * sigmoid) * up)
        if not math.isfinite(activation[row]):
            return 2
    stride = intermediate // 8 * 6
    for row in range(hidden):
        value = _rotated_dot(raw, 2 * area + row * stride, activation, intermediate, book, rotation)
        output[row] = np.float32(output[row] + np.float32(value * coefficient))
        if not math.isfinite(value) or not math.isfinite(output[row]):
            return 2
    return 0


class E8LatticeExpertBackend:
    def __init__(self, contract, *, max_scratch_bytes=64 * 1024 * 1024):
        if not isinstance(contract, LatticeContract):
            raise TypeError("an explicit LatticeContract is required")
        self._contract = contract
        self.max_scratch_bytes = _positive(max_scratch_bytes, "max_scratch_bytes")
        self.NAME = DTYPE + "-" + contract.identity + "-reference"

    @property
    def contract(self):
        return self._contract

    def info(self):
        return KernelBackendInfo(name=self.NAME, provider="openagent", available=True, priority=-12000,
            capabilities=KernelCapabilities(frozenset({DTYPE}), True, True),
            evidence=(KernelEvidence("contract", self.contract.identity),))

    def _executor(self):
        return _kernel

    def execute(self, source, inputs, routes, *, hidden_size, intermediate_size, dtype=DTYPE):
        vector = _finite_array(inputs, (hidden_size,), "inputs")
        return self.execute_batch(source, vector[None], [routes], hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, dtype=dtype)[0]

    def execute_batch(self, source, inputs, route_batch, *, hidden_size, intermediate_size, dtype=DTYPE):
        executor = self._executor()
        if dtype != DTYPE:
            raise ValueError("unsupported caller E8 dtype")
        length = payload_size(hidden_size, intermediate_size)
        if intermediate_size * 4 > self.max_scratch_bytes:
            raise MemoryError("E8 activation exceeds scratch budget")
        if not isinstance(source, ExpertPayloadSource):
            raise TypeError("source must implement acquire(key)")
        vectors = np.ascontiguousarray(_finite_matrix(inputs, hidden_size))
        routing = list(route_batch)
        if len(routing) != len(vectors):
            raise ValueError("one route list required per input row")
        routing = [_validate_routes(row, "routes") for row in routing]
        if any(weight > float(np.finfo(np.float32).max) for routes in routing for _, weight in routes):
            raise ValueError("route weight must be representable as float32")
        output = np.zeros_like(vectors)
        activation = np.empty(intermediate_size, np.float32)
        for row, routes in enumerate(routing):
            for key, coefficient in routes:
                failure = None
                with _lease(source, key) as payload:
                    if payload.nbytes != length:
                        raise ValueError("E8 payload size mismatch")
                    raw = np.frombuffer(payload, np.uint8)
                    try:
                        with np.errstate(over="ignore", invalid="ignore"):
                            status = executor(raw, vectors[row], hidden_size, intermediate_size,
                                              np.float32(coefficient), activation, output[row],
                                              self.contract.codebook, self.contract.transform)
                    except BaseException as error:
                        _detach_borrowed_frames(error)
                        failure = error
                    finally:
                        del raw
                if failure is not None:
                    raise failure from None
                if status == 1:
                    raise ValueError("E8 index or scale metadata is invalid")
                if status:
                    raise ArithmeticError("E8 computation or decoded weight is non-finite")
        return output
