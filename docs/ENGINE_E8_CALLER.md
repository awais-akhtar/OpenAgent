# Caller-Configured E8 Experts

This is an owned vector-quantization API, not the reference IQ3 codec or an
upstream checkpoint format. Its dtype is `e8-lattice-caller-v1`. A caller
supplies the codebook and the activation transform. The backend name includes
the SHA256 identity of those tables, so a registry cannot substitute another
configuration silently.

## Mathematical Contract

The [E8 lattice definition](https://math.ucr.edu/home/baez/week193.html)
is the union of even-sum integer points and the half-integer shift of that
lattice. The implementation validates that every eight-coordinate codebook
point has either all integer or all half-integer coordinates and that the
coordinate sum is even. Duplicate, non-finite and off-lattice points are
rejected. The codebook contains between 1 and 65,536 points.

`root_codebook()` constructs zero and the 240 norm-squared-two roots directly
from that definition: 112 signed coordinate-pair roots and 128 half-integer
sign patterns with even parity. This is a small demonstrator codebook, not a
reconstruction of an upstream IQ3 index table. `hadamard_transform()` constructs
a normalized order-eight Sylvester matrix. A caller can provide a different
8-by-8 transform, but it must be orthogonal within an absolute tolerance of
1e-6 after conversion to FP32.

`LatticeContract` owns immutable byte-backed FP32 arrays. Its hash covers the
format version, codebook length, ordered codebook bytes and transform bytes.
Changing point order or transform changes the identity. The backend's public
contract property cannot be reassigned.

## Layout and Execution

Each eight-weight row group occupies six bytes:

| Offset | Field |
| --- | --- |
| 0 | Little-endian uint16 codebook index |
| 2 | Little-endian FP32 scale |

Gate/up shapes are `[intermediate, hidden]`; down is `[hidden, intermediate]`.
Concatenate their row-major records without headers or padding. Both row
widths must be multiples of eight. The payload length is
`3*hidden*intermediate/8*6`. The codebook and transform are external metadata,
not embedded in this payload.

```text
Caller codebook + orthogonal transform
                |
          Contract validation
                |
    Immutable tables + content hash
                |
Weights -> rotate each eight-lane group -> fit scale/codebook index
                |
         Six-byte group records
                |
       Leased expert payload
                |
Input -> matching activation rotation -> direct codebook dot products
                |
              SwiGLU
```

Packing rotates a row group as `weights @ transform.T`, selects the codebook
point with least squared error after fitting a nonnegative scale, and stores
that scale as FP32. Execution rotates the activation as `transform @ input`.
With exact orthogonality, these transformations cancel in the dot product.
FP32 rounding and quantization still change numerical results; no unchanged
model-quality guarantee is made.

Execution validates every index, scale and reconstructed weight before
computing, including zero-weight routes. It borrows one expert at a time and
reuses one FP32 intermediate activation vector. Rotated activations are
accumulated in scalar temporaries, not full matrices. Contract tables are
resident and are not counted as activation scratch. Packing separately
allocates caller-sized output and codebook-sized fitting arrays.

## API

```python
from openagent.adaptive_engine.e8_lattice_expert import (
    E8LatticeExpertBackend, LatticeContract, root_codebook,
    hadamard_transform, pack_expert,
)
from openagent.adaptive_engine.expert_kernels import MappingPayloadSource

configuration = LatticeContract(root_codebook(), hadamard_transform())
payload = pack_expert(gate, up, down, configuration)
backend = E8LatticeExpertBackend(configuration)
result = backend.execute(
    MappingPayloadSource({"expert": payload}), hidden, [("expert", 1.0)],
    hidden_size=hidden_size, intermediate_size=intermediate_size,
)
metadata = configuration.metadata()
```

`NativeE8LatticeExpertBackend` in `native_e8_lattice_expert.py` uses actual
Numba nopython CPU execution over the same bytes and tables. Missing or disabled
Numba refuses execution; it does not fall back to the scalar kernel. The
registry requires the exact contract-bound backend name and disables fallback
for this dtype.

## Validation and Boundaries

`tests/test_e8_lattice_expert.py` compares hand-built records against a separate
struct decoder that reconstructs effective FP32 matrices. Tests cover signed
scales, non-square projections, transform application, root geometry,
immutable identities, malformed indices, invalid scales, arithmetic overflow,
scratch limits, actual nopython execution, disabled JIT and retained chained
exceptions over real mmap leases. Compiled-loop inspection checks that no
Numba-managed numeric arrays are allocated inside execution.

This caller codec is not accepted by persisted `ExpertLayer` manifests, the
SafeTensors model packer, OLMoE snapshots or the model gateway. Supporting those
paths requires serializing and validating the exact codebook and transform
contract alongside each model, not accepting arbitrary caller bytes under a
generic dtype.

The [reviewed format metadata](https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/docs/FORMATS.md)
describes an E8/IQ3 format with 98-byte, 256-weight super-blocks, parity signs,
binary16 super-scales and a block-diagonal FWHT activation rotation. Exact
compatibility still needs a normative ordered index table, bit offsets and
sign/parity interpretation, complete scale/sub-block equations, and the exact
rotation ordering and normalization. The six-byte caller records above do
not implement those contracts. Neither an IQ3 compression ratio nor a
reference-compatibility or GPU-speed claim follows from this API.
