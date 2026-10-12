# Q8_0 Expert Execution

Q8_0 is an explicit payload choice, not automatic model conversion. The kernel
registry includes `Q8_0ExpertBackend` as `q8_0-numpy`, priority `-11000`, supporting
only `q8_0`. Selecting a Q8_0 manifest/dtype opts in. It cannot replace the existing
FP16, FP32, row INT8 or row INT4 defaults. Execution uses NumPy FP32 operations;
the backend declares neither compiled execution nor native SIMD.

## Byte Contract

The block metadata comes from
[GGML's block declarations](https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h),
not its quantizer or kernel implementation. Each little-endian block is 34 bytes:
one IEEE binary16 scale followed by 32 signed INT8 codes. Decoding multiplies each
code by the scale in FP32. All signed codes, including -128, are valid. Scales may
be negative or zero but must be finite. There are no reserved codes or checksums
inside a block.

An expert concatenates row-major gate `[intermediate, hidden]`, up
`[intermediate, hidden]`, and down `[hidden, intermediate]`. No header, padding,
alignment gaps, row scales or trailing metadata are inserted. Both dimensions
must be positive integers divisible by 32. Exact expert length is:

```text
3 * hidden * intermediate / 32 * 34 bytes
```

Sparse manifests enforce this exact size and dimension constraint. Dimensions
are trusted metadata: equal-size dimension permutations cannot be detected from
bytes alone. DiskExpertStore supplies its existing region SHA256 and immutable-file
checks. The backend checks every scale before any projection arithmetic, including
experts routed with zero weight. Overflow in projections, activation, or routed
output raises `ArithmeticError`; malformed payloads raise `ValueError`.

## Matrix Converter

The public converter accepts existing NumPy-compatible matrix inputs; it does not
read GGUF, SafeTensors or model checkpoints. Example with caller-owned `gate`, `up`
and `down` matrices:

```python
from openagent.adaptive_engine.q8_0_conversion import convert_experts
from openagent.adaptive_engine.expert_runtime import SparseExpertRuntime

manifest = convert_experts(
    {"0": (gate, up, down)}, "experts.q8",
    hidden_size=gate.shape[1], intermediate_size=gate.shape[0],
    model_id="my-model", architecture="my-adapter",
    fingerprint="caller-supplied-source-identity",
)
manifest.write("experts.q8.json")
size = manifest.layers[0].regions["0"].size
with SparseExpertRuntime(
    ".", manifest, capacity_bytes=size, preferred_kernel="q8_0-numpy",
) as runtime:
    output = runtime.execute("block.0", hidden_vector, [("0", 1.0)])
```

The returned manifest uses paths relative to the container's parent directory;
use that directory as the runtime root. Caller identity metadata is not generated
or asserted to authenticate the source. The converter supports one layer per call,
writes consecutive expert regions with individual SHA256 hashes, and validates
the manifest before publishing the container. It refuses to overwrite an existing
path. A same-directory temporary file is flushed and published by a hard link;
filesystems without hard-link support raise an error. Failed conversion removes
its temporary file. JSON publication is a separate `manifest.write` call, not a
transaction spanning container and manifest files.

Packing uses FP32 inputs, binary16-rounded `max(abs(block))/127` scales, then
ties-to-even codes clipped to [-127,127]. Zero blocks have zero scale and codes;
nonzero scales that round to zero are floored to the smallest binary16 subnormal.
Scale overflow, non-real/nonfinite weights and FP32-unrepresentable values are
rejected. Clipping after scale rounding means error can exceed scale/2; tests
check both rounding and clipping allowances. This is an independent packing
policy, not a claim of byte parity with an upstream quantizer.

## SafeTensors And OLMoE Integration

Separately from the matrix converter, the existing SafeTensors expert packer and
OLMoE pack command accept `dtype="q8_0"` / `--dtype q8_0`. INT8 remains the
unchanged default. The Q8_0 command defaults to distinct files
`openagent-experts.q8` and `openagent-experts.q8.manifest.json` in the model
directory, keeping its container and manifest separate from INT8/INT4 artifacts.

```sh
openagent-olmoe pack MODEL --dtype q8_0
openagent-olmoe generate MODEL MODEL/openagent-experts.q8.manifest.json "Hello" --kernel q8_0-numpy
openagent-olmoe generate MODEL MODEL/openagent-experts.q8.manifest.json "Hello" --kernel q8_0-numba
openagent-olmoe verify MODEL MODEL/openagent-experts.q8.manifest.json oracle.json --kernel q8_0-numba
```

The CLI explicitly registers the optional
[Numba direct-byte backend](ENGINE_Q8_0_NATIVE.md) when `--kernel q8_0-numba` is
requested. `q8_0-numpy` uses the portable registered backend. No dependency is
downloaded automatically. Full `OlmoeModel` execution supports Q8_0 manifests;
full-model saved state binds expert content hashes and sizes, dimensions, dtype,
selected backend, and dense-weight identity. Cross-content, precision or backend
restore is rejected before KV mutation. Segment execution and its snapshot ABI
remain INT8-only and reject Q8_0 manifests; full-model support does not imply
segment support.

## Public Checkpoint Smoke Check

The reported local integration run used
`allenai/OLMoE-1B-7B-0125-Instruct` at revision
`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`. Conversion produced 1,024 experts
in a 6,845,104,128-byte Q8_0 container and took 116.823 seconds. The
`q8_0-numba` run passed the four-token public-checkpoint oracle in 25.291 seconds,
including loading and initial JIT compilation. This documents one smoke result,
not general token parity, preserved model quality, a controlled benchmark or a
speed comparison. These checkpoint observations were supplied by the integration
run, not rerun by the native-kernel fixture tests.

## Memory And Verification

`block_count` bounds each decoded block group to `32 * block_count` weights.
Execution scratch is O(32 * block_count + hidden + intermediate), independent of
expert count and batch length, excluding caller input/output, routing metadata,
borrowed expert bytes and NumPy/BLAS internal workspace. Only one expert lease is
active at a time; no weight cache or full-matrix decode is used during execution.
Disk-store capacity bounds its resident payloads separately; mmap mode does not
bound OS virtual mappings or page cache. Conversion holds one packed expert at a
time plus caller matrices and packing temporaries. `decode_expert` explicitly
allocates full decoded matrices and is only a convenience API.

Targeted tests:

```powershell
.venv/Scripts/python.exe -W error -m unittest tests.test_q8_0_expert tests.test_q8_0_integration -v
```

The tests include hand-constructed signed codes/scales, independent struct-based
decoding, routed FP32 differential checks, bounded block sizes, mmap lease cleanup,
exact manifest sizes, converter failure cleanup, and real buffered/mmap
DiskExpertStore execution with a one-expert cache and eviction. Checksum mismatch
rejection is tested in buffered mode; malformed scales are tested in both modes.
These are fixture-level numerical and storage checks, not upstream execution
parity, model-quality measurements or performance certification.

## Remaining Gaps

- No GGUF importer or implicit dimension padding.
- Segment execution/snapshot ABI still rejects Q8_0; only full-model state supports it.
- No accelerator validation or controlled benchmark; the optional CPU kernel is documented separately.
- No full-model quantized-quality evaluation or upstream quantizer parity test.

The existing [row INT4 packed format](ENGINE_PACKED_QUANTIZATION.md) remains a
separate byte contract.
