# Native Qwen3.5 Multimodal Blocks

`NativeQwen35Vision` remains in `openagent/adaptive_engine/native_qwen35_vision.py`.
Its constructor is unchanged: `(config, tensors, *, library, text_library,
max_bytes=256*1024*1024)`. There are no alternate wrappers or model aliases.
The public package exports the decoder and an exact-wrapper SafeTensors loader.

## Supported Boundary

This is a bounded synthetic FP32 multimodal implementation, not general
pretrained Qwen3.5 checkpoint support. It accepts the exact untied
`qwen3_5_moe` wrapper with text width 24 and vision geometry documented in
ENGINE_NATIVE_QWEN35_VISION.md, except vision depth now accepts integers 0-4.
Batch one, one initial image, processed patches `[4,3]`, and grid `[1,2,2]`
remain required. No raw-image processing, video, quantization, other geometry,
multiple images, or images introduced during cached decoding are supported.

Each real block executes original native C pre-attention LayerNorm, biased
QKV projection, two-axis vision rotary encoding, bidirectional per-head
scaled dot-product softmax attention, biased output projection and residual,
second LayerNorm, biased tanh-approximate GELU MLP and residual. The existing
merger uses exact erf GELU. All vision model arithmetic is in C, without
Torch, Transformers or NumPy arithmetic delegation at runtime. Oracle and
reference implementation source was not copied or translated.

The ordered C tensor list retains the nine existing patch/position/merger
tensors, followed by twelve tensors per block: norm1 weight/bias, norm2
weight/bias, attention QKV weight/bias, attention projection weight/bias,
MLP fc1 weight/bias, and MLP fc2 weight/bias. Counts are exactly `9+12*depth`;
the existing ABI signature/version remains compatible with depth zero.
Constructor validation requires the exact configured block names, extents,
finite FP32 values and immutable pinned ownership.

One merged image token has identical T/H/W positions, so the existing scalar
text RoPE is exact for this boundary, with zero cached position delta.
Prefill image substitution, candidate commit only after success, original
embedding restoration on reset, shared DLL lifetime and cached text decoding
remain unchanged. C validates all tensor extents and finite inputs before
execution and commits the vision output only when every output is finite.

## Local Checkpoint Loading

```python
from openagent.adaptive_engine import load_native_multimodal_decoder

decoder = load_native_multimodal_decoder(
    "local-checkpoint", model_type="qwen3_5_moe",
    library="native-vision.dll", text_library="native-text.dll",
    max_weight_bytes=64 * 1024**2, max_bytes=256 * 1024**2,
)
```

Both libraries must be built locally from the corresponding native sources.
The loader performs no download. It validates the exact wrapper, shard inventory,
tensor ranges and memory admission before constructing the bounded decoder.
Other wrappers are rejected, not reinterpreted as text checkpoints.

```text
Local config + SafeTensors shards
    -> exact identity and tensor validation
    -> native vision blocks and merger
    -> image embedding substitution
    -> native text decoder -> logits and cached text state
```

## Test Commands

Run with the already-installed official Transformers 5.16.1 black-box oracle:

```powershell
$env:PYTHONPATH='D:\openagent\.tmp-native-decoder-oracle;D:\openagent'
& .accelerator-venv/Scripts/python.exe -m unittest tests.test_native_qwen35_vision -v
```

Verified seven tests on Windows, compiling C with `/W4 /WX /O2 /fp:precise`.
The default fixture now has one real transformer block. It compares image
features, prefill logits/hidden states/attention, three cached text steps,
and exact official T/H/W position indices and zero delta. Additional feature
comparisons cover depths zero, two and four. Comparisons use tolerances
(`atol=2e-6, rtol=5e-5` for vision), not bit-identical floating-point claims.
Tests also demonstrate active transformer weights, malformed/null block
tensor rejection, nonfinite numerical output rollback, cache/candidate
rollback, reset/new-image/pure-text parity, immutable weights and DLL lifetime.
Compiler absence is a skip, not certification. Unix strict C99 compilation
is configured in the tests but was not executed in this Windows run.
