"""Source-inspected prerequisites, not decoder execution certificates.

Command-based full-model comparisons are opt-in and require complete evidence.
Publisher mappings cannot be promoted by extension exit codes.
"""

BACKENDS = ("linux", "cuda", "hip", "metal")
DECODERS = {
    "qwen35": {
        "candidate": "openagent.adaptive_engine.torch_qwen35_decoder.TorchQwen35Decoder",
        "loading_gap": "Complete device decoder execution requires pinned real-backend comparison evidence.",
    },
    "qwen4": {
        "candidate": "openagent.adaptive_engine.torch_qwen4_decoder.TorchQwen4Decoder",
        "loading_gap": "Hybrid device state and lexical continuation require whole-model comparison evidence.",
    },
    "glm-dsa": {
        "candidate": "openagent.adaptive_engine.torch_glm_dsa_decoder.TorchGlmDsaDecoder",
        "loading_gap": "Resident accelerator DSA execution requires pinned publisher and real-backend comparison evidence.",
    },
    "glm5-next": {
        "candidate": "openagent.adaptive_engine.torch_glm5_decoder.TorchGlm5NextDecoder",
        "loading_gap": "Bounded native family contract requires exact publisher and backend admission.",
    },
    "inkling": {
        "candidate": "openagent.adaptive_engine.torch_inkling_decoder.TorchInklingDecoder",
        "loading_gap": "Native CPU continuation does not establish publisher or accelerator certification.",
    },
    "mixtral": {
        "candidate": "openagent.adaptive_engine.torch_mixtral_decoder.TorchMixtralDecoder",
        "loading_gap": "Complete pinned publisher checkpoint admission is not certified here.",
    },
    "kimi-k3": {
        "candidate": "openagent.adaptive_engine.k3_execution.K3Execution",
        "loading_gap": "Publisher-shaped K3 loading and execution have not been compared on this backend.",
    },
    "deepseek-v41": {
        "candidate": "openagent.adaptive_engine.v41_execution.DevicePublisherV41Execution",
        "loading_gap": "Publisher V4.1 device arithmetic requires pinned whole-model and placement evidence.",
    },
}
LOADING_PREREQUISITES = (
    "Pinned local source, config, tokenizer, every weight shard and dependency identity",
    "Exact complete tensor inventory, shapes, dtypes and quantization scale/layout contracts",
    "Executable decoder admission with no missing tensors, subset loading or implicit fallback",
    "Observed requested runtime/device identity and placement for every decoder operation",
    "Memory/capacity admission and deterministic cleanup on loading or execution failure",
)
CPU_CANDIDATES = {
    'deepseek-v41':'openagent.adaptive_engine.v41_execution.PublisherV41Execution',
    'glm-dsa':'openagent.adaptive_engine.native_glm_dsa.NativeGlmDsaDecoder',
    'mixtral':'openagent.adaptive_engine.native_mixtral_decoder.NativeMixtralDecoder',
    'qwen35':'openagent.adaptive_engine.native_qwen35.NativeQwen35TextDecoder',
    'qwen4':'openagent.adaptive_engine.native_qwen4.NativeQwen4TextDecoder',
    'glm5-next':'openagent.adaptive_engine.native_glm5_next.NativeGlm5NextTextDecoder',
    'inkling':'openagent.adaptive_engine.native_inkling.NativeInklingDecoder',
}
COMPARISON_PREREQUISITES = (
    "Independent unchanged authoritative oracle and candidate for identical checkpoint pins",
    "Finite full-prefix logits from the complete token-to-logit path with explicit numerical tolerances",
    "Tokenwise and chunked incremental logits compared with fresh full-prefix execution",
    "Complete canonical cache leaves, positions and layer identities, not replay alone",
    "Greedy generation parity with fixed tokens, seed and positive generation steps",
    "Synchronized backend execution and placement evidence; CPU fallback is rejected",
    "No skipped, expected-failure, empty or missing mandatory comparisons",
)


def certification_matrix(backend):
    """Return fresh JSON-compatible unavailable rows without importing decoders."""
    if backend not in BACKENDS:
        raise ValueError("unknown decoder certification backend")
    return [
        {
            "decoder": name,
            "backend": backend,
            "candidate": CPU_CANDIDATES.get(name,entry['candidate']) if backend=='linux' else entry['candidate'],
            "status": "unavailable",
            "executed": False,
            "certified": False,
            "loading_prerequisites": list(LOADING_PREREQUISITES),
            "comparison_prerequisites": list(COMPARISON_PREREQUISITES),
            "gaps": [entry["loading_gap"],
                     "An explicit pinned candidate/oracle command manifest and validated placement evidence are required.",
                     "No real-machine full-model evidence has been collected by this runner."],
        }
        for name, entry in DECODERS.items()
    ]
