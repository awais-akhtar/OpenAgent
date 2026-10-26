"""Kind-specific Phase 1 runtime observations and their verification.

A P13/P15 receipt for one exact canonical route is accepted only when every
observation file is pinned, distinct, bound to that route and to the pinned
checkpoint config, and when its *measurements* agree with each other:

* ``tokens``     greedy prompt/continuation with an EOS or length finish.
* ``logits``     one FP32 vocabulary row per generated token; each row's argmax
                 must equal the generated token.
* ``comparison`` an independent reference implementation's tokens and logits;
                 the gate recomputes the maximum absolute logit error itself and
                 requires identical tokens.
* ``cache``      the physical cache position must equal the decoded positions.
* ``routing``    one expert-ID row per processed position and layer, with the
                 config's experts-per-token and expert range.
* ``recovery``   a snapshot/restore replay must reproduce the continuation.
* ``placement``  the measured device must match the route backend with zero
                 host-fallback operations.

Coverage checks (media, speculation, distributed, format/device, ABI/recovery)
are accepted as ``not-applicable`` only where the canonical route itself makes
them inapplicable; applicable checks need their own kind-specific evidence.

Claimed booleans such as ``numerical_parity_verified`` are never read.

Limits (what this gate cannot prove on its own):

* The comparison reference must be the family's reviewed in-repository
  implementation (``REFERENCE_SOURCES``), and its SHA-256 must equal the file
  currently in the repository, so evidence from an unknown reference or from
  older reference code is rejected.  The gate still cannot prove the reference
  was actually executed rather than its output fabricated; that rests on the
  bundled capture runner (``phase1_capture``) being the producer.
* Placement for host routes (native/public-range) is CPU by construction.
  For device routes the gate checks consistency only; a device placement
  receipt must come from instrumented execution, which no runner produces yet.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

OBSERVATION_SCHEMA = "openagent.phase1-observation.v1"
RECEIPTS_SCHEMA = "openagent.phase1-runtime-receipts.v1"
OBSERVATION_KINDS = ("tokens", "logits", "cache", "routing", "recovery", "comparison", "placement")
COVERAGE_REQUIREMENTS = ("media_lifecycle", "speculative_state", "distributed_recovery",
                         "format_device", "abi_recovery")
# Upper bound any receipt may declare for |candidate - reference| on logits.
MAX_LOGIT_ABS_TOLERANCE = 1e-2
MAX_OBSERVATION_BYTES = 512 * 1024 * 1024
# Checkpoint artifacts are hash-streamed from disk and are not parsed as
# receipt payloads.  Give them a separate bound so normal multi-gigabyte model
# shards can be pinned without allowing unbounded paths from a receipt.
MAX_ARTIFACT_BYTES = 128 * 1024**3
HOST_BACKENDS = ("native", "public-range")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Reviewed independent numerical references, including the in-repository
# helpers they import.  Pinning the complete local dependency closure prevents
# a receipt from remaining valid after a shared equation (for example RoPE or
# RMSNorm) changes while the top-level decoder file does not.  These entries
# make a family eligible for comparison evidence; they do not create runtime
# evidence or certify a route.
REFERENCE_SOURCES = {
    "olmoe": ("openagent/adaptive_engine/olmoe.py",),
    "mixtral": ("openagent/adaptive_engine/native_mixtral.py",),
    "qwen35": ("openagent/adaptive_engine/independent_qwen35.py",
               "openagent/adaptive_engine/native_mixtral.py"),
    "qwen4": ("openagent/adaptive_engine/independent_qwen4.py",
              "openagent/adaptive_engine/independent_qwen35.py",
              "openagent/adaptive_engine/native_mixtral.py"),
    "inkling": ("openagent/adaptive_engine/independent_inkling.py",),
    "glm-dsa": ("openagent/adaptive_engine/independent_glm_dsa.py",
                "openagent/adaptive_engine/independent_qwen35.py",
                "openagent/adaptive_engine/native_mixtral.py"),
    # GLM5-next's public independent contract intentionally has no numerical
    # forward implementation. Compare its scalar C candidate against the
    # separately implemented Torch device equations and pin the complete
    # numerical-reference source closure instead of treating the contract
    # owner as an oracle.
    "glm5-next": ("openagent/adaptive_engine/torch_glm5_decoder.py",
                  "openagent/adaptive_engine/device_glm5_forward.py",
                  "openagent/adaptive_engine/device_publisher_projections.py",
                  "openagent/adaptive_engine/device_qwen35_layers.py",
                  "openagent/adaptive_engine/device_logits.py",
                  "openagent/adaptive_engine/device_execution_policy.py",
                  "openagent/adaptive_engine/torch_inkling_decoder.py",
                  "openagent/adaptive_engine/device_inkling_forward.py",
                  "openagent/adaptive_engine/independent_inkling.py",
                  "openagent/adaptive_engine/independent_glm5_next.py"),
}
REFERENCE_IMPLEMENTATIONS = {
    "olmoe": "openagent.olmoe.OlmoeModel (NumPy reference)",
    "mixtral": "openagent.adaptive_engine.native_mixtral.IndependentMixtralDecoder (NumPy reference)",
    "qwen35": "openagent.adaptive_engine.independent_qwen35.IndependentQwen35TextDecoder (NumPy reference)",
    "qwen4": "openagent.adaptive_engine.independent_qwen4.IndependentQwen4TextDecoder (NumPy reference)",
    "inkling": "openagent.adaptive_engine.independent_inkling.IndependentInklingDecoder (NumPy reference)",
    "glm-dsa": "openagent.adaptive_engine.independent_glm_dsa.IndependentGlmDsaDecoder (NumPy reference)",
    "glm5-next": "openagent.adaptive_engine.torch_glm5_decoder.TorchGlm5NextDecoder (independent Torch reference)",
}

# Device GLM5 execution is compared with the separately compiled scalar C
# decoder.  This is intentionally a route-specific comparator: using the same
# Torch implementation on both sides would only prove repeatability, not
# numerical agreement between implementations.
_DEVICE_REFERENCE_SOURCES = {
    "glm5-next": ("openagent/adaptive_engine/native_glm5_next.py",
                  "openagent/adaptive_engine/independent_glm5_next.py",
                  "openagent/adaptive_engine/native_glm5_next.c"),
}
_DEVICE_REFERENCE_IMPLEMENTATIONS = {
    "glm5-next": "openagent.adaptive_engine.native_glm5_next.NativeGlm5NextTextDecoder (independent C reference)",
}


def _uses_device_native_glm_reference(family: str, backend: str | None,
                                      model_type: str | None) -> bool:
    return family == "glm5-next" and backend == "device" and model_type == "glm5_next_text"


def _reference_contract(family: str, backend: str | None, model_type: str | None):
    """Bind media references to both their text and independent tower equations."""
    if (family, model_type) == ('glm5-next', 'glm5_next'):
        device = backend == 'device'
        sources = (_DEVICE_REFERENCE_SOURCES if device else REFERENCE_SOURCES)['glm5-next']
        implementation = (_DEVICE_REFERENCE_IMPLEMENTATIONS if device else
                          REFERENCE_IMPLEMENTATIONS)['glm5-next']
        return (implementation + ' + IndependentGlm5VisionEncoder (independent media reference)',
                sources + ('openagent/adaptive_engine/independent_glm5_vision.py',
                           'openagent/adaptive_engine/independent_media_math.py',
                           'openagent/adaptive_engine/glm5_vision.py',
                           'openagent/adaptive_engine/publisher_glm5_config.py',
                           'openagent/adaptive_engine/publisher_glm5_loader.py',
                           'openagent/adaptive_engine/glm5_publisher_experts.py',
                           'openagent/adaptive_engine/publisher_tensor_reader.py',
                           'openagent/adaptive_engine/publisher_weight_mapping.py',
                           'openagent/adaptive_engine/publisher_family_contracts.py',
                           'openagent/adaptive_engine/native_media_ops.py',
                           'openagent/adaptive_engine/native_media_ops.c'))
    if (family, model_type) == ('qwen4', 'qwen4_exp'):
        return (REFERENCE_IMPLEMENTATIONS[family] +
                ' + IndependentQwen4VisionEncoder (independent media reference)',
                REFERENCE_SOURCES[family] + (
                    'openagent/adaptive_engine/independent_qwen4_vision.py',
                    'openagent/adaptive_engine/independent_media_math.py',
                    'openagent/adaptive_engine/qwen4_vision.py'))
    if (family, model_type) == ('inkling', 'inkling_mm_model'):
        return (REFERENCE_IMPLEMENTATIONS[family] +
                ' + IndependentInklingMediaEncoder (independent media reference)',
                REFERENCE_SOURCES[family] + (
                    'openagent/adaptive_engine/independent_media_math.py',
                    'openagent/adaptive_engine/inkling_media.py'))
    if _uses_device_native_glm_reference(family, backend, model_type):
        return _DEVICE_REFERENCE_IMPLEMENTATIONS[family], _DEVICE_REFERENCE_SOURCES[family]
    return REFERENCE_IMPLEMENTATIONS.get(family), REFERENCE_SOURCES.get(family)


def reference_implementation(family: str, backend: str | None = None,
                              model_type: str | None = None) -> str:
    """Return the reviewed comparator identity for this exact route backend."""
    value, _sources = _reference_contract(family, backend, model_type)
    if value is None:
        raise ObservationError("no reviewed independent reference is registered for this route")
    return value


def reference_source_pins(family: str, backend: str | None = None,
                          model_type: str | None = None) -> list[dict[str, str]]:
    """Return current SHA-256 pins for a family's complete local reference closure.

    No module is imported here: receipt verification remains a bounded,
    deterministic source-integrity check and never executes reference code.
    """
    _implementation, modules = _reference_contract(family, backend, model_type)
    if modules is None:
        raise ObservationError("no reviewed reference implementation is registered for this family")
    pins = []
    for module in modules:
        path = REPOSITORY_ROOT / module
        if not path.is_file():
            raise ObservationError("reviewed reference source is missing: " + module)
        pins.append({"module": module, "sha256": sha256_file(path)})
    return pins


def verify_reference_source_binding(family: str, reference: Any,
                                    backend: str | None = None,
                                    model_type: str | None = None) -> None:
    """Require a comparison observation to pin the family's exact source closure."""
    expected_implementation, expected_sources = _reference_contract(family, backend, model_type)
    if expected_sources is None or expected_implementation is None:
        raise ObservationError("no reviewed reference implementation is registered for this family")
    if type(reference) is not dict or reference.get("implementation") != expected_implementation:
        raise ObservationError("comparison reference is not the registered family implementation")
    expected_pins = reference_source_pins(family, backend, model_type)
    if len(expected_sources) == 1:
        # Preserve the original single-file receipt wire shape.
        source = reference.get("source") if type(reference) is dict else None
        if type(source) is not dict or source != expected_pins[0]:
            if type(source) is not dict or source.get("module") != expected_sources[0]:
                raise ObservationError("comparison reference is not the reviewed in-repository implementation")
            raise ObservationError("reference implementation changed since capture; recapture the evidence")
        return
    sources = reference.get("sources") if type(reference) is dict else None
    if type(sources) is not list or sources != expected_pins:
        if (type(sources) is not list or
                [pin.get("module") for pin in sources if type(pin) is dict] != list(expected_sources)):
            raise ObservationError("comparison reference is not the reviewed in-repository implementation")
        raise ObservationError("reference implementation changed since capture; recapture the evidence")


class ObservationError(ValueError):
    """A receipt or observation does not prove what it claims."""


# ---------------------------------------------------------------------------
# Encoding helpers (shared with the capture runner)
# ---------------------------------------------------------------------------

def encode_matrix(rows) -> dict[str, Any]:
    import numpy as np
    array = np.ascontiguousarray(np.asarray(rows, dtype="<f4"))
    if array.ndim != 2:
        raise ValueError("observation matrices are two-dimensional")
    return {"dtype": "float32-le", "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii")}


def decode_matrix(value: Any, *, label: str):
    import numpy as np
    if (type(value) is not dict or value.get("dtype") != "float32-le" or
            type(value.get("shape")) is not list or len(value["shape"]) != 2 or
            any(type(size) is not int or size <= 0 for size in value["shape"]) or
            type(value.get("data")) is not str):
        raise ObservationError(label + " must be a float32-le matrix")
    try:
        raw = base64.b64decode(value["data"].encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as error:
        raise ObservationError(label + " matrix data is not base64") from error
    rows, columns = value["shape"]
    if len(raw) != rows * columns * 4:
        raise ObservationError(label + " matrix size differs from its shape")
    array = np.frombuffer(raw, dtype="<f4").reshape(rows, columns)
    if not np.isfinite(array).all():
        raise ObservationError(label + " matrix contains non-finite values")
    return array


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Route applicability (derived from the canonical row, never from free text)
# ---------------------------------------------------------------------------

def coverage_applicability(row: dict[str, Any]) -> dict[str, bool]:
    """Which coverage checks the exact canonical route requires."""
    from .source_scope import _CONTRACT_CASES
    contract = _CONTRACT_CASES.get(row.get("family"), {})
    heads = tuple(contract.get("optional_heads", ()))
    media_cell = (row.get("cells") or {}).get("media") or {}
    family = str(row.get("family", ""))
    model_type = str(row.get("model_type", ""))
    # Inkling MTP is an optional auxiliary component on the top-level
    # multimodal wrapper. The standalone inkling_text identity rejects MTP
    # config/tensors in its own inventory planner and is not a speculative
    # target/draft route. Applicability follows the exact admitted identity,
    # not only the family's union of optional components.
    speculative_identity_applies = not (
        family == "inkling" and model_type == "inkling_text")
    return {
        "media_lifecycle": (contract.get("media", "none") != "none" and
                            not model_type.endswith("_text") and
                            media_cell.get("classification") == "implemented-source"),
        "speculative_state": (speculative_identity_applies and
                              any(head in heads for head in ("mtp", "speculative"))),
        "distributed_recovery": row.get("backend") == "distributed",
        "format_device": True,
        "abi_recovery": True,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _hex64(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _int_list(value: Any, *, label: str, upper: int, nonempty: bool = True) -> list[int]:
    if (type(value) is not list or (nonempty and not value) or
            any(type(item) is not int or not 0 <= item < upper for item in value)):
        raise ObservationError(label + " must be a list of in-range integer IDs")
    return value


def _pin(pin: Any, *, label: str, max_bytes: int = MAX_OBSERVATION_BYTES) -> tuple[Path, str]:
    if type(pin) is not dict:
        raise ObservationError(label + " pin is missing")
    path_text, digest = pin.get("path"), pin.get("sha256")
    if (type(path_text) is not str or not _hex64(digest) or
            "test_config" in path_text.lower() or "placeholder" in path_text.lower()):
        raise ObservationError(label + " pin is not a concrete path and SHA-256")
    path = Path(path_text)
    if not path.is_absolute() or not path.is_file():
        raise ObservationError(label + " pinned file does not exist")
    if path.stat().st_size > max_bytes:
        raise ObservationError(label + " pinned file exceeds its size limit")
    if sha256_file(path) != digest:
        raise ObservationError(label + " pinned file hash differs")
    return path, digest


def _load(path: Path, *, kind: str, case_id: str, config_sha: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ObservationError(kind + " observation is not JSON") from error
    if (type(payload) is not dict or payload.get("schema") != OBSERVATION_SCHEMA or
            payload.get("kind") != kind):
        raise ObservationError(kind + " observation has the wrong schema or kind")
    if payload.get("case_id") != case_id:
        raise ObservationError(kind + " observation is bound to another route")
    if payload.get("config_sha256") != config_sha:
        raise ObservationError(kind + " observation is bound to another checkpoint config")
    return payload


def _config_geometry(config: dict[str, Any]) -> dict[str, int]:
    config = config.get('text_config', config)
    if type(config) is not dict:
        raise ObservationError('pinned publisher text config must be an object')
    def first(*names):
        for name in names:
            value = config.get(name)
            if type(value) is int and value > 0:
                return value
        raise ObservationError("pinned config lacks " + "/".join(names))
    vocab = first('unpadded_vocab_size', 'vocab_size')
    if vocab > first('vocab_size'):
        raise ObservationError('unpadded vocabulary exceeds the pinned embedding vocabulary')
    return {"vocab": vocab, "layers": first("num_hidden_layers"),
            "experts": first("num_experts", "num_local_experts", "n_routed_experts"),
            "top_k": first("num_experts_per_tok")}


def routed_layer_mask(config, layers: int) -> tuple[bool, ...]:
    """Dense layers have no expert selection; bind this only to model config."""
    if isinstance(config, dict):
        config = config.get('text_config', config)
        schedule = config.get('mlp_layer_types')
        dense = config.get('dense_mlp_idx', config.get('first_k_dense_replace'))
        if schedule is None and dense is not None:
            if type(dense) is not int or not 0 <= dense <= layers:
                raise ObservationError('invalid pinned dense MLP prefix')
            schedule = ['dense'] * dense + ['sparse'] * (layers - dense)
    else:
        schedule = getattr(config, 'mlp_layer_types', None)
    if schedule is None:
        return (True,) * layers
    if (type(schedule) not in (list, tuple) or len(schedule) != layers or
            any(kind not in ('dense', 'sparse') for kind in schedule)):
        raise ObservationError('pinned MLP layer schedule differs from configured geometry')
    return tuple(kind == 'sparse' for kind in schedule)


_PUBLISHER_CONFIG_FIELDS = (
    "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "num_local_experts",
    "num_experts_per_tok", "vocab_size", "max_position_embeddings",
    "rope_theta", "sliding_window", "rms_norm_eps", "tie_word_embeddings",
)


def verify_publisher_config_contract(row: dict[str, Any], config: dict[str, Any]) -> None:
    """Compare pinned config geometry to publisher values carried by the row.

    Some canonical rows (currently Mixtral) carry an explicit publisher config
    contract. Check it here instead of trusting dimensions supplied by the
    observation payload. Fields intentionally absent from a publisher contract
    are not inferred or filled in.
    """
    publisher = row.get("publisher")
    if type(publisher) is not dict:
        return
    for name in _PUBLISHER_CONFIG_FIELDS:
        if name not in publisher:
            continue
        expected = publisher[name]
        default = {"rope_theta": 1e6, "sliding_window": None,
                   "rms_norm_eps": 1e-5, "tie_word_embeddings": False}.get(name, ...)
        actual = config.get(name, default)
        if actual is ...:
            raise ObservationError("pinned config lacks publisher field " + name)
        if type(expected) is bool:
            matches = type(actual) is bool and actual is expected
        elif type(expected) is int:
            matches = type(actual) is int and actual == expected
        elif type(expected) is float:
            matches = (not isinstance(actual, bool) and type(actual) in (int, float)
                       and math.isfinite(actual) and float(actual) == expected)
        else:
            matches = type(actual) is type(expected) and actual == expected
        if not matches:
            raise ObservationError("pinned config differs from publisher field " + name)
    if "head_dim" in publisher:
        hidden, heads = config.get("hidden_size"), config.get("num_attention_heads")
        actual = config.get("head_dim")
        if actual is None and type(hidden) is int and type(heads) is int and heads > 0 and hidden % heads == 0:
            actual = hidden // heads
        if type(actual) is not int or actual != publisher["head_dim"]:
            raise ObservationError("pinned config differs from publisher field head_dim")
    if "rope_type" in publisher or "rope_factor" in publisher:
        scaling = config.get("rope_parameters") or config.get("rope_scaling") or {}
        if type(scaling) is not dict:
            raise ObservationError("pinned config has invalid publisher RoPE contract")
        actual_rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        actual_rope_factor = scaling.get("factor", 1.0)
        if "rope_type" in publisher and actual_rope_type != publisher["rope_type"]:
            raise ObservationError("pinned config differs from publisher field rope_type")
        if "rope_factor" in publisher and (
                isinstance(actual_rope_factor, bool) or
                type(actual_rope_factor) not in (int, float) or
                not math.isfinite(actual_rope_factor) or
                float(actual_rope_factor) != publisher["rope_factor"]):
            raise ObservationError("pinned config differs from publisher field rope_factor")


def verify_route_receipt(row: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Verify one exact-route receipt; return the measurements it proved."""
    import numpy as np

    if row.get("combination_status") != "verified-admitted":
        raise ObservationError("runtime receipt cannot certify an unadmitted source route")
    case_id = row["case_id"]
    if receipt.get("case_id") != case_id:
        raise ObservationError("receipt is bound to another route")
    route_fields = ("family", "model_type", "backend", "arithmetic", "source_format",
                    "persisted_format")
    if receipt.get("route") != [row.get(name) for name in route_fields]:
        raise ObservationError("receipt route differs from the canonical source tuple")
    artifacts = receipt.get("artifacts")
    if type(artifacts) is not list:
        raise ObservationError("receipt lacks pinned artifacts")
    configs = [pin for pin in artifacts if type(pin) is dict and pin.get("role") == "config"]
    if len(configs) != 1:
        raise ObservationError("receipt must pin exactly one checkpoint config")
    config_path, config_sha = _pin(configs[0], label="config", max_bytes=MAX_ARTIFACT_BYTES)
    checkpoint_directory = config_path.parent.resolve()
    pinned_weights = []
    for index, pin in enumerate(artifacts):
        artifact_path, _ = _pin(pin, label=f"artifact {index}", max_bytes=MAX_ARTIFACT_BYTES)
        if pin.get("role") in ("weights", "expert_manifest", "expert_payload", "tokenizer"):
            if not artifact_path.resolve().is_relative_to(checkpoint_directory):
                raise ObservationError("checkpoint artifact is outside the pinned checkpoint")
        if pin.get("role") == "weights":
            pinned_weights.append(artifact_path.resolve())

    # Validate the observation manifest's structure and one-to-one file
    # binding before inspecting checkpoint contents. This reports duplicate
    # evidence directly even when a separate checkpoint artifact is also
    # malformed; no receipt is accepted until both checks pass.
    pins = receipt.get("observations")
    if type(pins) is not list:
        raise ObservationError("receipt lacks observation pins")
    by_kind: dict[str, tuple[Path, str]] = {}
    for pin in pins:
        kind = pin.get("kind") if type(pin) is dict else None
        if kind not in OBSERVATION_KINDS or kind in by_kind:
            raise ObservationError("observation kinds must be exactly one of each required kind")
        by_kind[kind] = _pin(pin, label=kind)
    if set(by_kind) != set(OBSERVATION_KINDS):
        raise ObservationError("missing observation kinds: " +
                               ", ".join(sorted(set(OBSERVATION_KINDS) - set(by_kind))))
    digests = [digest for _, digest in by_kind.values()]
    if len(set(digests)) != len(digests):
        raise ObservationError("one file cannot serve as several observation kinds")

    if len(pinned_weights) != len(set(pinned_weights)):
        raise ObservationError("receipt repeats a SafeTensors checkpoint weight shard pin")
    checkpoint_weight_shards = {
        path.resolve() for path in checkpoint_directory.glob("*.safetensors")
        if path.is_file()
    }
    if not checkpoint_weight_shards:
        raise ObservationError("pinned checkpoint contains no SafeTensors weight shards")
    if set(pinned_weights) != checkpoint_weight_shards:
        raise ObservationError("receipt must pin every SafeTensors checkpoint weight shard exactly")
    obs = {kind: _load(path, kind=kind, case_id=case_id, config_sha=config_sha)
           for kind, (path, _) in by_kind.items()}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ObservationError("pinned config is not JSON") from error
    if type(config) is not dict or config.get("model_type") != row.get("model_type"):
        raise ObservationError("pinned config identity differs from the route")
    verify_publisher_config_contract(row, config)
    geometry = _config_geometry(config)

    # tokens
    tokens = obs["tokens"]
    prompt = _int_list(tokens.get("prompt_ids"), label="prompt_ids", upper=geometry["vocab"])
    generated = _int_list(tokens.get("generated_ids"), label="generated_ids", upper=geometry["vocab"])
    implementation = tokens.get("implementation")
    if type(implementation) is not str or not implementation.strip():
        raise ObservationError("tokens observation must name the executing implementation")
    if tokens.get("decoding") != "greedy":
        raise ObservationError("only greedy decoding observations are comparable")
    finish = tokens.get("finish_reason")
    eos = config.get('text_config', config).get('eos_token_id', config.get('eos_token_id'))
    if finish == "eos":
        if generated[-1] != eos:
            raise ObservationError("EOS finish without the config EOS token")
    elif finish != "length" or (type(tokens.get("max_tokens")) is not int or
                                 tokens["max_tokens"] != len(generated)):
        raise ObservationError("finish reason must be EOS or an exact length limit")
    if eos in generated[:-1]:
        raise ObservationError("generation continued past EOS")

    # logits
    candidate = decode_matrix(obs["logits"].get("matrix"), label="logits")
    if candidate.shape != (len(generated), geometry["vocab"]):
        raise ObservationError("logits must have one vocabulary row per generated token")
    if [int(index) for index in candidate.argmax(axis=1)] != generated:
        raise ObservationError("generated tokens are not the greedy argmax of the observed logits")

    # comparison (gate recomputes the error)
    comparison = obs["comparison"]
    if comparison.get("candidate_logits_sha256") != by_kind["logits"][1]:
        raise ObservationError("comparison is not bound to the observed logits file")
    reference = comparison.get("reference")
    if type(reference) is not dict:
        raise ObservationError("comparison lacks an independent reference")
    ref_impl = reference.get("implementation")
    if type(ref_impl) is not str or not ref_impl.strip() or ref_impl == implementation:
        raise ObservationError("comparison reference must be a different implementation")
    ref_tokens = _int_list(reference.get("generated_ids"), label="reference generated_ids",
                           upper=geometry["vocab"])
    ref_logits = decode_matrix(reference.get("logits"), label="reference logits")
    tolerance = comparison.get("tolerance", {}).get("max_abs_logit") \
        if type(comparison.get("tolerance")) is dict else None
    if (type(tolerance) is not float or not math.isfinite(tolerance) or
            not 0.0 <= tolerance <= MAX_LOGIT_ABS_TOLERANCE):
        raise ObservationError(f"declared logit tolerance must be within [0, {MAX_LOGIT_ABS_TOLERANCE}]")
    verify_reference_source_binding(row.get("family"), reference, row.get("backend"),
                                    row.get("model_type"))
    if ref_tokens != generated:
        raise ObservationError("candidate and reference tokens differ")
    if ref_logits.shape != candidate.shape:
        raise ObservationError("candidate and reference logits differ in shape")
    max_abs = float(np.max(np.abs(candidate.astype(np.float64) - ref_logits.astype(np.float64))))
    if max_abs > tolerance:
        raise ObservationError(f"logit error {max_abs:.3g} exceeds tolerance {tolerance:.3g}")

    # cache
    processed = len(prompt) + len(generated) - 1
    cache = obs["cache"]
    if cache.get("position") != processed:
        raise ObservationError("cache position differs from the decoded positions")
    layer_digests = cache.get("layer_digests")
    if (cache.get("layers") != geometry["layers"] or type(layer_digests) is not list or
            len(layer_digests) != geometry["layers"] or not all(_hex64(d) for d in layer_digests)):
        raise ObservationError("cache must digest every configured layer")
    if type(cache.get("bytes")) is not int or cache["bytes"] <= 0:
        raise ObservationError("cache byte count must be measured")

    # routing
    layers = obs["routing"].get("layers")
    if type(layers) is not dict or len(layers) != geometry["layers"]:
        raise ObservationError("routing must cover every configured layer")
    layer_indices = []
    for layer_id in layers:
        suffix = layer_id.rsplit(".", 1)[-1] if type(layer_id) is str else ""
        if not suffix.isdecimal():
            raise ObservationError("routing layer IDs must end in a numeric layer index")
        layer_indices.append(int(suffix))
    if sorted(layer_indices) != list(range(geometry["layers"])):
        raise ObservationError("routing must cover every configured layer index exactly once")
    routed = routed_layer_mask(config, geometry['layers'])
    for layer_id, rows in layers.items():
        if type(rows) is not list or len(rows) != processed:
            raise ObservationError(f"routing layer {layer_id} must have one row per processed position")
        expected_top_k = geometry['top_k'] if routed[int(layer_id.rsplit('.', 1)[-1])] else 0
        for route in rows:
            _int_list(route, label=f"routing layer {layer_id}", upper=geometry["experts"],
                      nonempty=expected_top_k > 0)
            if len(route) != expected_top_k or len(set(route)) != len(route):
                raise ObservationError(f"routing layer {layer_id} rows must select distinct top-k experts")

    # recovery
    recovery = obs["recovery"]
    snapshot_position = recovery.get("snapshot_position")
    continuation = _int_list(recovery.get("continuation_ids"), label="recovery continuation",
                             upper=geometry["vocab"])
    replay = _int_list(recovery.get("replay_ids"), label="recovery replay", upper=geometry["vocab"])
    if (type(snapshot_position) is not int or snapshot_position < len(prompt) or
            recovery.get("restored_position") != snapshot_position):
        raise ObservationError("restore did not return to the snapshot position")
    if replay != continuation:
        raise ObservationError("replay after restore differs from the original continuation")

    # placement
    placement = obs["placement"]
    backend = row.get("backend")
    device = placement.get("device")
    if placement.get("backend") != backend or type(device) is not str or not device:
        raise ObservationError("placement backend/device is missing or differs from the route")
    if backend in HOST_BACKENDS and device != "cpu":
        raise ObservationError("host route observed on a non-CPU device")
    if backend not in HOST_BACKENDS and device == "cpu":
        raise ObservationError("device route observed on the host")
    if placement.get("host_fallback_operations") != 0:
        raise ObservationError("placement records host-fallback operations")
    if placement.get("implementation") != implementation:
        raise ObservationError("placement and tokens name different implementations")
    placement_constraints = receipt.get("placement_constraints")
    if (type(placement_constraints) is not dict or
            placement_constraints.get("backend") != backend or
            placement_constraints.get("device") != device):
        raise ObservationError("placement constraints differ from measured placement")
    if receipt.get("comparison_adapter") != ref_impl:
        raise ObservationError("comparison adapter differs from measured reference")
    cache_schema = receipt.get("cache_schema")
    if (type(cache_schema) is not dict or
            type(cache_schema.get("format")) is not str or not cache_schema["format"] or
            cache_schema.get("layers") != geometry["layers"]):
        raise ObservationError("cache schema differs from measured checkpoint layers")

    coverage = verify_coverage(row, receipt, observation_digests={kind: d for kind, (_, d) in by_kind.items()},
                               used_digests=set(digests), config_sha=config_sha)
    return {"case_id": case_id, "generated_tokens": len(generated), "processed_positions": processed,
            "max_abs_logit_error": max_abs, "tolerance": tolerance, "device": device,
            "implementation": implementation, "reference": ref_impl, "coverage": coverage}


def verify_coverage(row: dict[str, Any], receipt: dict[str, Any], *,
                    observation_digests: dict[str, str], used_digests: set[str],
                    config_sha: str) -> dict[str, str]:
    """Verify every coverage requirement against the route's applicability."""
    checks = receipt.get("checks")
    if type(checks) is not dict:
        raise ObservationError("receipt lacks coverage checks")
    applicable = coverage_applicability(row)
    chain = row.get("callable_chain") or {}
    result = {}
    seen = set(used_digests)
    for requirement in COVERAGE_REQUIREMENTS:
        check = checks.get(requirement)
        if type(check) is not dict:
            raise ObservationError(requirement + " coverage check is missing")
        status = check.get("status")
        if not applicable[requirement]:
            if status != "not-applicable" or not isinstance(check.get("reason"), str) or not check["reason"].strip():
                raise ObservationError(requirement + " is not applicable to this route and must say so")
            result[requirement] = "not-applicable"
            continue
        if status != "verified":
            raise ObservationError(requirement + " applies to this route and needs verified evidence")
        path, digest = _pin(check.get("evidence"), label=requirement)
        if digest in seen:
            raise ObservationError(requirement + " evidence reuses another observation file")
        seen.add(digest)
        payload = _load(path, kind=requirement, case_id=row["case_id"], config_sha=config_sha)
        _verify_coverage_payload(requirement, payload, row=row, chain=chain,
                                 observation_digests=observation_digests)
        result[requirement] = "verified"
    return result


def _verify_coverage_payload(requirement: str, payload: dict[str, Any], *, row: dict[str, Any],
                             chain: dict[str, Any], observation_digests: dict[str, str]) -> None:
    if requirement == "format_device":
        if payload.get("persisted_format") != row.get("persisted_format"):
            raise ObservationError("format evidence names another persisted format")
        if payload.get("consumer") not in tuple(chain.get("execution", ())):
            raise ObservationError("format evidence consumer is not in the route's execution chain")
        if payload.get("placement_sha256") != observation_digests["placement"]:
            raise ObservationError("format evidence is not bound to the placement observation")
    elif requirement == "abi_recovery":
        if payload.get("abi_owner") not in tuple(chain.get("native_abi", ())):
            raise ObservationError("ABI evidence owner is not in the route's native ABI chain")
        if payload.get("recovery_sha256") != observation_digests["recovery"]:
            raise ObservationError("ABI evidence is not bound to the recovery observation")
    elif requirement == "media_lifecycle":
        items = payload.get("items")
        if (type(items) is not list or not items or
                any(type(item) is not dict or item.get("kind") not in ("image", "audio", "video")
                    for item in items)):
            raise ObservationError("media evidence must list the exercised media items")
        before, after = payload.get("position_before_cancel"), payload.get("position_after_cancel")
        if type(before) is not int or before != after:
            raise ObservationError("media cancellation did not restore the cache position")
        if payload.get("released_items") != len(items):
            raise ObservationError("media cleanup did not release every item")
    elif requirement == "speculative_state":
        proposals, accepted, rejected = (payload.get(name) for name in ("proposals", "accepted", "rejected"))
        if (any(type(value) is not int or value < 0 for value in (proposals, accepted, rejected)) or
                proposals <= 0 or accepted + rejected != proposals):
            raise ObservationError("speculative evidence must account for every proposal")
        before, after = payload.get("rng_digest_before"), payload.get("rng_digest_after_restore")
        if not _hex64(before) or before != after:
            raise ObservationError("speculative restore did not reproduce the RNG state")
    elif requirement == "distributed_recovery":
        world = payload.get("world_size")
        ranks = payload.get("ranks")
        if type(world) is not int or world < 2 or type(ranks) is not list or len(ranks) != world:
            raise ObservationError("distributed evidence must cover every rank")
        if sorted(rank.get("rank") for rank in ranks if type(rank) is dict) != list(range(world)):
            raise ObservationError("distributed evidence ranks are incomplete")
        digests = {rank.get("rollback_digest") for rank in ranks}
        if len(digests) != 1 or not _hex64(next(iter(digests))):
            raise ObservationError("ranks did not agree on the rollback state")


def load_receipts(path: Path) -> dict[str, dict[str, Any]]:
    """Load the receipts manifest; malformed or duplicate entries fail closed."""
    path = Path(path)
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (type(manifest) is not dict or manifest.get("schema") != RECEIPTS_SCHEMA or
            type(manifest.get("routes")) is not list):
        raise ObservationError("receipt manifest has the wrong schema")
    receipts: dict[str, dict[str, Any]] = {}
    for receipt in manifest["routes"]:
        if type(receipt) is not dict or type(receipt.get("case_id")) is not str:
            raise ObservationError("receipt manifest entry lacks a case_id")
        if receipt["case_id"] in receipts:
            raise ObservationError("receipt manifest repeats a case_id")
        receipts[receipt["case_id"]] = receipt
    return receipts


__all__ = ["COVERAGE_REQUIREMENTS", "MAX_LOGIT_ABS_TOLERANCE", "OBSERVATION_KINDS",
           "REFERENCE_IMPLEMENTATIONS", "REFERENCE_SOURCES",
           "OBSERVATION_SCHEMA", "ObservationError", "RECEIPTS_SCHEMA", "coverage_applicability",
           "decode_matrix", "encode_matrix", "load_receipts", "reference_source_pins", "sha256_file",
           "reference_implementation", "routed_layer_mask", "verify_publisher_config_contract",
           "verify_reference_source_binding", "verify_coverage",
           "verify_route_receipt"]
