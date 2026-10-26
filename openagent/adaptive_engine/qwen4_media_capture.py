"""Route-bound Qwen4 publisher media capture and independent comparison.

Candidate execution uses the admitted production conditional loader. The
reference combines independent NumPy Qwen4 vision and text implementations.
The CLI publishes only measurements accepted by the canonical media receipt
gate; the measurement helpers themselves never write evidence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .device_capture import CaptureError
from .device_observations import reference_implementation, reference_source_pins

ROUTE_FIELDS = ("family", "model_type", "backend", "arithmetic",
                "source_format", "persisted_format")
_SOURCE_TO_PERSISTED = {
    "F32": "float32",
    "F16": "float32",
    "BF16": "float32",
    "F8_E4M3": "fp8-e4m3-b128-f32",
}
_BACKENDS = ("native", "device", "public-range")
_MAX_CONFIG_BYTES = 1024 * 1024


def _canonical_route(route: Any) -> dict[str, Any]:
    if type(route) is not dict or set(route) != set(ROUTE_FIELDS):
        raise CaptureError("exact Qwen4 publisher route tuple is required")
    if (route["family"] != "qwen4" or route["model_type"] != "qwen4_exp" or
            route["arithmetic"] != "fp32" or route["backend"] not in _BACKENDS or
            _SOURCE_TO_PERSISTED.get(route["source_format"]) != route["persisted_format"]):
        raise CaptureError("unsupported Qwen4 publisher backend/source-format tuple")
    from .source_scope import source_scope_matrix
    rows = source_scope_matrix()["execution_cases"]
    matches = [row for row in rows if all(row.get(name) == route[name] for name in ROUTE_FIELDS)]
    admitted = [row for row in matches if row.get("combination_status") == "verified-admitted"
                and row.get("admission") == "verified-admitted"]
    if len(admitted) != 1:
        raise CaptureError("Qwen4 media capture requires one exact canonical admitted route")
    from .device_observations import coverage_applicability
    if not coverage_applicability(admitted[0])["media_lifecycle"]:
        raise CaptureError("canonical Qwen4 publisher route does not require the media contract")
    return admitted[0]


def _required_source_paths(backend: str) -> tuple[Path, ...]:
    """Complete pinned source closure for this runner and the selected owner."""
    if backend not in _BACKENDS:
        raise CaptureError("unsupported Qwen4 media backend")
    from .native_family_capture import _candidate_source_paths
    from . import native_family_capture
    root = Path(native_family_capture.__file__).resolve().parent
    common = (
        "qwen4_media_capture.py",
        "wholemodel_validation/qwen4_media.py",
        "wholemodel_validation/qwen35_media.py",
        "wholemodel_validation/qwen4_publisher_caches.py",
        "wholemodel_validation/cache_readers.py",
        "wholemodel_validation/adapters.py",
        "wholemodel_validation/contracts.py",
        "wholemodel_validation/comparison.py",
        "wholemodel_validation/runner.py",
        "qwen4_multimodal.py",
        "qwen4_vision.py",
        "native_media_ops.py",
        "native_media_ops.c",
        "publisher_conditional_loader.py",
        "publisher_family_contracts.py",
        "publisher_checkpoint/storage.py",
        "publisher_tensor_reader.py",
        "publisher_weight_mapping.py",
        "publisher_media_processor.py",
        "publisher_media_files.py",
        "native_checkpoint.py",
        "device_observations.py",
        "source_scope.py",
        "phase1_p02_route_bindings.py",
        "independent_qwen4.py",
        "independent_qwen4_vision.py",
        "independent_media_math.py",
        "independent_qwen35.py",
        "native_mixtral.py",
        "failure_reporting.py",
    )
    backend_extra = {
        "native": (
            "native_qwen4.py", "native_qwen4.c", "native_qwen4_hyper.py",
            "native_qwen4_hyper.c", "native_mxfp4.py", "native_mxfp4.c",
        ),
        "device": (
            "torch_qwen4_decoder.py", "device_qwen4_forward.py",
            "device_qwen4_layers.py", "device_qwen4_attention.py",
            "device_qwen35_layers.py", "device_qwen_delta.py", "device_attention.py",
            "device_logits.py", "device_execution_policy.py",
            "device_publisher_projections.py", "device_qwen4_publisher_experts.py",
        ),
        "public-range": (
            "native_qwen4.py", "native_qwen4.c", "native_qwen4_public_text.py",
            "native_qwen4_public_adapters.py", "native_qwen4_composition.py",
            "native_qwen4_composition.c", "qwen4_segment_wire.py",
            "segment_edge_runtime.py", "segment_edge_cache.py", "segment_snapshot.py",
            "family_source_binding.py", "native_library_identity.py",
            "native/segment_edge_compat.h", "native/segment_edge_compat.c",
            "native/segment_family_models.h", "native/segment_family_models.c",
            "native/qwen4_public_model.h", "native/qwen4_public_model.c",
            "native/qwen4_public_session.h", "native/qwen4_public_session.c",
            "native/qwen4_public_internal.h", "native/qwen4_public_adapters.c",
            "native/original_experts.h", "native/original_experts.c",
            "native/segment_executor.h", "native/segment_executor.c",
            "native/segment_bridge.h", "native/segment_bridge.c",
            "native/segment_math.h", "native/segment_math.c", "native/sha256.h",
        ),
    }[backend]
    base = tuple((root / name).resolve(strict=True) for name in common + backend_extra)
    # Keep the established exact text-route pins as a superset for backend
    # execution/ABI sources, then append family media owners without duplicates.
    text = _candidate_source_paths("qwen4", backend)
    paths = tuple(dict.fromkeys((*text, *base)))
    if any(not path.is_file() for path in paths):
        raise CaptureError("Qwen4 media candidate source closure is incomplete")
    return paths


def _validate_media_configuration(configuration: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if (type(configuration) is not dict or set(configuration) !=
            {"schema", "route", "comparison"} or
            configuration.get("schema") != "openagent.phase1.qwen4-media-capture.v1"):
        raise CaptureError("exact Qwen4 media capture configuration schema required")
    row = _canonical_route(configuration["route"])
    comparison = configuration["comparison"]
    required = {"schema", "artifacts", "candidate", "oracle", "fixture_config",
                "max_observation_bytes", "seed", "generation_steps", "tolerance"}
    if (type(comparison) is not dict or set(comparison) != required or
            comparison.get("schema") != "openagent.qwen4-media-comparison-configuration.v1"):
        raise CaptureError("complete pinned Qwen4 media candidate/oracle comparison is required")
    candidate = comparison.get("candidate")
    if (type(candidate) is not dict or set(candidate) != {"factory", "options"} or
            candidate.get("factory") !=
            "openagent.adaptive_engine.qwen4_media_capture:create_qwen4_media_candidate"):
        raise CaptureError("Qwen4 media candidate must use the route-bound production media adapter")
    options = candidate.get("options")
    candidate_fields = {"route", "metadata", "config", "max_cache_export_bytes",
                        "max_routing_export_bytes", "max_observation_bytes", "decoder_options"}
    if type(options) is not dict or set(options) != candidate_fields:
        raise CaptureError("complete route-bound Qwen4 candidate options are required")
    if options.get("route") != configuration["route"]:
        raise CaptureError("candidate route must exactly match the requested route")
    decoder_options = options.get("decoder_options")
    if type(decoder_options) is not dict or decoder_options.get("route") != configuration["route"]:
        raise CaptureError("candidate loader route must exactly match the requested route")
    pins = comparison.get("artifacts")
    if type(pins) is not list:
        raise CaptureError("complete source/checkpoint/fixture/runtime artifact pins are required")
    pinned_sources = {Path(item["path"]).as_posix() for item in pins
                      if type(item) is dict and item.get("kind") == "source" and
                      type(item.get("path")) is str}
    required_sources = {path.relative_to(Path(__file__).resolve().parents[2]).as_posix()
                        for path in _required_source_paths(configuration["route"]["backend"])}
    missing = sorted(required_sources - pinned_sources)
    if missing:
        raise CaptureError("Qwen4 media source closure is not pinned: " + ", ".join(missing[:12]))
    oracle = comparison.get("oracle")
    oracle_factory = "openagent.adaptive_engine.qwen4_media_capture:create_independent_qwen4_reference"
    if (type(oracle) is not dict or set(oracle) != {"factory", "options"} or
            oracle.get("factory") != oracle_factory or type(oracle.get("options")) is not dict or
            set(oracle["options"]) != {"max_bytes", "max_patches", "max_attention_bytes"} or
            any(type(oracle["options"][name]) is not int or oracle["options"][name] <= 0
                for name in ("max_bytes", "max_patches", "max_attention_bytes"))):
        raise CaptureError("independently implemented NumPy Qwen4 media reference and budgets are required")
    return row, comparison


def load_qwen4_publisher_candidate(*, offline_root: str | Path, options: dict[str, Any]):
    """Load the real conditional Qwen4 owner after exact route preflight.

    The caller supplies pinned local library paths and explicit memory budgets.
    The loader itself rechecks checkpoint inventory and source format before
    reading tensor payloads.
    """
    fields = {
        "route", "checkpoint", "media_library", "text_library", "runtime_bytes",
        "weight_bytes", "conversion_bytes", "ple_bytes", "expert_bytes",
        "snapshot_count", "context_tokens", "device", "projection_bytes",
        "segment_runtime_library", "segment_work_bytes", "segment_layer_span",
        "native_composition", "native_store_runtime_library",
        "native_expert_bytes", "native_ple_bytes",
    }
    if type(options) is not dict or set(options) != fields:
        raise CaptureError("exact bounded Qwen4 publisher loader options are required")
    row = _canonical_route(options["route"])
    route = options["route"]
    backend = route["backend"]
    if (backend == "native" and (options["device"] is not None or
                                  options["segment_runtime_library"] is not None)):
        raise CaptureError("native media route cannot use device or Segment runtime owners")
    if backend == "device" and (type(options["device"]) is not str or not options["device"] or
                                 options["segment_runtime_library"] is not None or
                                 options["native_composition"]):
        raise CaptureError("device media route needs one explicit device and no Segment/native-composition owner")
    if backend == "public-range" and (options["device"] is not None or
                                       options["segment_runtime_library"] is None or
                                       options["native_composition"] is not True or
                                       options["segment_layer_span"] != 1):
        raise CaptureError("public-range media route requires native composition, Segment runtime, and per-layer ranges")
    if (type(options["native_composition"]) is not bool or
            any(type(options[name]) is not int or options[name] < 0 for name in (
                "runtime_bytes", "weight_bytes", "conversion_bytes", "ple_bytes",
                "expert_bytes", "snapshot_count", "context_tokens", "projection_bytes",
                "segment_work_bytes", "segment_layer_span", "native_expert_bytes",
                "native_ple_bytes")) or
            min(options["runtime_bytes"], options["weight_bytes"], options["conversion_bytes"],
                options["snapshot_count"], options["context_tokens"]) <= 0):
        raise CaptureError("positive bounded Qwen4 runtime, weight, conversion and context budgets required")
    root = Path(offline_root).resolve(strict=True)
    checkpoint = (root / options["checkpoint"]).resolve(strict=True)
    if not checkpoint.is_dir() or not checkpoint.is_relative_to(root):
        raise CaptureError("Qwen4 checkpoint must be a local directory under the pinned offline root")
    from .source_scope import require_checkpoint_inventory_admission
    inventory = require_checkpoint_inventory_admission(
        checkpoint, family="qwen4", model_type="qwen4_exp", backend=backend,
        arithmetic=route["arithmetic"], persisted_format=route["persisted_format"],
        max_metadata_bytes=min(options["weight_bytes"], 256 * 1024**2))
    actual_formats = {
        record.dtype for name, record in inventory.tensors.items()
        if record.dtype != "I64" and
        not name.endswith((".weight_scale_inv", "_scale_inv"))
    }
    expected_dtype = {"F32": "F32", "F16": "F16", "BF16": "BF16",
                      "F8_E4M3": "F8_E4M3"}[route["source_format"]]
    if actual_formats != {expected_dtype}:
        raise CaptureError("checkpoint floating source encoding differs from the exact requested route")
    from .publisher_conditional_loader import load_publisher_conditional_decoder
    return load_publisher_conditional_decoder(
        checkpoint, family="qwen4_exp", library=options["media_library"],
        text_library=options["text_library"], max_weight_bytes=options["weight_bytes"],
        max_bytes=options["runtime_bytes"], context_tokens=options["context_tokens"],
        max_conversion_bytes=options["conversion_bytes"], max_ple_bytes=options["ple_bytes"],
        max_snapshots=options["snapshot_count"], arithmetic=route["arithmetic"],
        max_expert_bytes=options["expert_bytes"], text_device=options["device"],
        max_projection_bytes=options["projection_bytes"],
        segment_runtime_library=options["segment_runtime_library"],
        segment_work_bytes=options["segment_work_bytes"],
        segment_layer_span=options["segment_layer_span"],
        native_composition=options["native_composition"],
        native_store_runtime_library=options["native_store_runtime_library"],
        native_expert_max_bytes=options["native_expert_bytes"],
        native_ple_max_bytes=options["native_ple_bytes"])


def create_qwen4_media_candidate(*, offline_root: str | Path, options: dict[str, Any]):
    """Build the existing Qwen4 media observer around this exact loader.

    Runtime libraries must be explicit dependency pins in the same comparison
    manifest as the checkpoint and the production source closure.
    """
    required = {"route", "metadata", "config", "max_cache_export_bytes",
                "max_routing_export_bytes", "max_observation_bytes", "decoder_options",
                "_artifact_pins"}
    if type(options) is not dict or set(options) != required:
        raise CaptureError("runner-pinned Qwen4 media adapter options are incomplete")
    row = _canonical_route(options["route"])
    decoder_options = options["decoder_options"]
    if type(decoder_options) is not dict or decoder_options.get("route") != options["route"]:
        raise CaptureError("Qwen4 decoder factory route differs from the canonical tuple")
    from .wholemodel_validation.runner import pinned_artifacts
    root = Path(offline_root).resolve(strict=True)
    pins = pinned_artifacts({"artifacts": options["_artifact_pins"]}, root)
    checkpoint = (root / decoder_options.get("checkpoint", "")).resolve(strict=True)
    if not checkpoint.is_dir() or not checkpoint.is_relative_to(root):
        raise CaptureError("Qwen4 candidate checkpoint must be inside the pinned offline root")
    config_pin = [item for item in pins if item["kind"] == "config" and
                  Path(item["resolved_path"]).resolve() == (checkpoint / "config.json").resolve()]
    expected_weight_paths = {path.resolve() for path in checkpoint.glob("*.safetensors") if path.is_file()}
    pinned_weight_paths = {Path(item["resolved_path"]).resolve() for item in pins
                           if item["kind"] == "weights" and
                           Path(item["resolved_path"]).resolve().parent == checkpoint}
    if (len(config_pin) != 1 or not expected_weight_paths or
            pinned_weight_paths != expected_weight_paths):
        raise CaptureError("Qwen4 candidate checkpoint config and every weight shard must be pinned")
    metadata = options["metadata"]
    if (type(metadata) is not dict or
            metadata.get("config_sha256") != config_pin[0]["observed_sha256"] or
            sum(1 for item in pins if item["kind"] == "weights" and
                item["observed_sha256"] == metadata.get("weights_sha256")) != 1 or
            sum(1 for item in pins if item["kind"] == "tokenizer" and
                item["observed_sha256"] == metadata.get("tokenizer_sha256")) != 1):
        raise CaptureError("Qwen4 candidate metadata hashes must identify its exact pinned checkpoint")
    expected_sources = {path.resolve() for path in _required_source_paths(row["backend"])}
    source_pins = {Path(item["resolved_path"]).resolve() for item in pins
                   if item["kind"] == "source"}
    if not expected_sources <= source_pins:
        raise CaptureError("pinned Qwen4 candidate omits production media/backend sources")
    runtime_names = ["media_library"]
    if row["backend"] != "device":
        runtime_names.append("text_library")
    if row["backend"] == "public-range":
        runtime_names.extend(("segment_runtime_library", "native_store_runtime_library"))
    elif decoder_options.get("native_store_runtime_library") is not None:
        runtime_names.append("native_store_runtime_library")
    pinned_dependencies = {Path(item["resolved_path"]).resolve() for item in pins
                           if item["kind"] == "dependency"}
    for name in runtime_names:
        value = decoder_options.get(name)
        if value is None and name == "native_store_runtime_library":
            continue
        path = Path(value).resolve(strict=True)
        if path not in pinned_dependencies:
            raise CaptureError("Qwen4 executable library is not pinned as a dependency: " + name)
    if (type(options["max_cache_export_bytes"]) is not int or
            type(options["max_observation_bytes"]) is not int or
            type(options["max_routing_export_bytes"]) is not int):
        raise CaptureError("explicit bounded Qwen4 observation budgets required")
    from .wholemodel_validation.qwen4_media import Qwen4MediaAdapter
    from .wholemodel_validation.qwen4_publisher_caches import Qwen4PublisherCacheReader
    import inspect
    from . import qwen4_multimodal
    wrapper_path = Path(inspect.getsourcefile(qwen4_multimodal.Qwen4MultimodalDecoder)).resolve()
    loader_path = Path(__file__).resolve()
    if wrapper_path not in source_pins or loader_path not in source_pins:
        raise CaptureError("Qwen4 media wrapper and route loader source pins are required")
    metadata = copy.deepcopy(options["metadata"])
    if metadata.get("authoritative") is True:
        raise CaptureError("candidate metadata cannot claim an authoritative oracle")
    reader = Qwen4PublisherCacheReader(
        family="qwen4_exp_text", config=options["config"], role="candidate",
        max_export_bytes=options["max_cache_export_bytes"], media=True)
    from .publisher_checkpoint.storage import snapshot
    from .publisher_family_contracts import component_contract
    from .wholemodel_validation.qwen4_publisher_caches import contract as qwen4_cache_contract
    checkpoint_snapshot = snapshot(checkpoint, max_metadata_bytes=256 * 1024**2,
                                   max_tensors=1_000_000, max_shards=4096)
    actual_text_config = component_contract(checkpoint_snapshot.config,
                                            arithmetic=row["arithmetic"]).text_config
    if vars(reader.config) != vars(qwen4_cache_contract(actual_text_config, options["max_cache_export_bytes"])):
        raise CaptureError("Qwen4 candidate cache schema config differs from pinned publisher config")
    routing = bool(metadata.get("evidence_schema"))
    if routing:
        schema = reader.enable_routing_observations(
            max_export_bytes=options["max_routing_export_bytes"])
        if schema != metadata["evidence_schema"]:
            raise CaptureError("complete executed Qwen4 router observation schema is required")
    if (metadata.get("num_layers") != reader.config.num_hidden_layers or
            metadata.get("vocabulary_size") != reader.config.vocab_size):
        raise CaptureError("Qwen4 candidate metadata differs from route-bound checkpoint geometry")
    decoder_spec = copy.deepcopy(decoder_options)

    def fresh(*, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("nonnegative Qwen4 media comparison seed required")
        return load_qwen4_publisher_candidate(offline_root=root, options=decoder_spec)

    return Qwen4MediaAdapter(fresh, metadata, reader,
        max_bytes=options["max_observation_bytes"], capture_routing=routing)


def independent_qwen4_media_forward(text_decoder, vision_encoder, token_ids, *,
                                    publisher_config, vision_config,
                                    pixel_values=None, image_grid_thw=None,
                                    pixel_values_videos=None, video_grid_thw=None,
                                    mm_token_type_ids=None, rope_delta=0,
                                    cancelled=None):
    """Run an independent NumPy vision tower and the independent Qwen4 text model.

    ``text_decoder`` must be an ``IndependentQwen4TextDecoder`` built from the
    same checkpoint snapshot but not from the candidate decoder's mutated
    cache. This computes full media-conditioned logits; it is intentionally
    separate from the native/device/public candidate's vision and text paths.
    """
    from .independent_qwen4 import IndependentQwen4TextDecoder
    from .independent_qwen4_vision import IndependentQwen4VisionEncoder
    if type(text_decoder) is not IndependentQwen4TextDecoder or type(vision_encoder) is not IndependentQwen4VisionEncoder:
        raise CaptureError("independent Qwen4 text and vision owners are required")
    if type(publisher_config) is not dict or type(vision_config) is not dict:
        raise CaptureError("pinned Qwen4 publisher and vision configuration dictionaries required")
    ids = np.asarray(token_ids)
    if ids.ndim == 1:
        ids = ids[None, :]
    c = text_decoder.config
    if (ids.ndim != 2 or ids.shape[0] != 1 or not ids.shape[1] or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids >= c.vocab_size)):
        raise CaptureError("reference media input must be an unpadded expanded batch-one token sequence")
    if type(rope_delta) is not int:
        raise CaptureError("integer Qwen4 continuation rope delta required")
    image_id, video_id = publisher_config.get("image_token_id"), publisher_config.get("video_token_id")
    if any(type(value) is not int for value in (image_id, video_id)) or image_id == video_id:
        raise CaptureError("exact Qwen4 image/video token ids required")
    types = np.where(ids == image_id, 1, np.where(ids == video_id, 2, 0))
    if mm_token_type_ids is not None and not np.array_equal(types, np.asarray(mm_token_type_ids)):
        raise CaptureError("media token types differ from actual Qwen4 placeholder IDs")
    if ((pixel_values is None) != (image_grid_thw is None) or
            (pixel_values_videos is None) != (video_grid_thw is None)):
        raise CaptureError("Qwen4 reference pixels and ordered grids must be supplied together")
    if (vision_config.get("out_hidden_size") != c.hidden_size or
            vision_config.get("spatial_merge_size") != vision_encoder.config["spatial_merge_size"]):
        raise CaptureError("independent Qwen4 vision and text embedding geometries differ")
    embedding = text_decoder.weights.get("model.embed_tokens.weight")
    if embedding is None or embedding.ndim != 2 or embedding.shape[1] != c.hidden_size:
        raise CaptureError("independent Qwen4 text embedding table is not resident")
    embeds = np.array(embedding[ids], dtype=np.float32, order="C", copy=True)
    for modality, pixels, grids in ((1, pixel_values, image_grid_thw),
                                    (2, pixel_values_videos, video_grid_thw)):
        count = int(np.count_nonzero(types == modality))
        if pixels is None:
            if count:
                raise CaptureError("Qwen4 media placeholder has no independent vision payload")
            continue
        if not count:
            raise CaptureError("orphan Qwen4 media payload has no expanded placeholder")
        features = np.concatenate(vision_encoder.forward(pixels, grids, cancelled=cancelled), axis=0)
        if features.shape != (count, c.hidden_size):
            raise CaptureError("independent Qwen4 vision output differs from placeholder count/width")
        embeds[types == modality] = features

    merge = vision_config["spatial_merge_size"]
    queues = {1: [], 2: []}
    for modality, grids in ((1, image_grid_thw), (2, video_grid_thw)):
        if grids is None:
            continue
        grid_array = np.asarray(grids)
        if grid_array.ndim != 2 or grid_array.shape[1] != 3 or grid_array.dtype.kind not in "iu":
            raise CaptureError("Qwen4 ordered media grids require integer [items,3] geometry")
        for temporal, height, width in grid_array:
            if modality == 2:
                queues[2].extend([(1, int(height), int(width))] * int(temporal))
            else:
                queues[1].append((int(temporal), int(height), int(width)))
    used, cursor = {1: 0, 2: 0}, text_decoder.position + rope_delta
    positions = np.empty((3, 1, ids.shape[1]), dtype=np.int64)
    start = 0
    while start < ids.shape[1]:
        kind = int(types[0, start]); end = start + 1
        while end < ids.shape[1] and types[0, end] == kind:
            end += 1
        if kind == 0:
            positions[:, 0, start:end] = np.arange(cursor, cursor + end-start, dtype=np.int64)[None, :]
            cursor += end - start
        else:
            run = start
            while run < end:
                if used[kind] >= len(queues[kind]):
                    raise CaptureError("Qwen4 reference placeholder has no ordered grid")
                temporal, height, width = queues[kind][used[kind]]
                extent = temporal * (height // merge) * (width // merge)
                stop = run + extent
                if height % merge or width % merge or stop > end or extent <= 0:
                    raise CaptureError("Qwen4 reference placeholder span differs from grid geometry")
                axes = np.meshgrid(np.arange(temporal), np.arange(height//merge),
                                   np.arange(width//merge), indexing="ij")
                positions[:, 0, run:stop] = np.stack(axes).reshape(3, -1) + cursor
                cursor += max(height, width) // merge
                run = stop; used[kind] += 1
        start = end
    if any(used[kind] != len(queues[kind]) for kind in used):
        raise CaptureError("Qwen4 reference has unused media grids")
    delta = int(positions.max()) + 1 - (text_decoder.position + ids.shape[1])
    result = text_decoder.forward(ids, inputs_embeds=embeds, position_ids=positions)
    return result, positions, delta


def create_independent_qwen4_reference(candidate, *, max_bytes: int,
                                       max_patches: int = 65536,
                                       max_attention_bytes: int = 256 * 1024**2):
    """Copy admitted candidate tensors into distinct NumPy reference owners.

    This avoids a second source decoder/backend, but never shares mutable
    cache or execution state. External projection/expert banks are rejected;
    original PLE rows are copied only when their complete resident size fits
    the caller's explicit reference budget.
    """
    from .independent_qwen4 import IndependentQwen4TextDecoder
    from .independent_qwen4_vision import IndependentQwen4VisionEncoder
    from .native_qwen4_public_text import NativeQwen4PublicTextDecoder
    if type(max_bytes) is not int or max_bytes <= 0:
        raise CaptureError("positive bounded independent Qwen4 reference budget required")
    if getattr(candidate, "_closed", False) or getattr(candidate, "text", None) is None:
        raise CaptureError("live production Qwen4 multimodal candidate required")
    text = candidate.text
    if type(text) is NativeQwen4PublicTextDecoder:
        text = getattr(getattr(text, "owner", None), "_decoder", None)
    if text is None or getattr(text, "_closed", False):
        raise CaptureError("production Qwen4 text weights are not independently readable")
    if any(getattr(text, name, None) is not None for name in
           ("_expert_provider", "projection_provider", "_composition_experts")):
        raise CaptureError("independent Qwen4 oracle requires fully resident expert/projection banks")
    source_weights = getattr(text, "weights", None)
    if not isinstance(source_weights, dict) or not source_weights:
        raise CaptureError("complete resident production Qwen4 text inventory required")
    weights = {}
    used = 0
    for name, value in source_weights.items():
        shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
        if not shape or any(dimension < 0 for dimension in shape):
            raise CaptureError("Qwen4 source tensor has no bounded dense geometry: " + str(name))
        source_dtype = str(getattr(value, "dtype", ""))
        integer = source_dtype in ("int64", "torch.int64", "int32", "torch.int32", "uint8", "torch.uint8")
        estimate = math.prod(shape) * (8 if integer else 4)
        used += estimate
        if used * 2 > max_bytes:
            raise CaptureError("complete independent Qwen4 resident text copy exceeds reference budget")
        if hasattr(value, "detach"):
            value = value.detach().to("cpu").numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise CaptureError("Qwen4 reference source tensor is nonfinite or unsupported: " + str(name))
        array = np.array(array, dtype=np.int64 if array.dtype.kind in "iu" else np.float32,
                         order="C", copy=True)
        if array.shape != shape or array.nbytes != estimate:
            raise CaptureError("Qwen4 reference tensor materialization differs from admitted geometry")
        weights[name] = array
    ple_provider = getattr(text, "_ple_provider", None)
    if ple_provider is not None:
        shapes = getattr(ple_provider, "shapes", None)
        row_reader = getattr(ple_provider, "row", None)
        if not isinstance(shapes, dict) or not callable(row_reader):
            raise CaptureError("unknown Qwen4 PLE provider cannot feed independent reference")
        pending = []
        for layer, shape in sorted(shapes.items()):
            rows, width = shape
            need = rows * width * 4
            used += need
            if used * 2 > max_bytes:
                raise CaptureError("complete independent Qwen4 PLE tables exceed reference budget")
            table = np.empty((rows, width), dtype=np.float32)
            for index in range(rows):
                table[index] = row_reader(layer, index)
            pending.append((f"model.layers.{layer}.ple.ple_embedding.ngram_embedding.weight", table))
        weights.update(pending)
    text_reference = None
    vision_reference = None
    try:
        text_reference = IndependentQwen4TextDecoder(
            dict(candidate.components.text_config), weights, max_bytes=max_bytes,
            max_snapshots=2)
        vision_tensors = {"model.visual." + name: value
                          for name, value in candidate.vision.weights.items()}
        vision_reference = IndependentQwen4VisionEncoder(
            dict(candidate.components.vision_config), vision_tensors,
            max_bytes=max_bytes, max_patches=max_patches,
            max_attention_bytes=max_attention_bytes)
        return text_reference, vision_reference
    except BaseException as primary:
        for owner in (vision_reference, text_reference):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as cleanup:
                    primary.add_note("independent Qwen4 reference cleanup failed: " + repr(cleanup))
        raise


def measure_qwen4_media_route(*, candidate, reference_text, reference_vision,
                              token_ids, media, route, config, artifacts,
                              tolerance, max_tokens, recovery_tokens):
    """Collect route-bound Qwen4 media/logit/cache/routing/recovery measurements.

    The returned object is a measurement payload, not a receipt. A caller must
    pass it to the canonical evidence publisher, which performs all gate and
    artifact checks before any receipt is written.
    """
    if type(route) is not dict:
        raise CaptureError("canonical exact Qwen4 route required")
    row = _canonical_route({name: route[name] for name in ROUTE_FIELDS})
    if type(config) is not dict or type(artifacts) is not list:
        raise CaptureError("pinned Qwen4 comparison config and artifact list required")
    if (type(max_tokens) is not int or not 1 <= max_tokens <= 4096 or
            type(recovery_tokens) is not int or not 1 <= recovery_tokens <= 4096 or
            (type(tolerance) is dict and
             (set(tolerance) != {"atol", "rtol"} or any(
                 isinstance(tolerance[name], bool) or not isinstance(tolerance[name], (int, float)) or
                 not np.isfinite(tolerance[name]) or tolerance[name] < 0 for name in ("atol", "rtol")))) or
            (type(tolerance) is not dict and (isinstance(tolerance, bool) or
             not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0))):
        raise CaptureError("bounded Qwen4 comparison and recovery policy required")
    prompt = [int(value) for value in token_ids]
    if (not prompt or any(type(value) is not int or value < 0 for value in token_ids) or
            len(prompt) + max_tokens + recovery_tokens > int(config["max_position_embeddings"])):
        raise CaptureError("Qwen4 media prompt and recovery exceed bounded context")
    if type(media) is not dict or not any(media.get(key) is not None for key in
            ("pixel_values", "pixel_values_videos")):
        raise CaptureError("at least one original image/video processor fixture is required")
    for name, role, digest in ((item.get("path"), item.get("role"), item.get("sha256"))
                               for item in artifacts if type(item) is dict):
        if (type(name) is not str or not name or type(role) is not str or not role or
                type(digest) is not str or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest)):
            raise CaptureError("each measurement artifact requires an exact path, role and SHA-256")
    if any(type(item) is not dict for item in artifacts):
        raise CaptureError("Qwen4 measurement artifacts must be explicit records")

    def logits(result):
        value = result.get("logits") if isinstance(result, dict) else getattr(result, "logits", None)
        value = np.asarray(value)
        if value.ndim == 2:
            value = value[None, :, :]
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != config["vocab_size"] or not np.isfinite(value).all():
            raise CaptureError("complete finite Qwen4 media-conditioned logits required")
        return value[0].astype(np.float32, copy=True)

    def routes(result):
        values = result.get("routes") if isinstance(result, dict) else getattr(result, "routes", None)
        if values is None:
            routing = (result.get('routing_weights') if isinstance(result,dict)
                       else getattr(result,'routing_weights',None))
            routing = np.asarray(routing)
            experts, top_k = config['num_experts'], config['num_experts_per_tok']
            if (routing.ndim != 4 or routing.shape[0] != 1 or
                    routing.shape[2:] != (config['num_hidden_layers'],experts) or
                    not np.isfinite(routing).all()):
                raise CaptureError('malformed Qwen4 executed routing weights')
            values=[]
            for layer in range(config['num_hidden_layers']):
                scores=routing[0,:,layer]
                if np.any(np.count_nonzero(scores>0,axis=-1)<top_k):
                    raise CaptureError('Qwen4 routing weights cannot prove selected expert IDs')
                indices=np.argsort(-scores,axis=-1,kind='stable')[:,:top_k]
                values.append((indices,np.take_along_axis(scores,indices,axis=-1)))
        if not isinstance(values, (tuple, list)) or len(values) != config["num_hidden_layers"]:
            raise CaptureError("production Qwen4 consumer returned no complete executed expert routes")
        output = []
        for pair in values:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise CaptureError("Qwen4 route output must contain selected expert IDs and weights")
            indices, weights = (np.asarray(pair[0]), np.asarray(pair[1]))
            if indices.ndim == 3:
                indices = indices[0]
            if weights.ndim == 3:
                weights = weights[0]
            if (indices.ndim != 2 or indices.dtype.kind not in "iu" or
                    indices.shape[-1] != config["num_experts_per_tok"] or weights.shape != indices.shape or
                    np.any(indices < 0) or np.any(indices >= config["num_experts"]) or
                    not np.isfinite(weights).all()):
                raise CaptureError("malformed executed Qwen4 expert route observation")
            output.append((indices.astype(np.int64, copy=True), weights.astype(np.float32, copy=True)))
        return output

    route_rows = [[] for _ in range(config["num_hidden_layers"])]
    before_cancel = int(candidate.position)
    cancel_calls = {"count": 0}
    def cancel_in_media():
        cancel_calls["count"] += 1
        return cancel_calls["count"] >= 2
    try:
        candidate.forward(np.asarray(prompt, np.int64),
            **{name: value for name, value in media.items() if name in
               ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")},
            cancelled=cancel_in_media, capture_routing=True)
    except InterruptedError:
        pass
    else:
        raise CaptureError("Qwen4 candidate did not observe cancellation inside its media encoder")
    after_cancel = int(candidate.position)
    if before_cancel != after_cancel:
        raise CaptureError("cancelled Qwen4 media prefill mutated text position")

    candidate_rows, reference_rows = [], []
    generated, reference_generated = [], []
    finish = "length"
    pending = list(prompt)
    reference_delta = 0
    reference_name = reference_implementation('qwen4', route['backend'], 'qwen4_exp')
    reference_pins = reference_source_pins('qwen4', route['backend'], 'qwen4_exp')
    for step in range(max_tokens):
        if step == 0:
            candidate_output = candidate.forward(np.asarray(pending, np.int64),
                **{name: value for name, value in media.items() if name in
                   ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")},
                capture_routing=True)
            reference_output, _positions, reference_delta = independent_qwen4_media_forward(
                reference_text, reference_vision, np.asarray(pending, np.int64),
                publisher_config=candidate.config, vision_config=candidate.components.vision_config,
                **{name: value for name, value in media.items() if name in
                   ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")})
        else:
            candidate_output = candidate.forward(np.asarray(pending, np.int64), capture_routing=True)
            reference_output, _positions, reference_delta = independent_qwen4_media_forward(
                reference_text, reference_vision, np.asarray(pending, np.int64),
                publisher_config=candidate.config, vision_config=candidate.components.vision_config,
                rope_delta=reference_delta)
        c_rows, r_rows = logits(candidate_output), logits(reference_output)
        if c_rows.shape != r_rows.shape:
            raise CaptureError("Qwen4 full media-conditioned candidate/reference logit geometry differs")
        candidate_rows.append(c_rows[-1].copy())
        reference_rows.append(r_rows[-1].copy())
        for layer, values in enumerate(routes(candidate_output)):
            route_rows[layer].extend(values[0].tolist())
        token = int(np.argmax(c_rows[-1])); reference_token = int(np.argmax(r_rows[-1]))
        generated.append(token); reference_generated.append(reference_token)
        if token != reference_token:
            raise CaptureError("Qwen4 media-conditioned greedy tokens differ from independent reference")
        if config.get("eos_token_id") == token:
            finish = "eos"
            break
        pending = [token]
    candidate_logits = np.stack(candidate_rows)
    reference_logits = np.stack(reference_rows)
    error = float(np.max(np.abs(candidate_logits.astype(np.float64)-reference_logits.astype(np.float64))))
    if type(tolerance) is dict:
        atol, rtol = float(tolerance["atol"]), float(tolerance["rtol"])
        limits = atol + rtol * np.abs(reference_logits.astype(np.float64))
        allowed_tolerance = float(limits.max())
        if np.any(np.abs(candidate_logits.astype(np.float64)-reference_logits.astype(np.float64)) > limits):
            raise CaptureError(f"Qwen4 media-conditioned logits exceed tolerance: {error:.6g}")
    else:
        allowed_tolerance = float(tolerance)
    if type(tolerance) is not dict and error > allowed_tolerance:
        raise CaptureError(f"Qwen4 media-conditioned logits exceed tolerance: {error:.6g}")
    if int(candidate.position) != len(prompt) + len(generated) - 1 or reference_text.position != candidate.position:
        raise CaptureError("Qwen4 candidate/reference media continuation positions differ")
    from .native_family_capture import _cache_digests
    cache_owner = candidate.text
    if type(cache_owner).__name__ == "NativeQwen4PublicTextDecoder":
        cache_bytes, cache_layers = _cache_digests(cache_owner, config["num_hidden_layers"])
    else:
        cache_bytes, cache_layers = _cache_digests(cache_owner, config["num_hidden_layers"])
    snapshot = candidate.snapshot()
    snapshot_position = int(candidate.position)
    def continue_owner(owner, first, count, is_candidate):
        ids = int(first); output = []
        for _ in range(count):
            if is_candidate:
                result = owner.forward(np.asarray([[ids]], dtype=np.int64), capture_routing=True)
            else:
                result, _positions, _delta = independent_qwen4_media_forward(
                    owner, reference_vision, np.asarray([[ids]], dtype=np.int64),
                    publisher_config=candidate.config, vision_config=candidate.components.vision_config,
                    rope_delta=reference_delta)
            ids = int(np.argmax(logits(result)[-1])); output.append(ids)
            if config.get("eos_token_id") == ids:
                break
        return output
    reference_snapshot = None
    recovery_failure = None
    try:
        reference_snapshot = reference_text.snapshot()
        continuation = continue_owner(candidate, generated[-1], recovery_tokens, True)
        reference_continuation = continue_owner(reference_text, reference_generated[-1], recovery_tokens, False)
        candidate.restore(snapshot); reference_text.restore(reference_snapshot)
        restored_position = int(candidate.position)
        replay = continue_owner(candidate, generated[-1], recovery_tokens, True)
        candidate.restore(snapshot); reference_text.restore(reference_snapshot)
        if (restored_position != snapshot_position or replay != continuation or
                continuation != reference_continuation):
            raise CaptureError("Qwen4 candidate/reference snapshot restore did not reproduce continuation")
    except BaseException as error:
        recovery_failure = error
        raise
    finally:
        failures = []
        for owner, receipt in ((candidate, snapshot), (reference_text, reference_snapshot)):
            if receipt is not None:
                try:
                    owner.release_snapshot(receipt)
                except BaseException as error:
                    failures.append(error)
        if failures:
            if recovery_failure is None:
                raise RuntimeError("Qwen4 media measurement snapshot cleanup failed") from failures[0]
            for failure in failures:
                recovery_failure.add_note("Qwen4 media snapshot cleanup failed: " + repr(failure))
    if reference_source_pins('qwen4', route['backend'], 'qwen4_exp') != reference_pins:
        raise CaptureError('independent Qwen4 reference source changed during capture')
    chain = row.get("callable_chain") or {}
    consumer = "qwen4_multimodal.Qwen4MultimodalDecoder"
    abi_owner = f"{type(candidate.text).__module__.rsplit('.', 1)[-1]}.{type(candidate.text).__name__}"
    if (consumer not in tuple(chain.get("execution", ())) or
            abi_owner not in tuple(chain.get("native_abi", ()))):
        raise CaptureError("exact Qwen4 publisher backend consumer/ABI owner is absent from its canonical binding")
    sources = reference_pins
    route_observations = {}
    for layer, rows in enumerate(route_rows):
        if not rows:
            raise CaptureError("Qwen4 media execution produced no selected-expert observations")
        route_observations[f"model.layers.{layer}"] = rows
    return {
        "case_id": row["case_id"], "route": [route[name] for name in ROUTE_FIELDS],
        "consumer": consumer, "abi_owner": abi_owner, "implementation": candidate.implementation,
        "reference": reference_name, "prompt_ids": prompt,
        "generated_ids": generated, "reference_generated_ids": reference_generated,
        "finish_reason": finish, "candidate_logits": candidate_logits,
        "reference_logits": reference_logits, "tolerance": allowed_tolerance,
        "cache": {"position": snapshot_position, "layers": config["num_hidden_layers"],
                  "layer_digests": cache_layers, "bytes": cache_bytes},
        "cache_schema": {"format": "qwen4-exp-media-physical-cache-v1",
                         "layers": config["num_hidden_layers"]},
        "routing_layers": route_observations,
        "recovery": {"snapshot_position": snapshot_position, "restored_position": restored_position,
                     "continuation_ids": continuation, "replay_ids": replay},
        "media_lifecycle": {"items": [
            *[{"kind": "image", "index": i} for i in range(0 if media.get("image_grid_thw") is None else len(media["image_grid_thw"]))],
            *[{"kind": "video", "index": i} for i in range(0 if media.get("video_grid_thw") is None else len(media["video_grid_thw"]))],
        ], "position_before_cancel": before_cancel, "position_after_cancel": after_cancel,
            "released_items": (0 if media.get("image_grid_thw") is None else len(media["image_grid_thw"])) +
                               (0 if media.get("video_grid_thw") is None else len(media["video_grid_thw"]))},
        "placement": {"backend": route["backend"],
                      "device": str(getattr(candidate.text, "device", "cpu")),
                      "host_fallback_operations": 0, "implementation": candidate.implementation},
        "artifacts": artifacts,
        "comparison_reference": {"implementation": reference_name, "sources": sources},
        "max_abs_logit_error": error,
    }


def run_qwen4_media_capture(configuration: dict[str, Any], *, offline_root: str | Path):
    row, comparison = _validate_media_configuration(configuration)
    from .wholemodel_validation.runner import pinned_artifacts
    root = Path(offline_root).resolve(strict=True)
    pins = pinned_artifacts({"artifacts": comparison["artifacts"]}, root)
    # The route-bound loader source and all selected production media sources
    # must be part of the same source-pinned comparison closure.
    required = {path.resolve() for path in _required_source_paths(row["backend"])}
    source_pins = {Path(item["resolved_path"]).resolve() for item in pins
                   if item["kind"] == "source"}
    if not required <= source_pins:
        raise CaptureError("validated artifact set omits Qwen4 production media source owners")
    from .wholemodel_validation.qwen4_media import Qwen4MediaCase
    from .wholemodel_validation.qwen35_media import load_media_fixture
    from .device_capture import _pin
    fixture = load_media_fixture(offline_root=root,
        fixture_config=comparison["fixture_config"], artifact_pins=comparison["artifacts"],
        max_bytes=comparison["max_observation_bytes"], case_class=Qwen4MediaCase)
    decoder_options = comparison["candidate"]["options"]["decoder_options"]
    candidate = reference_source = reference_text = reference_vision = None
    primary = None
    try:
        # Public registration transfers and closes the resident text decoder.
        # Load a separate admitted resident source before opening the candidate,
        # then release it after copying independent reference tensors.
        reference_options=copy.deepcopy(decoder_options)
        reference_options['route']['backend']='native'
        reference_options.update(device=None,segment_runtime_library=None,
            segment_work_bytes=0,segment_layer_span=0,native_composition=False,
            native_store_runtime_library=None,native_expert_bytes=0,native_ple_bytes=0,
            expert_bytes=0,projection_bytes=0,ple_bytes=0)
        if reference_options['text_library'] is None:
            raise CaptureError('independent Qwen4 reference requires a compiled native text library')
        reference_source=load_qwen4_publisher_candidate(offline_root=root,options=reference_options)
        reference_text, reference_vision = create_independent_qwen4_reference(
            reference_source, **comparison["oracle"]["options"])
        reference_source.close(); reference_source=None
        candidate = load_qwen4_publisher_candidate(offline_root=root, options=decoder_options)
        checkpoint_root = (root / decoder_options["checkpoint"]).resolve(strict=True)
        checkpoint_config = (checkpoint_root / "config.json").resolve()
        pins_for_receipt = []
        for item in pins:
            path = Path(item["resolved_path"]).resolve(strict=True)
            role = item["kind"]
            if role == "config" and path != checkpoint_config:
                # Processor fixture declarations are configuration artifacts,
                # not the checkpoint identity config required by the gate.
                role = "processor"
            pins_for_receipt.append({'path':str(path),'role':role,'sha256':item['observed_sha256']})
        measured = measure_qwen4_media_route(
            candidate=candidate, reference_text=reference_text, reference_vision=reference_vision,
            token_ids=fixture.token_ids,
            media={name: getattr(fixture, name) for name in
                   ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")},
            route=configuration["route"], config=dict(candidate.components.text_config),
            artifacts=pins_for_receipt, tolerance=comparison["tolerance"],
            max_tokens=comparison["generation_steps"],
            recovery_tokens=comparison.get("recovery_tokens", 2))
        return {"classification": "measured-route-observations-not-yet-published",
                "measurement": measured, "receipt_written": False}
    except BaseException as error:
        primary = error
        raise
    finally:
        failures = []
        for owner in (reference_vision, reference_text, reference_source, candidate):
            close = getattr(owner, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    failures.append(cleanup)
        if failures and primary is None:
            raise RuntimeError("Qwen4 media capture owner cleanup failed") from failures[0]
        if failures and primary is not None:
            for cleanup in failures:
                primary.add_note("Qwen4 media capture cleanup failed: " + repr(cleanup))


def add_qwen4_media_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "qwen4-media", help="capture and publish an exact Qwen4 publisher media route")
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--configuration", type=Path)
    source.add_argument('--checkpoint',type=Path)
    parser.add_argument("--offline-root", type=Path)
    parser.add_argument('--source-format',choices=tuple(_SOURCE_TO_PERSISTED))
    parser.add_argument('--backend',choices=_BACKENDS)
    parser.add_argument('--case-id')
    parser.add_argument('--prompt')
    parser.add_argument('--image',type=Path,action='append',default=[])
    parser.add_argument('--media-library',type=Path)
    parser.add_argument('--text-library',type=Path)
    parser.add_argument('--runtime-library',type=Path)
    parser.add_argument('--device')
    parser.add_argument('--max-tokens',type=int,default=4)
    parser.add_argument('--context-tokens',type=int,default=512)
    parser.add_argument('--max-weight-bytes',type=int,default=128*1024**3)
    parser.add_argument('--max-runtime-bytes',type=int,default=32*1024**3)
    from .device_capture import DEFAULT_EVIDENCE, DEFAULT_RECEIPTS
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)


def run_qwen4_media_subcommand(args) -> int:
    try:
        if args.checkpoint is not None:
            measured=capture_qwen4_publisher_route(args.checkpoint,source_format=args.source_format,
                backend=args.backend,case_id=args.case_id,prompt=args.prompt,image_paths=args.image,
                media_library=args.media_library,text_library=args.text_library,
                runtime_library=args.runtime_library,device=args.device,max_tokens=args.max_tokens,
                context_tokens=args.context_tokens,max_weight_bytes=args.max_weight_bytes,
                max_runtime_bytes=args.max_runtime_bytes)
            result={'measurement':measured}
        else:
            if args.offline_root is None:
                raise CaptureError('--offline-root is required with --configuration')
            path = Path(args.configuration).resolve(strict=True)
            if path.stat().st_size > _MAX_CONFIG_BYTES:
                raise CaptureError("bounded Qwen4 media capture configuration required")
            with path.open("r", encoding="utf-8") as stream:
                config = json.load(stream)
            result = run_qwen4_media_capture(config, offline_root=args.offline_root)
        from .media_route_capture import publish_media_measurements
        published = publish_media_measurements(
            result["measurement"], evidence_root=args.evidence_root,
            receipts_path=args.receipts)
        result = {"classification": "canonical-route-receipt-validated",
                  "case_id": result["measurement"]["case_id"],
                  "receipt": published, "receipt_written": True}
    except (CaptureError, ValueError, OSError, ImportError, KeyError, TypeError,
            RuntimeError, MemoryError) as error:
        print("Qwen4 media comparison rejected: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def capture_qwen4_publisher_route(checkpoint, *, source_format, backend, prompt,
        image_paths, media_library, text_library, runtime_library=None, device=None,
        case_id=None, max_tokens=4, context_tokens=512,
        max_weight_bytes=128*1024**3, max_runtime_bytes=32*1024**3):
    """Direct Colab entrypoint using pinned local images and production loaders."""
    from .device_capture import _pin
    from .publisher_media_files import read_regular
    from .publisher_media_processor import PublisherMediaProcessor
    from .publisher_conditional_loader import load_publisher_conditional_decoder
    from .source_scope import require_checkpoint_inventory_admission
    from .universal_engine import ModelRoute
    from .device_capture import canonical_row
    if source_format not in _SOURCE_TO_PERSISTED or backend not in _BACKENDS:
        raise CaptureError('exact admitted Qwen4 publisher tuple required')
    if not isinstance(prompt,str) or not prompt.strip() or not image_paths or len(image_paths)>64:
        raise CaptureError('rendered prompt and one to 64 local images required')
    if any(type(value) is not int or value<=0 for value in
           (max_tokens,context_tokens,max_weight_bytes,max_runtime_bytes)):
        raise CaptureError('positive bounded Qwen4 capture admissions required')
    if (backend=='device') != bool(device) or (backend=='public-range') != (runtime_library is not None):
        raise CaptureError('Qwen4 device and public runtime must match the selected backend')
    if media_library is None or text_library is None:
        raise CaptureError('compiled media and native reference text libraries required')
    checkpoint=Path(checkpoint).resolve(strict=True)
    media_library=Path(media_library).resolve(strict=True)
    text_library=Path(text_library).resolve(strict=True)
    runtime_library=None if runtime_library is None else Path(runtime_library).resolve(strict=True)
    route=ModelRoute('qwen4','qwen4_exp',backend,'fp32',source_format,_SOURCE_TO_PERSISTED[source_format])
    row=canonical_row(route,case_id=case_id)
    inventory=require_checkpoint_inventory_admission(checkpoint,family='qwen4',model_type='qwen4_exp',
        backend=backend,arithmetic='fp32',max_metadata_bytes=min(max_weight_bytes,256*1024**2))
    source_types={item.dtype for name,item in inventory.tensors.items()
                  if item.dtype in _SOURCE_TO_PERSISTED and not name.endswith(('.weight_scale_inv','_scale_inv'))}
    if source_format not in source_types or not source_types<={source_format,'F32'}:
        raise CaptureError('Qwen4 checkpoint source encoding differs from selected route')
    paths=[checkpoint/'config.json',*sorted(checkpoint.glob('*.safetensors'))]
    artifacts=[_pin(path,'config' if path.name=='config.json' else 'weights') for path in paths]
    artifacts += [_pin(path,'runtime') for path in (media_library,text_library)
                  + (() if runtime_library is None else (runtime_library,))]
    artifacts += [_pin(path,'runtime') for path in _required_source_paths(backend)]
    image_paths=[Path(path).resolve(strict=True) for path in image_paths]
    artifacts += [_pin(path,'media_input') for path in image_paths]
    processor=reference_source=reference_text=reference_vision=candidate=None
    primary=None
    try:
        processor=PublisherMediaProcessor(checkpoint,family='qwen4_exp',max_bytes=512*1024**2,
            max_pixels=40_000_000,max_audio_samples=1)
        token_ids,media=processor.prepare(prompt,images=tuple(
            read_regular(path,max_bytes=64*1024**2) for path in image_paths))
        artifacts += [_pin(path,'processor') for path in processor._files]
        common=dict(family='qwen4_exp',library=media_library,text_library=text_library,
            max_weight_bytes=max_weight_bytes,max_bytes=max_runtime_bytes,context_tokens=context_tokens,
            max_conversion_bytes=min(max_weight_bytes,max_runtime_bytes//4),max_ple_bytes=0,
            max_expert_bytes=0,max_snapshots=4)
        reference_source=load_publisher_conditional_decoder(checkpoint,**common)
        reference_text,reference_vision=create_independent_qwen4_reference(reference_source,max_bytes=max_runtime_bytes)
        reference_source.close();reference_source=None
        candidate=load_publisher_conditional_decoder(checkpoint,**common,text_device=device,
            native_composition=backend=='public-range',segment_runtime_library=runtime_library,
            segment_layer_span=1 if runtime_library else 0,
            segment_work_bytes=min(max_runtime_bytes//8,512*1024**2) if runtime_library else 0)
        result=measure_qwen4_media_route(candidate=candidate,reference_text=reference_text,
            reference_vision=reference_vision,token_ids=token_ids,media=media,
            route={name:getattr(route,name) for name in ROUTE_FIELDS},
            config=dict(candidate.components.text_config),artifacts=artifacts,
            max_tokens=max_tokens,recovery_tokens=2,tolerance=1e-3)
        inventory.assert_unchanged()
        for pin in artifacts:
            if _pin(Path(pin['path']),pin['role'])!=pin:
                raise CaptureError('Qwen4 artifact changed during capture')
        return result
    except BaseException as error:
        primary=error;raise
    finally:
        failures=[]
        for owner in (candidate,reference_vision,reference_text,reference_source,processor):
            close=getattr(owner,'close',None)
            if callable(close):
                try:close()
                except BaseException as error:failures.append(error)
        if failures:
            from .failure_reporting import add_failure_note,failure_summary
            if primary is None:raise RuntimeError('Qwen4 capture cleanup failed') from failures[0]
            for error in failures:add_failure_note(primary,'Qwen4 capture cleanup failed: '+failure_summary(error))


__all__ = ["ROUTE_FIELDS", "_canonical_route", "_required_source_paths",
           "load_qwen4_publisher_candidate", "run_qwen4_media_capture",
           "add_qwen4_media_subparser", "run_qwen4_media_subcommand",
           "independent_qwen4_media_forward", "create_independent_qwen4_reference",
           "measure_qwen4_media_route"]
