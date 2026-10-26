"""Exact, fail-closed GLM5 publisher image/video route observations.

This module returns measured values to the shared Phase 1 receipt finalizer;
it never creates or promotes evidence on its own.  The production composite
executes vision and text, while the comparison uses the separately reviewed
Torch/C text implementation and an independent NumPy vision implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from .device_capture import CaptureError, _pin, canonical_row, compile_segment_runtime
from .native_family_capture import compile_native_text_kernel, _torch_device_details, _torch_runtime_pin
from .publisher_media_files import read_regular


_PERSISTED = {"F32": "float32", "F16": "float32", "BF16": "float32",
              "F8_E4M3": "fp8-e4m3-b128-f32"}
_IDENTITY = ("glm5-next", "glm5_next")


def glm5_publisher_case_id(backend: str, source_format: str) -> str:
    """Return the pinned publisher route id, after canonical admission."""
    if backend not in ("native", "device", "public-range") or source_format not in _PERSISTED:
        raise CaptureError("unsupported exact GLM5 publisher backend/source format")
    from .universal_engine import ModelRoute
    route = ModelRoute(*_IDENTITY, backend, "fp32", source_format, _PERSISTED[source_format])
    row = canonical_row(route)
    return str(row["case_id"])


def capture_glm5_publisher_route(
        checkpoint: str | Path, *, source_format: str, backend: str,
        media_library: str | Path, text_library: str | Path | None = None,
        runtime_library: str | Path | None = None, device: str | None = None,
        reference_device: str | None = None, reference_library: str | Path | None = None,
        prompt: str, image_paths: Sequence[str | Path] = (),
        video_frame_paths: Sequence[str | Path] = (), video_indices: Sequence[int] = (),
        video_total_frames: int | None = None, video_fps: float | None = None,
        prompt_ids: Sequence[int] | None = None, max_tokens: int = 4,
        recovery_tokens: int = 2, context_tokens: int = 256,
        max_weight_bytes: int = 128 * 1024**3, max_runtime_bytes: int = 32 * 1024**3,
        max_conversion_bytes: int = 8 * 1024**3, max_expert_bytes: int = 8 * 1024**3,
        max_media_bytes: int = 2 * 1024**3, tolerance: float = 1e-3,
        case_id: str | None = None) -> dict[str, Any]:
    """Execute and return route-bound measurements; no receipt is fabricated.

    Exactly one media modality is accepted. ``prompt_ids`` is an optional
    explicit processor-independent override only for controlled capture; when
    supplied it must still agree with processor-produced expansion geometry.
    """
    from .device_observations import reference_implementation, reference_source_pins
    from .publisher_checkpoint.storage import snapshot
    from .publisher_conditional_loader import load_publisher_conditional_decoder
    from .publisher_media_processor import PublisherMediaProcessor
    from .source_scope import require_source_inventory_admission
    from .publisher_glm5_loader import load_publisher_glm5_text_decoder

    if source_format not in _PERSISTED:
        raise CaptureError("GLM5 publisher capture source format must be F32/F16/BF16/F8_E4M3")
    if backend not in ("native", "device", "public-range"):
        raise CaptureError("GLM5 publisher capture backend is not admitted")
    if type(prompt) is not str or not prompt.strip():
        raise CaptureError("nonempty rendered media prompt required")
    images, frames = tuple(image_paths), tuple(video_frame_paths)
    if bool(images) == bool(frames):
        raise CaptureError("provide images or selected video frames, exclusively")
    if len(images or frames) > 64:
        raise CaptureError("bounded publisher media item count required")
    if frames:
        if (len(video_indices) != len(frames) or type(video_total_frames) is not int or
                type(video_fps) not in (int, float) or isinstance(video_fps, bool)):
            raise CaptureError("video capture requires ordered indices, total frames, and FPS")
    elif video_indices or video_total_frames is not None or video_fps is not None:
        raise CaptureError("video metadata is inapplicable to image capture")
    if any(type(value) is not int or value <= 0 for value in
           (max_tokens, recovery_tokens, context_tokens, max_weight_bytes, max_runtime_bytes,
            max_conversion_bytes, max_expert_bytes, max_media_bytes)):
        raise CaptureError("positive bounded GLM5 capture budgets required")
    if (type(tolerance) not in (float, int) or isinstance(tolerance, bool) or
            not np.isfinite(tolerance) or tolerance < 0):
        raise CaptureError("finite nonnegative comparison tolerance required")

    # Backend/library combinations are checked before opening the checkpoint.
    media_library = Path(media_library).resolve(strict=True)
    if not media_library.is_file():
        raise CaptureError("compiled production GLM5 media library required")
    if backend in ("native", "public-range"):
        if text_library is None:
            raise CaptureError("compiled production GLM5 text library required")
        text_library = Path(text_library).resolve(strict=True)
        if not text_library.is_file():
            raise CaptureError("compiled GLM5 text library is not a file")
    elif text_library is not None:
        raise CaptureError("device text execution does not accept a native text candidate library")
    if backend == "device":
        if not device or reference_device is not None or reference_library is None or runtime_library is not None:
            raise CaptureError("device route requires --device and --reference-library only")
        reference_library = Path(reference_library).resolve(strict=True)
        if not reference_library.is_file():
            raise CaptureError("compiled independent native GLM5 reference library required")
    elif backend == "native":
        if not reference_device or device is not None or reference_library is not None or runtime_library is not None:
            raise CaptureError("native route requires an explicit Torch --reference-device only")
    else:
        if (not reference_device or device is not None or reference_library is not None or
                runtime_library is None):
            raise CaptureError("public-range route requires runtime library and Torch --reference-device")
        runtime_library = Path(runtime_library).resolve(strict=True)
        if not runtime_library.is_file():
            raise CaptureError("compiled Segment/Edge public runtime required")

    checkpoint = Path(checkpoint).resolve(strict=True)
    if not checkpoint.is_dir():
        raise CaptureError("exact local GLM5 publisher checkpoint directory required")
    persisted = _PERSISTED[source_format]
    from .universal_engine import ModelRoute
    route = ModelRoute(*_IDENTITY, backend, "fp32", source_format, persisted)
    row = canonical_row(route, case_id=case_id)
    resolved_case = row["case_id"]
    sources_before = reference_source_pins("glm5-next", backend, "glm5_next")
    source_paths_before = _candidate_sources(backend)
    source_pins_before = [_pin(path, "runtime") for path in source_paths_before]
    runtime_paths_before = [media_library]
    if text_library is not None: runtime_paths_before.append(text_library)
    if runtime_library is not None: runtime_paths_before.append(runtime_library)
    if reference_library is not None: runtime_paths_before.append(reference_library)
    runtime_pins_before = [_pin(path, "runtime") for path in runtime_paths_before]
    media_paths = tuple(images or frames)
    media_pins_before = [_pin(Path(path), "media_input") for path in media_paths]
    inventory = snapshot(checkpoint, max_metadata_bytes=min(max_weight_bytes, 256 * 1024**2),
                         max_tensors=1_000_000, max_shards=4096)
    scale_names = {name[:-7] + ".weight_scale_inv" if name.endswith(".weight")
                   else name + "_scale_inv"
                   for name, record in inventory.tensors.items() if record.dtype == "F8_E4M3"}
    source_tensors = {name: record for name, record in inventory.tensors.items()
                      if name not in scale_names and record.dtype in _PERSISTED}
    relevant = {name: record for name, record in source_tensors.items()
                if record.dtype == source_format}
    if not relevant or any(record.dtype not in (source_format,'F32') for record in source_tensors.values()):
        raise CaptureError("checkpoint tensor dtypes do not match the selected exact route")
    admitted = require_source_inventory_admission(
        family="glm5-next", model_type="glm5_next", backend=backend,
        arithmetic="fp32", persisted_format=persisted, inventory=relevant)
    if admitted != (source_format,):
        raise CaptureError("checkpoint inventory resolves to another GLM5 source format")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "glm5_next":
        raise CaptureError("pinned publisher config is not GLM5-next")

    image_blobs: list[bytes] = []
    video_blobs: list[bytes] = []
    media: dict[str, np.ndarray] = {}
    processor = candidate = reference = independent_vision = None
    candidate_forward = None
    recovery_snapshot = None
    primary: BaseException | None = None
    route_chunks: list[np.ndarray] = []
    try:
        input_paths = images if images else frames
        blobs = [read_regular(Path(path), max_bytes=max_media_bytes) for path in input_paths]
        if images:
            image_blobs = blobs
        else:
            video_blobs = blobs
        processor = PublisherMediaProcessor(checkpoint, family="glm5_next",
            max_bytes=max_media_bytes, max_pixels=100_000_000,
            max_audio_samples=1, source_version="5.16.1")
        metadata = None
        if frames:
            metadata = ({"frames_indices": tuple(video_indices),
                         "total_num_frames": video_total_frames, "fps": video_fps},)
        cooked_ids, media = processor.prepare(
            prompt, images=tuple(image_blobs),
            videos=((tuple(video_blobs),) if frames else ()), video_metadata=metadata)
        if prompt_ids is not None and tuple(prompt_ids) != cooked_ids:
            raise CaptureError("explicit prompt IDs differ from the pinned local processor output")
        prompt_ids = cooked_ids
        if (not prompt_ids or len(prompt_ids) + max_tokens + recovery_tokens > context_tokens or
                any(type(value) is not int or value < 0 for value in prompt_ids)):
            raise CaptureError("processor prompt and bounded generation exceed context admission")

        kwargs: dict[str, Any] = dict(
            family="glm5_next", library=media_library, text_library=text_library,
            max_weight_bytes=max_weight_bytes, max_bytes=max_runtime_bytes,
            context_tokens=context_tokens, max_conversion_bytes=max_conversion_bytes,
            max_expert_bytes=max_expert_bytes, max_snapshots=4)
        if backend == "device":
            kwargs.update(text_device=device)
        else:
            kwargs.update(text_library=text_library)
        if backend == "public-range":
            kwargs.update(segment_runtime_library=runtime_library,
                segment_work_bytes=max(4 * 1024**2, min(max_runtime_bytes // 4, 512 * 1024**2)),
                segment_layer_span=1)
        candidate = load_publisher_conditional_decoder(checkpoint, **kwargs)
        text_config = dict(candidate.components.text_config)
        # Publisher checkpoints use model_type=glm5_next, not the separate
        # resident glm5_next_text identity. Build references through the
        # publisher text inventory owner so config, expert namespace, FP8
        # scales, and source format are validated against this exact wrapper.
        if backend == "device":
            reference_library_for_text = reference_library
            reference_device_for_text = None
        else:
            reference_library_for_text = None
            reference_device_for_text = reference_device
        reference = load_publisher_glm5_text_decoder(
            checkpoint, library=reference_library_for_text,
            expert_library=media_library,
            context_tokens=context_tokens, max_weight_bytes=max_weight_bytes,
            max_bytes=max_runtime_bytes, max_expert_bytes=max_expert_bytes,
            max_conversion_bytes=max_conversion_bytes, text_device=reference_device_for_text,
            max_snapshots=4)

        from .independent_glm5_vision import IndependentGlm5VisionEncoder
        vision_tensors = {"model.visual." + name: value
                          for name, value in candidate.vision.weights.items()}
        independent_vision = IndependentGlm5VisionEncoder(
            candidate.components.vision_config, vision_tensors,
            max_bytes=max_runtime_bytes // 2)

        original_forward = candidate.text.forward
        def observe_forward(*args, **kwargs):
            kwargs["capture_routing"] = True
            result = original_forward(*args, **kwargs)
            routes = _routes_from_result(result, candidate.text)
            route_chunks.append(routes)
            return result
        candidate_forward = candidate.text.forward
        candidate.text.forward = observe_forward

        # Exercise cancellation after admission and at the media encoder's
        # first internal boundary, then prove the text/cache position did not
        # advance. Successful inference below uses the same owned media arrays.
        before_cancel = int(candidate.position)
        checks = {"count": 0}
        def cancel_in_vision():
            checks["count"] += 1
            # Composite boundary, placeholder scan, vision entry, then the
            # first block boundary: cancellation occurs after media work starts.
            return checks["count"] >= 4
        try:
            candidate.forward(np.asarray(prompt_ids, np.int64),
                **_media_kwargs(media), cancelled=cancel_in_vision)
        except InterruptedError:
            pass
        else:
            raise CaptureError("GLM media cancellation was not observed within the encoder")
        after_cancel = int(candidate.position)
        if before_cancel != after_cancel:
            raise CaptureError("cancelled GLM media prefill mutated text continuation state")

        candidate_logits, generated, finish = _greedy_publisher(
            candidate, prompt_ids, media, max_tokens, text_config, route_chunks)
        candidate.text.forward = candidate_forward
        candidate_forward = None
        reference_logits, reference_generated, _ = _greedy_reference(
            reference, prompt_ids, media, independent_vision,
            candidate.config, max_tokens, text_config)
        if generated != reference_generated or candidate_logits.shape != reference_logits.shape:
            raise CaptureError("GLM publisher candidate/reference token or logit geometry differs")
        error = float(np.max(np.abs(candidate_logits.astype(np.float64) -
                                    reference_logits.astype(np.float64))))
        if error > tolerance:
            raise CaptureError(f"GLM publisher media logits exceed tolerance: {error:.6g}")

        expected_processed = len(prompt_ids) + len(generated) - 1
        if int(candidate.position) != expected_processed:
            raise CaptureError("GLM composite cache position differs from processed media/text tokens")
        from .native_family_capture import _cache_digests
        text_config = dict(candidate.components.text_config)
        cache_bytes, cache_layers = _cache_digests(candidate.text, int(text_config["num_hidden_layers"]))

        # Recovery is measured at the same composite ownership boundary used
        # by serving. The snapshot belongs to candidate; it is never reused by
        # the reference or another checkpoint.
        recovery_snapshot = candidate.snapshot()
        snapshot_position = int(candidate.position)
        continuation = _continue_publisher(candidate, generated[-1], recovery_tokens, text_config)
        if candidate.position != snapshot_position + len(continuation):
            raise CaptureError("GLM recovery continuation position differs")
        candidate.restore(recovery_snapshot)
        restored_position = int(candidate.position)
        replay = _continue_publisher(candidate, generated[-1], recovery_tokens, text_config)
        candidate.restore(recovery_snapshot)
        candidate.release_snapshot(recovery_snapshot)
        recovery_snapshot = None
        if restored_position != snapshot_position or replay != continuation:
            raise CaptureError("GLM publisher restore did not reproduce continuation")
        processor._unchanged()
        inventory.assert_unchanged()

        implementation = str(candidate.implementation)
        consumer = "glm5_multimodal.Glm5MultimodalDecoder"
        abi_owner = _abi_owner(row, backend)
        sources = reference_source_pins("glm5-next", backend, "glm5_next")
        if sources != sources_before:
            raise CaptureError("independent GLM reference sources changed during capture")
        for pin in (*source_pins_before, *runtime_pins_before, *media_pins_before):
            if _pin(Path(pin["path"]), pin["role"]) != pin:
                raise CaptureError("GLM capture source/runtime/media artifact changed during execution")
        reference_name = reference_implementation("glm5-next", backend, "glm5_next")
        artifacts = [_pin(checkpoint / "config.json", "config")]
        artifacts.extend(_pin(path, "weights") for path in sorted(checkpoint.glob("*.safetensors")))
        artifacts.extend(source_pins_before)
        artifacts.extend(runtime_pins_before)
        artifacts.extend({"role": "processor", "path": str(path), "sha256": digest}
                         for path, digest in processor._files.items())
        artifacts.extend(media_pins_before)
        artifacts.extend({"role": "reference_source", "path": str(Path(__file__).resolve().parents[2] / item["module"]),
                          "sha256": item["sha256"]} for item in sources)
        placement = {"backend": backend,
                     "device": str(candidate.text.device) if backend == "device" else "cpu",
                     "host_fallback_operations": 0,
                     "vision_device": "cpu",
                     "vision_execution": "native-media-ops",
                     "media_preprocessing": "local-pinned-transformers-5.16.1",
                     "components": [
                         {"component": "vision", "device": "cpu",
                          "implementation": str(candidate.vision.implementation)},
                         {"component": "text",
                          "device": str(candidate.text.device) if backend == "device" else "cpu",
                          "implementation": str(candidate.text.implementation)},
                     ],
                     "implementation": implementation}
        if backend == "device":
            placement.update(_torch_device_details(candidate.text))
            runtime_pins_before.append(_torch_runtime_pin(candidate.text))
        if backend in ("native", "public-range"):
            placement["reference_runtime"] = _torch_device_details(reference)
            runtime_pins_before.append(_torch_runtime_pin(reference))
        elif backend == "device":
            runtime_pins_before.append(_pin(reference_library, "runtime"))
        return {
            "case_id": resolved_case, "route": list(route.as_tuple()),
            "consumer": consumer, "abi_owner": abi_owner,
            "implementation": implementation, "reference": reference_name,
            "prompt_ids": list(prompt_ids), "generated_ids": generated,
            "finish_reason": finish, "candidate_logits": candidate_logits,
            "reference_logits": reference_logits, "reference_generated_ids": reference_generated,
            "tolerance": float(tolerance),
            "cache": {"position": expected_processed, "layers": text_config["num_hidden_layers"],
                      "layer_digests": cache_layers, "bytes": cache_bytes},
            "cache_schema": {"format": "openagent.glm5-kda-dsa-kv.v1",
                             "layers": text_config["num_hidden_layers"]},
            "routing_layers": {f"model.layers.{index}": values
                               for index, values in enumerate(_merge_route_chunks(route_chunks))},
            "recovery": {"snapshot_position": snapshot_position,
                         "restored_position": restored_position,
                         "continuation_ids": continuation, "replay_ids": replay},
            "media_lifecycle": {"items": [{"kind": "video" if frames else "image",
                                             "index": index} for index in range(len(input_paths))],
                                "position_before_cancel": before_cancel,
                                "position_after_cancel": after_cancel,
                                "released_items": len(input_paths)},
            "placement": placement, "artifacts": artifacts,
            "comparison_reference": {"implementation": reference_name, "sources": sources},
            "max_abs_logit_error": error,
        }
    except BaseException as error:
        primary = error
        raise
    finally:
        # The processor returns owned arrays. Drop them on every exit path,
        # while attempting every independent owner close even if one fails.
        for value in tuple(media.values()):
            if isinstance(value, np.ndarray):
                value = None
        media.clear()
        image_blobs.clear(); video_blobs.clear()
        failures = []
        if candidate is not None and candidate_forward is not None:
            try:
                candidate.text.forward = candidate_forward
            except BaseException as cleanup:
                failures.append(cleanup)
        if candidate is not None and recovery_snapshot is not None:
            try:
                candidate.release_snapshot(recovery_snapshot)
            except BaseException as cleanup:
                failures.append(cleanup)
        for owner in (candidate, reference, independent_vision, processor):
            close = getattr(owner, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as cleanup:
                    failures.append(cleanup)
        if failures and primary is None:
            raise RuntimeError("GLM publisher capture cleanup failed") from failures[0]
        if failures and primary is not None and hasattr(primary, "add_note"):
            for cleanup in failures:
                primary.add_note("GLM capture cleanup remains pending: " + repr(cleanup))


def _load_native_reference_weights(checkpoint, max_weight_bytes, context_tokens):
    from .native_checkpoint import read_native_text_checkpoint
    return read_native_text_checkpoint(checkpoint, model_type="glm5_next_text",
        max_weight_bytes=max_weight_bytes, context_tokens=context_tokens,
        source_backend="device")


def _media_kwargs(media):
    return {name: media[name] for name in
            ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
            if name in media}


def _expand_video_grids(grids):
    grids = np.asarray(grids)
    if grids.ndim != 2 or grids.shape[1] != 3 or grids.dtype.kind not in "iu":
        raise CaptureError("publisher video grid must have exact [frames,height,width] rows")
    count = sum(int(row[0]) for row in grids)
    expanded = np.empty((count, 3), np.int64)
    offset = 0
    for temporal, height, width in grids:
        end = offset + int(temporal)
        expanded[offset:end] = (1, int(height), int(width))
        offset = end
    return expanded


def _reference_embeddings(decoder, ids, media, vision_encoder, config):
    ids = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    if hasattr(decoder, "embed_tokens"):
        embeddings = decoder.embed_tokens(ids)
    else:
        indices = (decoder.torch.tensor(ids, dtype=decoder.torch.long, device=decoder.device)
                   if hasattr(decoder, "torch") else ids)
        embeddings = decoder.weights["embed_tokens.weight"][indices]
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy().copy()
    embeddings = np.asarray(embeddings, dtype=np.float32).copy()
    text_config = decoder.contract
    image_id = int(config.get("image_token_id", 154854))
    start_id = int(config.get("video_start_token_id", 154832))
    end_id = int(config.get("video_end_token_id", 154833))
    spans = np.zeros(ids.shape, dtype=bool)
    inside = False
    for index, token in enumerate(ids[0]):
        if token == start_id:
            if inside: raise CaptureError("nested GLM video spans are unsupported")
            inside = True
        elif token == end_id:
            if not inside: raise CaptureError("unmatched GLM video end span")
            inside = False
        spans[0, index] = inside
    if inside: raise CaptureError("GLM media prompt has unterminated video span")
    placeholders = ids == image_id
    for is_video, pixels_key, grid_key in (
            (False, "pixel_values", "image_grid_thw"),
            (True, "pixel_values_videos", "video_grid_thw")):
        if pixels_key not in media:
            continue
        select = placeholders & (spans if is_video else ~spans)
        pixels, grids = media[pixels_key], media.get(grid_key)
        if grids is None or int(select.sum()) == 0:
            raise CaptureError("GLM processor media does not align with expanded placeholder tokens")
        if is_video:
            grids = _expand_video_grids(grids)
        features = np.concatenate(vision_encoder.forward(pixels, grids), axis=0)
        if features.shape != (int(select.sum()), text_config.hidden_size):
            raise CaptureError("independent GLM vision features differ from placeholder geometry")
        embeddings[select] = features
    return embeddings


def _result_logits(result):
    value = result.get("logits") if isinstance(result, dict) else getattr(result, "logits", None)
    if value is None: raise CaptureError("GLM decoder returned no logits")
    if hasattr(value, "detach"): value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3: array = array[0, -1]
    elif array.ndim == 2: array = array[-1]
    if array.ndim != 1 or not np.isfinite(array).all():
        raise CaptureError("GLM decoder returned malformed finite logits")
    return array.copy()


def _routes_from_result(result, decoder):
    value = result.get("routing_weights") if isinstance(result, dict) else getattr(result, "routing_weights", None)
    if value is None: raise CaptureError("GLM text route did not expose executed routing weights")
    if hasattr(value, "detach"): value = value.detach().cpu().numpy()
    values = np.asarray(value, dtype=np.float32)
    contract = getattr(decoder, "contract", None)
    experts, top_k = int(contract.n_routed_experts), int(contract.num_experts_per_tok)
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[-1] < experts:
        raise CaptureError("GLM executed routing tensor geometry differs")
    scores = values[0, :, :, :experts]
    layer_kinds = tuple(getattr(contract, "mlp_layer_types", ()))
    if len(layer_kinds) != values.shape[2]:
        raise CaptureError("GLM layer routing schedule differs from captured geometry")
    routes = []
    for layer, kind in enumerate(layer_kinds):
        if kind == "dense":
            routes.append([[] for _ in range(values.shape[1])])
            continue
        if kind != "sparse":
            raise CaptureError("unknown GLM MLP layer type in routing observation")
        layer_scores = scores[:, layer, :]
        if np.any(np.count_nonzero(layer_scores > 0, axis=-1) < top_k):
            raise CaptureError("GLM routing weights underflow; selected experts cannot be proven")
        chosen = np.argsort(-layer_scores, axis=-1, kind="stable")[:, :top_k]
        routes.append([[int(expert) for expert in row] for row in chosen])
    return routes


def _greedy_publisher(candidate, prompt_ids, media, max_tokens, config, route_chunks):
    pending = list(prompt_ids)
    rows, generated, finish = [], [], "length"
    eos = config.get("eos_token_id")
    for index in range(max_tokens):
        ids = np.asarray(pending, dtype=np.int64).reshape(1, -1)
        result = candidate.forward(ids, **(_media_kwargs(media) if index == 0 else {}))
        rows.append(_result_logits(result))
        token = int(np.argmax(rows[-1])); generated.append(token)
        if eos is not None and token == eos:
            finish = "eos"; break
        pending = [token]
    return np.stack(rows), generated, finish


def _greedy_reference(reference, prompt_ids, media, vision_encoder, original_config,
                      max_tokens, config):
    pending = list(prompt_ids)
    rows, generated, finish = [], [], "length"
    eos = config.get("eos_token_id")
    for index in range(max_tokens):
        ids = np.asarray(pending, dtype=np.int64).reshape(1, -1)
        kwargs = {}
        if index == 0:
            kwargs["inputs_embeds"] = _reference_embeddings(
                reference, ids, media, vision_encoder, original_config)
        result = reference.forward(ids, **kwargs)
        rows.append(_result_logits(result))
        token = int(np.argmax(rows[-1])); generated.append(token)
        if eos is not None and token == eos:
            finish = "eos"; break
        pending = [token]
    return np.stack(rows), generated, finish


def _merge_route_chunks(chunks):
    if not chunks:
        raise CaptureError("GLM route observer returned no rows")
    layers = len(chunks[0])
    if any(type(chunk) is not list or len(chunk) != layers for chunk in chunks):
        raise CaptureError("GLM route observer layer geometry changed")
    merged = [[] for _ in range(layers)]
    for chunk in chunks:
        for layer, rows in enumerate(chunk):
            merged[layer].extend(rows)
    return merged


def _continue_publisher(candidate, initial_token, count, config):
    output, pending = [], int(initial_token)
    for _ in range(count):
        result = candidate.forward(np.asarray([[pending]], dtype=np.int64))
        pending = int(np.argmax(_result_logits(result)))
        output.append(pending)
        if config.get("eos_token_id") == pending: break
    return output


def _abi_owner(row, backend):
    chain = row.get("callable_chain") or {}
    abi = tuple(chain.get("native_abi", ()))
    if backend == "device":
        preferred = "torch_glm5_decoder.TorchGlm5NextDecoder"
    elif backend == "public-range":
        preferred = "native_family_public_text.NativeFamilyPublicTextDecoder"
    else:
        preferred = "native_glm5_next.NativeGlm5NextTextDecoder"
    if preferred not in abi:
        raise CaptureError("canonical GLM route does not bind the actual text ABI owner")
    return preferred


def _candidate_sources(backend):
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "openagent/adaptive_engine/glm5_multimodal.py",
             root / "openagent/adaptive_engine/glm5_vision.py",
             root / "openagent/adaptive_engine/native_media_ops.py",
             root / "openagent/adaptive_engine/native_media_ops.c",
             root / "openagent/adaptive_engine/publisher_media_files.py",
             root / "openagent/adaptive_engine/publisher_media_processor.py",
             root / "openagent/adaptive_engine/publisher_glm5_loader.py",
             root / "openagent/adaptive_engine/publisher_conditional_loader.py",
             root / "openagent/adaptive_engine/publisher_family_contracts.py",
             root / "openagent/adaptive_engine/publisher_glm5_config.py",
             root / "openagent/adaptive_engine/publisher_tensor_reader.py",
             root / "openagent/adaptive_engine/publisher_weight_mapping.py",
             root / "openagent/adaptive_engine/glm5_publisher_experts.py",
             root / "openagent/adaptive_engine/native_expert.py",
             root / "openagent/adaptive_engine/native_fp8_block_expert.py",
             root / "openagent/adaptive_engine/native_checkpoint.py",
             root / "openagent/adaptive_engine/flat_native_session.py",
             root / "openagent/adaptive_engine/family_source_binding.py",
             root / "openagent/adaptive_engine/source_scope.py",
             root / "openagent/adaptive_engine/publisher_checkpoint/storage.py"]
    if backend in ("native", "public-range"):
        paths += [root / "openagent/adaptive_engine/native_glm5_next.py",
                  root / "openagent/adaptive_engine/native_glm5_next.c"]
    if backend == "device":
        paths += [root / "openagent/adaptive_engine/torch_glm5_decoder.py",
                  root / "openagent/adaptive_engine/device_glm5_forward.py",
                  root / "openagent/adaptive_engine/device_glm5_publisher_experts.py",
                  root / "openagent/adaptive_engine/device_logits.py",
                  root / "openagent/adaptive_engine/device_execution_policy.py"]
    if backend == "public-range":
        paths += [root / "openagent/adaptive_engine/native_family_public_text.py",
                  root / "openagent/adaptive_engine/native_family_public_adapters.py"]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise CaptureError("GLM capture source pin is missing: " + ", ".join(missing))
    return paths


def add_glm5_media_subparser(subparsers) -> None:
    parser = subparsers.add_parser("glm5-publisher-media",
        help="capture exact GLM5 publisher image/video route measurements")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-format", required=True, choices=tuple(_PERSISTED))
    parser.add_argument("--backend", required=True, choices=("native", "device", "public-range"))
    parser.add_argument("--media-library", required=True, type=Path)
    parser.add_argument("--text-library", type=Path)
    parser.add_argument("--runtime-library", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--reference-device")
    parser.add_argument("--reference-library", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", action="append", default=[], type=Path)
    parser.add_argument("--video-frame", action="append", default=[], type=Path)
    parser.add_argument("--video-indices", default="")
    parser.add_argument("--video-total-frames", type=int)
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--recovery-tokens", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--case-id")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    from .device_capture import DEFAULT_EVIDENCE, DEFAULT_RECEIPTS
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)


def run_glm5_media_subcommand(args) -> int:
    try:
        indices = tuple(int(item) for item in args.video_indices.split(",") if item.strip())
        measured = capture_glm5_publisher_route(
            args.checkpoint, source_format=args.source_format, backend=args.backend,
            media_library=args.media_library, text_library=args.text_library,
            runtime_library=args.runtime_library, device=args.device,
            reference_device=args.reference_device, reference_library=args.reference_library,
            prompt=args.prompt, image_paths=args.image, video_frame_paths=args.video_frame,
            video_indices=indices, video_total_frames=args.video_total_frames,
            video_fps=args.video_fps, max_tokens=args.max_tokens,
            recovery_tokens=args.recovery_tokens, context_tokens=args.context_tokens,
            case_id=args.case_id, tolerance=args.tolerance)
        from .media_route_capture import publish_media_measurements
        result = publish_media_measurements(
            measured, evidence_root=args.evidence_root, receipts_path=args.receipts)
    except (CaptureError, ValueError, OSError, ImportError, RuntimeError) as error:
        print("capture rejected: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


__all__ = ["add_glm5_media_subparser", "capture_glm5_publisher_route",
           "glm5_publisher_case_id", "run_glm5_media_subcommand"]
