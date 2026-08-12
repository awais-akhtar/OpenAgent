"""Local image measurements and explicit optional VLM observations."""

from __future__ import annotations

from io import BytesIO
import json
import math
from statistics import fmean
from typing import Any, Iterable


def summarise_image(values: Iterable[float], *, width: int = 0, height: int = 0) -> dict[str, float | int]:
    pixels = [float(value) for value in values]
    if not pixels:
        raise ValueError("scientific image needs at least one numeric sample")
    if any(not math.isfinite(value) for value in pixels):
        raise ValueError("scientific image samples must be finite")
    mean = fmean(pixels)
    variance = fmean((value - mean) ** 2 for value in pixels)
    return {
        "samples": len(pixels),
        "width": max(0, int(width)),
        "height": max(0, int(height)),
        "minimum": min(pixels),
        "maximum": max(pixels),
        "mean": mean,
        "stddev": math.sqrt(variance),
    }


def analyse_image_bytes(data: bytes, *, name: str = "image") -> dict[str, Any]:
    """Measure image structure locally without assigning semantic labels."""

    if not data:
        raise ValueError("image data is empty")
    try:
        from PIL import Image, ImageFilter, ImageStat
    except ImportError:
        return {
            "status": "unavailable",
            "analyser": "pillow",
            "name": name,
            "bytes": len(data),
            "reason": "Install Pillow to enable local image measurements.",
            "semantic_observations": [],
        }
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            rgb = image.convert("RGB")
            gray = image.convert("L")
            channel_stat = ImageStat.Stat(rgb)
            histogram = gray.histogram()
            total = max(1, sum(histogram))
            entropy = -sum((count / total) * math.log2(count / total) for count in histogram if count)
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_values = _pixel_values(edges)
            edge_density = sum(value >= 32 for value in edge_values) / max(1, len(edge_values))
            thirds = _region_brightness(gray)
            return {
                "status": "completed",
                "analyser": "pillow-local-measurements",
                "name": name,
                "format": image.format or "unknown",
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "channels": {name: mean for name, mean in zip(("red", "green", "blue"), channel_stat.mean)},
                "grayscale_mean": ImageStat.Stat(gray).mean[0],
                "grayscale_stddev": ImageStat.Stat(gray).stddev[0],
                "entropy_bits": entropy,
                "edge_density": edge_density,
                "region_brightness": thirds,
                "semantic_observations": [],
                "interpretation": "These are deterministic pixel measurements. No object, condition, or scientific meaning was inferred.",
                "privacy_boundary": "Image bytes were decoded in the local process.",
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"unsupported or damaged image: {exc}") from exc


def observe_with_gateway(
    data: bytes,
    *,
    gateway: Any,
    model: str,
    name: str = "image",
    media_type: str = "image/png",
    provider_id: str = "auto",
    allow_external: bool = False,
) -> dict[str, Any]:
    """Request structured observations from an explicitly selected VLM.

    A response is accepted only when the provider returns the documented JSON
    shape.  Provider errors and text-only fallbacks remain visibly unavailable.
    """

    if gateway is None or not model.strip():
        return {"status": "unavailable", "model": model, "observations": [], "reason": "A configured vision-capable model is required."}
    prompt = {
        "task": "scientific_image_observation",
        "name": name,
        "output_schema": {
            "observations": [{"statement": "string", "region": "string", "confidence": "0..1", "evidence": "visible pixels only"}],
            "limitations": ["string"],
        },
    }
    try:
        response = gateway.complete_vision(
            [
                {"role": "system", "content": "Return JSON only. Describe visible evidence, distinguish measurement from inference, and do not invent objects or values."},
                {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
            ],
            data,
            media_type=media_type,
            provider_id=provider_id,
            model=model,
            allow_external=allow_external,
            max_output_tokens=900,
        )
        payload = _json_object(str(response.content))
        observations = payload.get("observations") if isinstance(payload, dict) else None
        if not isinstance(observations, list):
            raise ValueError("provider did not return an observations list")
        clean = []
        for item in observations:
            if not isinstance(item, dict) or not str(item.get("statement", "")).strip():
                continue
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
            clean.append({
                "statement": str(item["statement"]).strip(),
                "region": str(item.get("region", "unspecified")).strip(),
                "confidence": confidence,
                "evidence": str(item.get("evidence", "model observation")).strip(),
            })
        if not clean:
            raise ValueError("provider returned no valid observations")
        return {
            "status": "completed",
            "provider": getattr(response, "provider", provider_id),
            "model": getattr(response, "model", model),
            "observations": clean,
            "limitations": [str(value) for value in payload.get("limitations", [])] if isinstance(payload.get("limitations"), list) else [],
            "privacy_boundary": "External access was permitted." if allow_external else "The request was restricted to configured local providers.",
        }
    except Exception as exc:
        return {"status": "unavailable", "model": model, "observations": [], "reason": str(exc), "privacy_boundary": "No VLM result was accepted."}


def _region_brightness(image: Any) -> dict[str, float]:
    width, height = image.size
    regions: dict[str, float] = {}
    for row, y_name in enumerate(("top", "middle", "bottom")):
        for column, x_name in enumerate(("left", "center", "right")):
            box = (column * width // 3, row * height // 3, (column + 1) * width // 3, (row + 1) * height // 3)
            values = _pixel_values(image.crop(box))
            regions[f"{y_name}_{x_name}"] = fmean(values) if values else 0.0
    return regions


def _pixel_values(image: Any) -> list[int]:
    flattened = getattr(image, "get_flattened_data", None)
    return list(flattened() if callable(flattened) else image.getdata())


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("provider response was not a JSON object")
    return value
