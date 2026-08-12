"""Validation, upload parsing, and reproducible materials dataset reports."""

from __future__ import annotations

import csv
import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence


def validate_records(records: Iterable[dict[str, Any]], *, target: str) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or target not in record:
            continue
        numeric = {str(key): float(value) for key, value in record.items() if _number(value)}
        if numeric:
            validated.append({**record, **numeric})
    if not validated:
        raise ValueError(f"no usable records contain numeric target '{target}'")
    return validated


def parse_dataset_upload(data: bytes, *, name: str, target: str) -> list[dict[str, Any]]:
    """Parse a local CSV or JSON upload and validate its numeric target."""

    lower_name = name.lower()
    if lower_name.endswith(".gz") or data[:2] == b"\x1f\x8b":
        data = _decompress_gzip(data)
        lower_name = lower_name[:-3] if lower_name.endswith(".gz") else lower_name
    suffix = Path(lower_name).suffix.lower()
    text = data.decode("utf-8-sig", errors="strict")
    if suffix == ".csv":
        records = [dict(row) for row in csv.DictReader(StringIO(text))]
        records = [{key: _coerce(value) for key, value in row.items()} for row in records]
    elif suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            records = _records_from_json_payload(payload)
    else:
        raise ValueError("materials upload must be CSV, JSON, or JSONL")
    records = [{key: _coerce(value) for key, value in row.items()} for row in records if isinstance(row, dict)]
    return validate_records(records, target=target)


def _records_from_json_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("records"), list):
        return payload["records"]
    columns = payload.get("columns")
    rows = payload.get("data")
    if isinstance(columns, list) and isinstance(rows, list):
        return [dict(zip(columns, row)) for row in rows if isinstance(row, list)]
    return []


def _decompress_gzip(data: bytes, *, max_uncompressed_bytes: int = 256 * 1024 * 1024) -> bytes:
    with gzip.GzipFile(fileobj=__import__("io").BytesIO(data)) as handle:
        content = handle.read(max_uncompressed_bytes + 1)
    if len(content) > max_uncompressed_bytes:
        raise ValueError("decompressed materials dataset exceeds the 256 MiB safety limit")
    return content


def split_records(records: Sequence[dict[str, Any]], *, test_fraction: float = 0.2, seed: int = 31) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) < 5:
        raise ValueError("at least five records are required for a held-out report")
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    indexes = list(range(len(records)))
    random.Random(seed).shuffle(indexes)
    test_size = max(1, min(len(records) - 2, round(len(records) * test_fraction)))
    test_indexes = set(indexes[:test_size])
    return (
        [dict(record) for index, record in enumerate(records) if index not in test_indexes],
        [dict(record) for index, record in enumerate(records) if index in test_indexes],
    )


def analyse_dataset_upload(
    data: bytes,
    *,
    name: str,
    target: str,
    artifact_dir: str | Path | None = None,
    test_fraction: float = 0.2,
    seed: int = 31,
) -> dict[str, Any]:
    """Fit a held-out baseline and optionally write machine-readable artifacts."""

    from .predictor import MaterialPropertyPredictor

    records = parse_dataset_upload(data, name=name, target=target)
    training, test = split_records(records, test_fraction=test_fraction, seed=seed)
    predictor = MaterialPropertyPredictor(seed=seed)
    fit = predictor.fit(training, target=target)
    metrics = predictor.evaluate(test)
    predictions = [
        {
            "row": index,
            "actual": float(record[target]),
            **predictor.predict_distribution(record),
        }
        for index, record in enumerate(test)
    ]
    report: dict[str, Any] = {
        "status": "completed",
        "dataset": name,
        "target": target,
        "rows": len(records),
        "training_rows": len(training),
        "test_rows": len(test),
        "split_seed": seed,
        "fit": fit,
        "test_metrics": metrics,
        "predictions": predictions,
        "interpretation": "Metrics are measured on a deterministic held-out split. Uncertainty is a model-disagreement and distribution-distance indicator, not a confidence probability.",
        "source_sha256": hashlib.sha256(data).hexdigest(),
    }
    report["artifacts"] = _write_artifacts(report, artifact_dir) if artifact_dir is not None else []
    return report


def _write_artifacts(report: dict[str, Any], artifact_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "materials-analysis.json"
    predictions_path = directory / "materials-predictions.csv"
    report_path.write_text(json.dumps({**report, "artifacts": []}, indent=2, sort_keys=True), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "actual", "prediction", "ensemble_mean", "ensemble_std", "nearest_training_distance", "uncertainty"])
        writer.writeheader()
        writer.writerows(report["predictions"])
    return [
        _artifact_manifest(report_path, "application/json"),
        _artifact_manifest(predictions_path, "text/csv"),
    ]


def _artifact_manifest(path: Path, media_type: str) -> dict[str, Any]:
    content = path.read_bytes()
    return {"name": path.name, "path": str(path), "media_type": media_type, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _coerce(value: Any) -> Any:
    if value is None:
        return value
    try:
        return float(str(value).strip())
    except ValueError:
        return str(value).strip()


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
