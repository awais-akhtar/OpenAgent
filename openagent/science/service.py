"""Persistent services for the Scientific AI workspace."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..models import AuditEvent, ContextEvent
from .materials.active_learning import ActiveLearner
from .materials.dataset import validate_records
from .materials.predictor import MaterialPropertyPredictor
from .safety import check_citations, check_consistency, detect_unsupported_claims
from .verification.scientific_verifier import ScientificVerifier


class ScientificService:
    """Keeps scientific records in the existing project-scoped event store."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.verifier = ScientificVerifier()

    def remember(
        self,
        *,
        project: str,
        category: str,
        name: str,
        value: Any,
        unit: str = "",
        source: str = "user",
    ) -> dict[str, Any]:
        record = {"category": category.strip(), "name": name.strip(), "value": value, "unit": unit.strip(), "source": source.strip() or "user"}
        if not record["category"] or not record["name"]:
            raise ValueError("category and name are required")
        event = self.store.add_event(ContextEvent(
            title=f"Scientific memory: {record['name']}",
            content=json.dumps(record, sort_keys=True, default=str),
            type="scientific_memory",
            source="scientific-ai",
            project=project,
            metadata=record,
        ))
        self._audit(project, "science.memory.create", event["id"], {"category": record["category"]})
        return event

    def list_memory(self, *, project: str, limit: int = 200) -> list[dict[str, Any]]:
        return [event for event in self.store.list_events(limit=min(500, max(1, limit)), project=project) if event.get("type") == "scientific_memory"]

    def save_material_dataset(self, *, project: str, name: str, target: str, records: Sequence[dict[str, Any]], source: str = "upload") -> dict[str, Any]:
        validated = validate_records(records, target=target)
        numeric_columns = sorted({key for row in validated for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
        payload = {"name": name.strip() or "Materials dataset", "target": target, "records": validated, "numeric_columns": numeric_columns, "source": source}
        event = self.store.add_event(ContextEvent(
            title=f"Materials dataset: {payload['name']}",
            content=json.dumps(payload, sort_keys=True, default=str),
            type="scientific_material_dataset",
            source="scientific-ai",
            project=project,
            metadata={"name": payload["name"], "target": target, "rows": len(validated), "numeric_columns": numeric_columns, "source": source},
        ))
        self._audit(project, "science.materials.dataset.create", event["id"], {"rows": len(validated), "target": target})
        return {"dataset": event, "summary": {"rows": len(validated), "target": target, "numeric_columns": numeric_columns}, "records": validated}

    def list_material_datasets(self, *, project: str, limit: int = 50) -> list[dict[str, Any]]:
        return [event for event in self.store.list_events(limit=min(500, max(1, limit)), project=project) if event.get("type") == "scientific_material_dataset"]

    def analyse_materials(
        self,
        *,
        project: str,
        records: Sequence[dict[str, Any]],
        target: str,
        candidates: Sequence[Mapping[str, float]] = (),
        maximize: bool = True,
    ) -> dict[str, Any]:
        validated = validate_records(records, target=target)
        predictor = MaterialPropertyPredictor()
        fit = predictor.fit(validated, target=target)
        source_candidates = list(candidates) or [{key: float(value) for key, value in row.items() if key != target and isinstance(value, (int, float))} for row in validated]
        ranked = predictor.rank_candidates(source_candidates, maximize=maximize)
        experiments = ActiveLearner().select(predictor, source_candidates, count=min(5, len(source_candidates)))
        chart = [{"label": f"Candidate {index + 1}", "prediction": float(item["prediction"]), "uncertainty": float(item["uncertainty"])} for index, item in enumerate(ranked)]
        verification = self.verifier.verify({"fit": fit, "ranked": ranked})
        output = {"fit": fit, "ranked_candidates": ranked, "next_experiments": experiments, "chart": chart, "verification": verification.to_dict()}
        self._audit(project, "science.materials.analyse", target, {"rows": len(validated), "candidates": len(source_candidates), "passed": verification.passed})
        return output

    def verify(self, *, project: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        values = payload.get("values") if isinstance(payload.get("values"), list) else None
        residuals = payload.get("residuals") if isinstance(payload.get("residuals"), list) else None
        units = payload.get("units") if isinstance(payload.get("units"), dict) else None
        constraints = payload.get("constraints") if isinstance(payload.get("constraints"), list) else None
        result = payload.get("result") if isinstance(payload.get("result"), dict) else dict(payload)
        equations = payload.get("equations") if isinstance(payload.get("equations"), list) else None
        scientific = self.verifier.verify(result, values=values, residuals=residuals, units=units, constraints=constraints, equations=equations)
        report_text = str(payload.get("text", ""))
        evidence_terms = payload.get("evidence_terms", []) if isinstance(payload.get("evidence_terms"), list) else []
        citations = payload.get("citations", []) if isinstance(payload.get("citations"), list) else []
        numeric_map = {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        safety = {
            "citations": check_citations(report_text, citations),
            "supported_claims": detect_unsupported_claims(report_text, evidence_terms),
            "consistency": check_consistency(numeric_map),
        }
        passed = scientific.passed and all(bool(item.get("passed")) for item in safety.values())
        self._audit(project, "science.verify", "scientific-result", {"passed": passed, "score": scientific.score})
        return {"passed": passed, "scientific": scientific.to_dict(), "safety": safety}

    def _audit(self, project: str, action: str, resource: str, metadata: dict[str, Any]) -> None:
        self.store.add_audit_event(AuditEvent(actor="scientific-agent", action=action, resource=resource, project=project, metadata=metadata))
