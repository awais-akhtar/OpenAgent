"""Persistent services for the Scientific AI workspace."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..models import AuditEvent, ContextEvent
from .citations import resolve_scientific_citations
from .computing.neural_operator import DeepONet
from .engineering.optimizer import EngineeringOptimizer, beam_problem
from .materials.active_learning import ActiveLearner
from .materials.catalog import analyse_public_materials_dataset, import_public_materials_dataset
from .materials.dataset import split_records, validate_records
from .materials.predictor import MaterialPropertyPredictor
from .safety import check_consistency
from .verification.scientific_verifier import ScientificVerifier


class ScientificService:
    """Keeps scientific records in the existing project-scoped event store."""

    def __init__(self, store: Any, *, citation_resolver: Any = None) -> None:
        self.store = store
        self.verifier = ScientificVerifier()
        self.citation_resolver = citation_resolver or resolve_scientific_citations

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

    def import_public_material_dataset(
        self,
        *,
        project: str,
        dataset_id: str,
        allow_external: bool,
        license_accepted: bool,
    ) -> dict[str, Any]:
        imported = import_public_materials_dataset(
            dataset_id,
            allow_external=allow_external,
            license_accepted=license_accepted,
        )
        provenance = imported["provenance"]
        saved = self.save_material_dataset(
            project=project,
            name=str(provenance["title"]),
            target=str(imported["target"]),
            records=imported["records"],
            source=str(provenance["source_url"]),
        )
        self._audit(project, "science.materials.catalog.import", dataset_id, {
            "rows": saved["summary"]["rows"],
            "sha256": provenance["verified_sha256"],
            "license": provenance["license_id"],
            "network_approved": True,
        })
        return {
            "status": "imported",
            "dataset": saved["dataset"],
            "summary": saved["summary"],
            "provenance": provenance,
            "privacy_boundary": imported["privacy_boundary"],
        }

    def analyse_public_material_dataset(
        self,
        *,
        project: str,
        dataset_id: str,
        allow_external: bool,
        license_accepted: bool,
        seed: int = 31,
    ) -> dict[str, Any]:
        report = analyse_public_materials_dataset(
            dataset_id,
            allow_external=allow_external,
            license_accepted=license_accepted,
            seed=seed,
        )
        provenance = report["provenance"]
        self._audit(project, "science.materials.catalog.analyse", dataset_id, {
            "rows": report["rows"],
            "training_rows": report["training_rows"],
            "test_rows": report["test_rows"],
            "seed": seed,
            "sha256": provenance["verified_sha256"],
            "license": provenance["license_id"],
            "network_approved": True,
        })
        return report

    def train_deeponet(
        self,
        *,
        project: str,
        inputs: Sequence[Sequence[float]],
        outputs: Sequence[Sequence[float]],
        evaluation_inputs: Sequence[Sequence[float]] = (),
        evaluation_outputs: Sequence[Sequence[float]] = (),
        backend: str = "auto",
        hidden: int = 48,
        basis: int = 24,
        epochs: int = 350,
        seed: int = 7,
    ) -> dict[str, Any]:
        """Train and evaluate one real branch/trunk operator in-process."""

        model = DeepONet(backend=backend, hidden=hidden, basis=basis, epochs=epochs, seed=seed)
        training = model.fit(inputs, outputs)
        if bool(evaluation_inputs) != bool(evaluation_outputs):
            raise ValueError("evaluation_inputs and evaluation_outputs must be supplied together")
        test_inputs = evaluation_inputs or inputs
        test_outputs = evaluation_outputs or outputs
        evaluation = model.evaluate(test_inputs, test_outputs)
        predictions = [model.predict(row) for row in test_inputs]
        report = {
            "status": "completed",
            "operator": "DeepONet",
            "training": training,
            "evaluation": evaluation,
            "evaluation_scope": "held-out" if evaluation_inputs else "training",
            "predictions": predictions,
            "progress": [
                {"stage": "validate", "status": "completed", "samples": len(inputs)},
                {"stage": "train", "status": "completed", "backend": training["backend"]},
                {"stage": "evaluate", "status": "completed", "samples": len(test_inputs)},
                {"stage": "report", "status": "completed"},
            ],
            "provenance": {
                "implementation": training["implementation"],
                "backend": training["backend"],
                "seed": seed,
                "external_network": False,
            },
            "privacy_boundary": "Training data and operator evaluation remain in the local OpenAgent process.",
        }
        self._audit(project, "science.deeponet.train", "DeepONet", {
            "training_samples": len(inputs),
            "evaluation_samples": len(test_inputs),
            "backend": training["backend"],
            "relative_rmse": evaluation["relative_rmse"],
        })
        return report

    def optimize_engineering(
        self,
        *,
        project: str,
        method: str = "random",
        iterations: int = 800,
        stress_limit: float = 250e6,
        deflection_limit: float = 0.005,
        benchmark: bool = False,
    ) -> dict[str, Any]:
        optimizer = EngineeringOptimizer(iterations=iterations, method=method)
        problem = beam_problem(stress_limit=stress_limit, deflection_limit=deflection_limit)
        result = optimizer.benchmark(problem) if benchmark else optimizer.solve(problem)
        output = {
            "status": "completed",
            "problem": "cantilever_beam",
            "requested_method": method,
            "available_methods": ["random", "differential_evolution", "minimize"],
            "benchmark": benchmark,
            "result": result,
            "privacy_boundary": "Optimisation runs locally and does not call an external solver.",
        }
        self._audit(project, "science.engineering.optimize", "cantilever-beam", {
            "method": method,
            "iterations": iterations,
            "benchmark": benchmark,
        })
        return output

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
        progress: list[dict[str, Any]] = [{"stage": "validate", "status": "completed", "rows": len(validated)}]
        training = validated
        held_out: list[dict[str, Any]] = []
        if len(validated) >= 5:
            training, held_out = split_records(validated)
        predictor = MaterialPropertyPredictor()
        fit = predictor.fit(training, target=target)
        progress.append({"stage": "fit", "status": "completed", "rows": len(training), "model": fit["model"]})
        test_metrics = predictor.evaluate(held_out) if held_out else None
        progress.append({
            "stage": "evaluate",
            "status": "completed" if held_out else "not_applicable",
            "rows": len(held_out),
            "reason": "At least five rows are required for a held-out split." if not held_out else "",
        })
        source_candidates = list(candidates) or [{key: float(value) for key, value in row.items() if key != target and isinstance(value, (int, float))} for row in validated]
        ranked = predictor.rank_candidates(source_candidates, maximize=maximize)
        experiments = ActiveLearner().select(predictor, source_candidates, count=min(5, len(source_candidates)))
        progress.append({"stage": "rank", "status": "completed", "candidates": len(source_candidates)})
        chart = [{"label": f"Candidate {index + 1}", "prediction": float(item["prediction"]), "uncertainty": float(item["uncertainty"])} for index, item in enumerate(ranked)]
        verification = self.verifier.verify({"fit": fit, "ranked": ranked})
        progress.append({"stage": "verify", "status": "completed", "passed": verification.passed})
        report = {
            "status": "completed",
            "target": target,
            "rows": len(validated),
            "training_rows": len(training),
            "test_rows": len(held_out),
            "test_metrics": test_metrics,
            "model": fit["model"],
            "uncertainty": fit["uncertainty_note"],
            "best_candidate": ranked[0] if ranked else None,
            "limitations": [
                "Predictions are produced by a transparent ridge bootstrap baseline.",
                "Uncertainty is a model-disagreement and distribution-distance indicator, not a calibrated probability.",
            ],
        }
        output = {"fit": fit, "ranked_candidates": ranked, "next_experiments": experiments, "chart": chart, "verification": verification.to_dict(), "progress": progress, "report": report}
        self._audit(project, "science.materials.analyse", target, {"rows": len(validated), "candidates": len(source_candidates), "passed": verification.passed})
        return output

    @staticmethod
    def serialize_agent_run(run: Mapping[str, Any]) -> dict[str, Any]:
        """Expose planner and verification provenance without changing the agent record."""

        metadata = run.get("metadata") if isinstance(run.get("metadata"), Mapping) else {}
        science = metadata.get("science") if isinstance(metadata.get("science"), Mapping) else {}
        plan = science.get("plan") if isinstance(science.get("plan"), Mapping) else {}
        verification = science.get("verification") if isinstance(science.get("verification"), Mapping) else {}
        return {
            "run": dict(run),
            "provenance": {
                "planner": plan.get("source", "unknown"),
                "workflow": science.get("workflow", plan.get("workflow", "")),
                "external_network": bool(metadata.get("external_network", False)),
                "context_items": len(run.get("context_used", [])) if isinstance(run.get("context_used"), list) else 0,
            },
            "verification_evidence": [
                dict(item.get("evidence", {}))
                for item in verification.get("findings", [])
                if isinstance(item, Mapping) and isinstance(item.get("evidence"), Mapping)
            ],
        }

    def verify(self, *, project: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        values = payload.get("values") if isinstance(payload.get("values"), list) else None
        residuals = payload.get("residuals") if isinstance(payload.get("residuals"), list) else None
        units = payload.get("units") if isinstance(payload.get("units"), dict) else None
        constraints = payload.get("constraints") if isinstance(payload.get("constraints"), list) else None
        result = payload.get("result") if isinstance(payload.get("result"), dict) else dict(payload)
        equations = payload.get("equations") if isinstance(payload.get("equations"), list) else None
        claims = payload.get("claims") if isinstance(payload.get("claims"), list) else None
        evidence_records = payload.get("evidence")
        evidence = evidence_records if isinstance(evidence_records, (Mapping, list, tuple)) else None
        scientific = self.verifier.verify(
            result,
            values=values,
            residuals=residuals,
            units=units,
            constraints=constraints,
            equations=equations,
            claims=claims,
            evidence=evidence,
        )
        citations = payload.get("citations", []) if isinstance(payload.get("citations"), list) else []
        numeric_map = {str(key): float(value) for key, value in result.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        citation_provenance = self._verify_citation_provenance(
            citations,
            approved=payload.get("external_citation_lookup_approved") is True,
        )
        claim_finding = next(
            (item for item in scientific.findings if item.check == "claim_evidence_provenance"),
            None,
        )
        supported_claims = {
            "status": "verified" if claim_finding and claim_finding.passed else "unverified" if claim_finding else "not_applicable",
            "passed": claim_finding.passed if claim_finding else True,
            "method": "structured-provenance",
            "evidence": dict(claim_finding.evidence) if claim_finding else {},
        }
        safety = {
            "consistency": check_consistency(numeric_map),
            "citation_provenance": citation_provenance,
            # Retain the established keys with stronger structured semantics.
            "citations": dict(citation_provenance),
            "supported_claims": supported_claims,
        }
        passed = scientific.passed and bool(safety["consistency"].get("passed")) and safety["citation_provenance"]["status"] in {"verified", "not_applicable"}
        evidence_checks = [
            {"check": finding.check, "passed": finding.passed, "evidence": dict(finding.evidence)}
            for finding in scientific.findings
        ]
        evidence_checks.extend(
            {"check": name, "passed": bool(item.get("passed")), "evidence": dict(item)}
            for name, item in safety.items()
        )
        provenance_checks = [item for item in evidence_checks if "provenance" in item["check"]]
        self._audit(project, "science.verify", "scientific-result", {
            "passed": passed,
            "score": scientific.score,
            "citation_status": safety["citation_provenance"]["status"],
        })
        return {
            "passed": passed,
            "scientific": scientific.to_dict(),
            "safety": safety,
            "evidence": evidence_checks,
            "provenance_checks": provenance_checks,
        }

    def _verify_citation_provenance(self, citations: Sequence[Any], *, approved: bool) -> dict[str, Any]:
        if not citations:
            return {
                "status": "not_applicable",
                "passed": True,
                "citations_checked": 0,
                "external_lookup": False,
                "reason": "No citations were supplied.",
            }
        normalized = [dict(item) for item in citations if isinstance(item, Mapping)]
        malformed = [
            index
            for index, item in enumerate(normalized)
            if not str(item.get("id", "")).strip()
            or not str(item.get("doi") or item.get("source") or item.get("url") or "").strip()
        ]
        if len(normalized) != len(citations) or malformed:
            return {
                "status": "unverified",
                "passed": False,
                "citations_checked": len(normalized),
                "external_lookup": False,
                "reason": "Citations must be structured objects with id and source or url fields.",
                "malformed_indexes": malformed,
            }
        if not approved:
            return {
                "status": "unverified",
                "passed": False,
                "citations_checked": len(normalized),
                "external_lookup": False,
                "reason": "External citation lookup was not explicitly approved.",
                "citations": normalized,
            }
        if not callable(self.citation_resolver):
            return {
                "status": "unverified",
                "passed": False,
                "citations_checked": len(normalized),
                "external_lookup": False,
                "reason": "No approved citation resolver is configured.",
                "citations": normalized,
            }
        try:
            resolved = self.citation_resolver(normalized)
        except Exception as exc:
            return {
                "status": "unverified",
                "passed": False,
                "citations_checked": len(normalized),
                "external_lookup": True,
                "reason": f"The configured citation resolver failed: {exc}",
                "citations": normalized,
            }
        if not isinstance(resolved, Mapping):
            raise ValueError("citation resolver must return a mapping")
        records = resolved.get("citations") if isinstance(resolved.get("citations"), list) else []
        verified = bool(resolved.get("verified")) and len(records) == len(normalized)
        return {
            "status": "verified" if verified else "unverified",
            "passed": verified,
            "citations_checked": len(normalized),
            "external_lookup": True,
            "resolver": str(resolved.get("resolver", "configured")),
            "citations": records,
            "reason": str(resolved.get("reason", "")),
            "claim_entailment_checked": False,
            "network_policy": dict(resolved.get("network_policy", {})) if isinstance(resolved.get("network_policy"), Mapping) else {},
        }

    def _audit(self, project: str, action: str, resource: str, metadata: dict[str, Any]) -> None:
        self.store.add_audit_event(AuditEvent(actor="scientific-agent", action=action, resource=resource, project=project, metadata=metadata))
