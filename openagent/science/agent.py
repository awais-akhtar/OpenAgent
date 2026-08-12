"""Scientific Agent built on the existing OpenAgent agent registry contract."""

from __future__ import annotations

import json
import re
from typing import Any

from ..models import AgentResponse
from .models import ScienceRun
from .multimodal.fusion import MultimodalScientificContext
from .planner import ScientificPlanner
from .reports.scientific_report import build_report
from .tool_router import ScientificToolRouter
from .execution import repair_scientific_code
from .verification.scientific_verifier import ScientificVerifier
from .papers import PaperSearch


class ScientificAgent:
    spec = type(
        "AgentSpecValue",
        (),
        {
            "id": "scientific",
            "name": "Scientific Agent",
            "description": "Plans, executes, and verifies local scientific and engineering workflows.",
        },
    )()

    def __init__(self, context: Any, *, gateway: Any = None) -> None:
        self.context = context
        self.gateway = gateway
        self.planner = ScientificPlanner(gateway=gateway)
        self.router = ScientificToolRouter(repair=self._repair_from_traceback)
        self.verifier = ScientificVerifier()

    def run(self, goal: str, *, project: str = "default") -> AgentResponse:
        clean_goal = goal.strip() or "Plan a reproducible scientific investigation"
        plan = self.planner.create_plan(clean_goal)
        memory = self._scientific_memory(project)
        result, verification = self.router.execute(plan, memory=memory)
        result["model_analysis"] = self._grounded_model_analysis(clean_goal, result)
        report = build_report(goal=clean_goal, plan=plan.to_dict(), result=result, verification=verification.to_dict())
        context = self.context.search(clean_goal, project=project, limit=5)
        titles = [str(item.get("title", "")) for item in context if item.get("title")]
        papers = PaperSearch([
            {"title": item.get("title", ""), "abstract": item.get("content", ""), "citation": item.get("metadata", {}).get("citation", ""), "source": item.get("source", "local")}
            for item in context
            if "paper" in str(item.get("type", "")).lower() or "paper" in str(item.get("source", "")).lower() or item.get("metadata", {}).get("citation")
        ]).search(clean_goal)
        science_run = ScienceRun(plan.workflow, clean_goal, "completed" if verification.passed else "needs_review", result, verification)
        return AgentResponse(
            agent=self.spec.id,
            goal=clean_goal,
            response=report,
            next_actions=list(plan.steps[-2:]) + (["Review failed checks before using the result."] if not verification.passed else ["Record the verified result and assumptions in the project memory."]),
            confidence=verification.score,
            context_used=titles,
            metadata={"science": {**science_run.to_dict(), "plan": plan.to_dict(), "memory_context": memory, "local_papers": [paper.to_dict() for paper in papers]}, "engine_routed": bool(result.get("execution")), "external_network": False},
        )

    def _scientific_memory(self, project: str) -> list[dict[str, Any]]:
        store = getattr(self.context, "store", None)
        if store is None:
            return []
        records: list[dict[str, Any]] = []
        for event in store.list_events(limit=500, project=project):
            if event.get("type") != "scientific_memory":
                continue
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            records.append({
                "category": metadata.get("category", "assumption"),
                "name": metadata.get("name", event.get("title", "")),
                "value": metadata.get("value"),
                "unit": metadata.get("unit", ""),
                "source": metadata.get("source", event.get("source", "scientific-memory")),
            })
        return records

    def _grounded_model_analysis(self, goal: str, result: dict[str, Any]) -> dict[str, Any]:
        evidence = _result_evidence(result)
        if self.gateway is None:
            return {"status": "unavailable", "reason": "No model gateway is configured.", "claims": []}
        try:
            response = self.gateway.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Interpret a completed scientific calculation. Return JSON only with keys summary, "
                            "claims, and limitations. Each claim must contain text and evidence_ids, and every "
                            "evidence id must come from the supplied evidence object. Do not invent measurements."
                        ),
                    },
                    {"role": "user", "content": json.dumps({"goal": goal, "evidence": evidence}, default=str)},
                ],
                provider_id="auto",
                allow_external=False,
                max_output_tokens=700,
            )
            payload = _json_payload(str(response.content))
            claims = payload.get("claims")
            if not isinstance(payload.get("summary"), str) or not isinstance(claims, list):
                raise ValueError("The model analysis did not match the required schema.")
            checked_claims = []
            for claim in claims:
                if not isinstance(claim, dict) or not str(claim.get("text", "")).strip():
                    raise ValueError("The model analysis contained an invalid claim.")
                references = claim.get("evidence_ids")
                if not isinstance(references, list) or not references or any(str(item) not in evidence for item in references):
                    raise ValueError("The model analysis cited unavailable evidence.")
                checked_claims.append({"text": str(claim["text"]).strip(), "evidence_ids": [str(item) for item in references]})
            limitations = payload.get("limitations", [])
            if not isinstance(limitations, list):
                raise ValueError("The model analysis limitations must be a list.")
            return {
                "status": "completed",
                "provider": response.provider,
                "model": response.model,
                "summary": payload["summary"].strip(),
                "claims": checked_claims,
                "limitations": [str(item).strip() for item in limitations if str(item).strip()],
                "evidence_ids": sorted(evidence),
            }
        except Exception as exc:
            return {"status": "unavailable", "reason": str(exc), "claims": [], "evidence_ids": sorted(evidence)}

    def _repair_from_traceback(self, code: str, traceback: str) -> str:
        repaired = repair_scientific_code(code, traceback)
        if repaired != code or self.gateway is None:
            return repaired
        try:
            response = self.gateway.complete(
                [
                    {"role": "system", "content": "Repair the supplied Python program from its traceback. Preserve the scientific method and output schema. Return only complete Python source, with no explanation."},
                    {"role": "user", "content": f"PROGRAM\n{code}\n\nTRACEBACK\n{traceback[-4000:]}"},
                ],
                provider_id="auto",
                allow_external=False,
                max_output_tokens=1200,
            )
            candidate = _python_source(response.content)
            if candidate and candidate != code:
                compile(candidate, "<scientific-repair>", "exec")
                return candidate
        except Exception:
            pass
        return code

    def run_multimodal(self, goal: str, context: MultimodalScientificContext, *, project: str = "default") -> AgentResponse:
        clean_goal = goal.strip() or "Explain the supplied scientific evidence"
        summary = context.summary()
        verification = self.verifier.verify(summary)
        report = build_report(goal=clean_goal, plan={"workflow": "multimodal_fusion", "steps": ["summarise local evidence", "verify numeric values", "state supported findings"]}, result=summary, verification=verification.to_dict())
        return AgentResponse(
            agent=self.spec.id,
            goal=clean_goal,
            response=report,
            next_actions=["Add a domain-specific equation or constraint check before making a decision."],
            confidence=verification.score,
            context_used=[],
            metadata={"science": {"workflow": "multimodal_fusion", "evidence_count": summary["count"]}, "external_network": False},
        )


def _python_source(value: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", value, flags=re.DOTALL | re.IGNORECASE)
    source = (match.group(1) if match else value).strip()
    return source if "\n" in source and not source.lower().startswith(("openagent", "request:")) else ""


def _result_evidence(result: dict[str, Any], *, limit: int = 80) -> dict[str, Any]:
    evidence: dict[str, Any] = {}

    def collect(value: Any, path: str) -> None:
        if len(evidence) >= limit:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"code", "stdout", "stderr", "lifecycle", "history", "model_analysis"}:
                    continue
                collect(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(item, (int, float, str, bool)) for item in value[:20]):
                evidence[path] = list(value[:20])
            else:
                for index, child in enumerate(value[:10]):
                    collect(child, f"{path}[{index}]")
        elif isinstance(value, (int, float, str, bool)) or value is None:
            evidence[path] = value

    collect(result, "result")
    return evidence


def _json_payload(value: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    parsed = json.loads(match.group(1) if match else value.strip())
    if not isinstance(parsed, dict):
        raise ValueError("The model analysis was not a JSON object.")
    return parsed
