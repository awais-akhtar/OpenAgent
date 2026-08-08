"""Scientific AI route contracts and local workflow handlers."""

from __future__ import annotations

from dataclasses import dataclass
import base64
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs

from ..science.agent import ScientificAgent
from ..science.computing.neural_operator import DeepONet, FNO, PINN
from ..science.experiment import ExperimentPlanner
from ..science.execution import ScientificCodeRunner
from ..science.multimodal.fusion import MultimodalScientificContext
from ..science.service import ScientificService
from ..science.tools import COMSOLTool, FenicsTool, LAMMPSTool, OpenFOAMTool, discover_scientific_tools
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "science"
ROUTES = (
    ("GET", "/api/science/status"),
    ("GET", "/api/science/tools"),
    ("GET", "/api/science/memory"),
    ("GET", "/api/science/materials/datasets"),
    ("POST", "/api/science/run"),
    ("POST", "/api/science/multimodal"),
    ("POST", "/api/science/experiments"),
    ("POST", "/api/science/execute"),
    ("POST", "/api/science/memory"),
    ("POST", "/api/science/materials/datasets"),
    ("POST", "/api/science/materials/analyse"),
    ("POST", "/api/science/verify"),
    ("POST", "/api/science/simulate"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ScienceRouteDependencies:
    agent: ScientificAgent
    run_agent: Any
    read_json: Any
    service: ScientificService | None = None


def handle_get(path: str, query: str, deps: ScienceRouteDependencies) -> JsonRouteResult | None:
    if path == "/api/science/tools":
        return JsonRouteResult({
            "tools": [item.to_dict() for item in discover_scientific_tools()],
            "operator_learning": [FNO().family, DeepONet().family, PINN().family],
            "privacy_boundary": "Only local package discovery is performed.",
        })
    params = parse_qs(query)
    project = str(params.get("project", ["default"])[0]).strip() or "default"
    if path == "/api/science/memory":
        if deps.service is None:
            return JsonRouteResult({"error": "Scientific persistence is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
        return JsonRouteResult({"memory": deps.service.list_memory(project=project)})
    if path == "/api/science/materials/datasets":
        if deps.service is None:
            return JsonRouteResult({"error": "Scientific persistence is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
        return JsonRouteResult({"datasets": deps.service.list_material_datasets(project=project)})
    if path != "/api/science/status":
        return None
    return JsonRouteResult(
        {
            "status": "ready",
            "workflows": [
                {"id": "scientific_review", "label": "Scientific review"},
                {"id": "heat_equation", "label": "Heat equation and operator surrogate"},
                {"id": "beam_optimization", "label": "Engineering beam optimisation"},
                {"id": "heat_exchanger_optimization", "label": "Profiled heat-exchanger optimisation"},
                {"id": "materials_discovery", "label": "Materials property discovery"},
                {"id": "multimodal_fusion", "label": "Multimodal scientific evidence"},
            ],
            "backends": {"numpy": _optional_module("numpy"), "scipy": _optional_module("scipy")},
            "capabilities": ["scientific_memory", "experiment_planning", "bounded_execution", "profiling"],
            "privacy_boundary": "Scientific calculations and evidence stay local. This API does not perform network research or upload inputs.",
        }
    )


def handle_post(path: str, deps: ScienceRouteDependencies) -> JsonRouteResult | None:
    if path not in {item[1] for item in ROUTES if item[0] == "POST"}:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/science/run":
            goal = str(payload.get("goal", "")).strip()
            return JsonRouteResult({"run": deps.run_agent(goal, project=project)}, HTTPStatus.CREATED)
        if path == "/api/science/experiments":
            candidates = payload.get("candidates", [])
            if not isinstance(candidates, list) or not all(isinstance(item, dict) for item in candidates):
                raise ValueError("candidates must be a list of objects")
            proposals = ExperimentPlanner().propose(str(payload.get("goal", "")), candidates)
            return JsonRouteResult({"experiments": [item.to_dict() for item in proposals]}, HTTPStatus.CREATED)
        if path == "/api/science/execute":
            if payload.get("approved") is not True:
                return JsonRouteResult({"error": "Scientific code execution requires explicit approval."}, HTTPStatus.FORBIDDEN)
            code = str(payload.get("code", ""))
            if not code.strip():
                raise ValueError("code is required")
            record = ScientificCodeRunner().run(
                code,
                timeout_seconds=min(30.0, max(0.1, float(payload.get("timeout_seconds", 10.0)))),
                retries=min(3, max(0, int(payload.get("retries", 0)))),
            )
            return JsonRouteResult({"execution": record.to_dict()}, HTTPStatus.CREATED)
        if path == "/api/science/memory":
            if deps.service is None:
                return JsonRouteResult({"error": "Scientific persistence is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
            event = deps.service.remember(
                project=project,
                category=str(payload.get("category", "assumption")),
                name=str(payload.get("name", "")),
                value=payload.get("value"),
                unit=str(payload.get("unit", "")),
                source=str(payload.get("source", "user")),
            )
            return JsonRouteResult({"memory": event}, HTTPStatus.CREATED)
        if path == "/api/science/materials/datasets":
            if deps.service is None:
                return JsonRouteResult({"error": "Scientific persistence is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
            records = payload.get("records", [])
            if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
                raise ValueError("records must be a list of objects")
            saved = deps.service.save_material_dataset(project=project, name=str(payload.get("name", "Materials dataset")), target=str(payload.get("target", "")), records=records, source=str(payload.get("source", "upload")))
            return JsonRouteResult(saved, HTTPStatus.CREATED)
        if path == "/api/science/materials/analyse":
            if deps.service is None:
                return JsonRouteResult({"error": "Scientific persistence is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
            records = payload.get("records", [])
            candidates = payload.get("candidates", [])
            if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
                raise ValueError("records must be a list of objects")
            if not isinstance(candidates, list) or not all(isinstance(item, dict) for item in candidates):
                raise ValueError("candidates must be a list of objects")
            result = deps.service.analyse_materials(project=project, records=records, target=str(payload.get("target", "")), candidates=candidates, maximize=bool(payload.get("maximize", True)))
            return JsonRouteResult({"analysis": result}, HTTPStatus.CREATED)
        if path == "/api/science/verify":
            if deps.service is None:
                return JsonRouteResult({"error": "Scientific verification is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
            return JsonRouteResult({"verification": deps.service.verify(project=project, payload=payload)}, HTTPStatus.CREATED)
        if path == "/api/science/simulate":
            simulator_id = str(payload.get("simulator", "fenics")).lower()
            adapters = {"fenics": FenicsTool(), "openfoam": OpenFOAMTool(), "comsol": COMSOLTool(), "lammps": LAMMPSTool()}
            adapter = adapters.get(simulator_id)
            if adapter is None:
                raise ValueError("simulator must be fenics, openfoam, comsol, or lammps")
            inputs = payload.get("inputs", {})
            if not isinstance(inputs, dict):
                raise ValueError("inputs must be an object")
            result = adapter.run(inputs=inputs) if adapter.capability().available and payload.get("approved") is True else adapter.mock_example(inputs=inputs)
            return JsonRouteResult({"simulation": result, "capability": adapter.capability().to_dict()}, HTTPStatus.CREATED)
        context = MultimodalScientificContext()
        for item in payload.get("documents", []) if isinstance(payload.get("documents"), list) else []:
            if isinstance(item, dict):
                encoded = str(item.get("base64", ""))
                if encoded:
                    context.add_document(str(item.get("name", "document")), base64.b64decode(encoded, validate=True), media_type=str(item.get("media_type", "")))
        for item in payload.get("text", []) if isinstance(payload.get("text"), list) else []:
            if isinstance(item, dict):
                context.add_text(str(item.get("name", "text")), str(item.get("content", "")))
        for item in payload.get("sensor_csv", []) if isinstance(payload.get("sensor_csv"), list) else []:
            if isinstance(item, dict):
                context.add_sensor_csv(str(item.get("name", "sensor.csv")), str(item.get("content", "")))
        for item in payload.get("images", []) if isinstance(payload.get("images"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("values"), list):
                context.add_image(str(item.get("name", "image")), [float(value) for value in item["values"]], width=int(item.get("width", 0) or 0), height=int(item.get("height", 0) or 0))
        for item in payload.get("simulations", []) if isinstance(payload.get("simulations"), list) else []:
            if isinstance(item, dict):
                context.add_simulation(str(item.get("name", "simulation.json")), item.get("content", {}))
        response = deps.agent.run_multimodal(str(payload.get("goal", "Explain the supplied evidence")), context, project=project)
        return JsonRouteResult({"run": response.to_record(), "context": context.summary()}, HTTPStatus.CREATED)
    except (TypeError, ValueError, RuntimeError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def _optional_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True
