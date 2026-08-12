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
from ..science.materials.catalog import list_public_materials_datasets
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
    ("GET", "/api/science/materials/catalog"),
    ("POST", "/api/science/run"),
    ("POST", "/api/science/multimodal"),
    ("POST", "/api/science/experiments"),
    ("POST", "/api/science/execute"),
    ("POST", "/api/science/memory"),
    ("POST", "/api/science/materials/datasets"),
    ("POST", "/api/science/materials/analyse"),
    ("POST", "/api/science/materials/catalog/import"),
    ("POST", "/api/science/materials/catalog/analyse"),
    ("POST", "/api/science/operators/deeponet"),
    ("POST", "/api/science/engineering/optimize"),
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
    gateway: Any = None


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
    if path == "/api/science/materials/catalog":
        return JsonRouteResult({
            "datasets": list_public_materials_datasets(),
            "privacy_boundary": "Listing the bundled metadata is local. Importing records requires separate network and license approval.",
        })
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
            "backends": {"numpy": _optional_module("numpy"), "scipy": _optional_module("scipy"), "torch": _optional_module("torch")},
            "capabilities": ["scientific_memory", "experiment_planning", "bounded_execution", "profiling", "deeponet_training", "materials_reporting", "multimodal_observations", "optimizer_selection", "approved_citation_resolution"],
            "optimizer_methods": ["random", "differential_evolution", "minimize"],
            "privacy_boundary": "Scientific calculations and evidence stay local unless a verification request explicitly approves external citation resolution.",
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
            run = deps.run_agent(goal, project=project)
            serialized = deps.service.serialize_agent_run(run) if deps.service is not None and isinstance(run, dict) else {"run": run}
            return JsonRouteResult(serialized, HTTPStatus.CREATED)
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
                memory_limit_mb=min(16_384, max(32, int(payload.get("memory_limit_mb", 512)))),
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
        if path == "/api/science/materials/catalog/import":
            if deps.service is None:
                return JsonRouteResult({"error": "Scientific persistence is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
            result = deps.service.import_public_material_dataset(
                project=project,
                dataset_id=str(payload.get("dataset_id", "")),
                allow_external=payload.get("allow_external") is True,
                license_accepted=payload.get("license_accepted") is True,
            )
            return JsonRouteResult({"import": result}, HTTPStatus.CREATED)
        if path == "/api/science/materials/catalog/analyse":
            if deps.service is None:
                return JsonRouteResult({"error": "Scientific persistence is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
            result = deps.service.analyse_public_material_dataset(
                project=project,
                dataset_id=str(payload.get("dataset_id", "")),
                allow_external=payload.get("allow_external") is True,
                license_accepted=payload.get("license_accepted") is True,
                seed=int(payload.get("seed", 31)),
            )
            return JsonRouteResult({"analysis": result}, HTTPStatus.CREATED)
        if path == "/api/science/operators/deeponet":
            if deps.service is None:
                return JsonRouteResult({"error": "Scientific operator training is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
            inputs = _numeric_matrix(payload.get("inputs"), "inputs")
            outputs = _numeric_matrix(payload.get("outputs"), "outputs")
            evaluation_inputs = _numeric_matrix(payload.get("evaluation_inputs", []), "evaluation_inputs", allow_empty=True)
            evaluation_outputs = _numeric_matrix(payload.get("evaluation_outputs", []), "evaluation_outputs", allow_empty=True)
            result = deps.service.train_deeponet(
                project=project,
                inputs=inputs,
                outputs=outputs,
                evaluation_inputs=evaluation_inputs,
                evaluation_outputs=evaluation_outputs,
                backend=str(payload.get("backend", "auto")),
                hidden=min(512, max(2, int(payload.get("hidden", 48)))),
                basis=min(512, max(2, int(payload.get("basis", 24)))),
                epochs=min(20_000, max(1, int(payload.get("epochs", 350)))),
                seed=int(payload.get("seed", 7)),
            )
            return JsonRouteResult({"operator": result}, HTTPStatus.CREATED)
        if path == "/api/science/engineering/optimize":
            if deps.service is None:
                return JsonRouteResult({"error": "Scientific optimisation is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
            result = deps.service.optimize_engineering(
                project=project,
                method=str(payload.get("method", "random")),
                iterations=min(50_000, max(1, int(payload.get("iterations", 800)))),
                stress_limit=float(payload.get("stress_limit", 250e6)),
                deflection_limit=float(payload.get("deflection_limit", 0.005)),
                benchmark=bool(payload.get("benchmark", False)),
            )
            return JsonRouteResult({"optimization": result}, HTTPStatus.CREATED)
        if path == "/api/science/verify":
            if deps.service is None:
                return JsonRouteResult({"error": "Scientific verification is unavailable."}, HTTPStatus.SERVICE_UNAVAILABLE)
            if "external_citation_lookup_approved" in payload and not isinstance(payload.get("external_citation_lookup_approved"), bool):
                raise ValueError("external_citation_lookup_approved must be a boolean")
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
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("values"), list):
                context.add_image(str(item.get("name", "image")), [float(value) for value in item["values"]], width=int(item.get("width", 0) or 0), height=int(item.get("height", 0) or 0))
            elif str(item.get("base64", "")):
                context.add_image_bytes(
                    str(item.get("name", "image")),
                    base64.b64decode(str(item["base64"]), validate=True),
                    media_type=str(item.get("media_type", "image/png")),
                    gateway=deps.gateway,
                    vlm_model=str(item.get("vlm_model", payload.get("vlm_model", ""))),
                    provider_id=str(item.get("provider_id", payload.get("provider_id", "auto"))),
                    allow_external=bool(item.get("allow_external", payload.get("allow_external", False))),
                )
        for item in payload.get("simulations", []) if isinstance(payload.get("simulations"), list) else []:
            if isinstance(item, dict):
                context.add_simulation(str(item.get("name", "simulation.json")), item.get("content", {}))
        response = deps.agent.run_multimodal(str(payload.get("goal", "Explain the supplied evidence")), context, project=project)
        summary = context.summary()
        observations = [
            {"name": item.get("name", ""), "kind": item.get("kind", ""), "observations": item.get("vlm", {}).get("observations", [])}
            for item in summary["items"]
            if isinstance(item, dict) and isinstance(item.get("vlm"), dict)
        ]
        return JsonRouteResult({"run": response.to_record(), "context": summary, "observations": observations}, HTTPStatus.CREATED)
    except PermissionError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.FORBIDDEN)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def _optional_module(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def _numeric_matrix(value: Any, field: str, *, allow_empty: bool = False) -> list[list[float]]:
    if allow_empty and value == []:
        return []
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValueError(f"{field} must be a non-empty matrix")
    try:
        return [[float(item) for item in row] for row in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only numeric values") from exc
