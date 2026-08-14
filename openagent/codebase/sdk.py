"""Public extension SDK for CodeQ software-generation targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .frameworks import FRAMEWORKS, FrameworkSpec, preflight_for_spec
from .profiles import artifact_contract, classify_request
from .repository import RepositoryIntelligence


PLUGIN_GROUP = "openagent.codeq.frameworks"


@dataclass(frozen=True)
class GenerationRequest:
    goal: str
    repository: str = ""
    acceptance_criteria: tuple[str, ...] = ()
    target_environment: str = ""


@dataclass(frozen=True)
class GenerationContract:
    profile: dict[str, Any]
    framework: dict[str, Any]
    preflight: dict[str, Any]
    artifacts: dict[str, Any]
    validation: dict[str, Any]
    environment_plan: tuple[dict[str, Any], ...]
    model_contract: dict[str, Any]
    acceptance_criteria: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Extension contract for a software target understood by CodeQ."""

    @property
    def id(self) -> str: ...

    def describe(self) -> dict[str, Any]: ...

    def matches(self, goal: str, project_map: dict[str, Any]) -> float: ...

    def artifact_contract(self, profile: dict[str, Any]) -> dict[str, Any]: ...

    def validation_commands(self, project_map: dict[str, Any]) -> dict[str, str]: ...

    def preflight(self, goal: str, project_map: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DeclarativeFrameworkAdapter:
    """A framework adapter defined without subclassing CodeQ internals."""

    spec: FrameworkSpec
    keywords: tuple[str, ...] = ()
    artifacts: dict[str, Any] = field(default_factory=dict)
    commands: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.spec.id

    def describe(self) -> dict[str, Any]:
        result = asdict(self.spec)
        result["plugin"] = False
        return result

    def matches(self, goal: str, project_map: dict[str, Any]) -> float:
        framework = str(project_map.get("framework", "")).casefold()
        if framework == self.id.casefold():
            return 1.0
        text = str(goal).casefold()
        hits = sum(1 for keyword in self.keywords if keyword.casefold() in text)
        return min(0.95, 0.45 + hits * 0.15) if hits else 0.0

    def artifact_contract(self, profile: dict[str, Any]) -> dict[str, Any]:
        return dict(self.artifacts or artifact_contract({**profile, "framework": self.id}))

    def validation_commands(self, project_map: dict[str, Any]) -> dict[str, str]:
        discovered = project_map.get("commands", {}) if isinstance(project_map.get("commands"), dict) else {}
        return {**self.commands, **{str(key): str(value) for key, value in discovered.items() if str(value).strip()}}

    def preflight(self, goal: str, project_map: dict[str, Any]) -> dict[str, Any]:
        return preflight_for_spec(self.spec, goal, {**project_map, "framework": self.id})


class GeneratorRegistry:
    """Register built-in or third-party framework adapters."""

    def __init__(self, *, load_plugins: bool = True) -> None:
        self._adapters: dict[str, FrameworkAdapter] = {}
        for spec in FRAMEWORKS:
            self.register(DeclarativeFrameworkAdapter(spec), replace=True)
        self.plugin_errors: list[dict[str, str]] = []
        if load_plugins:
            self.load_entry_points()

    def register(self, adapter: FrameworkAdapter, *, replace: bool = False) -> None:
        if not isinstance(adapter, FrameworkAdapter):
            raise TypeError("framework adapter does not implement the CodeQ SDK contract")
        identifier = str(adapter.id).strip().casefold()
        if not identifier or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in identifier):
            raise ValueError("framework adapter id must use letters, digits, hyphens, or underscores")
        if identifier in self._adapters and not replace:
            raise ValueError(f"framework adapter already registered: {identifier}")
        self._adapters[identifier] = adapter

    def unregister(self, identifier: str) -> None:
        key = str(identifier).strip().casefold()
        if key not in self._adapters:
            raise KeyError(identifier)
        del self._adapters[key]

    def get(self, identifier: str) -> FrameworkAdapter:
        key = str(identifier).strip().casefold()
        if key not in self._adapters:
            raise KeyError(f"unknown framework adapter: {identifier}")
        return self._adapters[key]

    def catalog(self) -> list[dict[str, Any]]:
        return [self._adapters[key].describe() for key in sorted(self._adapters)]

    def resolve(self, goal: str, project_map: dict[str, Any]) -> FrameworkAdapter:
        profile = classify_request(goal, project_map)
        identifier = str(profile.get("framework", "model-selected")).casefold()
        if identifier in self._adapters and identifier != "model-selected":
            return self._adapters[identifier]
        ranked = sorted(
            ((float(adapter.matches(goal, project_map)), adapter) for adapter in self._adapters.values()),
            key=lambda item: (-item[0], item[1].id),
        )
        if ranked and ranked[0][0] > 0:
            return ranked[0][1]
        return self._adapters["model-selected"]

    def load_entry_points(self) -> None:
        try:
            points = metadata.entry_points()
            selected = points.select(group=PLUGIN_GROUP) if hasattr(points, "select") else points.get(PLUGIN_GROUP, [])
        except Exception as exc:
            self.plugin_errors.append({"plugin": PLUGIN_GROUP, "error": str(exc)})
            return
        for point in selected:
            try:
                value = point.load()
                adapter = value() if isinstance(value, type) else value
                self.register(adapter)
            except Exception as exc:
                self.plugin_errors.append({"plugin": str(point.name), "error": str(exc)})


class CodeQSDK:
    """Programmatic software-generation contracts used by CodeQ and extensions."""

    def __init__(self, registry: GeneratorRegistry | None = None) -> None:
        self.registry = registry or GeneratorRegistry()

    def inspect(self, repository: str | Path) -> dict[str, Any]:
        return RepositoryIntelligence(repository).scan()

    def prepare(
        self,
        request: GenerationRequest | str,
        project_map: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        value = request if isinstance(request, GenerationRequest) else GenerationRequest(goal=str(request))
        if not value.goal.strip():
            raise ValueError("software-generation goal is required")
        if project_map is None and value.repository:
            project_map = self.inspect(value.repository)
        project_map = project_map or {"file_count": 0, "framework": "", "languages": [], "commands": {}}
        profile = classify_request(value.goal, project_map)
        adapter = self.registry.resolve(value.goal, project_map)
        if profile.get("framework") == "model-selected" and adapter.id != "model-selected":
            profile = {**profile, "framework": adapter.id, "language": adapter.describe().get("languages", ["model-selected"])[0]}
        preflight = adapter.preflight(value.goal, {**project_map, "framework": adapter.id})
        commands = adapter.validation_commands(project_map)
        contract = GenerationContract(
            profile=profile,
            framework=adapter.describe(),
            preflight=preflight,
            artifacts=adapter.artifact_contract(profile),
            validation=self._validation_contract(profile, commands),
            environment_plan=tuple(self._environment_plan(preflight)),
            model_contract=self._model_contract(value.goal, project_map, profile),
            acceptance_criteria=tuple(item for item in value.acceptance_criteria if str(item).strip()),
        ).to_dict()
        # Preserve the original preflight shape for existing API clients while
        # adding the complete SDK contract alongside it.
        return {**preflight, "generation_contract": contract, "sdk": {"name": "CodeQ", "version": "1", "plugin_group": PLUGIN_GROUP}}

    @staticmethod
    def _validation_contract(profile: dict[str, Any], commands: dict[str, str]) -> dict[str, Any]:
        required = ["artifacts", "independent_review"]
        required.extend(name for name in ("lint", "typecheck", "test", "build", "browser", "runtime") if commands.get(name))
        if profile.get("requires_runtime") and "runtime" not in required:
            required.append("runtime_smoke_or_target_evidence")
        return {"required_gates": required, "commands": commands, "completion_requires_all_configured_gates": True}

    @staticmethod
    def _environment_plan(preflight: dict[str, Any]) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for group in preflight.get("missing_toolchains", []):
            steps.append({"kind": "host_toolchain", "status": "required", "alternatives": list(group), "automatic": False})
        if preflight.get("host", {}).get("supported") is False:
            steps.append({"kind": "target_host", "status": "required", "system": preflight.get("framework", {}).get("host_platforms", []), "automatic": False})
        for requirement in preflight.get("external_requirements", []):
            steps.append({"kind": "external_infrastructure", "status": "required", "requirement": requirement, "automatic": False})
        for boundary in preflight.get("external_boundaries", []):
            steps.append({"kind": boundary.get("boundary", "external"), "status": boundary.get("status", "approval_required"), "automatic": False})
        return steps

    @staticmethod
    def _model_contract(goal: str, project_map: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        size = int(project_map.get("file_count", 0) or 0)
        words = len(str(goal).split())
        complexity = "large" if size > 2_000 or words > 500 else "medium" if size > 100 or words > 80 else "focused"
        roles = ["project_manager", "programmer", "tester", "reviewer"]
        if profile.get("architecture") == "full-stack":
            roles.extend(["architect", "backend", "database", "security"])
        return {
            "complexity": complexity,
            "required_capabilities": ["structured_json", "multi_file_code", "tool_use", "error_repair"],
            "roles": list(dict.fromkeys(roles)),
            "selection": "role-routed",
            "acceptance": "model output is provisional until repository validation and independent review pass",
        }
