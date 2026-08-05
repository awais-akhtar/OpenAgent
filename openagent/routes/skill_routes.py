"""Skill editor, validation, packaging, and supply-chain routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..skill_ecosystem import SkillEcosystem
from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "skills"
ROUTES = (
    ("GET", "/api/skills"), ("GET", "/api/skills/{id}"),
    ("GET", "/api/skills/{id}/permissions"), ("GET", "/api/skills/{id}/dependency-scan"),
    ("GET", "/api/skills/marketplace"), ("GET", "/api/skills/installations"),
    ("POST", "/api/skills"), ("POST", "/api/skills/import-url"),
    ("POST", "/api/skills/{id}/test"), ("POST", "/api/skills/{id}/package"),
    ("POST", "/api/skills/{id}/export"), ("POST", "/api/skills/{id}/sandbox"),
    ("POST", "/api/skills/{id}/publish"),
    ("POST", "/api/skills/marketplace/{id}/install"), ("POST", "/api/skills/marketplace/{id}/remove"),
    ("POST", "/api/skills/installations/{id}/update"), ("POST", "/api/skills/installations/{id}/uninstall"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class SkillRouteDependencies:
    skills: SkillEcosystem
    read_json: Any


def handle_get(path: str, query: str, deps: SkillRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/skills"):
        return None
    params = parse_qs(query)
    if path == "/api/skills":
        return JsonRouteResult({"skills": deps.skills.list(project=(params.get("project") or ["default"])[0])})
    if path == "/api/skills/marketplace":
        return JsonRouteResult({"listings": deps.skills.marketplace(project=(params.get("project") or ["default"])[0])})
    if path == "/api/skills/installations":
        return JsonRouteResult({"installations": deps.skills.installations(project=(params.get("project") or ["default"])[0])})
    suffix = unquote(path.removeprefix("/api/skills/")).strip("/")
    try:
        if suffix.endswith("/permissions"):
            return JsonRouteResult(deps.skills.permission_preview(suffix.removesuffix("/permissions").strip("/")))
        if suffix.endswith("/dependency-scan"):
            return JsonRouteResult(deps.skills.dependency_scan(suffix.removesuffix("/dependency-scan").strip("/")))
        return JsonRouteResult({"skill": deps.skills.get(suffix)})
    except KeyError:
        return JsonRouteResult({"error": "Skill not found."}, HTTPStatus.NOT_FOUND)


def handle_post(path: str, deps: SkillRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/skills"):
        return None
    payload = deps.read_json()
    try:
        if path == "/api/skills":
            skill = deps.skills.save(
                name=str(payload.get("name", "")), markdown=str(payload.get("markdown", "")),
                project=str(payload.get("project", "default")), version=str(payload.get("version", "0.1.0")),
                manifest=payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {},
                skill_id=str(payload.get("skill_id", "")),
            )
            return JsonRouteResult({"skill": skill}, HTTPStatus.CREATED)
        if path == "/api/skills/import-url":
            skill = deps.skills.import_url(
                str(payload.get("url", "")), allow_network=bool(payload.get("allow_network")),
                project=str(payload.get("project", "default")),
            )
            return JsonRouteResult({"skill": skill}, HTTPStatus.CREATED)
        if path.startswith("/api/skills/marketplace/"):
            suffix = unquote(path.removeprefix("/api/skills/marketplace/")).strip("/")
            listing_id, action = suffix.rsplit("/", 1)
            if action == "install":
                return JsonRouteResult({"installation": deps.skills.install(
                    listing_id, signing_key=str(payload.get("signing_key", "")),
                    project=str(payload.get("project", "default")) or "default",
                )}, HTTPStatus.CREATED)
            if action == "remove":
                return JsonRouteResult({"listing": deps.skills.remove_listing(listing_id)})
        if path.startswith("/api/skills/installations/"):
            suffix = unquote(path.removeprefix("/api/skills/installations/")).strip("/")
            installation_id, action = suffix.rsplit("/", 1)
            if action == "uninstall":
                return JsonRouteResult({"installation": deps.skills.uninstall(installation_id)})
            if action == "update":
                return JsonRouteResult(deps.skills.update_installation(
                    installation_id, listing_id=str(payload.get("listing_id", "")),
                    signing_key=str(payload.get("signing_key", "")),
                ))
        suffix = unquote(path.removeprefix("/api/skills/")).strip("/")
        skill_id, action = suffix.rsplit("/", 1)
        if action == "test":
            return JsonRouteResult(deps.skills.test(skill_id))
        if action == "package":
            return JsonRouteResult(deps.skills.package(skill_id, signing_key=str(payload.get("signing_key", ""))), HTTPStatus.CREATED)
        if action == "export":
            return JsonRouteResult(deps.skills.export_markdown(skill_id), HTTPStatus.CREATED)
        if action == "sandbox":
            return JsonRouteResult(deps.skills.run_sandbox(
                skill_id,
                input_data=payload.get("input") if isinstance(payload.get("input"), dict) else {},
                approved_permissions=[str(item) for item in payload.get("approved_permissions", [])]
                if isinstance(payload.get("approved_permissions"), list) else [],
            ), HTTPStatus.CREATED)
        if action == "publish":
            return JsonRouteResult({"listing": deps.skills.publish(
                skill_id, signing_key=str(payload.get("signing_key", "")),
                visibility=str(payload.get("visibility", "local")),
            )}, HTTPStatus.CREATED)
    except KeyError:
        return JsonRouteResult({"error": "Skill not found."}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None
