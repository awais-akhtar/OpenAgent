"""Personal OS HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs, unquote

from . import JsonRouteResult
from ._contracts import route_contract

ROUTE_GROUP = "personal_os"
ROUTES = (
    ("GET", "/api/personal-os/status"),
    ("GET", "/api/personal-os/presets"),
    ("POST", "/api/personal-os/presets"),
    ("GET", "/api/personal-os/signatures"),
    ("POST", "/api/personal-os/signatures"),
    ("GET", "/api/personal-os/drafts"),
    ("POST", "/api/personal-os/drafts"),
    ("GET", "/api/personal-os/drafts/{id}/versions"),
    ("GET", "/api/personal-os/notes"),
    ("POST", "/api/personal-os/notes"),
    ("GET", "/api/personal-os/gallery"),
    ("POST", "/api/personal-os/gallery"),
    ("POST", "/api/personal-os/gallery/favorite"),
    ("GET", "/api/personal-os/compare"),
    ("POST", "/api/personal-os/compare/start"),
    ("POST", "/api/personal-os/compare/vote"),
    ("GET", "/api/personal-os/export"),
    ("POST", "/api/personal-os/import"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class PersonalOSRouteDependencies:
    personal_os: Any
    read_json: Callable[[], dict[str, Any]]
    auth_scope_owner: Callable[[], str]
    query_owner: Callable[[dict[str, list[str]]], str]
    payload_owner: Callable[[dict[str, object], str], str]
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_int: Callable[..., int]


def handle_get(path: str, query: str, deps: PersonalOSRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/personal-os/"):
        return None
    params = parse_qs(query)
    first = deps.first
    pos = deps.personal_os

    project = first(params, "project") or "default"
    # Local-mode writes default to the ``local`` owner, so reads must use the
    # same owner unless authentication supplies a user-specific identity.
    owner = deps.query_owner(params) or "local"

    if path == "/api/personal-os/status":
        return JsonRouteResult(pos.status(project=project, owner=owner))

    if path == "/api/personal-os/presets":
        return JsonRouteResult({
            "presets": pos.presets(
                project=project,
                owner=owner,
                kind=first(params, "kind") or "",
            )
        })

    if path == "/api/personal-os/signatures":
        return JsonRouteResult({"signatures": pos.signatures(project=project, owner=owner)})

    if path == "/api/personal-os/drafts":
        return JsonRouteResult({
            "drafts": pos.drafts(
                project=project,
                owner=owner,
                status=first(params, "status") or "",
            )
        })

    if path.startswith("/api/personal-os/drafts/") and path.endswith("/versions"):
        draft_id = unquote(path.removeprefix("/api/personal-os/drafts/").removesuffix("/versions")).strip()
        if not draft_id or "/" in draft_id:
            return JsonRouteResult({"error": "draft_id is required"}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"versions": pos.draft_versions(draft_id)})

    if path == "/api/personal-os/notes":
        return JsonRouteResult({
            "notes": pos.notes(
                project=project,
                owner=owner,
                status=first(params, "status") or "",
                query=first(params, "q") or "",
            )
        })

    if path == "/api/personal-os/gallery":
        return JsonRouteResult({
            "assets": pos.gallery(project=project, owner=owner, tag=first(params, "tag") or ""),
            "stats": pos.gallery_stats(project=project, owner=owner),
        })

    if path == "/api/personal-os/compare":
        return JsonRouteResult({"compares": pos.compare_history(project=project, owner=owner)})

    if path == "/api/personal-os/export":
        return JsonRouteResult(pos.export_bundle(project=project, owner=owner))

    return None


def handle_post(path: str, deps: PersonalOSRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/personal-os/"):
        return None
    payload = deps.read_json()
    pos = deps.personal_os
    project = str(payload.get("project", "default")).strip() or "default"
    owner = deps.payload_owner(payload, "local")

    try:
        if path == "/api/personal-os/presets":
            name = str(payload.get("name", "")).strip()
            if not name:
                return JsonRouteResult({"error": "name is required"}, HTTPStatus.BAD_REQUEST)
            preset = pos.upsert_preset(
                name=name,
                kind=str(payload.get("kind", "chat")),
                instructions=str(payload.get("instructions", "")),
                provider=str(payload.get("provider", "auto")),
                model=str(payload.get("model", "")),
                parameters=payload.get("parameters") if isinstance(payload.get("parameters"), dict) else None,
                project=project,
                owner=owner,
            )
            return JsonRouteResult({"preset": preset}, HTTPStatus.CREATED)

        if path == "/api/personal-os/signatures":
            label = str(payload.get("label", payload.get("name", ""))).strip()
            content = str(payload.get("content", "")).strip()
            if not label or not content:
                return JsonRouteResult({"error": "label and content are required"}, HTTPStatus.BAD_REQUEST)
            signature = pos.upsert_signature(
                label=label,
                content=content,
                is_default=bool(payload.get("is_default", False)),
                project=project,
                owner=owner,
            )
            return JsonRouteResult({"signature": signature}, HTTPStatus.CREATED)

        if path == "/api/personal-os/drafts":
            draft = pos.save_draft(
                title=str(payload.get("title", "")).strip() or "Untitled draft",
                content=str(payload.get("content", "")),
                status=str(payload.get("status", "draft")),
                draft_id=str(payload.get("draft_id", "")).strip(),
                project=project,
                owner=owner,
            )
            return JsonRouteResult({"draft": draft}, HTTPStatus.CREATED)

        if path == "/api/personal-os/notes":
            title = str(payload.get("title", payload.get("content", ""))).strip()
            if not title:
                return JsonRouteResult({"error": "title is required"}, HTTPStatus.BAD_REQUEST)
            note = pos.save_note(
                title=title,
                body=str(payload.get("body", payload.get("content", ""))),
                tags=payload.get("tags") if isinstance(payload.get("tags"), list) else [],
                due_at=str(payload.get("due_at", "")),
                status=str(payload.get("status", "open")),
                project=project,
                owner=owner,
            )
            return JsonRouteResult({"note": note}, HTTPStatus.CREATED)

        if path == "/api/personal-os/gallery":
            title = str(payload.get("title", payload.get("item", {}).get("title", "") if isinstance(payload.get("item"), dict) else "")).strip()
            if not title:
                return JsonRouteResult({"error": "title is required"}, HTTPStatus.BAD_REQUEST)
            asset = pos.add_gallery_asset(
                title=title,
                media_type=str(payload.get("media_type", "image")),
                source_path=str(payload.get("source_path", "")),
                content=str(payload.get("content", "")),
                tags=payload.get("tags") if isinstance(payload.get("tags"), list) else [],
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                project=project,
                owner=owner,
            )
            return JsonRouteResult({"asset": asset}, HTTPStatus.CREATED)

        if path == "/api/personal-os/gallery/favorite":
            asset_id = str(payload.get("asset_id", payload.get("item_id", payload.get("id", "")))).strip()
            if not asset_id:
                return JsonRouteResult({"error": "item_id is required"}, HTTPStatus.BAD_REQUEST)
            asset = pos.set_gallery_favorite(
                asset_id=asset_id,
                favorite=bool(payload.get("favorite", True)),
                project=project,
                owner=owner,
            )
            return JsonRouteResult({"asset": asset})

        if path == "/api/personal-os/compare/start":
            compare = pos.start_compare(
                prompt=str(payload.get("prompt", payload.get("left", ""))),
                left_output=str(payload.get("left_output", payload.get("left", ""))),
                right_output=str(payload.get("right_output", payload.get("right", ""))),
                left_label=str(payload.get("left_label", "A")),
                right_label=str(payload.get("right_label", "B")),
                project=project,
                owner=owner,
            )
            return JsonRouteResult({"compare": compare}, HTTPStatus.CREATED)

        if path == "/api/personal-os/compare/vote":
            session_id = str(payload.get("session_id", payload.get("compare_id", ""))).strip()
            vote = str(payload.get("vote", "")).strip()
            if not session_id or not vote:
                return JsonRouteResult({"error": "session_id and vote are required"}, HTTPStatus.BAD_REQUEST)
            compare = pos.vote_compare(
                compare_id=session_id,
                vote=vote,
                project=project,
                owner=owner,
            )
            return JsonRouteResult({"compare": compare})

        if path == "/api/personal-os/import":
            data = payload.get("bundle", payload.get("data", payload))
            if not isinstance(data, dict):
                return JsonRouteResult({"error": "data must be an object"}, HTTPStatus.BAD_REQUEST)
            result = pos.import_bundle(data, project=project, owner=owner)
            return JsonRouteResult(result)

    except Exception as exc:  # noqa: BLE001
        from ..personal_os import PersonalOSError
        if isinstance(exc, PersonalOSError):
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        raise

    return None
