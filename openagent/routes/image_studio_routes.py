"""Image studio route contracts."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..image_studio import ImageStudio
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "image_studio"
ROUTES = (
    ("GET", "/api/gallery/library"),
    ("GET", "/api/gallery/{id}/operations"),
    ("GET", "/api/gallery/{id}/versions"),
    ("GET", "/api/gallery/albums"),
    ("POST", "/api/gallery/upload"),
    ("POST", "/api/gallery/{id}/rotate"),
    ("POST", "/api/gallery/{id}/replace"),
    ("POST", "/api/gallery/upscale"),
    ("POST", "/api/gallery/albums"),
    ("POST", "/api/gallery/albums/{id}/assets"),
    ("POST", "/api/gallery/{id}/metadata"),
    ("POST", "/api/gallery/batch-tags"),
    ("POST", "/api/gallery/{id}/transform"),
    ("POST", "/api/gallery/{id}/undo"),
    ("POST", "/api/gallery/deduplicate"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ImageStudioRouteDependencies:
    studio: ImageStudio
    first: Any
    parse_int: Any
    query_owner: Any
    payload_owner: Any
    read_json: Any


def handle_get(path: str, query: str, deps: ImageStudioRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/gallery"):
        return None
    params = parse_qs(query)
    try:
        if path == "/api/gallery/library":
            return JsonRouteResult(
                deps.studio.library(
                    project=deps.first(params, "project") or "default",
                    owner=deps.query_owner(params),
                    query=deps.first(params, "q") or "",
                    tag=deps.first(params, "tag") or "",
                    limit=deps.parse_int(deps.first(params, "limit"), default=100),
                )
            )
        if path == "/api/gallery/albums":
            return JsonRouteResult({"albums": deps.studio.albums(project=deps.first(params, "project") or "default", owner=deps.query_owner(params))})
        if path.startswith("/api/gallery/") and path.endswith("/operations"):
            asset_id = unquote(path.removeprefix("/api/gallery/").removesuffix("/operations")).strip("/")
            return JsonRouteResult({"operations": deps.studio.operations(asset_id, owner=deps.query_owner(params))})
        if path.startswith("/api/gallery/") and path.endswith("/versions"):
            asset_id = unquote(path.removeprefix("/api/gallery/").removesuffix("/versions")).strip("/")
            return JsonRouteResult({"versions": deps.studio.versions(asset_id, owner=deps.query_owner(params))})
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return None


def handle_post(path: str, deps: ImageStudioRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/gallery"):
        return None
    payload = deps.read_json()
    owner = deps.payload_owner(payload)
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/gallery/upload":
            asset = deps.studio.upload_asset(
                filename=str(payload.get("filename", "image.bin")),
                content=deps.studio.decode_base64(str(payload.get("content_base64", ""))),
                project=project,
                owner=owner,
                content_type=str(payload.get("content_type", "")),
                tags=[str(item) for item in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else [],
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"asset": asset}, HTTPStatus.CREATED)
        if path == "/api/gallery/albums":
            return JsonRouteResult(
                {"album": deps.studio.create_album(str(payload.get("name", "")), project=project, owner=owner, description=str(payload.get("description", "")))},
                HTTPStatus.CREATED,
            )
        if path.startswith("/api/gallery/albums/") and path.endswith("/assets"):
            album_id = unquote(path.removeprefix("/api/gallery/albums/").removesuffix("/assets")).strip("/")
            return JsonRouteResult(deps.studio.add_to_album(album_id, str(payload.get("asset_id", "")), owner=owner))
        if path == "/api/gallery/deduplicate":
            return JsonRouteResult(deps.studio.deduplicate(project=project, owner=owner if owner != "local" else ""))
        if path == "/api/gallery/batch-tags":
            asset_ids = payload.get("asset_ids")
            tags = payload.get("tags")
            return JsonRouteResult(deps.studio.batch_tag(
                [str(item) for item in asset_ids] if isinstance(asset_ids, list) else [],
                [str(item) for item in tags] if isinstance(tags, list) else [],
                owner=owner,
                replace=bool(payload.get("replace")),
            ))
        if path.startswith("/api/gallery/") and path.endswith("/rotate"):
            asset_id = unquote(path.removeprefix("/api/gallery/").removesuffix("/rotate")).strip("/")
            return JsonRouteResult(
                deps.studio.rotate(
                    asset_id,
                    degrees=deps.parse_int(str(payload.get("degrees", "90")), default=90),
                    owner=owner,
                )
            )
        if path.startswith("/api/gallery/") and path.endswith("/replace"):
            asset_id = unquote(path.removeprefix("/api/gallery/").removesuffix("/replace")).strip("/")
            return JsonRouteResult(
                deps.studio.replace(
                    asset_id,
                    content=deps.studio.decode_base64(str(payload.get("content_base64", ""))),
                    filename=str(payload.get("filename", "")),
                    content_type=str(payload.get("content_type", "")),
                    owner=owner,
                )
            )
        if path == "/api/gallery/upscale":
            asset_id = str(payload.get("asset_id", "")).strip()
            return JsonRouteResult(
                deps.studio.upscale(
                    asset_id,
                    scale=deps.parse_int(str(payload.get("scale", "2")), default=2),
                    owner=owner,
                )
            )
        if path.startswith("/api/gallery/") and path.endswith("/metadata"):
            asset_id = unquote(path.removeprefix("/api/gallery/").removesuffix("/metadata")).strip("/")
            tags = payload.get("tags")
            return JsonRouteResult({"asset": deps.studio.edit_metadata(
                asset_id, owner=owner, filename=str(payload.get("filename", "")),
                tags=[str(item) for item in tags] if isinstance(tags, list) else None,
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                favorite=bool(payload["favorite"]) if "favorite" in payload else None,
            )})
        if path.startswith("/api/gallery/") and path.endswith("/transform"):
            asset_id = unquote(path.removeprefix("/api/gallery/").removesuffix("/transform")).strip("/")
            mask_value = str(payload.get("mask_base64", ""))
            return JsonRouteResult(deps.studio.transform(
                asset_id, str(payload.get("operation", "")), owner=owner,
                params=payload.get("params") if isinstance(payload.get("params"), dict) else {},
                mask=deps.studio.decode_base64(mask_value) if mask_value else None,
            ))
        if path.startswith("/api/gallery/") and path.endswith("/undo"):
            asset_id = unquote(path.removeprefix("/api/gallery/").removesuffix("/undo")).strip("/")
            return JsonRouteResult(deps.studio.undo(asset_id, owner=owner))
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None
