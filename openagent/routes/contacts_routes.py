"""Contacts and personal CRM route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote

from ..contacts import ContactsManager
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "contacts"
ROUTES = (
    ("GET", "/api/contacts"),
    ("POST", "/api/contacts"),
    ("GET", "/api/contacts/search"),
    ("POST", "/api/contacts/import"),
    ("GET", "/api/contacts/export"),
    ("PATCH", "/api/contacts/{id}"),
    ("DELETE", "/api/contacts/{id}"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ContactsRouteDependencies:
    contacts: ContactsManager
    query_owner: Any
    payload_owner: Any
    read_json: Any


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


def handle_get(path: str, query: str, deps: ContactsRouteDependencies) -> JsonRouteResult | None:
    params = parse_qs(query)
    project = _first(params, "project") or "default"
    owner = deps.query_owner(params)
    if path == "/api/contacts":
        return JsonRouteResult(
            {
                "contacts": deps.contacts.list_contacts(
                    project=project,
                    query=_first(params, "q"),
                    owner=owner,
                )
            }
        )
    if path == "/api/contacts/search":
        return JsonRouteResult(
            {
                "contacts": deps.contacts.search(
                    _first(params, "q"),
                    project=project,
                    owner=owner,
                )
            }
        )
    if path == "/api/contacts/export":
        return JsonRouteResult(deps.contacts.export_contacts(project=project, owner=owner))
    return None


def handle_post(path: str, deps: ContactsRouteDependencies) -> JsonRouteResult | None:
    if path not in {"/api/contacts", "/api/contacts/import"}:
        return None
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    owner = deps.payload_owner(payload, "local")
    try:
        if path == "/api/contacts":
            contact = deps.contacts.create_contact(
                name=str(payload.get("name", "")).strip(),
                project=project,
                owner=owner,
                handles=payload.get("handles") if isinstance(payload.get("handles"), dict) else {},
                notes=str(payload.get("notes", "")),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"contact": contact}, HTTPStatus.CREATED)
        records = payload.get("contacts")
        if not isinstance(records, list):
            return JsonRouteResult({"error": "contacts must be a list."}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(deps.contacts.import_contacts(records, project=project, owner=owner), HTTPStatus.CREATED)
    except (KeyError, ValueError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def handle_patch(path: str, deps: ContactsRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/contacts/"):
        return None
    contact_id = unquote(path.removeprefix("/api/contacts/")).strip()
    if "/" in contact_id:
        return None
    payload = deps.read_json()
    try:
        contact = deps.contacts.update_contact(
            contact_id,
            owner=deps.payload_owner(payload, ""),
            name=str(payload["name"]) if "name" in payload else None,
            handles=payload.get("handles") if isinstance(payload.get("handles"), dict) else None,
            notes=str(payload["notes"]) if "notes" in payload else None,
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
        )
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult({"contact": contact})


def handle_delete(path: str, deps: ContactsRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/contacts/"):
        return None
    contact_id = unquote(path.removeprefix("/api/contacts/")).strip()
    if "/" in contact_id:
        return None
    try:
        result = deps.contacts.delete_contact(contact_id, owner="")
    except KeyError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
    return JsonRouteResult(result)
