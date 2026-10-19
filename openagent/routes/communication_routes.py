"""Communication hub HTTP routes.

The stdlib server delegates here so communication protocol details stay out of
the runtime facade while existing URLs keep their shape.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "communication"
ROUTES = (
    ("GET", "/api/communication/connectors"),
    ("GET", "/api/communication/inbox"),
    ("GET", "/api/communication/drafts"),
    ("GET", "/api/communication/sent"),
    ("GET", "/api/communication/contacts"),
    ("GET", "/api/communication/calendar"),
    ("GET", "/api/communication/analytics"),
    ("GET", "/api/communication/sync-runs"),
    ("GET", "/api/communication/dead-letters"),
    ("GET", "/api/communication/oauth/providers"),
    ("GET", "/api/communication/oauth/accounts"),
    ("GET", "/api/communication/oauth/callback"),
    ("GET", "/api/communication/oauth/callbacks"),
    ("GET", "/api/communication/oauth/mailbox-policy"),
    ("GET", "/api/communication/provider-certification"),
    ("GET", "/api/communication/attachments"),
    ("GET", "/api/communication/approvals"),
    ("GET", "/api/communication/scheduled"),
    ("POST", "/api/communication/connectors/configure"),
    ("POST", "/api/communication/contacts"),
    ("POST", "/api/communication/messages/import"),
    ("POST", "/api/communication/drafts/reply"),
    ("POST", "/api/communication/drafts"),
    ("POST", "/api/communication/send/request-approval"),
    ("POST", "/api/communication/send/resolve"),
    ("POST", "/api/communication/send"),
    ("POST", "/api/communication/calendar"),
    ("POST", "/api/communication/sync"),
    ("POST", "/api/communication/sync-resilient"),
    ("POST", "/api/communication/oauth/configure"),
    ("POST", "/api/communication/oauth/accounts/configure"),
    ("POST", "/api/communication/oauth/authorize-url"),
    ("POST", "/api/communication/oauth/refresh"),
    ("POST", "/api/communication/attachments"),
    ("POST", "/api/communication/messages/state"),
    ("POST", "/api/communication/scheduled"),
    ("POST", "/api/communication/scheduled/process"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class CommunicationRouteDependencies:
    communication: Any
    first: Callable[[dict[str, list[str]], str], str | None]
    parse_bool: Callable[[object, bool], bool]
    parse_int: Callable[[object, int], int]
    query_owner: Callable[[dict[str, list[str]]], str]
    payload_owner: Callable[[dict[str, object], str], str]
    read_json: Callable[[], dict[str, object]]


def handle_get(path: str, query: str, deps: CommunicationRouteDependencies) -> JsonRouteResult | None:
    params = parse_qs(query)
    first = deps.first
    communication = deps.communication

    if path == "/api/communication/connectors":
        return JsonRouteResult({"connectors": communication.connectors()})
    if path == "/api/communication/approvals":
        return JsonRouteResult({"approvals": communication.pending_approvals(project=first(params, "project") or "default", owner=deps.query_owner(params), limit=deps.parse_int(first(params, "limit"), default=100))})
    if path == "/api/communication/scheduled":
        return JsonRouteResult({"scheduled": communication.scheduled_sends(project=first(params, "project") or "default", owner=deps.query_owner(params), limit=deps.parse_int(first(params, "limit"), default=100))})
    if path == "/api/communication/inbox":
        follow_up = first(params, "follow_up")
        follow_filter = None if follow_up is None else deps.parse_bool(follow_up, default=False)
        return JsonRouteResult({
            "messages": communication.inbox(
                project=first(params, "project") or "default",
                query=first(params, "q") or "",
                follow_up=follow_filter,
                owner=deps.query_owner(params),
            )
        })
    if path == "/api/communication/drafts":
        return JsonRouteResult({
            "drafts": communication.list_drafts(
                project=first(params, "project") or "default",
                owner=deps.query_owner(params),
                limit=deps.parse_int(first(params, "limit"), default=100),
            )
        })
    if path == "/api/communication/sent":
        return JsonRouteResult({
            "sent": communication.list_sent_messages(
                project=first(params, "project") or "default",
                owner=deps.query_owner(params),
                limit=deps.parse_int(first(params, "limit"), default=100),
            )
        })
    if path == "/api/communication/contacts":
        return JsonRouteResult({
            "contacts": communication.contacts(
                project=first(params, "project") or "default",
                query=first(params, "q") or "",
                owner=deps.query_owner(params),
            )
        })
    if path == "/api/communication/calendar":
        return JsonRouteResult({
            "events": communication.calendar(
                project=first(params, "project") or "default",
                owner=deps.query_owner(params),
            )
        })
    if path == "/api/communication/analytics":
        return JsonRouteResult(
            communication.analytics(project=first(params, "project") or "default", owner=deps.query_owner(params))
        )
    if path == "/api/communication/sync-runs":
        return JsonRouteResult({
            "runs": communication.sync_runs(
                project=first(params, "project") or "default",
                platform=first(params, "platform") or "",
                owner=deps.query_owner(params),
                limit=deps.parse_int(first(params, "limit"), default=50),
            )
        })
    if path == "/api/communication/dead-letters":
        return JsonRouteResult({
            "dead_letters": communication.list_dead_letters(
                project=first(params, "project") or "default",
                owner=deps.query_owner(params),
            )
        })
    if path == "/api/communication/oauth/providers":
        return JsonRouteResult({"providers": communication.oauth_providers()})
    if path == "/api/communication/provider-certification":
        return JsonRouteResult(communication.provider_certification_matrix())
    if path == "/api/communication/oauth/accounts":
        return JsonRouteResult({
            "accounts": communication.oauth_accounts(
                project=first(params, "project") or "default",
                owner=deps.query_owner(params),
                platform=first(params, "platform") or "",
            )
        })
    if path == "/api/communication/oauth/callback":
        callback = communication.record_oauth_callback(
            state=first(params, "state") or "",
            code=first(params, "code") or "",
            error=first(params, "error") or "",
            account_id=first(params, "account_id") or "",
            redirect_uri=first(params, "redirect_uri") or "",
            owner=deps.query_owner(params) or "local",
            project=first(params, "project") or "default",
            metadata={"source": "local-callback-route"},
        )
        return JsonRouteResult({"callback": callback})
    if path == "/api/communication/oauth/callbacks":
        return JsonRouteResult({
            "callbacks": communication.oauth_callbacks(
                project=first(params, "project") or "default",
                owner=deps.query_owner(params),
                limit=deps.parse_int(first(params, "limit"), default=50),
            )
        })
    if path == "/api/communication/oauth/mailbox-policy":
        try:
            policy = communication.mailbox_policy(
                provider=first(params, "provider") or "generic",
                account_id=first(params, "account_id") or "",
                owner=deps.query_owner(params),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(policy)
    if path == "/api/communication/attachments":
        return JsonRouteResult({
            "attachments": communication.attachments(
                project=first(params, "project") or "default",
                owner=deps.query_owner(params),
                limit=deps.parse_int(first(params, "limit"), default=50),
            )
        })
    return None


def handle_post(path: str, deps: CommunicationRouteDependencies) -> JsonRouteResult | None:
    if not path.startswith("/api/communication/"):
        return None
    communication = deps.communication
    payload = deps.read_json()
    project = str(payload.get("project", "default")).strip() or "default"
    owner_for = lambda default="local": deps.payload_owner(payload, default)

    try:
        if path == "/api/communication/messages/state":
            return JsonRouteResult({"message": communication.mutate_message(
                str(payload.get("message_id", "")), owner=owner_for(""),
                folder=str(payload["folder"]) if "folder" in payload else None,
                read=bool(payload["read"]) if "read" in payload else None,
                answered=bool(payload["answered"]) if "answered" in payload else None,
                spam=bool(payload["spam"]) if "spam" in payload else None,
                archived=bool(payload["archived"]) if "archived" in payload else None,
            )})
        if path == "/api/communication/scheduled":
            return JsonRouteResult({"scheduled": communication.schedule_draft(
                str(payload.get("draft_id", "")), send_at=str(payload.get("send_at", "")),
                approval_id=str(payload.get("approval_id", "")), project=project, owner=owner_for(),
            )}, HTTPStatus.CREATED)
        if path == "/api/communication/scheduled/process":
            return JsonRouteResult(communication.process_scheduled_sends(
                project=project, owner=owner_for(""),
                allow_external=deps.parse_bool(payload.get("allow_external"), default=False),
            ))
        if path == "/api/communication/connectors/configure":
            connector = communication.configure_connector(
                str(payload.get("platform", "")).strip(),
                enabled=deps.parse_bool(payload.get("enabled"), default=False),
                config=payload.get("config") if isinstance(payload.get("config"), dict) else {},
            )
            return JsonRouteResult({"connector": connector})
        if path == "/api/communication/contacts":
            contact = communication.upsert_contact(
                str(payload.get("name", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
                handles=payload.get("handles") if isinstance(payload.get("handles"), dict) else {},
                notes=str(payload.get("notes", "")),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"contact": contact}, HTTPStatus.CREATED)
        if path == "/api/communication/messages/import":
            message = communication.import_message(
                platform=str(payload.get("platform", "email")).strip() or "email",
                sender=str(payload.get("sender", "")).strip(),
                recipients=[str(item) for item in payload.get("recipients", [])] if isinstance(payload.get("recipients"), list) else [],
                subject=str(payload.get("subject", "")),
                body=str(payload.get("body", "")),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
                thread_id=str(payload.get("thread_id", "")),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"message": message}, HTTPStatus.CREATED)
        if path == "/api/communication/drafts/reply":
            draft = communication.draft_reply(
                str(payload.get("message_id", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
                instructions=str(payload.get("instructions", "")),
            )
            return JsonRouteResult({"draft": draft}, HTTPStatus.CREATED)
        if path == "/api/communication/drafts":
            draft = communication.create_draft(
                platform=str(payload.get("platform", "email")).strip() or "email",
                recipients=[str(item) for item in payload.get("recipients", [])] if isinstance(payload.get("recipients"), list) else [],
                subject=str(payload.get("subject", "")),
                body=str(payload.get("body", "")),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
            )
            return JsonRouteResult({"draft": draft}, HTTPStatus.CREATED)
        if path == "/api/communication/send/request-approval":
            approval = communication.request_send_approval(
                str(payload.get("draft_id", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
                reason=str(payload.get("reason", "")),
            )
            return JsonRouteResult({"approval": approval}, HTTPStatus.CREATED)
        if path == "/api/communication/send/resolve":
            approval = communication.resolve_send_approval(
                str(payload.get("approval_id", "")).strip(),
                approved=deps.parse_bool(payload.get("approved"), default=False),
                owner=owner_for(""),
            )
            return JsonRouteResult({"approval": approval})
        if path == "/api/communication/send":
            sent = communication.send_draft(
                str(payload.get("draft_id", "")).strip(),
                approval_id=str(payload.get("approval_id", "")).strip() or None,
                allow_external=deps.parse_bool(payload.get("allow_external"), default=False),
                owner=owner_for(""),
            )
            return JsonRouteResult({"sent": sent}, HTTPStatus.CREATED)
        if path == "/api/communication/calendar":
            event = communication.add_calendar_event(
                title=str(payload.get("title", "")).strip(),
                starts_at=str(payload.get("starts_at", "")).strip(),
                ends_at=str(payload.get("ends_at", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
                calendar_id=str(payload.get("calendar_id", "local")).strip() or "local",
                attendees=[str(item) for item in payload.get("attendees", [])] if isinstance(payload.get("attendees"), list) else [],
                source=str(payload.get("source", "local")).strip() or "local",
            )
            return JsonRouteResult({"event": event}, HTTPStatus.CREATED)
        if path == "/api/communication/sync":
            run = communication.sync_connector(
                str(payload.get("platform", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
                payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                allow_external=deps.parse_bool(payload.get("allow_external"), default=False),
            )
            return JsonRouteResult({"run": run}, HTTPStatus.CREATED)
        if path == "/api/communication/sync-resilient":
            result = communication.sync_connector_resilient(
                str(payload.get("platform", "")).strip(),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
                payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                allow_external=deps.parse_bool(payload.get("allow_external"), default=False),
                max_attempts=deps.parse_int(str(payload.get("max_attempts", "")), default=3),
            )
            status = HTTPStatus.OK if result.get("status") == "dead_lettered" else HTTPStatus.CREATED
            return JsonRouteResult(result, status)
        if path in {"/api/communication/oauth/configure", "/api/communication/oauth/accounts/configure"}:
            account = communication.configure_oauth_account(
                provider=str(payload.get("provider", "generic")),
                platform=str(payload.get("platform", "email")),
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
                label=str(payload.get("label", "")),
                client_id=str(payload.get("client_id", "")),
                client_secret=str(payload.get("client_secret", "")),
                auth_endpoint=str(payload.get("auth_endpoint", "")),
                token_endpoint=str(payload.get("token_endpoint", "")),
                redirect_uri=str(payload.get("redirect_uri", "")),
                scopes=[str(item) for item in payload.get("scopes", [])] if isinstance(payload.get("scopes"), list) else None,
                mailbox_policy=payload.get("mailbox_policy") if isinstance(payload.get("mailbox_policy"), dict) else None,
            )
            return JsonRouteResult({"account": account}, HTTPStatus.CREATED)
        if path == "/api/communication/oauth/authorize-url":
            auth = communication.oauth_authorization_url(
                str(payload.get("account_id", "")).strip(),
                owner=owner_for(""),
                state=str(payload.get("state", "")),
                code_challenge=str(payload.get("code_challenge", "")),
            )
            return JsonRouteResult(auth)
        if path == "/api/communication/oauth/refresh":
            refreshed = communication.refresh_oauth_token(
                str(payload.get("account_id", "")).strip(),
                owner=owner_for(""),
                allow_external=deps.parse_bool(payload.get("allow_external"), default=False),
                authorization_code=str(payload.get("authorization_code", "")),
                code_verifier=str(payload.get("code_verifier", "")),
            )
            return JsonRouteResult(refreshed)
        if path == "/api/communication/attachments":
            encoded = str(payload.get("content_base64", ""))
            try:
                content = base64.b64decode(encoded.encode("ascii"), validate=True)
            except (ValueError, binascii.Error):
                return JsonRouteResult({"error": "content_base64 must be valid base64."}, HTTPStatus.BAD_REQUEST)
            attachment = communication.ingest_attachment(
                filename=str(payload.get("filename", "attachment.bin")),
                content=content,
                project=str(payload.get("project", "default")).strip() or "default",
                owner=owner_for(),
                message_id=str(payload.get("message_id", "")),
                content_type=str(payload.get("content_type", "application/octet-stream")),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                create_document=deps.parse_bool(payload.get("create_document"), default=True),
            )
            return JsonRouteResult({"attachment": attachment}, HTTPStatus.CREATED)
    except Exception as exc:  # noqa: BLE001 - route module maps domain failures to JSON errors
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None
