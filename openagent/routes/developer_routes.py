"""Developer platform HTTP routes."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable

from ..developer import DeveloperPlatformError
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "developer"
ROUTES = (
    ("GET", "/api/developer/status"),
    ("GET", "/api/developer/schema"),
    ("GET", "/api/developer/webhooks"),
    ("GET", "/api/developer/webhook-deliveries"),
    ("GET", "/api/developer/plugins"),
    ("GET", "/api/developer/tools"),
    ("GET", "/api/developer/mcp"),
    ("GET", "/api/developer/mcp/servers"),
    ("GET", "/api/developer/extensions"),
    ("GET", "/api/developer/sdk"),
    ("GET", "/api/developer/api-keys"),
    ("POST", "/api/developer/api-keys"),
    ("POST", "/api/developer/api-keys/verify"),
    ("POST", "/api/developer/webhooks"),
    ("POST", "/api/developer/webhooks/dispatch"),
    ("POST", "/api/developer/plugins"),
    ("POST", "/api/developer/tools"),
    ("POST", "/api/developer/tools/invoke"),
    ("POST", "/api/developer/mcp/servers"),
    ("POST", "/api/developer/extensions"),
    ("POST", "/api/developer/sdk"),
    ("POST", "/api/graphql"),
    ("POST", "/api/developer/graphql"),
)
POST_PATHS = frozenset(path for method, path in ROUTES if method == "POST")


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class DeveloperRouteDependencies:
    developer: Any
    parse_bool: Callable[..., bool]
    read_json: Callable[[], dict[str, object]]


def handle_get(path: str, _query: str, deps: DeveloperRouteDependencies) -> JsonRouteResult | None:
    developer = deps.developer
    handlers: dict[str, Callable[[], dict[str, Any]]] = {
        "/api/developer/status": developer.status,
        "/api/developer/schema": developer.graphql_schema,
        "/api/developer/mcp": developer.mcp_manifest,
    }
    if path in handlers:
        return JsonRouteResult(handlers[path]())

    collections: dict[str, tuple[str, Callable[[], list[dict[str, Any]]]]] = {
        "/api/developer/webhooks": ("webhooks", developer.webhooks),
        "/api/developer/webhook-deliveries": ("deliveries", developer.webhook_deliveries),
        "/api/developer/plugins": ("plugins", developer.plugins),
        "/api/developer/tools": ("tools", developer.tools),
        "/api/developer/mcp/servers": ("servers", developer.mcp_servers),
        "/api/developer/extensions": ("packages", developer.extension_packages),
        "/api/developer/sdk": ("artifacts", developer.sdk_artifacts),
        "/api/developer/api-keys": ("api_keys", developer.api_keys),
    }
    collection = collections.get(path)
    if collection is None:
        return None
    key, provider = collection
    return JsonRouteResult({key: provider()})


def handle_post(path: str, deps: DeveloperRouteDependencies) -> JsonRouteResult | None:
    if path not in POST_PATHS:
        return None

    payload = deps.read_json()
    developer = deps.developer
    parse_bool = deps.parse_bool
    try:
        if path in {"/api/graphql", "/api/developer/graphql"}:
            return JsonRouteResult(developer.execute_graphql(
                str(payload.get("query", "")).strip(),
                variables=payload.get("variables") if isinstance(payload.get("variables"), dict) else {},
            ))
        if path == "/api/developer/api-keys":
            value = developer.create_api_key(
                label=str(payload.get("label", "")).strip(),
                scopes=[str(item) for item in payload.get("scopes", [])]
                if isinstance(payload.get("scopes"), list) else None,
            )
            return JsonRouteResult({"api_key": value}, HTTPStatus.CREATED)
        if path == "/api/developer/api-keys/verify":
            return JsonRouteResult(developer.verify_api_key(
                str(payload.get("key", "")),
                scope=str(payload.get("scope", "")).strip() or None,
            ))
        if path == "/api/developer/webhooks":
            value = developer.register_webhook(
                event=str(payload.get("event", "*")).strip() or "*",
                target_url=str(payload.get("target_url", "")).strip(),
                enabled=parse_bool(payload.get("enabled"), default=True),
                secret_hint=str(payload.get("secret_hint", "")),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"webhook": value}, HTTPStatus.CREATED)
        if path == "/api/developer/webhooks/dispatch":
            value = developer.dispatch_webhook(
                event=str(payload.get("event", "")).strip(),
                payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                allow_network=parse_bool(payload.get("allow_network"), default=False),
            )
            return JsonRouteResult(value, HTTPStatus.CREATED)
        if path == "/api/developer/plugins":
            manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else payload
            return JsonRouteResult({"plugin": developer.register_plugin(manifest)}, HTTPStatus.CREATED)
        if path == "/api/developer/tools":
            value = developer.register_tool(
                tool_id=str(payload.get("tool_id", "")).strip(),
                name=str(payload.get("name", "")).strip(),
                description=str(payload.get("description", "")),
                schema=payload.get("schema") if isinstance(payload.get("schema"), dict) else {},
                handler=str(payload.get("handler", "registered")).strip() or "registered",
                permissions=[str(item) for item in payload.get("permissions", [])]
                if isinstance(payload.get("permissions"), list) else [],
                enabled=parse_bool(payload.get("enabled"), default=True),
            )
            return JsonRouteResult({"tool": value}, HTTPStatus.CREATED)
        if path == "/api/developer/tools/invoke":
            try:
                value = developer.invoke_tool(
                    str(payload.get("tool_id", "")).strip(),
                    input_data=payload.get("input") if isinstance(payload.get("input"), dict) else {},
                    approved=parse_bool(payload.get("approved"), default=False),
                )
            except (DeveloperPlatformError, ValueError) as exc:
                return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return JsonRouteResult({"invocation": value}, HTTPStatus.CREATED)
        if path == "/api/developer/mcp/servers":
            value = developer.register_mcp_server(
                name=str(payload.get("name", "")).strip(),
                command=str(payload.get("command", "")).strip(),
                args=[str(item) for item in payload.get("args", [])]
                if isinstance(payload.get("args"), list) else [],
                env={str(key): str(value) for key, value in payload.get("env", {}).items()}
                if isinstance(payload.get("env"), dict) else {},
                enabled=parse_bool(payload.get("enabled"), default=True),
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
            )
            return JsonRouteResult({"server": value}, HTTPStatus.CREATED)
        if path == "/api/developer/extensions":
            value = developer.publish_extension_package(
                package_type=str(payload.get("package_type", "")).strip(),
                name=str(payload.get("name", "")).strip(),
                version=str(payload.get("version", "")).strip(),
                source=str(payload.get("source", "local")).strip() or "local",
                status=str(payload.get("status", "available")).strip() or "available",
                manifest=payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {},
            )
            return JsonRouteResult({"package": value}, HTTPStatus.CREATED)
        if path == "/api/developer/sdk":
            value = developer.generate_sdk(language=str(payload.get("language", "python")).strip() or "python")
            return JsonRouteResult({"artifact": value}, HTTPStatus.CREATED)
    except ValueError as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    return None
