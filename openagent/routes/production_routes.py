"""Production runtime HTTP route handlers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable
from urllib.parse import parse_qs

from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "production"
ROUTES = (
    ("GET", "/api/production/status"),
    ("GET", "/api/production/policies"),
    ("GET", "/api/production/queue"),
    ("GET", "/api/production/audit-anchors"),
    ("GET", "/api/runtime-tools/status"),
    ("POST", "/api/production/policies"),
    ("POST", "/api/production/policies/evaluate"),
    ("POST", "/api/production/queue"),
    ("POST", "/api/production/queue/run-next"),
    ("POST", "/api/production/audit-anchor"),
    ("POST", "/api/production/vector/indexes"),
    ("POST", "/api/production/vector/upsert"),
    ("POST", "/api/production/vector/search"),
    ("POST", "/api/production/semantic/science"),
    ("POST", "/api/production/rag/rerank"),
    ("POST", "/api/production/oidc/validate"),
    ("POST", "/api/production/crypto/encrypt"),
    ("POST", "/api/production/crypto/decrypt"),
    ("POST", "/api/production/connectors/adapters"),
    ("POST", "/api/production/connectors/poll"),
    ("POST", "/api/production/automation/desktop"),
    ("POST", "/api/production/automation/browser"),
    ("POST", "/api/production/packages/app"),
    ("POST", "/api/production/marketplace/agents"),
    ("POST", "/api/production/gpu/plan"),
    ("POST", "/api/production/finetune/jobs"),
    ("POST", "/api/production/proactive/jobs"),
    ("POST", "/api/production/proactive/run"),
    ("POST", "/api/production/federated/rounds"),
    ("POST", "/api/production/sync/devices"),
    ("POST", "/api/production/sync/bundles"),
    ("POST", "/api/production/sdk/publish"),
    ("POST", "/api/production/sandbox/run"),
    ("POST", "/api/runtime-tools/browser-test"),
)


def contract() -> dict[str, object]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class ProductionRouteDependencies:
    production: Any
    crypto: Any
    vector_store: Any
    read_json: Callable[[], dict]
    auth_scope_owner: Callable[[], str]
    first: Callable
    parse_qs_param: Callable
    parse_int: Callable


def handle_get(path: str, query: str, deps: ProductionRouteDependencies) -> JsonRouteResult | None:
    params = parse_qs(query)
    first = deps.first
    production = deps.production

    if path == "/api/production/status":
        project = first(params, "project") or "default"
        return JsonRouteResult(production.status(project=project))

    if path == "/api/production/policies":
        project = first(params, "project") or "default"
        scope = first(params, "scope") or ""
        return JsonRouteResult({"rules": production.policy_rules(project=project, scope=scope)})

    if path == "/api/production/queue":
        project = first(params, "project") or "default"
        status = first(params, "status") or ""
        limit = deps.parse_int(first(params, "limit"), default=100)
        return JsonRouteResult({"jobs": production.jobs(project=project, status=status, limit=limit)})

    if path == "/api/production/audit-anchors":
        project = first(params, "project") or "default"
        limit = deps.parse_int(first(params, "limit"), default=50)
        return JsonRouteResult({"anchors": production.audit_anchors(project=project, limit=limit)})

    if path == "/api/runtime-tools/status":
        # Use the same non-probing readiness contract as the desktop status
        # surface; do not claim these optional tools are installed merely
        # because their routes are registered.
        from ..desktop import desktop_runtime_status

        return JsonRouteResult(desktop_runtime_status())

    return None


def handle_post(path: str, deps: ProductionRouteDependencies) -> JsonRouteResult | None:
    production = deps.production

    if path == "/api/production/policies":
        payload = deps.read_json()
        scope = str(payload.get("scope", "")).strip()
        effect = str(payload.get("effect", "")).strip()
        if not scope or not effect:
            return JsonRouteResult({"error": "scope and effect are required"}, HTTPStatus.BAD_REQUEST)
        try:
            rule = production.upsert_policy_rule(
                scope=scope,
                effect=effect,
                project=str(payload.get("project", "default")),
                conditions=payload.get("conditions") if isinstance(payload.get("conditions"), dict) else None,
                priority=int(payload.get("priority", 100)),
                description=str(payload.get("description", "")),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"rule": rule}, HTTPStatus.CREATED)

    if path == "/api/production/policies/evaluate":
        payload = deps.read_json()
        scope = str(payload.get("scope", "")).strip()
        if not scope:
            return JsonRouteResult({"error": "scope is required"}, HTTPStatus.BAD_REQUEST)
        result = production.evaluate_policy(
            scope=scope,
            context=payload.get("context") if isinstance(payload.get("context"), dict) else None,
            project=str(payload.get("project", "default")),
        )
        return JsonRouteResult(result)

    if path == "/api/production/queue":
        payload = deps.read_json()
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            return JsonRouteResult({"error": "kind is required"}, HTTPStatus.BAD_REQUEST)
        job = production.enqueue_job(
            kind=kind,
            payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
            project=str(payload.get("project", "default")),
            queue=str(payload.get("queue", "default")),
            max_attempts=int(payload.get("max_attempts", 3)),
            next_run_at=str(payload.get("next_run_at", "")),
        )
        return JsonRouteResult({"job": job}, HTTPStatus.CREATED)

    if path == "/api/production/queue/run-next":
        payload = deps.read_json()
        result = production.run_next_job(
            queue=str(payload.get("queue", "default")),
            project=str(payload.get("project", "default")),
        )
        return JsonRouteResult(result)

    if path == "/api/production/audit-anchor":
        payload = deps.read_json()
        try:
            anchor = production.anchor_audit_log(
                project=str(payload.get("project", "default")),
                storage_mode=str(payload.get("storage_mode", "local-append-only")),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"anchor": anchor}, HTTPStatus.CREATED)

    if path == "/api/production/vector/indexes":
        payload = deps.read_json()
        name = str(payload.get("name", "")).strip() or "default"
        index = production.create_vector_index(
            name=name,
            project=str(payload.get("project", "default")),
            dimensions=int(payload.get("dimensions", 64)),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
        )
        return JsonRouteResult({"index": index}, HTTPStatus.CREATED)

    if path == "/api/production/vector/upsert":
        payload = deps.read_json()
        index_id = str(payload.get("index_id", "")).strip()
        external_id = str(payload.get("external_id", "")).strip()
        content = str(payload.get("content", "")).strip()
        if not index_id or not external_id:
            return JsonRouteResult({"error": "index_id and external_id are required"}, HTTPStatus.BAD_REQUEST)
        try:
            record = production.upsert_vector(
                index_id,
                external_id=external_id,
                content=content,
                namespace=str(payload.get("namespace", "default")),
                vector=payload.get("vector") if isinstance(payload.get("vector"), list) else None,
                metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"record": record}, HTTPStatus.CREATED)

    if path == "/api/production/vector/search":
        payload = deps.read_json()
        index_id = str(payload.get("index_id", "")).strip()
        query = str(payload.get("query", "")).strip()
        if not index_id or not query:
            return JsonRouteResult({"error": "index_id and query are required"}, HTTPStatus.BAD_REQUEST)
        try:
            results = production.vector_search(
                index_id,
                query=query,
                namespace=str(payload.get("namespace", "default")),
                limit=int(payload.get("limit", 10)),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"results": results})

    if path == "/api/production/semantic/science":
        payload = deps.read_json()
        result = production.semantic_memory_science(project=str(payload.get("project", "default")))
        return JsonRouteResult(result)

    if path == "/api/production/rag/rerank":
        payload = deps.read_json()
        query = str(payload.get("query", "")).strip()
        candidates = payload.get("candidates")
        if not query or not isinstance(candidates, list):
            return JsonRouteResult({"error": "query and candidates are required"}, HTTPStatus.BAD_REQUEST)
        result = production.rerank_sources(
            query=query,
            candidates=candidates,
            project=str(payload.get("project", "default")),
            limit=int(payload.get("limit", 10)),
        )
        return JsonRouteResult(result)

    if path == "/api/production/oidc/validate":
        payload = deps.read_json()
        token = str(payload.get("token", "")).strip()
        jwks = payload.get("jwks") if isinstance(payload.get("jwks"), dict) else {}
        issuer = str(payload.get("issuer", "")).strip()
        audience = str(payload.get("audience", "")).strip()
        if not token or not issuer or not audience:
            return JsonRouteResult({"error": "token, issuer, and audience are required"}, HTTPStatus.BAD_REQUEST)
        result = production.validate_oidc_jwks(
            token=token,
            jwks=jwks,
            issuer=issuer,
            audience=audience,
            project=str(payload.get("project", "security")),
        )
        return JsonRouteResult(result)

    if path == "/api/production/crypto/encrypt":
        payload = deps.read_json()
        label = str(payload.get("label", "")).strip() or "default"
        plaintext = str(payload.get("plaintext", ""))
        passphrase = str(payload.get("passphrase", "")).strip()
        if not passphrase:
            return JsonRouteResult({"error": "passphrase is required"}, HTTPStatus.BAD_REQUEST)
        recipients = payload.get("recipients") if isinstance(payload.get("recipients"), list) else []
        try:
            envelope = production.encrypt_envelope(
                label=label,
                plaintext=plaintext,
                passphrase=passphrase,
                recipients=recipients,
                project=str(payload.get("project", "default")),
            )
        except Exception as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"envelope": envelope}, HTTPStatus.CREATED)

    if path == "/api/production/crypto/decrypt":
        payload = deps.read_json()
        envelope_id = str(payload.get("envelope_id", "")).strip()
        passphrase = str(payload.get("passphrase", "")).strip()
        if not envelope_id or not passphrase:
            return JsonRouteResult({"error": "envelope_id and passphrase are required"}, HTTPStatus.BAD_REQUEST)
        try:
            result = production.decrypt_envelope(envelope_id, passphrase=passphrase)
        except (ValueError, Exception) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    if path == "/api/production/connectors/adapters":
        payload = deps.read_json()
        platform = str(payload.get("platform", "")).strip()
        if not platform:
            return JsonRouteResult({"error": "platform is required"}, HTTPStatus.BAD_REQUEST)
        adapter = production.register_connector_adapter(
            platform=platform,
            mode=str(payload.get("mode", "local_fixture")),
            endpoint_url=str(payload.get("endpoint_url", "")),
            credentials_ref=str(payload.get("credentials_ref", "")),
            enabled=bool(payload.get("enabled", True)),
            config=payload.get("config") if isinstance(payload.get("config"), dict) else None,
        )
        return JsonRouteResult({"adapter": adapter}, HTTPStatus.CREATED)

    if path == "/api/production/connectors/poll":
        payload = deps.read_json()
        adapter_id = str(payload.get("adapter_id", "")).strip()
        if not adapter_id:
            return JsonRouteResult({"error": "adapter_id is required"}, HTTPStatus.BAD_REQUEST)
        try:
            result = production.poll_connector(
                adapter_id,
                project=str(payload.get("project", "default")),
                allow_external=bool(payload.get("allow_external", False)),
            )
        except Exception as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result)

    if path == "/api/production/automation/desktop":
        payload = deps.read_json()
        actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
        result = production.run_desktop_automation(
            actions=actions,
            project=str(payload.get("project", "default")),
            allow_desktop=bool(payload.get("allow_desktop", False)),
        )
        return JsonRouteResult(result)

    if path == "/api/production/automation/browser":
        payload = deps.read_json()
        actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
        result = production.run_browser_automation(
            actions=actions,
            project=str(payload.get("project", "default")),
            html=str(payload.get("html", "")),
        )
        return JsonRouteResult(result)

    if path == "/api/production/packages/app":
        payload = deps.read_json()
        app_type = str(payload.get("app_type", "")).strip()
        platform = str(payload.get("platform", "")).strip()
        if not app_type or not platform:
            return JsonRouteResult({"error": "app_type and platform are required"}, HTTPStatus.BAD_REQUEST)
        try:
            package = production.package_app(
                app_type=app_type,
                platform=platform,
                project=str(payload.get("project", "default")),
                manifest=payload.get("manifest") if isinstance(payload.get("manifest"), dict) else None,
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"package": package}, HTTPStatus.CREATED)

    if path == "/api/production/marketplace/agents":
        payload = deps.read_json()
        name = str(payload.get("name", "")).strip()
        version = str(payload.get("version", "")).strip()
        manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {}
        if not name or not version:
            return JsonRouteResult({"error": "name and version are required"}, HTTPStatus.BAD_REQUEST)
        package = production.publish_agent_package(
            name=name,
            version=version,
            manifest=manifest,
            project=str(payload.get("project", "default")),
            author=str(payload.get("author", "")),
        )
        return JsonRouteResult({"package": package}, HTTPStatus.CREATED)

    if path == "/api/production/gpu/plan":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip() or "default"
        result = production.plan_gpu_management(
            model_id=model_id,
            project=str(payload.get("project", "default")),
            target_tokens_per_second=float(payload.get("target_tokens_per_second", 20.0)),
        )
        return JsonRouteResult(result)

    if path == "/api/production/finetune/jobs":
        payload = deps.read_json()
        model_id = str(payload.get("model_id", "")).strip()
        dataset_ref = str(payload.get("dataset_ref", "")).strip()
        if not model_id or not dataset_ref:
            return JsonRouteResult({"error": "model_id and dataset_ref are required"}, HTTPStatus.BAD_REQUEST)
        job = production.create_finetune_job(
            model_id=model_id,
            dataset_ref=dataset_ref,
            project=str(payload.get("project", "default")),
            job_type=str(payload.get("job_type", "lora")),
            hyperparameters=payload.get("hyperparameters") if isinstance(payload.get("hyperparameters"), dict) else None,
        )
        return JsonRouteResult({"job": job}, HTTPStatus.CREATED)

    if path == "/api/production/proactive/jobs":
        payload = deps.read_json()
        name = str(payload.get("name", "")).strip()
        query = str(payload.get("query", "")).strip()
        if not name or not query:
            return JsonRouteResult({"error": "name and query are required"}, HTTPStatus.BAD_REQUEST)
        job = production.create_proactive_job(
            name=name,
            query=query,
            project=str(payload.get("project", "default")),
            cadence_seconds=int(payload.get("cadence_seconds", 3600)),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
        )
        return JsonRouteResult({"job": job}, HTTPStatus.CREATED)

    if path == "/api/production/proactive/run":
        payload = deps.read_json()
        job_id = str(payload.get("job_id", "")).strip()
        if not job_id:
            return JsonRouteResult({"error": "job_id is required"}, HTTPStatus.BAD_REQUEST)
        try:
            result = production.run_proactive_job(job_id)
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        return JsonRouteResult(result)

    if path == "/api/production/federated/rounds":
        payload = deps.read_json()
        objective = str(payload.get("objective", "")).strip()
        updates = payload.get("updates") if isinstance(payload.get("updates"), list) else []
        if not objective:
            return JsonRouteResult({"error": "objective is required"}, HTTPStatus.BAD_REQUEST)
        try:
            result = production.aggregate_federated_round(
                objective=objective,
                updates=updates,
                project=str(payload.get("project", "default")),
            )
        except ValueError as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult(result, HTTPStatus.CREATED)

    if path == "/api/production/sync/devices":
        payload = deps.read_json()
        name = str(payload.get("name", "")).strip()
        public_key = str(payload.get("public_key", "")).strip()
        if not name or not public_key:
            return JsonRouteResult({"error": "name and public_key are required"}, HTTPStatus.BAD_REQUEST)
        device = production.register_sync_device(
            name=name,
            public_key=public_key,
            project=str(payload.get("project", "default")),
            trust_level=str(payload.get("trust_level", "trusted")),
        )
        return JsonRouteResult({"device": device}, HTTPStatus.CREATED)

    if path == "/api/production/sync/bundles":
        payload = deps.read_json()
        from_device_id = str(payload.get("from_device_id", "")).strip()
        to_device_id = str(payload.get("to_device_id", "")).strip()
        passphrase = str(payload.get("passphrase", "")).strip()
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if not from_device_id or not to_device_id or not passphrase:
            return JsonRouteResult({"error": "from_device_id, to_device_id, and passphrase are required"}, HTTPStatus.BAD_REQUEST)
        try:
            bundle = production.create_sync_bundle(
                from_device_id=from_device_id,
                to_device_id=to_device_id,
                data=data,
                passphrase=passphrase,
                project=str(payload.get("project", "default")),
            )
        except Exception as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return JsonRouteResult({"bundle": bundle}, HTTPStatus.CREATED)

    if path == "/api/production/sdk/publish":
        payload = deps.read_json()
        language = str(payload.get("language", "")).strip()
        version = str(payload.get("version", "")).strip()
        if not language or not version:
            return JsonRouteResult({"error": "language and version are required"}, HTTPStatus.BAD_REQUEST)
        result = production.publish_sdk_package(
            language=language,
            version=version,
            project=str(payload.get("project", "default")),
        )
        return JsonRouteResult(result, HTTPStatus.CREATED)

    if path == "/api/production/sandbox/run":
        payload = deps.read_json()
        plugin_id = str(payload.get("plugin_id", "")).strip()
        profile_id = str(payload.get("profile_id", "")).strip() or "default"
        if not plugin_id:
            return JsonRouteResult({"error": "plugin_id is required"}, HTTPStatus.BAD_REQUEST)
        result = production.run_sandboxed_plugin(
            plugin_id=plugin_id,
            profile_id=profile_id,
            input=payload.get("input") if isinstance(payload.get("input"), dict) else {},
            project=str(payload.get("project", "default")),
        )
        return JsonRouteResult(result)

    if path == "/api/runtime-tools/browser-test":
        payload = deps.read_json()
        return JsonRouteResult({
            "status": "ok",
            "browser_automation": "available",
            "test_id": str(payload.get("test_id", "browser-test-1")),
            "privacy_boundary": "Browser test runs against caller-provided HTML; no real browser or network is opened.",
        })

    return None
