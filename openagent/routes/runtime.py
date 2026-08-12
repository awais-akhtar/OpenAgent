"""Local HTTP API and static workspace server."""

from __future__ import annotations

import base64
import binascii
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import struct
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4
import zlib

from .. import __version__
from ..advanced_rag import AdvancedRAGEngine
from ..adaptive_engine.runtime import AdaptiveExecutionEngine
from ..agent_bridge import AgentBridge
from ..agents import AgentRegistry
from ..assistant import AssistantManager
from ..auth import AuthError, AuthManager, required_scope_for_path
from ..backup_restore import BackupRestoreManager
from ..calendar_sync import CalendarSyncError, CalendarSyncManager
from ..calendar_system import CalendarSystem
from ..capabilities import vision_coverage
from ..chat import ChatService
from ..chat_artifacts import ChatArtifactService
from ..codebase import CodebaseService
from ..communication import CommunicationApprovalRequired, CommunicationHub
from ..conversation_organization import ConversationOrganization
from ..config import auth_enforcement_enabled, web_root as default_web_root
from ..contacts import ContactsManager
from ..context import ContextEngine
from ..developer import DeveloperPlatform, DeveloperPlatformError
from ..document_processing import DocumentProcessing
from ..digital_twin import DigitalTwin
from ..documents import DocumentError, DocumentManager
from ..enterprise import EnterpriseManager
from ..enterprise_identity import EnterpriseIdentity
from ..fine_tuning import FineTuningManager
from ..hard_audit import run_hard_audit
from ..image_studio import ImageStudio
from ..integration_hub import IntegrationHub, IntegrationHubError
from ..knowledge import KnowledgeEngine
from ..local_ai import LocalAIInfrastructure, ModelDownloadError
from ..media import MediaManager
from ..mcp_runtime import MCPRuntime
from ..mission_control import MissionControl
from ..model_cookbook import ModelCookbook
from ..model_gateway import ModelGateway, ProviderConfigurationError, runtime_settings_snapshot
from ..models import AuditEvent, utc_now
from ..multi_agent import ApprovalRequired, MultiAgentRuntime, PermissionDenied
from ..personal_os import PersonalOS, PersonalOSError
from ..phd_workspace import PhDWorkspace
from ..production_runtime import ProductionRuntime, ProductionRuntimeError
from ..research import ResearchEngine
from ..research_lifecycle import ResearchLifecycle
from ..science.agent import ScientificAgent
from ..science.service import ScientificService
from ..safe_terminal import SafeTerminal
from ..scheduler import Scheduler
from ..skill_ecosystem import SkillEcosystem
from .agent_bridge_routes import AgentBridgeRouteDependencies, handle_get as handle_agent_bridge_get, handle_post as handle_agent_bridge_post
from .advanced_rag_routes import AdvancedRAGRouteDependencies, handle_get as handle_advanced_rag_get, handle_post as handle_advanced_rag_post
from .adaptive_engine_routes import AdaptiveEngineRouteDependencies, handle_get as handle_adaptive_engine_get, handle_post as handle_adaptive_engine_post
from .backup_routes import BackupRouteDependencies, handle_get as handle_backup_get, handle_post as handle_backup_post
from .calendar_routes import CalendarRouteDependencies, handle_get as handle_calendar_get, handle_post as handle_calendar_post
from .codebase_routes import CodebaseRouteDependencies, handle_get as handle_codebase_get, handle_post as handle_codebase_post
from .communication_routes import CommunicationRouteDependencies, handle_get as handle_communication_get, handle_post as handle_communication_post
from .conversation_routes import ConversationRouteDependencies, handle_delete as handle_conversation_delete, handle_get as handle_conversation_get, handle_patch as handle_conversation_patch, handle_post as handle_conversation_post
from .contacts_routes import ContactsRouteDependencies, handle_delete as handle_contacts_delete, handle_get as handle_contacts_get, handle_patch as handle_contacts_patch, handle_post as handle_contacts_post
from .document_processing_routes import DocumentProcessingRouteDependencies, handle_post as handle_document_processing_post
from .handlers import handler_specs
from .health_routes import HealthRouteDependencies, handle_get as handle_health_get
from .image_studio_routes import ImageStudioRouteDependencies, handle_get as handle_image_studio_get, handle_post as handle_image_studio_post
from .identity_routes import IdentityRouteDependencies, handle_delete as handle_identity_delete, handle_get as handle_identity_get, handle_post as handle_identity_post
from .inference_routes import InferenceRouteDependencies, handle_get as handle_inference_get, handle_post as handle_inference_post
from .mcp_runtime_routes import MCPRuntimeRouteDependencies, handle_get as handle_mcp_runtime_get, handle_patch as handle_mcp_runtime_patch, handle_post as handle_mcp_runtime_post
from .model_cookbook_routes import ModelCookbookRouteDependencies, handle_get as handle_model_cookbook_get, handle_post as handle_model_cookbook_post
from .phd_routes import PhDRouteDependencies, handle_delete as handle_phd_delete, handle_get as handle_phd_get, handle_patch as handle_phd_patch, handle_post as handle_phd_post
from .scheduler_routes import SchedulerRouteDependencies, handle_get as handle_scheduler_get, handle_patch as handle_scheduler_patch, handle_post as handle_scheduler_post
from .science_routes import ScienceRouteDependencies, handle_get as handle_science_get, handle_post as handle_science_post
from .skill_routes import SkillRouteDependencies, handle_get as handle_skill_get, handle_post as handle_skill_post
from .terminal_routes import TerminalRouteDependencies, handle_get as handle_terminal_get, handle_post as handle_terminal_post
from .ultra_model_routes import UltraModelRouteDependencies, handle_get as handle_ultra_model_get, handle_post as handle_ultra_model_post
from .vector_routes import VectorRouteDependencies, handle_get as handle_vector_get, handle_post as handle_vector_post
from .worker_routes import WorkerRouteDependencies, handle_get as handle_worker_get, handle_post as handle_worker_post
from ..scraper import WebScraper
from ..security_privacy import SecurityPrivacyError, SecurityPrivacyManager
from ..semantic_memory import SemanticMemoryManager
from ..storage import OpenAgentStore
from ..system_metrics import system_metrics
from ..ultra_model_engine import UltraModelEngine
from ..vault import EncryptedVault
from ..voice import VoiceRuntime
from ..vector_store import VectorStore
from ..worker import WorkerRuntime
from ..workspace import WorkspaceManager
from ..workflows import WorkflowEngine, WorkflowSafetyError


mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("text/html; charset=utf-8", ".html")


def _parse_multipart_parts(body: bytes, boundary: bytes) -> list[dict[str, object]]:
    """Return multipart parts with headers, form name, filename, and raw content."""

    import re as _re

    parts: list[dict[str, object]] = []
    delimiter = b"--" + boundary
    for raw_part in body.split(delimiter):
        raw_part = raw_part.strip(b"\r\n")
        if not raw_part or raw_part == b"--":
            continue
        if raw_part.endswith(b"--"):
            raw_part = raw_part[:-2].rstrip(b"\r\n")
        header_end = raw_part.find(b"\r\n\r\n")
        offset = 4
        if header_end == -1:
            header_end = raw_part.find(b"\n\n")
            offset = 2
        if header_end == -1:
            continue
        headers_raw = raw_part[:header_end].decode("utf-8", errors="replace")
        content = raw_part[header_end + offset:]
        headers: dict[str, str] = {}
        for line in headers_raw.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        disposition = headers.get("content-disposition", "")
        name_match = _re.search(r'name="([^"]*)"', disposition)
        filename_match = _re.search(r'filename="([^"]*)"', disposition)
        parts.append(
            {
                "headers": headers,
                "name": name_match.group(1) if name_match else "",
                "filename": filename_match.group(1) if filename_match else "",
                "content_type": headers.get("content-type", ""),
                "content": content.rstrip(b"\r\n"),
            }
        )
    return parts


def _parse_multipart(body: bytes, boundary: bytes) -> tuple[str, str]:
    """Extract filename and text content from a multipart/form-data body.

    Returns (filename, extracted_text). Minimal stdlib-only implementation.
    """
    for part in _parse_multipart_parts(body, boundary):
        content = part["content"]
        if not isinstance(content, bytes) or not content:
            continue
        filename = str(part.get("filename", ""))
        if not content:
            continue
        # Best-effort PDF text extraction (BT/ET blocks)
        if filename.lower().endswith(".pdf"):
            text = _extract_pdf_text(content)
            if text.strip():
                return filename, text
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        if text.strip():
            return filename, text
    return "", ""


def _extract_pdf_text(raw: bytes) -> str:
    """Extract readable text from a PDF using stdlib only (BT/ET scan)."""
    import re as _re
    pieces: list[str] = []
    for block in _re.findall(rb"BT(.*?)ET", raw, _re.DOTALL):
        for s in _re.findall(rb"\((.*?)\)", block):
            try:
                piece = s.decode("latin-1", errors="replace").strip()
                if piece:
                    pieces.append(piece)
            except Exception:
                pass
    return " ".join(pieces)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    data = kind + payload
    return struct.pack(">I", len(payload)) + data + struct.pack(">I", binascii.crc32(data) & 0xFFFFFFFF)


@lru_cache(maxsize=4)
def _generated_icon_png(size: int) -> bytes:
    """Generate a small local PNG icon so the PWA manifest never points at missing assets."""

    size = max(16, min(int(size), 512))
    rows: list[bytes] = []
    center = size / 2
    radius = size * 0.34
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            dx = x - center
            dy = y - center
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= radius:
                row.extend((248, 250, 252))
            else:
                mix = int(26 + 42 * (x + y) / max(1, size * 2))
                row.extend((8, mix, 38 + mix // 2))
        rows.append(bytes(row))
    raw = b"".join(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )


def build_http_server(
    host: str,
    port: int,
    store: OpenAgentStore,
    web_root: Path | None = None,
    *,
    warm_openvino: bool = False,
) -> ThreadingHTTPServer:
    context = ContextEngine(store)
    adaptive_engine = AdaptiveExecutionEngine(store)
    gateway = ModelGateway(direct_hardware=adaptive_engine.direct)
    adaptive_engine.bind_gateway(gateway)
    agents = AgentRegistry(store, gateway)
    chat_artifacts = ChatArtifactService(store)
    openvino_warm_event = threading.Event()
    openvino_warm_thread: threading.Thread | None = None
    if warm_openvino:
        def warm_managed_openvino() -> None:
            audit_event: AuditEvent
            try:
                result = gateway.warmup(provider_id="openvino-direct")
                audit_event = AuditEvent(
                    actor="model-gateway", action="openvino.warmup", resource="model:openvino-direct",
                    project="default", metadata=result,
                )
            except Exception as exc:
                audit_event = AuditEvent(
                    actor="model-gateway", action="openvino.warmup", resource="model:openvino-direct",
                    project="default", status="failed", metadata={"error": str(exc)},
                )
            finally:
                try:
                    store.add_audit_event(audit_event)
                except Exception:
                    # A short-lived test or desktop shutdown may remove the data
                    # directory before an optional background warmup finishes.
                    pass
                openvino_warm_event.set()

        openvino_warm_thread = threading.Thread(target=warm_managed_openvino, name="openagent-openvino-warmup", daemon=True)
        openvino_warm_thread.start()
    else:
        openvino_warm_event.set()
    vector_store = VectorStore(store)
    chat = ChatService(store, gateway)
    codebase = CodebaseService(store, gateway)
    workspace = WorkspaceManager(store)
    assistant = AssistantManager(store)
    workflows = WorkflowEngine(store, gateway)
    knowledge = KnowledgeEngine(store, vector_store=vector_store)
    advanced_rag = AdvancedRAGEngine(
        store,
        knowledge=knowledge,
        vector_store=vector_store,
        gateway=gateway,
    )
    semantic_memory = SemanticMemoryManager(store)
    media = MediaManager(store)
    voice = VoiceRuntime(store)
    production = ProductionRuntime(store, gateway)
    multi_agent = MultiAgentRuntime(store, gateway)
    communication = CommunicationHub(store)
    contacts_manager = ContactsManager(store)
    local_ai = LocalAIInfrastructure(store, gateway)
    ultra_model = UltraModelEngine(store)
    enterprise = EnterpriseManager(store)
    auth = AuthManager(store)
    enterprise_identity = EnterpriseIdentity(store, auth)
    documents = DocumentManager(store)

    def search_context_resolver(query: str, *, project: str = "default", limit: int = 10) -> list[dict[str, object]]:
        return context.search(query, project=project, limit=limit)

    developer = DeveloperPlatform(
        store,
        resolvers={
            "health": lambda args: store.health(),
            "context": lambda args: context.snapshot(project=str(args.get("project", "")).strip() or None),
            "agents": lambda args: {"agents": agents.specs()},
            "models": lambda args: {"providers": gateway.specs()},
            "workflows": lambda args: {
                "workflows": store.list_workflows(
                    project=str(args.get("project", "")).strip() or None,
                    limit=parse_int(str(args.get("limit", "100")), default=100),
                )
            },
            "workspace": lambda args: workspace.dashboard(
                project=str(args.get("project", "default")).strip() or "default"
            ),
            "assistant": lambda args: assistant.daily_brief(
                project=str(args.get("project", "default")).strip() or "default"
            ),
            "semantic": lambda args: semantic_memory.status(
                project=str(args.get("project", "default")).strip() or "default"
            ),
            "media": lambda args: media.status(
                project=str(args.get("project", "default")).strip() or "default"
            ),
            "voice": lambda args: voice.status(
                project=str(args.get("project", "default")).strip() or "default"
            ),
            "production": lambda args: production.status(
                project=str(args.get("project", "default")).strip() or "default"
            ),
            "search_context": search_context_resolver,
        },
    )
    security_privacy = SecurityPrivacyManager(store)
    research_engine = ResearchEngine(store, gateway)
    research_lifecycle = ResearchLifecycle(store)
    calendar_system = CalendarSystem(store)
    calendar_sync = CalendarSyncManager(store, calendar_system)
    digital_twin = DigitalTwin(store)
    vault = EncryptedVault(store)
    backup_restore = BackupRestoreManager(store)
    scraper = WebScraper(store)
    phd_workspace = PhDWorkspace(store, gateway=gateway, scraper=scraper)
    scientific_agent = ScientificAgent(context, gateway=gateway)
    scientific_service = ScientificService(store)
    fine_tuning = FineTuningManager(store)
    mission_control = MissionControl(store, gateway)
    personal_os = PersonalOS(store)
    worker_runtime = WorkerRuntime(store, worker_id="openagent-http-worker", gateway=gateway)
    model_cookbook = ModelCookbook(store, gateway)
    document_processing = DocumentProcessing(store)
    image_studio = ImageStudio(store)
    scheduler = Scheduler(store)
    skills = SkillEcosystem(store)
    conversation_organization = ConversationOrganization(store)
    safe_terminal = SafeTerminal(store, workspace_root=Path.cwd())
    mcp_runtime = MCPRuntime(store)
    agent_bridge = AgentBridge(store)
    root = (web_root or default_web_root()).resolve()

    def ui_build_info() -> dict[str, object]:
        assets: dict[str, object] = {}
        for name in ("index.html", "app.js", "styles.css", "sw.js", "manifest.json"):
            path = root / name
            if path.is_file():
                stat = path.stat()
                assets[name] = {
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            else:
                assets[name] = {"path": str(path), "missing": True}
        return {
            "project": "OpenAgent",
            "version": __version__,
            "web_root": str(root),
            "assets": assets,
            "cache_policy": "Static shell assets are served with no-store headers in local workbench mode.",
            "privacy_boundary": "UI build status reads local web asset metadata only.",
        }

    def module_health(project: str = "default") -> dict[str, object]:
        modules: list[dict[str, object]] = []

        def add_module(
            module_id: str,
            label: str,
            fn: object,
            *,
            degraded: bool = False,
            reason: str = "",
        ) -> None:
            try:
                evidence = fn()  # type: ignore[operator]
                status = "degraded" if degraded else "ready"
                modules.append(
                    {
                        "id": module_id,
                        "label": label,
                        "status": status,
                        "reason": reason,
                        "evidence": evidence,
                    }
                )
            except Exception as exc:
                modules.append(
                    {
                        "id": module_id,
                        "label": label,
                        "status": "error",
                        "reason": str(exc),
                        "evidence": {},
                    }
                )

        model_tools = local_ai.model_tool_status()
        provider_specs = gateway.specs()
        chat_ready = any(
            bool(provider.get("available"))
            for provider in provider_specs
            if provider.get("id") in {"openvino-direct", "ollama", "llamacpp"}
        )
        add_module(
            "chat",
            "Chat and model gateway",
            lambda: {
                "providers": provider_specs,
                "tool_status": model_tools,
                "recent_jobs": local_ai.model_jobs(project=project, limit=3),
            },
            degraded=not chat_ready,
            reason="" if chat_ready else "Offline fallback is available, but no real local chat model is ready.",
        )
        add_module("dashboard", "Dashboard telemetry", lambda: {"health": store.health(), "metrics": system_metrics(store.data_dir)})
        add_module("voice", "Voice runtime", lambda: voice.status(project=project))
        add_module("research", "Research engine", lambda: research_engine.status(project=project))
        add_module("phd", "PhD application workspace", lambda: phd_workspace.status(project=project))
        add_module("science", "Scientific AI", lambda: {"status": "ready", "privacy_boundary": "Local scientific computation only."})
        add_module("codebase", "Autonomous software engineering", lambda: {"projects": len(codebase.list_projects(project=project)), "runs": len(codebase.list_runs(project=project, limit=20)), "sandbox_root": str(codebase.sandboxes.root)})
        add_module("agents", "Multi-agent system", lambda: {"agents": agents.specs(), "marketplace": multi_agent.marketplace()})
        add_module("knowledge", "Knowledge and RAG", lambda: knowledge.status(project=project))
        add_module("memory", "Semantic memory", lambda: semantic_memory.status(project=project))
        add_module("workspace", "Workspace", lambda: workspace.dashboard(project=project))
        add_module("automation", "Workflow automation", lambda: {"workflows": store.list_workflows(project=project, limit=5)})
        add_module("assistant", "Personal assistant", lambda: assistant.daily_brief(project=project))
        add_module("communications", "Communication hub", lambda: {"connectors": communication.connectors(), "inbox": communication.inbox(project=project, limit=5)})
        add_module("models", "Local AI infrastructure", lambda: local_ai.status())
        add_module("ultra_models", "Ultra-scale CPU model engine", lambda: ultra_model.status(project=project))
        add_module("adaptive_engine", "Adaptive CPU, GPU, and NPU compiler", lambda: adaptive_engine.readiness(project=project))
        add_module("security", "Security and privacy", lambda: security_privacy.status())
        add_module("enterprise_identity", "Passkeys and device trust", lambda: enterprise_identity.status())
        add_module("developer", "Developer platform", lambda: developer.status())
        add_module("calendar", "Calendar", lambda: calendar_system.status(project=project))
        add_module("twin", "Digital twin", lambda: digital_twin.status(project=project))
        add_module("vault", "Encrypted vault", lambda: vault.status(project=project))
        add_module("scraper", "Web scraper", lambda: scraper.status(project=project))
        add_module("finetune", "Fine-tuning", lambda: fine_tuning.status(project=project))
        add_module("production", "Production runtime", lambda: production.status(project=project))
        add_module("image_studio", "Image Studio", lambda: image_studio.library(project=project, limit=3))
        add_module("scheduler", "Scheduler and notifications", lambda: {"tasks": scheduler.list_tasks(project=project, limit=3), "notifications": scheduler.notifications(project=project, limit=3)})
        add_module("conversation_organization", "Conversation organization", lambda: {"conversations": conversation_organization.list(project=project, limit=3)})
        add_module("mcp_runtime", "MCP runtime", lambda: {"servers": mcp_runtime.list_servers(project=project, limit=3), "tools": mcp_runtime.tools(project=project)})
        add_module("agent_bridge", "Agent bridge", lambda: agent_bridge.capabilities())
        counts = {
            "ready": sum(1 for item in modules if item["status"] == "ready"),
            "degraded": sum(1 for item in modules if item["status"] == "degraded"),
            "error": sum(1 for item in modules if item["status"] == "error"),
        }
        return {
            "project": project,
            "status": "ok" if counts["error"] == 0 else "needs_attention",
            "counts": counts,
            "modules": modules,
            "privacy_boundary": "Module health calls local status/read-only APIs only. It does not download models, sync connectors, or call external providers.",
        }

    integrations = IntegrationHub(
        store,
        health_provider=store.health,
        module_health_provider=module_health,
        metrics_provider=lambda: system_metrics(store.data_dir),
        model_provider=local_ai.status,
        voice_provider=lambda project: voice.status(project=project),
        developer_provider=developer.status,
        tool_provider=developer.tools,
        backup_provider=enterprise.create_backup,
        backups_provider=enterprise.backups,
    )
    health_routes = HealthRouteDependencies(
        health=store.health,
        system_metrics=lambda: system_metrics(store.data_dir),
        ui_build_info=ui_build_info,
        module_health=module_health,
        handler_specs=handler_specs,
    )
    def worker_route_dependencies(read_json) -> WorkerRouteDependencies:
        return WorkerRouteDependencies(
            status=lambda project: store.worker_status(project=project),
            list_jobs=lambda project, state, limit: store.list_jobs(project=project, state=state, limit=limit),
            get_job=store.get_job,
            dead_letters=lambda project, limit: store.list_dead_letter_jobs(project=project, limit=limit),
            enqueue=lambda job_type, payload, project, priority, max_attempts: worker_runtime.enqueue(
                job_type,
                payload,
                project=project,
                priority=priority,
                max_attempts=max_attempts,
            ),
            run_once=worker_runtime.run_once,
            cancel=lambda job_id, reason: store.cancel_job(job_id, reason=reason),
            retry=store.retry_job,
            read_json=read_json,
        )

    def vector_route_dependencies(read_json) -> VectorRouteDependencies:
        return VectorRouteDependencies(vector_store=vector_store, read_json=read_json)

    def advanced_rag_route_dependencies(read_json) -> AdvancedRAGRouteDependencies:
        return AdvancedRAGRouteDependencies(rag=advanced_rag, read_json=read_json)

    def inference_route_dependencies(read_json) -> InferenceRouteDependencies:
        return InferenceRouteDependencies(gateway=gateway, store=store, read_json=read_json)

    def ultra_model_route_dependencies(read_json) -> UltraModelRouteDependencies:
        return UltraModelRouteDependencies(engine=ultra_model, read_json=read_json)

    def adaptive_engine_route_dependencies(read_json) -> AdaptiveEngineRouteDependencies:
        return AdaptiveEngineRouteDependencies(engine=adaptive_engine, read_json=read_json)

    def contacts_route_dependencies(read_json, query_owner, payload_owner) -> ContactsRouteDependencies:
        return ContactsRouteDependencies(
            contacts=contacts_manager,
            query_owner=query_owner,
            payload_owner=payload_owner,
            read_json=read_json,
        )

    def backup_route_dependencies(read_json) -> BackupRouteDependencies:
        return BackupRouteDependencies(backups=backup_restore, read_json=read_json)

    def model_cookbook_route_dependencies(read_json) -> ModelCookbookRouteDependencies:
        return ModelCookbookRouteDependencies(cookbook=model_cookbook, read_json=read_json)

    def phd_route_dependencies(read_json, query_owner, payload_owner) -> PhDRouteDependencies:
        return PhDRouteDependencies(
            workspace=phd_workspace,
            first=first,
            parse_bool=parse_bool,
            parse_int=parse_int,
            query_owner=query_owner,
            payload_owner=payload_owner,
            read_json=read_json,
        )

    def document_processing_route_dependencies(read_json, payload_owner) -> DocumentProcessingRouteDependencies:
        return DocumentProcessingRouteDependencies(processing=document_processing, payload_owner=payload_owner, read_json=read_json)

    def image_studio_route_dependencies(read_json, query_owner, payload_owner) -> ImageStudioRouteDependencies:
        return ImageStudioRouteDependencies(
            studio=image_studio,
            first=first,
            parse_int=parse_int,
            query_owner=query_owner,
            payload_owner=payload_owner,
            read_json=read_json,
        )

    def scheduler_route_dependencies(read_json, query_owner, payload_owner) -> SchedulerRouteDependencies:
        return SchedulerRouteDependencies(
            scheduler=scheduler,
            first=first,
            parse_bool=parse_bool,
            parse_int=parse_int,
            query_owner=query_owner,
            payload_owner=payload_owner,
            read_json=read_json,
        )

    def science_route_dependencies(read_json) -> ScienceRouteDependencies:
        return ScienceRouteDependencies(
            agent=scientific_agent,
            run_agent=lambda goal, project="default": agents.run("scientific", goal, project=project),
            read_json=read_json,
            service=scientific_service,
            gateway=gateway,
        )

    def codebase_route_dependencies(read_json, query_owner, payload_owner) -> CodebaseRouteDependencies:
        return CodebaseRouteDependencies(
            codebase=codebase,
            first=first,
            parse_bool=parse_bool,
            parse_int=parse_int,
            query_owner=query_owner,
            payload_owner=payload_owner,
            read_json=read_json,
        )

    def conversation_route_dependencies(read_json, query_owner, payload_owner) -> ConversationRouteDependencies:
        return ConversationRouteDependencies(
            organization=conversation_organization,
            first=first,
            parse_bool=parse_bool,
            parse_int=parse_int,
            query_owner=query_owner,
            payload_owner=payload_owner,
            read_json=read_json,
        )

    def identity_route_dependencies(handler) -> IdentityRouteDependencies:
        return IdentityRouteDependencies(
            identity=enterprise_identity,
            auth=auth,
            read_json=handler.read_json,
            resolve_user=lambda: auth.resolve_bearer(handler._bearer_token()),
            request_origin=lambda: handler.headers.get("Origin", "").strip()
                or f"http://{handler.headers.get('Host', f'{host}:{port}')}",
            request_ip=lambda: str(handler.client_address[0]) if handler.client_address else "",
            user_agent=lambda: handler.headers.get("User-Agent", ""),
        )

    def terminal_route_dependencies(read_json, query_owner, payload_owner) -> TerminalRouteDependencies:
        return TerminalRouteDependencies(
            terminal=safe_terminal,
            first=first,
            parse_bool=parse_bool,
            parse_int=parse_int,
            query_owner=query_owner,
            payload_owner=payload_owner,
            read_json=read_json,
        )

    def mcp_runtime_route_dependencies(read_json, query_owner, payload_owner) -> MCPRuntimeRouteDependencies:
        return MCPRuntimeRouteDependencies(
            runtime=mcp_runtime,
            first=first,
            parse_bool=parse_bool,
            parse_int=parse_int,
            query_owner=query_owner,
            payload_owner=payload_owner,
            read_json=read_json,
        )

    def agent_bridge_route_dependencies(read_json, query_owner, payload_owner) -> AgentBridgeRouteDependencies:
        return AgentBridgeRouteDependencies(
            bridge=agent_bridge,
            first=first,
            parse_int=parse_int,
            query_owner=query_owner,
            payload_owner=payload_owner,
            read_json=read_json,
        )

    class OpenAgentHandler(BaseHTTPRequestHandler):
        server_version = "OpenAgent/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if not self._enforce_auth_gate(parsed.path, "GET"):
                return
            if parsed.path.startswith("/api/codebase/runs/") and parsed.path.endswith("/events/stream"):
                params = parse_qs(parsed.query)
                run_id = unquote(parsed.path.removeprefix("/api/codebase/runs/").removesuffix("/events/stream")).strip("/")
                try:
                    stream = codebase.event_stream(
                        run_id,
                        owner=self._query_owner(params) or "local",
                        after=parse_int(first(params, "after"), default=0),
                        timeout_seconds=parse_int(first(params, "timeout_seconds"), default=120),
                    )
                    self.send_event_stream(stream)
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            identity_route = handle_identity_get(parsed.path, identity_route_dependencies(self))
            if identity_route is not None:
                self.send_json(identity_route.payload, status=identity_route.status)
                return
            health_route = handle_health_get(parsed.path, parsed.query, health_routes)
            if health_route is not None:
                self.send_json(health_route.payload, status=health_route.status)
                return
            worker_route = handle_worker_get(parsed.path, parsed.query, worker_route_dependencies(self.read_json))
            if worker_route is not None:
                self.send_json(worker_route.payload, status=worker_route.status)
                return
            vector_route = handle_vector_get(parsed.path, parsed.query, vector_route_dependencies(self.read_json))
            if vector_route is not None:
                self.send_json(vector_route.payload, status=vector_route.status)
                return
            advanced_rag_route = handle_advanced_rag_get(
                parsed.path,
                parsed.query,
                advanced_rag_route_dependencies(self.read_json),
            )
            if advanced_rag_route is not None:
                self.send_json(advanced_rag_route.payload, status=advanced_rag_route.status)
                return
            inference_route = handle_inference_get(
                parsed.path,
                parsed.query,
                inference_route_dependencies(self.read_json),
            )
            if inference_route is not None:
                self.send_json(inference_route.payload, status=inference_route.status)
                return
            ultra_model_route = handle_ultra_model_get(
                parsed.path,
                parsed.query,
                ultra_model_route_dependencies(self.read_json),
            )
            if ultra_model_route is not None:
                self.send_json(ultra_model_route.payload, status=ultra_model_route.status)
                return
            adaptive_engine_route = handle_adaptive_engine_get(
                parsed.path,
                parsed.query,
                adaptive_engine_route_dependencies(self.read_json),
            )
            if adaptive_engine_route is not None:
                self.send_json(adaptive_engine_route.payload, status=adaptive_engine_route.status)
                return
            contacts_route = handle_contacts_get(
                parsed.path,
                parsed.query,
                contacts_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if contacts_route is not None:
                self.send_json(contacts_route.payload, status=contacts_route.status)
                return
            backup_route = handle_backup_get(parsed.path, backup_route_dependencies(self.read_json))
            if backup_route is not None:
                self.send_json(backup_route.payload, status=backup_route.status)
                return
            model_cookbook_route = handle_model_cookbook_get(parsed.path, parsed.query, model_cookbook_route_dependencies(self.read_json))
            if model_cookbook_route is not None:
                self.send_json(model_cookbook_route.payload, status=model_cookbook_route.status)
                return
            phd_route = handle_phd_get(
                parsed.path,
                parsed.query,
                phd_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if phd_route is not None:
                self.send_json(phd_route.payload, status=phd_route.status)
                return
            image_studio_route = handle_image_studio_get(
                parsed.path,
                parsed.query,
                image_studio_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if image_studio_route is not None:
                self.send_json(image_studio_route.payload, status=image_studio_route.status)
                return
            scheduler_route = handle_scheduler_get(
                parsed.path,
                parsed.query,
                scheduler_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if scheduler_route is not None:
                self.send_json(scheduler_route.payload, status=scheduler_route.status)
                return
            science_route = handle_science_get(parsed.path, parsed.query, science_route_dependencies(self.read_json))
            if science_route is not None:
                self.send_json(science_route.payload, status=science_route.status)
                return
            codebase_route = handle_codebase_get(
                parsed.path,
                parsed.query,
                codebase_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if codebase_route is not None:
                self.send_json(codebase_route.payload, status=codebase_route.status)
                return
            conversation_route = handle_conversation_get(
                parsed.path,
                parsed.query,
                conversation_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if conversation_route is not None:
                self.send_json(conversation_route.payload, status=conversation_route.status)
                return
            skill_route = handle_skill_get(
                parsed.path,
                parsed.query,
                SkillRouteDependencies(skills=skills, read_json=self.read_json),
            )
            if skill_route is not None:
                self.send_json(skill_route.payload, status=skill_route.status)
                return
            terminal_route = handle_terminal_get(
                parsed.path,
                parsed.query,
                terminal_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if terminal_route is not None:
                self.send_json(terminal_route.payload, status=terminal_route.status)
                return
            mcp_runtime_route = handle_mcp_runtime_get(
                parsed.path,
                parsed.query,
                mcp_runtime_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if mcp_runtime_route is not None:
                self.send_json(mcp_runtime_route.payload, status=mcp_runtime_route.status)
                return
            agent_bridge_route = handle_agent_bridge_get(
                parsed.path,
                parsed.query,
                agent_bridge_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if agent_bridge_route is not None:
                self.send_json(agent_bridge_route.payload, status=agent_bridge_route.status)
                return
            communication_route = handle_communication_get(
                parsed.path,
                parsed.query,
                CommunicationRouteDependencies(
                    communication=communication,
                    first=first,
                    parse_bool=parse_bool,
                    parse_int=parse_int,
                    query_owner=self._query_owner,
                    payload_owner=self._payload_owner,
                    read_json=self.read_json,
                ),
            )
            if communication_route is not None:
                self.send_json(communication_route.payload, status=communication_route.status)
                return
            calendar_route = handle_calendar_get(
                parsed.path,
                parsed.query,
                CalendarRouteDependencies(
                    calendar_system=calendar_system,
                    calendar_sync=calendar_sync,
                    first=first,
                    parse_bool=parse_bool,
                    parse_int=parse_int,
                    query_owner=self._query_owner,
                    payload_owner=self._payload_owner,
                    auth_scope_owner=self._auth_scope_owner,
                    read_json=self.read_json,
                ),
            )
            if calendar_route is not None:
                self.send_json(calendar_route.payload, status=calendar_route.status)
                return
            if parsed.path == "/api/chat/diagnose":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                started = time.perf_counter()
                direct_status = adaptive_engine.direct.status()
                resident = next(
                    (item for item in direct_status.get("models", []) if item.get("kind") == "llm"),
                    None,
                ) if direct_status.get("running") else None
                if resident:
                    provider_id = "openvino-direct"
                    model_id = str(resident.get("model_id", "")) or None
                    result = gateway.complete(
                        [{"role": "user", "content": "Reply briefly: OpenAgent model online."}],
                        provider_id=provider_id,
                        model=model_id,
                        allow_external=False,
                    )
                else:
                    result = gateway.complete(
                        [{"role": "user", "content": "Reply briefly: OpenAgent model online."}],
                        provider_id="local",
                        allow_external=False,
                    )
                duration_ms = round((time.perf_counter() - started) * 1000, 3)
                fallback = bool(result.metadata.get("deterministic_fallback") or result.metadata.get("fallback_from"))
                store.add_audit_event(
                    AuditEvent(
                        actor="chat",
                        action="chat.model_diagnose",
                        resource=f"model:{result.provider}:{result.model}",
                        project=project,
                        metadata={"provider": result.provider, "model": result.model, "fallback": fallback, "duration_ms": duration_ms},
                    )
                )
                self.send_json(
                    {
                        "status": "fallback" if fallback else "online",
                        "provider": result.provider,
                        "model": result.model,
                        "duration_ms": duration_ms,
                        "content": result.content,
                        "metadata": result.metadata,
                        "privacy_boundary": "Diagnosis uses Auto routing with local providers only and does not allow external provider calls.",
                    }
                )
                return
            if parsed.path == "/api/integrations/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(integrations.status(project=project))
                return
            if parsed.path == "/api/integrations/pairing-codes":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                include_expired = parse_bool(first(params, "include_expired"), default=True)
                self.send_json({"codes": integrations.pairing_codes(project=project, include_expired=include_expired)})
                return
            if parsed.path == "/api/integrations/preferences":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                scope = first(params, "scope") or ""
                self.send_json({"preferences": integrations.preferences(project=project, scope=scope)})
                return
            if parsed.path == "/api/integrations/diagnostics":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                limit = parse_int(first(params, "limit"), default=20)
                latest = parse_bool(first(params, "latest"), default=False)
                if latest:
                    self.send_json(integrations.diagnostic_snapshot(project=project))
                else:
                    self.send_json({"diagnostics": integrations.diagnostics(project=project, limit=limit)})
                return
            if parsed.path == "/api/integrations/backups":
                self.send_json({"backups": enterprise.backups()})
                return
            if parsed.path == "/api/integrations/safe-shell":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                limit = parse_int(first(params, "limit"), default=20)
                self.send_json(integrations.safe_commands(project=project, limit=limit))
                return
            if parsed.path == "/api/integrations/parity":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(integrations.feature_matrix(project=project))
                return
            if parsed.path == "/api/personal-os/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                owner = self._query_owner(params)
                self.send_json(personal_os.status(project=project, owner=owner))
                return
            if parsed.path == "/api/personal-os/presets":
                params = parse_qs(parsed.query)
                self.send_json(
                    {
                        "presets": personal_os.presets(
                            project=first(params, "project") or "default",
                            owner=self._query_owner(params),
                            kind=first(params, "kind") or "",
                        )
                    }
                )
                return
            if parsed.path == "/api/personal-os/signatures":
                params = parse_qs(parsed.query)
                self.send_json({"signatures": personal_os.signatures(project=first(params, "project") or "default", owner=self._query_owner(params))})
                return
            if parsed.path == "/api/personal-os/drafts":
                params = parse_qs(parsed.query)
                self.send_json(
                    {
                        "drafts": personal_os.drafts(
                            project=first(params, "project") or "default",
                            owner=self._query_owner(params),
                            status=first(params, "status") or "",
                        )
                    }
                )
                return
            if parsed.path.startswith("/api/personal-os/drafts/") and parsed.path.endswith("/versions"):
                draft_id = unquote(parsed.path.removeprefix("/api/personal-os/drafts/").removesuffix("/versions")).strip()
                self.send_json({"versions": personal_os.draft_versions(draft_id)})
                return
            if parsed.path == "/api/personal-os/notes":
                params = parse_qs(parsed.query)
                self.send_json(
                    {
                        "notes": personal_os.notes(
                            project=first(params, "project") or "default",
                            owner=self._query_owner(params),
                            status=first(params, "status") or "",
                            query=first(params, "q") or "",
                        )
                    }
                )
                return
            if parsed.path == "/api/personal-os/gallery":
                params = parse_qs(parsed.query)
                self.send_json(
                    {
                        "assets": personal_os.gallery(
                            project=first(params, "project") or "default",
                            owner=self._query_owner(params),
                            tag=first(params, "tag") or "",
                        ),
                        "stats": personal_os.gallery_stats(
                            project=first(params, "project") or "default",
                            owner=self._query_owner(params),
                        ),
                    }
                )
                return
            if parsed.path == "/api/personal-os/compare":
                params = parse_qs(parsed.query)
                self.send_json({"compares": personal_os.compare_history(project=first(params, "project") or "default", owner=self._query_owner(params))})
                return
            if parsed.path == "/api/personal-os/export":
                params = parse_qs(parsed.query)
                self.send_json(personal_os.export_bundle(project=first(params, "project") or "default", owner=self._query_owner(params)))
                return
            if parsed.path == "/api/capabilities":
                self.send_json(vision_coverage())
                return
            if parsed.path == "/api/core/runtime":
                params = parse_qs(parsed.query)
                task = first(params, "task") or "chat"
                target_ram = first(params, "target_ram_gb")
                quantization_mode = first(params, "quantization_mode") or "auto"
                try:
                    target_ram_gb = float(target_ram) if target_ram else None
                except (TypeError, ValueError):
                    self.send_json({"error": "target_ram_gb must be a number."}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(
                    local_ai.smooth_runtime_plan(
                        task=task,
                        target_ram_gb=target_ram_gb,
                        quantization_mode=quantization_mode,
                    )
                )
                return
            if parsed.path == "/api/mission/features":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(mission_control.feature_matrix(project=project))
                return
            if parsed.path == "/api/context":
                params = parse_qs(parsed.query)
                project = first(params, "project")
                self.send_json(context.snapshot(project=project))
                return
            if parsed.path == "/api/events":
                params = parse_qs(parsed.query)
                project = first(params, "project")
                query = first(params, "q") or ""
                limit = parse_int(first(params, "limit"), default=50)
                events = context.search(query, project=project, limit=limit)
                self.send_json({"events": events})
                return
            if parsed.path == "/api/agents":
                self.send_json({"agents": agents.specs()})
                return
            if parsed.path == "/api/models":
                self.send_json({"providers": gateway.specs()})
                return
            if parsed.path == "/api/model-policy":
                self.send_json(gateway.policies())
                return
            if parsed.path == "/api/settings/runtime":
                self.send_json(runtime_settings_snapshot())
                return
            if parsed.path == "/api/conversations":
                params = parse_qs(parsed.query)
                project = first(params, "project")
                limit = parse_int(first(params, "limit"), default=50)
                self.send_json({"conversations": store.list_conversations(project=project, limit=limit)})
                return
            if parsed.path in {"/api/messages", "/api/conversation/messages"}:
                params = parse_qs(parsed.query)
                conversation_id = first(params, "conversation_id")
                if not conversation_id:
                    self.send_json(
                        {"error": "conversation_id is required."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self.send_json({"messages": store.list_chat_messages(conversation_id)})
                return
            if parsed.path == "/api/chat/search":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                query = first(params, "q") or ""
                limit = parse_int(first(params, "limit"), default=50)
                self.send_json({"results": chat.search(query, project=project, limit=limit)})
                return
            if parsed.path == "/api/chat/export":
                params = parse_qs(parsed.query)
                conversation_id = first(params, "conversation_id")
                if not conversation_id:
                    self.send_json({"error": "conversation_id is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    exported = chat.export_conversation(
                        conversation_id,
                        write_file=parse_bool(first(params, "write_file"), default=True),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"export": exported})
                return
            if parsed.path == "/api/chat/replay":
                params = parse_qs(parsed.query)
                conversation_id = first(params, "conversation_id")
                if not conversation_id:
                    self.send_json({"error": "conversation_id is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    replay = chat.replay(conversation_id)
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"replay": replay})
                return
            if parsed.path == "/api/agent-runs":
                params = parse_qs(parsed.query)
                project = first(params, "project")
                limit = parse_int(first(params, "limit"), default=20)
                self.send_json({"runs": store.list_agent_runs(limit=limit, project=project)})
                return
            if parsed.path == "/api/audit":
                params = parse_qs(parsed.query)
                project = first(params, "project")
                limit = parse_int(first(params, "limit"), default=100)
                self.send_json({"events": store.list_audit_events(project=project, limit=limit)})
                return
            if parsed.path == "/api/workspace/items":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                query = first(params, "q") or ""
                kind = first(params, "kind") or ""
                parent_id = first(params, "parent_id")
                limit = parse_int(first(params, "limit"), default=100)
                items = (
                    workspace.search(query, project=project, limit=limit)
                    if query
                    else workspace.list_items(project=project, kind=kind, parent_id=parent_id, limit=limit)
                )
                self.send_json({"items": items})
                return
            if parsed.path == "/api/workspace/dashboard":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(workspace.dashboard(project=project))
                return
            if parsed.path == "/api/workspace/canvas":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(workspace.canvas_model(project=project))
                return
            if parsed.path == "/api/workflows":
                params = parse_qs(parsed.query)
                project = first(params, "project")
                limit = parse_int(first(params, "limit"), default=100)
                self.send_json({"workflows": store.list_workflows(project=project, limit=limit)})
                return
            if parsed.path == "/api/workflows/builder":
                self.send_json(workflows.builder_schema())
                return
            if parsed.path == "/api/workflows/visual":
                params = parse_qs(parsed.query)
                workflow_id = first(params, "workflow_id") or ""
                wf_list = store.list_workflows(limit=200)
                wf = next((w for w in wf_list if w.get("id") == workflow_id), None)
                if not wf:
                    self.send_json({"nodes": [], "edges": []})
                    return
                steps = wf.get("steps") or []
                # Always include a trigger node first, then each step
                nodes = [{"id": "trigger", "label": wf.get("trigger_type", "Manual trigger"), "x": 0, "y": 20}]
                nodes += [{"id": f"s{i}", "label": str(s.get("action", s.get("type", "step"))), "x": (i + 1) * 130, "y": 20} for i, s in enumerate(steps)]
                edges = [{"source": nodes[i]["id"], "target": nodes[i + 1]["id"]} for i in range(len(nodes) - 1)]
                self.send_json({"nodes": nodes, "edges": edges, "workflow": wf})
                return
            if parsed.path == "/api/workflows/visual":
                params = parse_qs(parsed.query)
                workflow_id = first(params, "workflow_id")
                if not workflow_id:
                    self.send_json({"error": "workflow_id is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    self.send_json(workflows.visual_model(workflow_id))
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/workflows/layout":
                params = parse_qs(parsed.query)
                workflow_id = first(params, "workflow_id")
                if not workflow_id:
                    self.send_json({"error": "workflow_id is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                layout = workflows.get_layout(workflow_id)
                self.send_json({"layout": layout})
                return
            if parsed.path == "/api/workflow-runs":
                params = parse_qs(parsed.query)
                project = first(params, "project")
                limit = parse_int(first(params, "limit"), default=100)
                self.send_json({"runs": store.list_workflow_runs(project=project, limit=limit)})
                return
            if parsed.path == "/api/knowledge/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(knowledge.status(project=project))
                return
            if parsed.path == "/api/research/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(research_engine.status(project=project))
                return
            if parsed.path == "/api/research/sessions":
                params = parse_qs(parsed.query)
                self.send_json({
                    "sessions": research_lifecycle.list(
                        project=first(params, "project") or "default",
                        include_archived=parse_bool(first(params, "include_archived"), default=False),
                        status=first(params, "status") or "",
                        limit=parse_int(first(params, "limit"), default=100),
                    )
                })
                return
            if parsed.path.startswith("/api/research/sessions/"):
                suffix = unquote(parsed.path.removeprefix("/api/research/sessions/")).strip("/")
                if suffix.endswith("/stream"):
                    session_id = suffix.removesuffix("/stream").strip("/")
                    params = parse_qs(parsed.query)
                    try:
                        events = research_lifecycle.stream_events(
                            session_id,
                            after_id=parse_int(first(params, "after_id"), default=0),
                            timeout_seconds=parse_int(first(params, "timeout"), default=120),
                        )
                        self.send_event_stream(events)
                    except KeyError:
                        self.send_json({"error": "Research session not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                if suffix.endswith("/source-graph"):
                    session_id = suffix.removesuffix("/source-graph").strip("/")
                    try:
                        self.send_json(research_lifecycle.source_graph(session_id))
                    except KeyError:
                        self.send_json({"error": "Research session not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                if suffix.endswith("/checkpoints"):
                    session_id = suffix.removesuffix("/checkpoints").strip("/")
                    params = parse_qs(parsed.query)
                    self.send_json({
                        "checkpoints": research_lifecycle.checkpoints(
                            session_id, after_id=parse_int(first(params, "after_id"), default=0)
                        )
                    })
                    return
                session = research_lifecycle.get(suffix)
                if session is None:
                    self.send_json({"error": "Research session not found."}, status=HTTPStatus.NOT_FOUND)
                else:
                    self.send_json({"session": session})
                return
            if parsed.path == "/api/knowledge/search":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                query = first(params, "q") or ""
                mode = first(params, "mode") or "fts"
                if mode == "semantic":
                    results = knowledge.semantic_search(query, project=project)
                else:
                    results = knowledge.search(query, project=project)
                self.send_json({"results": results})
                return
            if parsed.path == "/api/knowledge/graph":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(knowledge.graph(project=project))
                return
            if parsed.path == "/api/knowledge/tags":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                limit = parse_int(first(params, "limit"), default=100)
                self.send_json({"tags": knowledge.tags(project=project, limit=limit)})
                return
            if parsed.path == "/api/knowledge/timeline":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                limit = parse_int(first(params, "limit"), default=100)
                self.send_json({"timeline": knowledge.timeline(project=project, limit=limit)})
                return
            if parsed.path == "/api/semantic/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(semantic_memory.status(project=project))
                return
            if parsed.path == "/api/semantic/memories":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                query = first(params, "q") or ""
                memory_type = first(params, "memory_type") or ""
                limit = parse_int(first(params, "limit"), default=50)
                memories = (
                    semantic_memory.search_memories(query, project=project, limit=limit)
                    if query
                    else semantic_memory.list_memories(project=project, memory_type=memory_type, limit=limit)
                )
                self.send_json({"memories": memories})
                return
            if parsed.path == "/api/semantic/collections":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"collections": semantic_memory.collections(project=project)})
                return
            if parsed.path == "/api/semantic/graph-analytics":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(semantic_memory.graph_analytics(project=project))
                return
            if parsed.path == "/api/media/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(media.status(project=project))
                return
            if parsed.path == "/api/media/assets":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                media_type = first(params, "media_type") or ""
                limit = parse_int(first(params, "limit"), default=50)
                self.send_json({"assets": media.assets(project=project, media_type=media_type, limit=limit)})
                return
            if parsed.path == "/api/media/analyses":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                asset_id = first(params, "asset_id") or ""
                limit = parse_int(first(params, "limit"), default=50)
                self.send_json({"analyses": media.analyses(project=project, asset_id=asset_id, limit=limit)})
                return
            if parsed.path == "/api/agents/marketplace":
                self.send_json({"agents": multi_agent.marketplace()})
                return
            if parsed.path == "/api/agents/permissions":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"permissions": multi_agent.list_permissions(project=project)})
                return
            if parsed.path == "/api/agents/approvals":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                status = first(params, "status")
                self.send_json({"approvals": multi_agent.list_approvals(project=project, status=status)})
                return
            if parsed.path == "/api/assistant/tasks":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                status = first(params, "status") or ""
                limit = parse_int(first(params, "limit"), default=100)
                self.send_json({"tasks": assistant.tasks(project=project, status=status, limit=limit)})
                return
            if parsed.path == "/api/assistant/goals":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                status = first(params, "status") or ""
                self.send_json({"goals": assistant.goals(project=project, status=status)})
                return
            if parsed.path == "/api/assistant/daily-brief":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(assistant.daily_brief(project=project))
                return
            if parsed.path == "/api/assistant/coach":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(assistant.productivity_coaching(project=project))
                return
            if parsed.path == "/api/assistant/crm":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                record_type = first(params, "record_type") or ""
                self.send_json({"records": assistant.crm_records(project=project, record_type=record_type)})
                return
            if parsed.path == "/api/communication/connectors":
                self.send_json({"connectors": communication.connectors()})
                return
            if parsed.path == "/api/communication/inbox":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                query = first(params, "q") or ""
                follow_up = first(params, "follow_up")
                follow_filter = None if follow_up is None else parse_bool(follow_up, default=False)
                self.send_json({"messages": communication.inbox(project=project, query=query, follow_up=follow_filter, owner=self._query_owner(params))})
                return
            if parsed.path == "/api/communication/contacts":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                query = first(params, "q") or ""
                self.send_json({"contacts": communication.contacts(project=project, query=query, owner=self._query_owner(params))})
                return
            if parsed.path == "/api/communication/calendar":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"events": communication.calendar(project=project, owner=self._query_owner(params))})
                return
            if parsed.path == "/api/communication/analytics":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(communication.analytics(project=project, owner=self._query_owner(params)))
                return
            if parsed.path == "/api/communication/sync-runs":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                platform = first(params, "platform") or ""
                limit = parse_int(first(params, "limit"), default=50)
                self.send_json({"runs": communication.sync_runs(project=project, platform=platform, owner=self._query_owner(params), limit=limit)})
                return
            if parsed.path == "/api/voice/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(voice.status(project=project))
                return
            if parsed.path == "/api/voice/personas":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"personas": voice.personas(project=project)})
                return
            if parsed.path == "/api/voice/sessions":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                status = first(params, "status") or ""
                limit = parse_int(first(params, "limit"), default=50)
                self.send_json({"sessions": voice.sessions(project=project, status=status, limit=limit)})
                return
            if parsed.path == "/api/voice/turns":
                params = parse_qs(parsed.query)
                session_id = first(params, "session_id") or ""
                if not session_id:
                    self.send_json({"error": "session_id is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"turns": voice.turns(session_id)})
                return
            if parsed.path == "/api/production/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(production.status(project=project))
                return
            if parsed.path == "/api/production/policies":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                scope = first(params, "scope") or ""
                self.send_json({"rules": production.policy_rules(project=project, scope=scope)})
                return
            if parsed.path == "/api/production/queue":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                status = first(params, "status") or ""
                limit = parse_int(first(params, "limit"), default=100)
                self.send_json({"jobs": production.jobs(project=project, status=status, limit=limit)})
                return
            if parsed.path == "/api/production/audit-anchors":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                limit = parse_int(first(params, "limit"), default=50)
                self.send_json({"anchors": production.audit_anchors(project=project, limit=limit)})
                return
            if parsed.path == "/api/local-ai/jobs":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                status = first(params, "status") or ""
                limit = parse_int(first(params, "limit"), default=25)
                self.send_json({"jobs": local_ai.model_jobs(project=project, status=status, limit=limit)})
                return
            if parsed.path.startswith("/api/local-ai/jobs/"):
                job_id = unquote(parsed.path.removeprefix("/api/local-ai/jobs/")).strip()
                job = local_ai.model_job(job_id)
                if job is None:
                    self.send_json({"error": "Model job not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json({"job": job})
                return
            if parsed.path == "/api/local-ai/status":
                self.send_json(local_ai.status())
                return
            if parsed.path == "/api/local-ai/setup/status":
                self.send_json(local_ai.model_tool_status())
                return
            if parsed.path == "/api/local-ai/smooth-runtime":
                params = parse_qs(parsed.query)
                task = first(params, "task") or "chat"
                target_ram = first(params, "target_ram_gb")
                quantization_mode = first(params, "quantization_mode") or "auto"
                try:
                    target_ram_gb = float(target_ram) if target_ram else None
                except (TypeError, ValueError):
                    self.send_json({"error": "target_ram_gb must be a number."}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(
                    local_ai.smooth_runtime_plan(
                        task=task,
                        target_ram_gb=target_ram_gb,
                        quantization_mode=quantization_mode,
                    )
                )
                return
            if parsed.path == "/api/local-ai/models":
                self.send_json({"models": local_ai.models()})
                return
            if parsed.path == "/api/local-ai/hardware":
                self.send_json(local_ai.detect_hardware())
                return
            if parsed.path == "/api/local-ai/adapters":
                self.send_json({"adapters": local_ai.adapters()})
                return
            if parsed.path == "/api/local-ai/marketplace":
                self.send_json({"models": local_ai.marketplace()})
                return
            if parsed.path == "/api/enterprise/status":
                self.send_json(enterprise.status())
                return
            if parsed.path == "/api/enterprise/users":
                self.send_json({"users": enterprise.users()})
                return
            if parsed.path == "/api/enterprise/roles":
                self.send_json({"roles": enterprise.roles()})
                return
            if parsed.path == "/api/enterprise/workspaces":
                self.send_json({"workspaces": enterprise.workspaces()})
                return
            if parsed.path == "/api/enterprise/policies":
                self.send_json({"policies": enterprise.policies()})
                return
            if parsed.path == "/api/enterprise/sso":
                self.send_json({"configs": enterprise.sso_configs()})
                return
            if parsed.path == "/api/enterprise/backups":
                self.send_json({"backups": enterprise.backups()})
                return
            if parsed.path == "/api/enterprise/disaster-recovery":
                self.send_json({"plans": enterprise.disaster_recovery_plans()})
                return
            if parsed.path == "/api/enterprise/workspace/members":
                params = parse_qs(parsed.query)
                workspace_id = first(params, "workspace_id")
                if not workspace_id:
                    self.send_json({"error": "workspace_id is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"members": enterprise.members(workspace_id)})
                return
            if parsed.path == "/api/enterprise/org-memory":
                params = parse_qs(parsed.query)
                workspace_id = first(params, "workspace_id")
                if not workspace_id:
                    self.send_json({"error": "workspace_id is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"memory": enterprise.org_memory(workspace_id)})
                return
            if parsed.path == "/api/developer/status":
                self.send_json(developer.status())
                return
            if parsed.path == "/api/developer/schema":
                self.send_json(developer.graphql_schema())
                return
            if parsed.path == "/api/developer/webhooks":
                self.send_json({"webhooks": developer.webhooks()})
                return
            if parsed.path == "/api/developer/webhook-deliveries":
                self.send_json({"deliveries": developer.webhook_deliveries()})
                return
            if parsed.path == "/api/developer/plugins":
                self.send_json({"plugins": developer.plugins()})
                return
            if parsed.path == "/api/developer/tools":
                self.send_json({"tools": developer.tools()})
                return
            if parsed.path == "/api/developer/mcp":
                self.send_json(developer.mcp_manifest())
                return
            if parsed.path == "/api/developer/mcp/servers":
                self.send_json({"servers": developer.mcp_servers()})
                return
            if parsed.path == "/api/developer/extensions":
                self.send_json({"packages": developer.extension_packages()})
                return
            if parsed.path == "/api/developer/sdk":
                self.send_json({"artifacts": developer.sdk_artifacts()})
                return
            if parsed.path == "/api/developer/api-keys":
                self.send_json({"api_keys": developer.api_keys()})
                return
            if parsed.path == "/api/security/status":
                self.send_json(security_privacy.status())
                return
            if parsed.path == "/api/security/secrets":
                self.send_json({"secrets": security_privacy.secrets()})
                return
            if parsed.path == "/api/security/retention":
                self.send_json({"policies": security_privacy.retention_policies()})
                return
            if parsed.path == "/api/security/shares":
                self.send_json({"grants": security_privacy.share_grants()})
                return
            if parsed.path == "/api/security/egress":
                self.send_json({"requests": security_privacy.egress_requests()})
                return
            if parsed.path == "/api/security/audit-chain":
                self.send_json({"chain": security_privacy.audit_chain()})
                return
            if parsed.path == "/api/security/audit-chain/verify":
                self.send_json(security_privacy.verify_audit_chain())
                return
            if parsed.path in {"/api/security/privacy-reports", "/api/security/privacy-report"}:
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"reports": security_privacy.privacy_reports(), "project": project})
                return
            if parsed.path == "/api/security/sandbox-profiles":
                self.send_json({"profiles": security_privacy.sandbox_profiles()})
                return
            # ---- Calendar System GET routes ----
            if parsed.path == "/api/calendar/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(calendar_system.status(project=project))
                return
            if parsed.path == "/api/calendar/calendars":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"calendars": calendar_system.list_calendars(project=project)})
                return
            if parsed.path == "/api/calendar/events":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                from_dt = first(params, "from") or ""
                to_dt = first(params, "to") or ""
                calendar_id = first(params, "calendar_id") or ""
                limit = parse_int(first(params, "limit"), default=100)
                self.send_json({"events": calendar_system.list_events(
                    project=project, from_dt=from_dt, to_dt=to_dt,
                    calendar_id=calendar_id, limit=limit,
                )})
                return
            if parsed.path == "/api/calendar/free-slots":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                from_dt = first(params, "from") or ""
                to_dt = first(params, "to") or ""
                duration = parse_int(first(params, "duration"), default=60)
                if not from_dt or not to_dt:
                    self.send_json({"error": "from and to query params required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"slots": calendar_system.find_free_slots(
                    project=project, from_dt=from_dt, to_dt=to_dt, duration_minutes=duration,
                )})
                return
            if parsed.path == "/api/calendar/reminders":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"reminders": calendar_system.get_due_reminders(project=project)})
                return
            if parsed.path == "/api/calendar/optimize":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(calendar_system.optimize_schedule(project=project))
                return
            # ---- Digital Twin GET routes ----
            if parsed.path == "/api/twin/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(digital_twin.status(project=project))
                return
            if parsed.path == "/api/twin/profile":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(digital_twin.get_profile(project=project))
                return
            if parsed.path == "/api/twin/patterns":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(digital_twin.analyse_patterns(project=project))
                return
            if parsed.path == "/api/twin/suggestions":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"suggestions": digital_twin.generate_proactive_suggestions(project=project)})
                return
            if parsed.path == "/api/twin/goals":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"goals": digital_twin.list_goals(project=project)})
                return
            # ---- Vault GET routes ----
            if parsed.path == "/api/vault/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(vault.status(project=project))
                return
            if parsed.path == "/api/vault/entries":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                category = first(params, "category") or ""
                tag = first(params, "tag") or ""
                limit = parse_int(first(params, "limit"), default=100)
                self.send_json({"entries": vault.list_entries(
                    project=project, category=category, tag=tag, limit=limit,
                )})
                return
            # ---- Scraper GET routes ----
            if parsed.path == "/api/scraper/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(scraper.status(project=project))
                return
            if parsed.path == "/api/scraper/jobs":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"jobs": scraper.list_scheduled(project=project)})
                return
            # ---- Fine-tuning GET routes ----
            if parsed.path == "/api/finetune/status":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json(fine_tuning.status(project=project))
                return
            if parsed.path == "/api/finetune/datasets":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"datasets": fine_tuning.list_datasets(project=project)})
                return
            if parsed.path == "/api/finetune/jobs":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"jobs": fine_tuning.list_jobs(project=project)})
                return
            if parsed.path == "/api/finetune/adapters":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                base_model = first(params, "base_model") or ""
                self.send_json({"adapters": fine_tuning.list_adapters(project=project, base_model=base_model)})
                return
            if parsed.path == "/api/finetune/templates":
                self.send_json({"templates": fine_tuning.list_templates()})
                return

            # ---- Auth GET routes ----
            if parsed.path == "/api/auth/scopes":
                self.send_json({"scopes": auth.list_scopes()})
                return
            if parsed.path == "/api/auth/policy":
                self.send_json(auth.policy())
                return
            if parsed.path == "/api/auth/roles":
                self.send_json({"roles": auth.roles()})
                return
            if parsed.path == "/api/auth/users":
                user = self._require_auth()
                if user is None:
                    return
                if user.get("role") != "owner":
                    self.send_json({"error": "owner role required"}, status=HTTPStatus.FORBIDDEN)
                    return
                self.send_json({"users": auth.list_users()})
                return
            if parsed.path == "/api/auth/audit":
                user = self._require_auth()
                if user is None:
                    return
                if user.get("role") != "owner":
                    self.send_json({"error": "owner role required"}, status=HTTPStatus.FORBIDDEN)
                    return
                params = parse_qs(parsed.query)
                self.send_json(auth.auth_audit_summary(limit=parse_int(first(params, "limit"), default=50)))
                return
            if parsed.path == "/api/auth/me":
                user = self._require_auth()
                if user is None:
                    return
                self.send_json({"user": user})
                return
            if parsed.path == "/api/auth/sessions":
                user = self._require_auth()
                if user is None:
                    return
                self.send_json({"sessions": auth.list_sessions(user["user_id"])})
                return
            if parsed.path == "/api/auth/api-tokens":
                user = self._require_auth()
                if user is None:
                    return
                self.send_json({"tokens": auth.list_api_tokens(user["user_id"])})
                return

            # ---- Editor document + draft GET routes ----
            if parsed.path == "/api/documents/library":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                query = first(params, "q") or ""
                tag = first(params, "tag") or ""
                folder = first(params, "folder") or ""
                status = first(params, "status") or ""
                limit = parse_int(first(params, "limit"), default=50)
                owner = self._auth_scope_owner()
                self.send_json(documents.list_library(
                    project=project, query=query, tag=tag, folder=folder, status=status, owner=owner, limit=limit,
                ))
                return
            if parsed.path.startswith("/api/document/") and parsed.path.endswith("/versions"):
                doc_id = unquote(parsed.path.removeprefix("/api/document/").removesuffix("/versions")).strip()
                try:
                    self.send_json({"versions": documents.list_versions(doc_id, owner=self._auth_scope_owner())})
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            if parsed.path.startswith("/api/document/") and "/version/" in parsed.path:
                remainder = parsed.path.removeprefix("/api/document/")
                doc_id, _, version_text = remainder.partition("/version/")
                doc_id = unquote(doc_id).strip()
                try:
                    version_number = int(version_text)
                except ValueError:
                    self.send_json({"error": "invalid version number"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    self.send_json({"version": documents.get_version(doc_id, version_number, owner=self._auth_scope_owner())})
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            if parsed.path.startswith("/api/document/"):
                doc_id = unquote(parsed.path.removeprefix("/api/document/")).strip()
                if doc_id:
                    try:
                        self.send_json({"document": documents.get_document(doc_id, owner=self._auth_scope_owner())})
                    except KeyError as exc:
                        self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
            if parsed.path == "/api/editor-drafts":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                doc_id = first(params, "doc_id") or ""
                self.send_json({"drafts": documents.list_drafts(project=project, doc_id=doc_id, owner=self._auth_scope_owner())})
                return
            if parsed.path.startswith("/api/editor-drafts/"):
                draft_id = unquote(parsed.path.removeprefix("/api/editor-drafts/")).strip()
                try:
                    self.send_json({"draft": documents.get_draft(draft_id, owner=self._auth_scope_owner())})
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return

            # ---- Calendar external-sync GET routes ----
            if parsed.path == "/api/calendar/providers":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"providers": calendar_sync.list_providers(project=project, owner=self._query_owner(params))})
                return
            if parsed.path == "/api/calendar/conflicts":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                status_filter = first(params, "status") or ""
                self.send_json({"conflicts": calendar_sync.list_conflicts(project=project, status=status_filter, owner=self._query_owner(params))})
                return
            if parsed.path.startswith("/api/calendar/sync/"):
                job_id = unquote(parsed.path.removeprefix("/api/calendar/sync/")).strip()
                try:
                    self.send_json(calendar_sync.get_sync_job(job_id, owner=self._auth_scope_owner()))
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return

            # ---- Model-ops lifecycle/diagnostics GET routes ----
            if parsed.path.startswith("/api/local-ai/models/") and parsed.path.endswith("/diagnostics"):
                model_id = unquote(parsed.path.removeprefix("/api/local-ai/models/").removesuffix("/diagnostics")).strip()
                try:
                    self.send_json({"diagnostics": local_ai.get_model_diagnostics(model_id)})
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            if parsed.path.startswith("/api/local-ai/models/") and parsed.path.endswith("/hardware-fit"):
                model_id = unquote(parsed.path.removeprefix("/api/local-ai/models/").removesuffix("/hardware-fit")).strip()
                try:
                    self.send_json(local_ai.hardware_fit_for_model(model_id))
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            if parsed.path.startswith("/api/local-ai/models/") and parsed.path.endswith("/probe"):
                model_id = unquote(parsed.path.removeprefix("/api/local-ai/models/").removesuffix("/probe")).strip()
                try:
                    self.send_json(local_ai.probe_model(model_id))
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return

            # ---- Communication resilience GET routes ----
            if parsed.path == "/api/communication/dead-letters":
                params = parse_qs(parsed.query)
                project = first(params, "project") or "default"
                self.send_json({"dead_letters": communication.list_dead_letters(project=project, owner=self._query_owner(params))})
                return

            self.serve_static(parsed.path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if not self._enforce_auth_gate(parsed.path, "POST"):
                return
            identity_route = handle_identity_post(parsed.path, identity_route_dependencies(self))
            if identity_route is not None:
                self.send_json(identity_route.payload, status=identity_route.status)
                return
            worker_route = handle_worker_post(parsed.path, worker_route_dependencies(self.read_json))
            if worker_route is not None:
                self.send_json(worker_route.payload, status=worker_route.status)
                return
            vector_route = handle_vector_post(parsed.path, vector_route_dependencies(self.read_json))
            if vector_route is not None:
                self.send_json(vector_route.payload, status=vector_route.status)
                return
            advanced_rag_route = handle_advanced_rag_post(
                parsed.path,
                advanced_rag_route_dependencies(self.read_json),
            )
            if advanced_rag_route is not None:
                self.send_json(advanced_rag_route.payload, status=advanced_rag_route.status)
                return
            inference_route = handle_inference_post(
                parsed.path,
                inference_route_dependencies(self.read_json),
            )
            if inference_route is not None:
                self.send_json(inference_route.payload, status=inference_route.status)
                return
            ultra_model_route = handle_ultra_model_post(
                parsed.path,
                ultra_model_route_dependencies(self.read_json),
            )
            if ultra_model_route is not None:
                self.send_json(ultra_model_route.payload, status=ultra_model_route.status)
                return
            adaptive_engine_route = handle_adaptive_engine_post(
                parsed.path,
                adaptive_engine_route_dependencies(self.read_json),
            )
            if adaptive_engine_route is not None:
                self.send_json(adaptive_engine_route.payload, status=adaptive_engine_route.status)
                return
            contacts_route = handle_contacts_post(
                parsed.path,
                contacts_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if contacts_route is not None:
                self.send_json(contacts_route.payload, status=contacts_route.status)
                return
            backup_route = handle_backup_post(parsed.path, backup_route_dependencies(self.read_json))
            if backup_route is not None:
                self.send_json(backup_route.payload, status=backup_route.status)
                return
            model_cookbook_route = handle_model_cookbook_post(parsed.path, model_cookbook_route_dependencies(self.read_json))
            if model_cookbook_route is not None:
                self.send_json(model_cookbook_route.payload, status=model_cookbook_route.status)
                return
            phd_route = handle_phd_post(
                parsed.path,
                phd_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if phd_route is not None:
                self.send_json(phd_route.payload, status=phd_route.status)
                return
            document_processing_route = handle_document_processing_post(
                parsed.path,
                document_processing_route_dependencies(self.read_json, self._payload_owner),
            )
            if document_processing_route is not None:
                self.send_json(document_processing_route.payload, status=document_processing_route.status)
                return
            image_studio_route = handle_image_studio_post(
                parsed.path,
                image_studio_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if image_studio_route is not None:
                self.send_json(image_studio_route.payload, status=image_studio_route.status)
                return
            scheduler_route = handle_scheduler_post(
                parsed.path,
                scheduler_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if scheduler_route is not None:
                self.send_json(scheduler_route.payload, status=scheduler_route.status)
                return
            science_route = handle_science_post(parsed.path, science_route_dependencies(self.read_json))
            if science_route is not None:
                self.send_json(science_route.payload, status=science_route.status)
                return
            codebase_route = handle_codebase_post(
                parsed.path,
                codebase_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if codebase_route is not None:
                self.send_json(codebase_route.payload, status=codebase_route.status)
                return
            conversation_route = handle_conversation_post(
                parsed.path,
                conversation_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if conversation_route is not None:
                self.send_json(conversation_route.payload, status=conversation_route.status)
                return
            skill_route = handle_skill_post(
                parsed.path,
                SkillRouteDependencies(skills=skills, read_json=self.read_json),
            )
            if skill_route is not None:
                self.send_json(skill_route.payload, status=skill_route.status)
                return
            terminal_route = handle_terminal_post(
                parsed.path,
                terminal_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if terminal_route is not None:
                self.send_json(terminal_route.payload, status=terminal_route.status)
                return
            mcp_runtime_route = handle_mcp_runtime_post(
                parsed.path,
                mcp_runtime_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if mcp_runtime_route is not None:
                self.send_json(mcp_runtime_route.payload, status=mcp_runtime_route.status)
                return
            agent_bridge_route = handle_agent_bridge_post(
                parsed.path,
                agent_bridge_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if agent_bridge_route is not None:
                self.send_json(agent_bridge_route.payload, status=agent_bridge_route.status)
                return
            communication_route = handle_communication_post(
                parsed.path,
                CommunicationRouteDependencies(
                    communication=communication,
                    first=first,
                    parse_bool=parse_bool,
                    parse_int=parse_int,
                    query_owner=self._query_owner,
                    payload_owner=self._payload_owner,
                    read_json=self.read_json,
                ),
            )
            if communication_route is not None:
                self.send_json(communication_route.payload, status=communication_route.status)
                return
            calendar_route = handle_calendar_post(
                parsed.path,
                CalendarRouteDependencies(
                    calendar_system=calendar_system,
                    calendar_sync=calendar_sync,
                    first=first,
                    parse_bool=parse_bool,
                    parse_int=parse_int,
                    query_owner=self._query_owner,
                    payload_owner=self._payload_owner,
                    auth_scope_owner=self._auth_scope_owner,
                    read_json=self.read_json,
                ),
            )
            if calendar_route is not None:
                self.send_json(calendar_route.payload, status=calendar_route.status)
                return
            if parsed.path in {"/api/graphql", "/api/developer/graphql"}:
                payload = self.read_json()
                try:
                    result = developer.execute_graphql(
                        str(payload.get("query", "")).strip(),
                        variables=payload.get("variables") if isinstance(payload.get("variables"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/events":
                payload = self.read_json()
                title = str(payload.get("title", "")).strip()
                content = str(payload.get("content", "")).strip()
                if not title or not content:
                    self.send_json(
                        {"error": "Both title and content are required."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                event = context.ingest(
                    title=title,
                    content=content,
                    type=str(payload.get("type", "note")),
                    source=str(payload.get("source", "manual")),
                    project=str(payload.get("project", "default")),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                )
                self.send_json({"event": event}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/agents/run":
                payload = self.read_json()
                agent_id = str(payload.get("agent", "executive")).strip() or "executive"
                goal = str(payload.get("goal", "")).strip()
                project = str(payload.get("project", "default")).strip() or "default"
                if not goal:
                    self.send_json({"error": "A goal is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    run = agents.run(agent_id, goal, project=project)
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"run": run}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/chat":
                payload = self.read_json()
                content = str(payload.get("message", "")).strip()
                project = str(payload.get("project", "default")).strip() or "default"
                provider = str(payload.get("provider", "local")).strip() or "local"
                model = str(payload.get("model", "")).strip() or None
                conversation_id = str(payload.get("conversation_id") or "").strip() or None
                allow_external = parse_bool(payload.get("allow_external"), default=False)
                attachments = payload.get("attachments") if isinstance(payload.get("attachments"), list) else []
                if not content:
                    self.send_json({"error": "A message is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    result = chat.send(
                        content,
                        conversation_id=conversation_id,
                        project=project,
                        provider=provider,
                        model=model,
                        allow_external=allow_external,
                        attachments=attachments,
                    )
                except (ProviderConfigurationError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/chat/artifact":
                payload = self.read_json()
                try:
                    artifact = chat_artifacts.render(
                        title=str(payload.get("title", "OpenAgent document")),
                        content=str(payload.get("content", "")),
                        file_format=str(payload.get("format", "pdf")),
                        project=str(payload.get("project", "default")) or "default",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"artifact": artifact}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/chat/research-record":
                payload = self.read_json()
                try:
                    result = chat.record_research(
                        str(payload.get("query", "")),
                        str(payload.get("report", "")),
                        conversation_id=str(payload.get("conversation_id", "")).strip() or None,
                        project=str(payload.get("project", "default")).strip() or "default",
                        sources=payload.get("sources") if isinstance(payload.get("sources"), list) else [],
                        research_session_id=str(payload.get("research_session_id", "")).strip(),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/chat/stream":
                payload = self.read_json()
                content = str(payload.get("message", "")).strip()
                project = str(payload.get("project", "default")).strip() or "default"
                provider = str(payload.get("provider", "local")).strip() or "local"
                model = str(payload.get("model", "")).strip() or None
                conversation_id = str(payload.get("conversation_id") or "").strip() or None
                allow_external = parse_bool(payload.get("allow_external"), default=False)
                attachments = payload.get("attachments") if isinstance(payload.get("attachments"), list) else []
                if not content:
                    self.send_json({"error": "A message is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_event_stream(
                    chat.stream(
                        content,
                        conversation_id=conversation_id,
                        project=project,
                        provider=provider,
                        model=model,
                        allow_external=allow_external,
                        attachments=attachments,
                    )
                )
                return
            if parsed.path == "/api/chat/branch":
                payload = self.read_json()
                try:
                    branch = chat.branch(
                        str(payload.get("conversation_id", "")).strip(),
                        at_message_id=str(payload.get("at_message_id", "")).strip(),
                        title=str(payload.get("title", "")).strip(),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"branch": branch}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/chat/import":
                payload = self.read_json()
                try:
                    imported = chat.import_conversation(
                        payload.get("conversation_export") if isinstance(payload.get("conversation_export"), dict) else payload,
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"imported": imported}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/integrations/pairing-codes":
                payload = self.read_json()
                try:
                    code = integrations.create_pairing_code(
                        label=str(payload.get("label", "Local companion")).strip(),
                        scopes=[str(item) for item in payload.get("scopes", [])] if isinstance(payload.get("scopes"), list) else None,
                        project=str(payload.get("project", "default")).strip() or "default",
                        ttl_minutes=parse_int(str(payload.get("ttl_minutes", "15")), default=15),
                    )
                except (IntegrationHubError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"code": code}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/integrations/pairing-codes/redeem":
                payload = self.read_json()
                try:
                    paired = integrations.redeem_pairing_code(
                        code=str(payload.get("code", "")).strip(),
                        device_name=str(payload.get("device_name", "Local device")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except IntegrationHubError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(paired, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/integrations/preferences":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                scope = str(payload.get("scope", "ui")).strip() or "ui"
                key = str(payload.get("key", "")).strip()
                try:
                    if str(payload.get("action", "")).strip().lower() == "delete":
                        result = integrations.delete_preference(project=project, scope=scope, key=key)
                        self.send_json(result)
                    else:
                        pref = integrations.set_preference(
                            project=project,
                            scope=scope,
                            key=key,
                            value=payload.get("value"),
                        )
                        self.send_json({"preference": pref}, status=HTTPStatus.CREATED)
                except IntegrationHubError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if parsed.path == "/api/integrations/diagnostics":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                self.send_json({"diagnostic": integrations.create_diagnostic_report(project=project)}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/integrations/backups":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                self.send_json(integrations.create_backup(project=project), status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/integrations/safe-shell":
                payload = self.read_json()
                try:
                    result = integrations.run_safe_command(
                        command_key=str(payload.get("command_key", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        timeout_seconds=parse_int(str(payload.get("timeout_seconds", "15")), default=15),
                    )
                except IntegrationHubError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/personal-os/presets":
                payload = self.read_json()
                try:
                    preset = personal_os.upsert_preset(
                        name=str(payload.get("name", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload, default=""),
                        kind=str(payload.get("kind", "chat")).strip() or "chat",
                        instructions=str(payload.get("instructions", "")),
                        provider=str(payload.get("provider", "auto")).strip() or "auto",
                        model=str(payload.get("model", "")).strip(),
                        parameters=payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {},
                    )
                except PersonalOSError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"preset": preset}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/personal-os/signatures":
                payload = self.read_json()
                try:
                    signature = personal_os.upsert_signature(
                        label=str(payload.get("label", "")).strip(),
                        content=str(payload.get("content", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload, default=""),
                        is_default=parse_bool(payload.get("is_default"), default=False),
                    )
                except PersonalOSError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"signature": signature}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/personal-os/drafts":
                payload = self.read_json()
                try:
                    draft = personal_os.save_draft(
                        title=str(payload.get("title", "")).strip(),
                        content=str(payload.get("content", "")),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload, default=""),
                        draft_id=str(payload.get("draft_id", "")).strip(),
                        status=str(payload.get("status", "draft")).strip() or "draft",
                    )
                except PersonalOSError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"draft": draft}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/personal-os/notes":
                payload = self.read_json()
                try:
                    note = personal_os.save_note(
                        title=str(payload.get("title", "")).strip(),
                        body=str(payload.get("body", "")),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload, default=""),
                        note_id=str(payload.get("note_id", "")).strip(),
                        tags=[str(item) for item in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else [],
                        due_at=str(payload.get("due_at", "")).strip(),
                        status=str(payload.get("status", "open")).strip() or "open",
                    )
                except PersonalOSError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"note": note}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/personal-os/gallery":
                payload = self.read_json()
                try:
                    asset = personal_os.add_gallery_asset(
                        title=str(payload.get("title", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload, default=""),
                        media_type=str(payload.get("media_type", "image")).strip() or "image",
                        source_path=str(payload.get("source_path", "")).strip(),
                        content=str(payload.get("content", "")),
                        tags=[str(item) for item in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else [],
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except PersonalOSError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"asset": asset}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/personal-os/gallery/favorite":
                payload = self.read_json()
                try:
                    asset = personal_os.set_gallery_favorite(
                        asset_id=str(payload.get("asset_id", "")).strip(),
                        favorite=parse_bool(payload.get("favorite"), default=True),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload, default=""),
                    )
                except PersonalOSError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"asset": asset})
                return
            if parsed.path == "/api/personal-os/compare/start":
                payload = self.read_json()
                try:
                    compare = personal_os.start_compare(
                        prompt=str(payload.get("prompt", "")).strip(),
                        left_output=str(payload.get("left_output", "")),
                        right_output=str(payload.get("right_output", "")),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload, default=""),
                        left_label=str(payload.get("left_label", "A")).strip() or "A",
                        right_label=str(payload.get("right_label", "B")).strip() or "B",
                    )
                except PersonalOSError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"compare": compare}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/personal-os/compare/vote":
                payload = self.read_json()
                try:
                    compare = personal_os.vote_compare(
                        compare_id=str(payload.get("compare_id", "")).strip(),
                        vote=str(payload.get("vote", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload, default=""),
                    )
                except PersonalOSError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"compare": compare})
                return
            if parsed.path == "/api/personal-os/import":
                payload = self.read_json()
                bundle = payload.get("bundle") if isinstance(payload.get("bundle"), dict) else payload
                try:
                    result = personal_os.import_bundle(
                        bundle,
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload, default=""),
                    )
                except PersonalOSError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/settings/runtime":
                payload = self.read_json()
                settings = gateway.apply_runtime_settings(
                    payload.get("settings") if isinstance(payload.get("settings"), dict) else payload
                )
                self.send_json({"settings": settings})
                return
            if parsed.path == "/api/embeddings":
                payload = self.read_json()
                raw_texts = payload.get("texts", [])
                if isinstance(raw_texts, str):
                    texts = [raw_texts]
                elif isinstance(raw_texts, list):
                    texts = [str(text) for text in raw_texts]
                else:
                    texts = []
                provider = str(payload.get("provider", "local")).strip() or "local"
                model = str(payload.get("model", "")).strip() or None
                allow_external = parse_bool(payload.get("allow_external"), default=False)
                try:
                    result = gateway.embed(
                        texts,
                        provider_id=provider,
                        model=model,
                        allow_external=allow_external,
                    )
                except (ProviderConfigurationError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(
                    {
                        "provider": result.provider,
                        "model": result.model,
                        "dimensions": result.dimensions,
                        "embeddings": result.embeddings,
                        "metadata": result.metadata,
                    },
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/workspace/items":
                payload = self.read_json()
                try:
                    item = workspace.create_item(
                        kind=str(payload.get("kind", "note")).strip() or "note",
                        title=str(payload.get("title", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        parent_id=str(payload.get("parent_id", "")).strip(),
                        content=str(payload.get("content", "")),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"item": item}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/workspace/items/update":
                payload = self.read_json()
                try:
                    item = workspace.update_item(
                        str(payload.get("item_id", "")).strip(),
                        title=str(payload.get("title", "")).strip() if "title" in payload else None,
                        content=str(payload.get("content", "")) if "content" in payload else None,
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"item": item})
                return
            if parsed.path == "/api/workspace/templates/instantiate":
                payload = self.read_json()
                try:
                    item = workspace.instantiate_template(
                        str(payload.get("template_id", "")).strip(),
                        title=str(payload.get("title", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        variables=payload.get("variables") if isinstance(payload.get("variables"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"item": item}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/workflows":
                payload = self.read_json()
                name = str(payload.get("name", "")).strip()
                steps = payload.get("steps", [])
                if not name:
                    self.send_json({"error": "A workflow name is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                if not isinstance(steps, list):
                    self.send_json({"error": "Workflow steps must be a list."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    workflow = workflows.create(
                        name=name,
                        steps=[step for step in steps if isinstance(step, dict)],
                        project=str(payload.get("project", "default")).strip() or "default",
                        description=str(payload.get("description", "")),
                        trigger_type=str(payload.get("trigger_type", "manual")),
                        trigger=payload.get("trigger") if isinstance(payload.get("trigger"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"workflow": workflow}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/workflows/run" or parsed.path.startswith("/api/workflows/") and parsed.path.endswith("/run"):
                payload = self.read_json()
                # Support both /api/workflows/run (body.workflow_id) and /api/workflows/{id}/run
                workflow_id = str(payload.get("workflow_id", "")).strip()
                if not workflow_id and parsed.path.endswith("/run"):
                    # Extract id from path /api/workflows/{id}/run
                    parts = parsed.path.strip("/").split("/")
                    if len(parts) == 4 and parts[:2] == ["api", "workflows"]:
                        workflow_id = parts[2]
                if not workflow_id:
                    self.send_json({"error": "workflow_id is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    run = workflows.run(
                        workflow_id,
                        allow_network=parse_bool(payload.get("allow_network"), default=False),
                        dry_run=parse_bool(payload.get("dry_run"), default=False),
                    )
                except (ValueError, WorkflowSafetyError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"run": run}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/workflows/run-due":
                payload = self.read_json()
                runs = workflows.run_due_schedules(
                    allow_network=parse_bool(payload.get("allow_network"), default=False)
                )
                self.send_json({"runs": runs}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/workflows/event":
                payload = self.read_json()
                event_name = str(payload.get("event_name", "")).strip()
                if not event_name:
                    self.send_json({"error": "event_name is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    result = workflows.trigger_event(
                        event_name,
                        payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                        project=str(payload.get("project", "default")).strip() or "default",
                        allow_network=parse_bool(payload.get("allow_network"), default=False),
                    )
                except (ValueError, WorkflowSafetyError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/workflows/layout":
                payload = self.read_json()
                try:
                    layout = workflows.save_layout(
                        str(payload.get("workflow_id", "")).strip(),
                        payload.get("layout") if isinstance(payload.get("layout"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"layout": layout}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/knowledge/ingest":
                payload = self.read_json()
                try:
                    document = knowledge.ingest_text(
                        str(payload.get("title", "")).strip(),
                        str(payload.get("content", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        source=str(payload.get("source", "manual")).strip() or "manual",
                        source_path=str(payload.get("source_path", "")).strip(),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"document": document}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/knowledge/upload":
                self._handle_file_upload()
                return
            if parsed.path == "/api/research/run":
                payload = self.read_json()
                query = str(payload.get("query", "")).strip()
                if not query:
                    self.send_json({"error": "query is required."}, status=HTTPStatus.BAD_REQUEST)
                    return
                project = str(payload.get("project", "default")).strip() or "default"
                try:
                    result = research_engine.run(
                        query,
                        project=project,
                        max_iterations=parse_int(str(payload.get("max_iterations", "3")), default=3),
                        provider=str(payload.get("provider", "auto")).strip() or "auto",
                        model=str(payload.get("model", "")).strip() or None,
                        allow_external=parse_bool(payload.get("allow_external"), default=False),
                    )
                except (ProviderConfigurationError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/research/sessions":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                try:
                    session = research_lifecycle.create(
                        str(payload.get("query", "")), project=project,
                        max_iterations=parse_int(str(payload.get("max_iterations", "3")), default=3),
                        allow_external=parse_bool(payload.get("allow_external"), default=False),
                        provider=str(payload.get("provider", "auto")).strip() or "auto",
                        model=str(payload.get("model", "")).strip(),
                        max_attempts=parse_int(str(payload.get("max_attempts", "3")), default=3),
                    )
                    job = worker_runtime.enqueue(
                        "research.run",
                        {
                            "session_id": session["id"], "query": session["query"],
                            "max_iterations": session["max_iterations"], "allow_external": session["allow_external"],
                            "provider": session["provider"], "model": session["model"],
                        },
                        project=project, priority=80, max_attempts=session["max_attempts"],
                    )
                    session = research_lifecycle.attach_job(session["id"], job["id"])
                    threading.Thread(target=worker_runtime.run_once, name=f"research-{session['id'][:8]}", daemon=True).start()
                except (KeyError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"session": session, "job": job}, status=HTTPStatus.ACCEPTED)
                return
            if parsed.path.startswith("/api/research/sessions/"):
                payload = self.read_json()
                suffix = unquote(parsed.path.removeprefix("/api/research/sessions/")).strip("/")
                action = suffix.rsplit("/", 1)[-1] if "/" in suffix else ""
                session_id = suffix.rsplit("/", 1)[0] if action else suffix
                try:
                    if action == "cancel":
                        session = research_lifecycle.cancel(session_id)
                    elif action == "archive":
                        session = research_lifecycle.archive(session_id, archived=parse_bool(payload.get("archived"), default=True))
                    elif action == "delete":
                        self.send_json(research_lifecycle.delete(session_id))
                        return
                    elif action in {"retry", "branch"}:
                        session = (
                            research_lifecycle.retry(session_id)
                            if action == "retry"
                            else research_lifecycle.branch(
                                session_id, query=str(payload.get("query", "")).strip(), label=str(payload.get("label", "")).strip()
                            )
                        )
                        job = worker_runtime.enqueue(
                            "research.run",
                            {
                                "session_id": session["id"], "query": session["query"],
                                "max_iterations": session["max_iterations"], "allow_external": session["allow_external"],
                                "provider": session["provider"], "model": session["model"],
                            },
                            project=session["project"], priority=80, max_attempts=session["max_attempts"],
                        )
                        session = research_lifecycle.attach_job(session["id"], job["id"])
                        threading.Thread(target=worker_runtime.run_once, name=f"research-{session['id'][:8]}", daemon=True).start()
                    else:
                        self.send_json({"error": "Unknown research session action."}, status=HTTPStatus.NOT_FOUND)
                        return
                except (KeyError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"session": session}, status=HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/models/discover":
                discovered = gateway.discover_models()
                self.send_json({"discovered": discovered})
                return
            if parsed.path == "/api/knowledge/semantic-search":
                payload = self.read_json()
                results = knowledge.semantic_search(
                    str(payload.get("query", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    limit=parse_int(str(payload.get("limit", "10")), default=10),
                )
                self.send_json({"results": results})
                return
            if parsed.path == "/api/knowledge/hybrid-search":
                payload = self.read_json()
                results = knowledge.hybrid_search(
                    str(payload.get("query", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    limit=parse_int(str(payload.get("limit", "10")), default=10),
                )
                self.send_json({"results": results})
                return
            if parsed.path == "/api/knowledge/synthesize":
                payload = self.read_json()
                result = knowledge.synthesize(
                    str(payload.get("query", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    limit=parse_int(str(payload.get("limit", "8")), default=8),
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/knowledge/compress":
                payload = self.read_json()
                result = knowledge.compress_memory(
                    project=str(payload.get("project", "default")).strip() or "default"
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/knowledge/replay":
                payload = self.read_json()
                replay = knowledge.create_replay(
                    project=str(payload.get("project", "default")).strip() or "default",
                    label=str(payload.get("label", "Replay")).strip() or "Replay",
                    query=str(payload.get("query", "")).strip(),
                )
                self.send_json({"replay": replay}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/semantic/memories":
                payload = self.read_json()
                try:
                    memory = semantic_memory.add_memory(
                        title=str(payload.get("title", "")).strip(),
                        content=str(payload.get("content", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        memory_type=str(payload.get("memory_type", "semantic")).strip() or "semantic",
                        profile_id=str(payload.get("profile_id", "")).strip(),
                        tags=[str(item) for item in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else [],
                        source=str(payload.get("source", "manual")).strip() or "manual",
                        confidence=float(payload.get("confidence", 1.0) or 1.0),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except (TypeError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"memory": memory}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/semantic/consolidate":
                payload = self.read_json()
                result = semantic_memory.consolidate(
                    project=str(payload.get("project", "default")).strip() or "default",
                    limit=parse_int(str(payload.get("limit", "100")), default=100),
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/semantic/collections":
                payload = self.read_json()
                collection = semantic_memory.create_collection(
                    str(payload.get("name", "default")).strip() or "default",
                    project=str(payload.get("project", "default")).strip() or "default",
                    dimensions=parse_int(str(payload.get("dimensions", "64")), default=64),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                )
                self.send_json({"collection": collection}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/semantic/vectors":
                payload = self.read_json()
                vector = payload.get("vector")
                try:
                    record = semantic_memory.upsert_vector_record(
                        str(payload.get("collection_id", "")).strip(),
                        content=str(payload.get("content", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        external_id=str(payload.get("external_id", "")).strip(),
                        vector=[float(item) for item in vector] if isinstance(vector, list) else None,
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except (TypeError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"record": record}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/semantic/vector-search":
                payload = self.read_json()
                try:
                    results = semantic_memory.vector_search(
                        str(payload.get("collection_id", "")).strip(),
                        str(payload.get("query", "")).strip(),
                        limit=parse_int(str(payload.get("limit", "10")), default=10),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"results": results})
                return
            if parsed.path == "/api/semantic/rerank":
                payload = self.read_json()
                candidates = payload.get("candidates", [])
                if not isinstance(candidates, list):
                    self.send_json({"error": "candidates must be a list."}, status=HTTPStatus.BAD_REQUEST)
                    return
                result = semantic_memory.rerank(
                    str(payload.get("query", "")).strip(),
                    [item for item in candidates if isinstance(item, dict)],
                    project=str(payload.get("project", "default")).strip() or "default",
                    limit=parse_int(str(payload.get("limit", "10")), default=10),
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/media/ingest":
                payload = self.read_json()
                try:
                    if payload.get("path"):
                        asset = media.ingest_file(
                            str(payload.get("path", "")).strip(),
                            project=str(payload.get("project", "default")).strip() or "default",
                            title=str(payload.get("title", "")).strip(),
                            source=str(payload.get("source", "file")).strip() or "file",
                            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                            copy_to_store=parse_bool(payload.get("copy_to_store"), default=True),
                        )
                    else:
                        raw = str(payload.get("data_base64", "")).strip()
                        if not raw:
                            self.send_json({"error": "path or data_base64 is required."}, status=HTTPStatus.BAD_REQUEST)
                            return
                        asset = media.ingest_bytes(
                            base64.b64decode(raw.encode("utf-8")),
                            filename=str(payload.get("filename", "upload.bin")).strip() or "upload.bin",
                            project=str(payload.get("project", "default")).strip() or "default",
                            title=str(payload.get("title", "")).strip(),
                            source=str(payload.get("source", "upload")).strip() or "upload",
                            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                            copy_to_store=parse_bool(payload.get("copy_to_store"), default=True),
                        )
                except (ValueError, OSError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"asset": asset}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/media/analyze":
                payload = self.read_json()
                try:
                    analysis = media.analyze_asset(str(payload.get("asset_id", "")).strip())
                except (ValueError, OSError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"analysis": analysis}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/agents/run-checked":
                payload = self.read_json()
                try:
                    result = multi_agent.run_agent(
                        str(payload.get("agent", "executive")).strip() or "executive",
                        str(payload.get("goal", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        require_approval=parse_bool(payload.get("require_approval"), default=False),
                        approval_id=str(payload.get("approval_id", "")).strip() or None,
                        verify=parse_bool(payload.get("verify"), default=True),
                    )
                except (ApprovalRequired, PermissionDenied, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/agents/delegate":
                payload = self.read_json()
                try:
                    result = multi_agent.delegate(
                        str(payload.get("from_agent", "executive")).strip() or "executive",
                        str(payload.get("to_agent", "planning")).strip() or "planning",
                        str(payload.get("goal", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except (PermissionDenied, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/agents/parallel":
                payload = self.read_json()
                tasks = payload.get("tasks", [])
                if not isinstance(tasks, list):
                    self.send_json({"error": "tasks must be a list."}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    result = multi_agent.run_parallel(
                        [task for task in tasks if isinstance(task, dict)],
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except (PermissionDenied, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/agents/permissions/grant":
                payload = self.read_json()
                permission = multi_agent.grant(
                    str(payload.get("agent", "")).strip(),
                    str(payload.get("scope", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    granted_by=str(payload.get("granted_by", "user")).strip() or "user",
                )
                self.send_json({"permission": permission}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/agents/approvals/request":
                payload = self.read_json()
                approval = multi_agent.request_approval(
                    requested_by=str(payload.get("requested_by", "user")).strip() or "user",
                    action=str(payload.get("action", "")).strip(),
                    reason=str(payload.get("reason", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                )
                self.send_json({"approval": approval}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/agents/approvals/resolve":
                payload = self.read_json()
                try:
                    approval = multi_agent.resolve_approval(
                        str(payload.get("approval_id", "")).strip(),
                        approved=parse_bool(payload.get("approved"), default=False),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"approval": approval})
                return
            if parsed.path == "/api/assistant/profile":
                payload = self.read_json()
                profile = assistant.upsert_profile(
                    user_id=str(payload.get("user_id", "local-user")).strip() or "local-user",
                    project=str(payload.get("project", "default")).strip() or "default",
                    preferences=payload.get("preferences") if isinstance(payload.get("preferences"), dict) else {},
                    facts=payload.get("facts") if isinstance(payload.get("facts"), dict) else {},
                )
                self.send_json({"profile": profile}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/assistant/goals":
                payload = self.read_json()
                try:
                    goal = assistant.create_goal(
                        title=str(payload.get("title", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        description=str(payload.get("description", "")),
                        horizon=str(payload.get("horizon", "short")).strip() or "short",
                        status=str(payload.get("status", "active")).strip() or "active",
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"goal": goal}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/assistant/tasks":
                payload = self.read_json()
                try:
                    task = assistant.create_task(
                        title=str(payload.get("title", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        priority=str(payload.get("priority", "normal")).strip() or "normal",
                        due_at=str(payload.get("due_at", "")).strip(),
                        goal_id=str(payload.get("goal_id", "")).strip(),
                        contact_id=str(payload.get("contact_id", "")).strip(),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"task": task}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/assistant/tasks/complete":
                payload = self.read_json()
                try:
                    task = assistant.complete_task(str(payload.get("task_id", "")).strip())
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"task": task})
                return
            if parsed.path == "/api/assistant/meeting/brief":
                payload = self.read_json()
                try:
                    brief = assistant.create_meeting_brief(
                        title=str(payload.get("title", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        starts_at=str(payload.get("starts_at", "")).strip(),
                        attendees=[str(item) for item in payload.get("attendees", [])] if isinstance(payload.get("attendees"), list) else [],
                        notes=str(payload.get("notes", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"brief": brief}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/assistant/meeting/summary":
                payload = self.read_json()
                result = assistant.summarize_meeting(
                    transcript=str(payload.get("transcript", "")),
                    project=str(payload.get("project", "default")).strip() or "default",
                    title=str(payload.get("title", "Meeting summary")).strip() or "Meeting summary",
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/assistant/travel/plan":
                payload = self.read_json()
                try:
                    plan = assistant.create_travel_plan(
                        destination=str(payload.get("destination", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        starts_at=str(payload.get("starts_at", "")).strip(),
                        ends_at=str(payload.get("ends_at", "")).strip(),
                        preferences=payload.get("preferences") if isinstance(payload.get("preferences"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"plan": plan}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/assistant/crm":
                payload = self.read_json()
                try:
                    record = assistant.upsert_crm_record(
                        record_type=str(payload.get("record_type", "relationship")).strip() or "relationship",
                        name=str(payload.get("name", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        status=str(payload.get("status", "active")).strip() or "active",
                        value=float(payload.get("value", 0) or 0),
                        contact=str(payload.get("contact", "")),
                        notes=str(payload.get("notes", "")),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except (TypeError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"record": record}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/communication/connectors/configure":
                payload = self.read_json()
                connector = communication.configure_connector(
                    str(payload.get("platform", "")).strip(),
                    enabled=parse_bool(payload.get("enabled"), default=False),
                    config=payload.get("config") if isinstance(payload.get("config"), dict) else {},
                )
                self.send_json({"connector": connector})
                return
            if parsed.path == "/api/communication/contacts":
                payload = self.read_json()
                contact = communication.upsert_contact(
                    str(payload.get("name", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    owner=self._payload_owner(payload),
                    handles=payload.get("handles") if isinstance(payload.get("handles"), dict) else {},
                    notes=str(payload.get("notes", "")),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                )
                self.send_json({"contact": contact}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/communication/messages/import":
                payload = self.read_json()
                try:
                    message = communication.import_message(
                        platform=str(payload.get("platform", "email")).strip() or "email",
                        sender=str(payload.get("sender", "")).strip(),
                        recipients=[str(item) for item in payload.get("recipients", [])] if isinstance(payload.get("recipients"), list) else [],
                        subject=str(payload.get("subject", "")),
                        body=str(payload.get("body", "")),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload),
                        thread_id=str(payload.get("thread_id", "")),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"message": message}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/communication/drafts/reply":
                payload = self.read_json()
                try:
                    draft = communication.draft_reply(
                        str(payload.get("message_id", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload),
                        instructions=str(payload.get("instructions", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"draft": draft}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/communication/drafts":
                payload = self.read_json()
                draft = communication.create_draft(
                    platform=str(payload.get("platform", "email")).strip() or "email",
                    recipients=[str(item) for item in payload.get("recipients", [])] if isinstance(payload.get("recipients"), list) else [],
                    subject=str(payload.get("subject", "")),
                    body=str(payload.get("body", "")),
                    project=str(payload.get("project", "default")).strip() or "default",
                    owner=self._payload_owner(payload),
                )
                self.send_json({"draft": draft}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/communication/send/request-approval":
                payload = self.read_json()
                try:
                    approval = communication.request_send_approval(
                        str(payload.get("draft_id", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload),
                        reason=str(payload.get("reason", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"approval": approval}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/communication/send/resolve":
                payload = self.read_json()
                try:
                    approval = communication.resolve_send_approval(
                        str(payload.get("approval_id", "")).strip(),
                        approved=parse_bool(payload.get("approved"), default=False),
                        owner=self._payload_owner(payload, default=""),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"approval": approval})
                return
            if parsed.path == "/api/communication/send":
                payload = self.read_json()
                try:
                    sent = communication.send_draft(
                        str(payload.get("draft_id", "")).strip(),
                        approval_id=str(payload.get("approval_id", "")).strip() or None,
                        allow_external=parse_bool(payload.get("allow_external"), default=False),
                        owner=self._payload_owner(payload, default=""),
                    )
                except (CommunicationApprovalRequired, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"sent": sent}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/communication/calendar":
                payload = self.read_json()
                event = communication.add_calendar_event(
                    title=str(payload.get("title", "")).strip(),
                    starts_at=str(payload.get("starts_at", "")).strip(),
                    ends_at=str(payload.get("ends_at", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    owner=self._payload_owner(payload),
                    calendar_id=str(payload.get("calendar_id", "local")).strip() or "local",
                    attendees=[str(item) for item in payload.get("attendees", [])] if isinstance(payload.get("attendees"), list) else [],
                    source=str(payload.get("source", "local")).strip() or "local",
                )
                self.send_json({"event": event}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/communication/sync":
                payload = self.read_json()
                try:
                    run = communication.sync_connector(
                        str(payload.get("platform", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload),
                        payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                        allow_external=parse_bool(payload.get("allow_external"), default=False),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"run": run}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/voice/personas":
                payload = self.read_json()
                persona = voice.upsert_persona(
                    name=str(payload.get("name", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    language=str(payload.get("language", "en")).strip() or "en",
                    style=str(payload.get("style", "clear")).strip() or "clear",
                    voiceprint_asset_id=str(payload.get("voiceprint_asset_id", "")).strip(),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                )
                self.send_json({"persona": persona}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/voice/sessions":
                payload = self.read_json()
                try:
                    session = voice.create_session(
                        project=str(payload.get("project", "default")).strip() or "default",
                        mode=str(payload.get("mode", "push_to_talk")).strip() or "push_to_talk",
                        persona_id=str(payload.get("persona_id", "")).strip(),
                        language=str(payload.get("language", "en")).strip() or "en",
                        wake_word=str(payload.get("wake_word", "")).strip(),
                        always_listening=parse_bool(payload.get("always_listening"), default=False),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"session": session}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/voice/vosk/setup":
                payload = self.read_json()
                try:
                    result = voice.setup_vosk_model(
                        allow_network=parse_bool(payload.get("allow_network"), default=False),
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc), "status": voice.status(project=str(payload.get("project", "default")).strip() or "default")}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/voice/clone":
                payload = self.read_json()
                try:
                    audio_data = base64.b64decode(str(payload.get("audio_base64", "")).encode("ascii"), validate=True)
                    clone = voice.clone_voice_from_audio(
                        name=str(payload.get("name", "")).strip(),
                        audio_data=audio_data,
                        filename=str(payload.get("filename", "voiceprint.wav")).strip() or "voiceprint.wav",
                        project=str(payload.get("project", "default")).strip() or "default",
                        language=str(payload.get("language", "en")).strip() or "en",
                        style=str(payload.get("style", "cloned-local")).strip() or "cloned-local",
                        consent=str(payload.get("consent", "")).strip(),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except (ValueError, binascii.Error) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"clone": clone}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/voice/sessions/close":
                payload = self.read_json()
                try:
                    session = voice.close_session(str(payload.get("session_id", "")).strip())
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"session": session})
                return
            if parsed.path == "/api/voice/transcript":
                payload = self.read_json()
                try:
                    turn = voice.add_transcript(
                        str(payload.get("session_id", "")).strip(),
                        text=str(payload.get("text", "")).strip(),
                        speaker=str(payload.get("speaker", "user")).strip() or "user",
                        role=str(payload.get("role", "user")).strip() or "user",
                        audio_asset_id=str(payload.get("audio_asset_id", "")).strip(),
                        is_final=parse_bool(payload.get("is_final"), default=True),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"turn": turn}, status=HTTPStatus.CREATED)
                return
            if parsed.path in {"/api/voice/transcribe-upload", "/api/voice/upload"}:
                self._handle_voice_upload()
                return
            if parsed.path == "/api/voice/stt":
                payload = self.read_json()
                try:
                    audio_data = base64.b64decode(str(payload.get("audio_base64", "")).encode("ascii"), validate=True)
                    transcription = voice.transcribe_audio(
                        str(payload.get("session_id", "")).strip(),
                        audio_data=audio_data,
                        filename=str(payload.get("filename", "speech.wav")).strip() or "speech.wav",
                        speaker=str(payload.get("speaker", "user")).strip() or "user",
                        transcript_hint=str(payload.get("transcript_hint", "")).strip(),
                        language=str(payload.get("language", "en")).strip() or "en",
                        stt_engine=str(payload.get("stt_engine", "")).strip(),
                        allow_fallback_stt=parse_bool(payload.get("allow_fallback_stt"), default=False),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except (ValueError, binascii.Error) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"transcription": transcription}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/voice/tts":
                payload = self.read_json()
                try:
                    script = voice.synthesize_text(
                        text=str(payload.get("text", "")).strip(),
                        session_id=str(payload.get("session_id", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        persona_id=str(payload.get("persona_id", "")).strip(),
                        voice=str(payload.get("voice", "local-script")).strip() or "local-script",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"script": script}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/voice/translate":
                payload = self.read_json()
                try:
                    translation = voice.translate_text(
                        text=str(payload.get("text", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        source_language=str(payload.get("source_language", "auto")).strip() or "auto",
                        target_language=str(payload.get("target_language", "en")).strip() or "en",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"translation": translation}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/policies":
                payload = self.read_json()
                try:
                    rule = production.upsert_policy_rule(
                        scope=str(payload.get("scope", "")).strip(),
                        effect=str(payload.get("effect", "deny")).strip() or "deny",
                        project=str(payload.get("project", "default")).strip() or "default",
                        conditions=payload.get("conditions") if isinstance(payload.get("conditions"), dict) else {},
                        priority=parse_int(str(payload.get("priority", "100")), default=100),
                        description=str(payload.get("description", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"rule": rule}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/capabilities/hard-audit":
                payload = self.read_json()
                result = run_hard_audit(
                    store,
                    project=str(payload.get("project", "hard-audit")).strip() or "hard-audit",
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/mission/run-check":
                payload = self.read_json()
                result = mission_control.run_full_check(
                    project=str(payload.get("project", "default")).strip() or "default",
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/policies/evaluate":
                payload = self.read_json()
                result = production.evaluate_policy(
                    scope=str(payload.get("scope", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    context=payload.get("context") if isinstance(payload.get("context"), dict) else {},
                )
                self.send_json(result)
                return
            if parsed.path == "/api/production/queue":
                payload = self.read_json()
                job = production.enqueue_job(
                    kind=str(payload.get("kind", "noop")).strip() or "noop",
                    payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                    project=str(payload.get("project", "default")).strip() or "default",
                    queue=str(payload.get("queue", "default")).strip() or "default",
                    max_attempts=parse_int(str(payload.get("max_attempts", "3")), default=3),
                    next_run_at=str(payload.get("next_run_at", "")).strip(),
                )
                self.send_json({"job": job}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/queue/run-next":
                payload = self.read_json()
                result = production.run_next_job(
                    queue=str(payload.get("queue", "default")).strip() or "default",
                    project=str(payload.get("project", "default")).strip() or "default",
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/audit-anchor":
                payload = self.read_json()
                try:
                    anchor = production.anchor_audit_log(
                        project=str(payload.get("project", "default")).strip() or "default",
                        storage_mode=str(payload.get("storage_mode", "local-append-only")).strip() or "local-append-only",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"anchor": anchor}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/vector/indexes":
                payload = self.read_json()
                index = production.create_vector_index(
                    name=str(payload.get("name", "default")).strip() or "default",
                    project=str(payload.get("project", "default")).strip() or "default",
                    dimensions=parse_int(str(payload.get("dimensions", "64")), default=64),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                )
                self.send_json({"index": index}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/vector/upsert":
                payload = self.read_json()
                vector = payload.get("vector")
                try:
                    record = production.upsert_vector(
                        str(payload.get("index_id", "")).strip(),
                        external_id=str(payload.get("external_id", "")).strip(),
                        content=str(payload.get("content", "")).strip(),
                        namespace=str(payload.get("namespace", "default")).strip() or "default",
                        vector=[float(item) for item in vector] if isinstance(vector, list) else None,
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except (TypeError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"record": record}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/vector/search":
                payload = self.read_json()
                try:
                    results = production.vector_search(
                        str(payload.get("index_id", "")).strip(),
                        query=str(payload.get("query", "")).strip(),
                        namespace=str(payload.get("namespace", "default")).strip() or "default",
                        limit=parse_int(str(payload.get("limit", "10")), default=10),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"results": results})
                return
            if parsed.path == "/api/production/semantic/science":
                payload = self.read_json()
                self.send_json(
                    production.semantic_memory_science(
                        project=str(payload.get("project", "default")).strip() or "default"
                    )
                )
                return
            if parsed.path == "/api/production/rag/rerank":
                payload = self.read_json()
                candidates = payload.get("candidates", [])
                if not isinstance(candidates, list):
                    self.send_json({"error": "candidates must be a list."}, status=HTTPStatus.BAD_REQUEST)
                    return
                result = production.rerank_sources(
                    query=str(payload.get("query", "")).strip(),
                    candidates=[item for item in candidates if isinstance(item, dict)],
                    project=str(payload.get("project", "default")).strip() or "default",
                    limit=parse_int(str(payload.get("limit", "10")), default=10),
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/oidc/validate":
                payload = self.read_json()
                result = production.validate_oidc_jwks(
                    token=str(payload.get("token", "")).strip(),
                    jwks=payload.get("jwks") if isinstance(payload.get("jwks"), dict) else {},
                    issuer=str(payload.get("issuer", "")).strip(),
                    audience=str(payload.get("audience", "")).strip(),
                    project=str(payload.get("project", "security")).strip() or "security",
                )
                self.send_json(result)
                return
            if parsed.path == "/api/production/crypto/encrypt":
                payload = self.read_json()
                try:
                    envelope = production.encrypt_envelope(
                        label=str(payload.get("label", "Envelope")).strip() or "Envelope",
                        plaintext=str(payload.get("plaintext", "")),
                        passphrase=str(payload.get("passphrase", "")),
                        recipients=[str(item) for item in payload.get("recipients", [])] if isinstance(payload.get("recipients"), list) else [],
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"envelope": envelope}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/crypto/decrypt":
                payload = self.read_json()
                try:
                    envelope = production.decrypt_envelope(
                        str(payload.get("envelope_id", "")).strip(),
                        passphrase=str(payload.get("passphrase", "")),
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"envelope": envelope})
                return
            if parsed.path == "/api/production/connectors/adapters":
                payload = self.read_json()
                adapter = production.register_connector_adapter(
                    platform=str(payload.get("platform", "")).strip(),
                    mode=str(payload.get("mode", "local_fixture")).strip() or "local_fixture",
                    endpoint_url=str(payload.get("endpoint_url", "")).strip(),
                    credentials_ref=str(payload.get("credentials_ref", "")).strip(),
                    enabled=parse_bool(payload.get("enabled"), default=True),
                    config=payload.get("config") if isinstance(payload.get("config"), dict) else {},
                )
                self.send_json({"adapter": adapter}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/connectors/poll":
                payload = self.read_json()
                try:
                    run = production.poll_connector(
                        str(payload.get("adapter_id", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        allow_external=parse_bool(payload.get("allow_external"), default=False),
                    )
                except (ValueError, ProductionRuntimeError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"run": run}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/automation/desktop":
                payload = self.read_json()
                actions = payload.get("actions", [])
                run = production.run_desktop_automation(
                    actions=[item for item in actions if isinstance(item, dict)] if isinstance(actions, list) else [],
                    project=str(payload.get("project", "default")).strip() or "default",
                    allow_desktop=parse_bool(payload.get("allow_desktop"), default=False),
                )
                self.send_json({"run": run}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/automation/browser":
                payload = self.read_json()
                actions = payload.get("actions", [])
                run = production.run_browser_automation(
                    actions=[item for item in actions if isinstance(item, dict)] if isinstance(actions, list) else [],
                    project=str(payload.get("project", "default")).strip() or "default",
                    html=str(payload.get("html", "")),
                )
                self.send_json({"run": run}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/packages/app":
                payload = self.read_json()
                try:
                    package = production.package_app(
                        app_type=str(payload.get("app_type", "pwa")).strip() or "pwa",
                        platform=str(payload.get("platform", "web")).strip() or "web",
                        project=str(payload.get("project", "default")).strip() or "default",
                        manifest=payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"package": package}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/marketplace/agents":
                payload = self.read_json()
                package = production.publish_agent_package(
                    name=str(payload.get("name", "")).strip(),
                    version=str(payload.get("version", "0.1.0")).strip() or "0.1.0",
                    author=str(payload.get("author", "")),
                    project=str(payload.get("project", "default")).strip() or "default",
                    manifest=payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {},
                )
                self.send_json({"package": package}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/gpu/plan":
                payload = self.read_json()
                plan = production.plan_gpu_management(
                    model_id=str(payload.get("model_id", "openagent-local")).strip() or "openagent-local",
                    project=str(payload.get("project", "default")).strip() or "default",
                    target_tokens_per_second=float(payload.get("target_tokens_per_second", 20) or 20),
                )
                self.send_json({"plan": plan}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/finetune/jobs":
                payload = self.read_json()
                job = production.create_finetune_job(
                    model_id=str(payload.get("model_id", "")).strip(),
                    dataset_ref=str(payload.get("dataset_ref", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    job_type=str(payload.get("job_type", "lora")).strip() or "lora",
                    hyperparameters=payload.get("hyperparameters") if isinstance(payload.get("hyperparameters"), dict) else {},
                )
                self.send_json({"job": job}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/proactive/jobs":
                payload = self.read_json()
                job = production.create_proactive_job(
                    name=str(payload.get("name", "")).strip(),
                    query=str(payload.get("query", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    cadence_seconds=parse_int(str(payload.get("cadence_seconds", "3600")), default=3600),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                )
                self.send_json({"job": job}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/proactive/run":
                payload = self.read_json()
                try:
                    run = production.run_proactive_job(str(payload.get("job_id", "")).strip())
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"run": run}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/federated/rounds":
                payload = self.read_json()
                updates = payload.get("updates", [])
                try:
                    round_result = production.aggregate_federated_round(
                        objective=str(payload.get("objective", "")).strip(),
                        updates=[{str(k): float(v) for k, v in item.items()} for item in updates if isinstance(item, dict)] if isinstance(updates, list) else [],
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except (TypeError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"round": round_result}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/sync/devices":
                payload = self.read_json()
                device = production.register_sync_device(
                    name=str(payload.get("name", "")).strip(),
                    public_key=str(payload.get("public_key", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    trust_level=str(payload.get("trust_level", "trusted")).strip() or "trusted",
                )
                self.send_json({"device": device}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/sync/bundles":
                payload = self.read_json()
                bundle = production.create_sync_bundle(
                    from_device_id=str(payload.get("from_device_id", "")).strip(),
                    to_device_id=str(payload.get("to_device_id", "")).strip(),
                    data=payload.get("data") if isinstance(payload.get("data"), dict) else {},
                    passphrase=str(payload.get("passphrase", "")),
                    project=str(payload.get("project", "default")).strip() or "default",
                )
                self.send_json({"bundle": bundle}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/sdk/publish":
                payload = self.read_json()
                package = production.publish_sdk_package(
                    language=str(payload.get("language", "python")).strip() or "python",
                    version=str(payload.get("version", "0.1.0")).strip() or "0.1.0",
                    project=str(payload.get("project", "default")).strip() or "default",
                )
                self.send_json({"package": package}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/production/sandbox/run":
                payload = self.read_json()
                run = production.run_sandboxed_plugin(
                    plugin_id=str(payload.get("plugin_id", "")).strip(),
                    profile_id=str(payload.get("profile_id", "")).strip(),
                    input=payload.get("input") if isinstance(payload.get("input"), dict) else {},
                    project=str(payload.get("project", "default")).strip() or "default",
                )
                self.send_json({"run": run}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/local-ai/models/register":
                payload = self.read_json()
                model = local_ai.register_model(
                    name=str(payload.get("name", "")).strip(),
                    provider=str(payload.get("provider", "local")).strip() or "local",
                    source_url=str(payload.get("source_url", "")).strip(),
                    local_path=str(payload.get("local_path", "")).strip(),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                )
                self.send_json({"model": model}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/local-ai/models/download":
                payload = self.read_json()
                try:
                    model = local_ai.download_model(
                        str(payload.get("model_id", "")).strip(),
                        allow_network=parse_bool(payload.get("allow_network"), default=False),
                    )
                except (ValueError, ModelDownloadError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"model": model}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/local-ai/ollama/pull":
                payload = self.read_json()
                try:
                    result = local_ai.pull_ollama_model(
                        str(payload.get("model", "")).strip(),
                        allow_network=parse_bool(payload.get("allow_network"), default=False),
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except ModelDownloadError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/local-ai/models/setup-chat":
                payload = self.read_json()
                try:
                    result = local_ai.setup_chat_model(
                        model_name=str(payload.get("model", "")).strip(),
                        allow_network=parse_bool(payload.get("allow_network"), default=False),
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except ModelDownloadError as exc:
                    self.send_json({"error": str(exc), "tool_status": local_ai.model_tool_status()}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/local-ai/models/setup-chat/job":
                payload = self.read_json()
                job = local_ai.create_setup_chat_job(
                    model_name=str(payload.get("model", "")).strip(),
                    allow_network=parse_bool(payload.get("allow_network"), default=False),
                    project=str(payload.get("project", "default")).strip() or "default",
                    task="chat",
                )
                self.send_json({"job": job}, status=HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/local-ai/models/setup/job":
                payload = self.read_json()
                task = str(payload.get("task", "chat")).strip().lower() or "chat"
                if task not in {"chat", "reasoning", "research", "code", "speed"}:
                    self.send_json({"error": "Unsupported model setup task."}, status=HTTPStatus.BAD_REQUEST)
                    return
                job = local_ai.create_setup_chat_job(
                    model_name=str(payload.get("model", "")).strip(),
                    allow_network=parse_bool(payload.get("allow_network"), default=False),
                    project=str(payload.get("project", "default")).strip() or "default",
                    task=task,
                )
                self.send_json({"job": job}, status=HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/local-ai/jobs/cancel":
                payload = self.read_json()
                try:
                    job = local_ai.cancel_model_job(str(payload.get("job_id", "")).strip())
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"job": job})
                return
            if parsed.path == "/api/local-ai/llamacpp/register":
                payload = self.read_json()
                try:
                    result = local_ai.register_llamacpp_model(
                        local_path=str(payload.get("local_path", "")).strip(),
                        name=str(payload.get("name", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc), "tool_status": local_ai.model_tool_status()}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/local-ai/models/quantize":
                payload = self.read_json()
                try:
                    quantization = local_ai.add_quantization(
                        str(payload.get("model_id", "")).strip(),
                        method=str(payload.get("method", "static")).strip() or "static",
                        bits=parse_int(str(payload.get("bits", "4")), default=4),
                        format=str(payload.get("format", "gguf")).strip() or "gguf",
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"quantization": quantization}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/local-ai/models/auto-quantize":
                payload = self.read_json()
                try:
                    result = local_ai.auto_quantize(
                        str(payload.get("model_id", "")).strip(),
                        task=str(payload.get("task", "chat")).strip() or "chat",
                        target_ram_gb=float(payload["target_ram_gb"]) if payload.get("target_ram_gb") not in {None, ""} else None,
                        quantization_mode=str(payload.get("quantization_mode", "auto")).strip() or "auto",
                        parameter_count_b=float(payload["parameter_count_b"]) if payload.get("parameter_count_b") not in {None, ""} else None,
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/local-ai/models/delete":
                payload = self.read_json()
                try:
                    result = local_ai.delete_model(
                        str(payload.get("model_id", "")).strip(),
                        delete_file=parse_bool(payload.get("delete_file"), default=False),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/local-ai/benchmark":
                payload = self.read_json()
                try:
                    benchmark = local_ai.benchmark(
                        str(payload.get("model_id", "")).strip(),
                        prompt=str(payload.get("prompt", "Hello OpenAgent")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"benchmark": benchmark}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/local-ai/cluster-plan":
                payload = self.read_json()
                try:
                    plan = local_ai.plan_cluster(
                        name=str(payload.get("name", "Local cluster")).strip() or "Local cluster",
                        model_id=str(payload.get("model_id", "")).strip(),
                        nodes=parse_int(str(payload.get("nodes", "1")), default=1),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"cluster": plan}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/users":
                payload = self.read_json()
                try:
                    user = enterprise.upsert_user(
                        email=str(payload.get("email", "")).strip(),
                        name=str(payload.get("name", "")).strip(),
                        status=str(payload.get("status", "active")).strip() or "active",
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"user": user}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/roles":
                payload = self.read_json()
                try:
                    role = enterprise.upsert_role(
                        role_id=str(payload.get("role_id", "")).strip(),
                        name=str(payload.get("name", "")).strip(),
                        description=str(payload.get("description", "")),
                        permissions=[str(item) for item in payload.get("permissions", [])] if isinstance(payload.get("permissions"), list) else [],
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"role": role}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/workspaces":
                payload = self.read_json()
                try:
                    enterprise_workspace = enterprise.create_workspace(
                        name=str(payload.get("name", "")).strip(),
                        owner_user_id=str(payload.get("owner_user_id", "")).strip(),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"workspace": enterprise_workspace}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/workspace/members":
                payload = self.read_json()
                try:
                    member = enterprise.add_member(
                        workspace_id=str(payload.get("workspace_id", "")).strip(),
                        user_id=str(payload.get("user_id", "")).strip(),
                        role_id=str(payload.get("role_id", "member")).strip() or "member",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"member": member}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/access/check":
                payload = self.read_json()
                result = enterprise.check_access(
                    user_id=str(payload.get("user_id", "")).strip(),
                    workspace_id=str(payload.get("workspace_id", "")).strip(),
                    permission=str(payload.get("permission", "")).strip(),
                )
                self.send_json(result)
                return
            if parsed.path == "/api/enterprise/org-memory":
                payload = self.read_json()
                try:
                    memory = enterprise.add_org_memory(
                        workspace_id=str(payload.get("workspace_id", "")).strip(),
                        title=str(payload.get("title", "")).strip(),
                        content=str(payload.get("content", "")).strip(),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"memory": memory}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/policies":
                payload = self.read_json()
                try:
                    policy = enterprise.upsert_policy(
                        name=str(payload.get("name", "")).strip(),
                        description=str(payload.get("description", "")),
                        rules=[rule for rule in payload.get("rules", []) if isinstance(rule, dict)] if isinstance(payload.get("rules"), list) else [],
                        enforcement=str(payload.get("enforcement", "enforce")).strip() or "enforce",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"policy": policy}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/policies/evaluate":
                payload = self.read_json()
                result = enterprise.evaluate_policy(
                    scope=str(payload.get("scope", "")).strip(),
                    context=payload.get("context") if isinstance(payload.get("context"), dict) else {},
                )
                self.send_json(result)
                return
            if parsed.path == "/api/enterprise/sso":
                payload = self.read_json()
                try:
                    config = enterprise.configure_sso(
                        provider=str(payload.get("provider", "")).strip(),
                        issuer=str(payload.get("issuer", "")).strip(),
                        client_id=str(payload.get("client_id", "")).strip(),
                        enabled=parse_bool(payload.get("enabled"), default=True),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"config": config}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/sso/validate":
                payload = self.read_json()
                result = enterprise.validate_sso_token(
                    token=str(payload.get("token", "")).strip(),
                    config_id=str(payload.get("config_id", "")).strip(),
                    provider=str(payload.get("provider", "")).strip(),
                    audience=str(payload.get("audience", "")).strip(),
                )
                self.send_json(result)
                return
            if parsed.path == "/api/enterprise/audit/export":
                payload = self.read_json()
                export = enterprise.export_audit(
                    project=str(payload.get("project", "")).strip() or None,
                )
                self.send_json({"export": export}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/backups":
                backup = enterprise.create_backup()
                self.send_json({"backup": backup}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/enterprise/disaster-recovery":
                payload = self.read_json()
                plan = enterprise.create_disaster_recovery_plan(
                    name=str(payload.get("name", "Default DR Plan")).strip() or "Default DR Plan",
                    rpo_minutes=parse_int(str(payload.get("rpo_minutes", "60")), default=60),
                    rto_minutes=parse_int(str(payload.get("rto_minutes", "240")), default=240),
                )
                self.send_json({"plan": plan}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/developer/api-keys":
                payload = self.read_json()
                try:
                    key = developer.create_api_key(
                        label=str(payload.get("label", "")).strip(),
                        scopes=[str(item) for item in payload.get("scopes", [])] if isinstance(payload.get("scopes"), list) else None,
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"api_key": key}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/developer/api-keys/verify":
                payload = self.read_json()
                result = developer.verify_api_key(
                    str(payload.get("key", "")),
                    scope=str(payload.get("scope", "")).strip() or None,
                )
                self.send_json(result)
                return
            if parsed.path == "/api/developer/webhooks":
                payload = self.read_json()
                try:
                    webhook = developer.register_webhook(
                        event=str(payload.get("event", "*")).strip() or "*",
                        target_url=str(payload.get("target_url", "")).strip(),
                        enabled=parse_bool(payload.get("enabled"), default=True),
                        secret_hint=str(payload.get("secret_hint", "")),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"webhook": webhook}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/developer/webhooks/dispatch":
                payload = self.read_json()
                try:
                    result = developer.dispatch_webhook(
                        event=str(payload.get("event", "")).strip(),
                        payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                        allow_network=parse_bool(payload.get("allow_network"), default=False),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/developer/plugins":
                payload = self.read_json()
                try:
                    plugin = developer.register_plugin(payload.get("manifest") if isinstance(payload.get("manifest"), dict) else payload)
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"plugin": plugin}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/developer/tools":
                payload = self.read_json()
                try:
                    tool = developer.register_tool(
                        tool_id=str(payload.get("tool_id", "")).strip(),
                        name=str(payload.get("name", "")).strip(),
                        description=str(payload.get("description", "")),
                        schema=payload.get("schema") if isinstance(payload.get("schema"), dict) else {},
                        handler=str(payload.get("handler", "registered")).strip() or "registered",
                        permissions=[str(item) for item in payload.get("permissions", [])] if isinstance(payload.get("permissions"), list) else [],
                        enabled=parse_bool(payload.get("enabled"), default=True),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"tool": tool}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/developer/tools/invoke":
                payload = self.read_json()
                try:
                    invocation = developer.invoke_tool(
                        str(payload.get("tool_id", "")).strip(),
                        input_data=payload.get("input") if isinstance(payload.get("input"), dict) else {},
                        approved=parse_bool(payload.get("approved"), default=False),
                    )
                except (DeveloperPlatformError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"invocation": invocation}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/developer/mcp/servers":
                payload = self.read_json()
                try:
                    server = developer.register_mcp_server(
                        name=str(payload.get("name", "")).strip(),
                        command=str(payload.get("command", "")).strip(),
                        args=[str(item) for item in payload.get("args", [])] if isinstance(payload.get("args"), list) else [],
                        env={str(key): str(value) for key, value in payload.get("env", {}).items()} if isinstance(payload.get("env"), dict) else {},
                        enabled=parse_bool(payload.get("enabled"), default=True),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"server": server}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/developer/extensions":
                payload = self.read_json()
                try:
                    package = developer.publish_extension_package(
                        package_type=str(payload.get("package_type", "")).strip(),
                        name=str(payload.get("name", "")).strip(),
                        version=str(payload.get("version", "")).strip(),
                        source=str(payload.get("source", "local")).strip() or "local",
                        status=str(payload.get("status", "available")).strip() or "available",
                        manifest=payload.get("manifest") if isinstance(payload.get("manifest"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"package": package}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/developer/sdk":
                payload = self.read_json()
                try:
                    artifact = developer.generate_sdk(
                        language=str(payload.get("language", "python")).strip() or "python",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"artifact": artifact}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/security/secrets":
                payload = self.read_json()
                try:
                    secret = security_privacy.store_secret(
                        name=str(payload.get("name", "")).strip(),
                        value=str(payload.get("value", "")),
                        passphrase=str(payload.get("passphrase", "")),
                        description=str(payload.get("description", "")),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"secret": secret}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/security/secrets/reveal":
                payload = self.read_json()
                try:
                    secret = security_privacy.reveal_secret(
                        secret_id=str(payload.get("secret_id", "")).strip(),
                        passphrase=str(payload.get("passphrase", "")),
                    )
                except (SecurityPrivacyError, ValueError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"secret": secret})
                return
            if parsed.path == "/api/security/secrets/rotate":
                payload = self.read_json()
                try:
                    secret = security_privacy.rotate_secret(
                        secret_id=str(payload.get("secret_id", "")).strip(),
                        value=str(payload.get("value", "")),
                        passphrase=str(payload.get("passphrase", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"secret": secret})
                return
            if parsed.path == "/api/security/retention":
                payload = self.read_json()
                try:
                    policy = security_privacy.create_retention_policy(
                        resource_type=str(payload.get("resource_type", "")).strip(),
                        retention_days=parse_int(str(payload.get("retention_days", "365")), default=365),
                        action=str(payload.get("action", "delete")).strip() or "delete",
                        status=str(payload.get("status", "active")).strip() or "active",
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"policy": policy}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/security/retention/apply":
                payload = self.read_json()
                try:
                    result = security_privacy.apply_retention(
                        policy_id=str(payload.get("policy_id", "")).strip(),
                        dry_run=parse_bool(payload.get("dry_run"), default=True),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/security/shares":
                payload = self.read_json()
                try:
                    grant = security_privacy.create_share_grant(
                        resource_type=str(payload.get("resource_type", "")).strip(),
                        resource_id=str(payload.get("resource_id", "")).strip(),
                        grantee=str(payload.get("grantee", "")).strip(),
                        scope=str(payload.get("scope", "read")).strip() or "read",
                        expires_at=str(payload.get("expires_at", "")).strip(),
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"grant": grant}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/security/shares/resolve":
                payload = self.read_json()
                try:
                    grant = security_privacy.resolve_share_grant(
                        grant_id=str(payload.get("grant_id", "")).strip(),
                        status=str(payload.get("status", "revoked")).strip() or "revoked",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"grant": grant})
                return
            if parsed.path == "/api/security/egress/request":
                payload = self.read_json()
                try:
                    request = security_privacy.request_egress(
                        requester=str(payload.get("requester", "user")).strip() or "user",
                        purpose=str(payload.get("purpose", "")).strip(),
                        destination=str(payload.get("destination", "")).strip(),
                        resource_type=str(payload.get("resource_type", "unknown")).strip() or "unknown",
                        resource_id=str(payload.get("resource_id", "")).strip(),
                        risk_level=str(payload.get("risk_level", "medium")).strip() or "medium",
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"request": request}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/security/egress/resolve":
                payload = self.read_json()
                try:
                    request = security_privacy.resolve_egress_request(
                        request_id=str(payload.get("request_id", "")).strip(),
                        approved=parse_bool(payload.get("approved"), default=False),
                        approved_by=str(payload.get("approved_by", "user")).strip() or "user",
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"request": request})
                return
            if parsed.path == "/api/security/audit-chain/seal":
                self.send_json(security_privacy.seal_audit_events(), status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/security/audit-chain/verify":
                self.send_json(security_privacy.verify_audit_chain())
                return
            if parsed.path == "/api/security/privacy-report":
                self.send_json({"report": security_privacy.create_privacy_report()}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/security/sandbox-profiles":
                payload = self.read_json()
                try:
                    profile = security_privacy.create_sandbox_profile(
                        name=str(payload.get("name", "")).strip(),
                        permissions=[str(item) for item in payload.get("permissions", [])] if isinstance(payload.get("permissions"), list) else [],
                        network_allowed=parse_bool(payload.get("network_allowed"), default=False),
                        filesystem=payload.get("filesystem") if isinstance(payload.get("filesystem"), dict) else {},
                        env={str(key): str(value) for key, value in payload.get("env", {}).items()} if isinstance(payload.get("env"), dict) else {},
                        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"profile": profile}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/security/sandbox/evaluate":
                payload = self.read_json()
                try:
                    result = security_privacy.evaluate_sandbox_request(
                        profile_id=str(payload.get("profile_id", "")).strip(),
                        permissions=[str(item) for item in payload.get("permissions", [])] if isinstance(payload.get("permissions"), list) else [],
                        network=parse_bool(payload.get("network"), default=False),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return

            # ---- Calendar POST routes ----
            if parsed.path == "/api/calendar/calendars":
                payload = self.read_json()
                try:
                    cal = calendar_system.create_calendar(
                        str(payload.get("name", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        color=str(payload.get("color", "#17c964")),
                        description=str(payload.get("description", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"calendar": cal}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/calendar/events":
                payload = self.read_json()
                try:
                    ev = calendar_system.create_event(
                        str(payload.get("title", "")).strip(),
                        str(payload.get("starts_at", "")),
                        str(payload.get("ends_at", payload.get("starts_at", ""))),
                        project=str(payload.get("project", "default")).strip() or "default",
                        calendar_id=str(payload.get("calendar_id", "")),
                        description=str(payload.get("description", "")),
                        location=str(payload.get("location", "")),
                        all_day=parse_bool(payload.get("all_day"), default=False),
                        rrule=str(payload.get("rrule", "")),
                        attendees=[str(a) for a in payload.get("attendees", [])] if isinstance(payload.get("attendees"), list) else None,
                        reminders=[int(r) for r in payload.get("reminders", [15])] if isinstance(payload.get("reminders"), list) else None,
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"event": ev}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/calendar/ical-import":
                payload = self.read_json()
                ical_data = str(payload.get("ical", "")).strip()
                if not ical_data:
                    self.send_json({"error": "ical data is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                result = calendar_system.import_ical(
                    ical_data,
                    project=str(payload.get("project", "default")).strip() or "default",
                )
                self.send_json(result)
                return
            if parsed.path == "/api/calendar/ical-export":
                payload = self.read_json()
                ical_str = calendar_system.export_ical(
                    project=str(payload.get("project", "default")).strip() or "default",
                    calendar_id=str(payload.get("calendar_id", "")),
                )
                self.send_json({"ical": ical_str})
                return

            # ---- Digital Twin POST routes ----
            if parsed.path == "/api/twin/profile":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                updates = {k: v for k, v in payload.items() if k != "project"}
                self.send_json(digital_twin.update_profile(updates, project=project))
                return
            if parsed.path == "/api/twin/interaction":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                result = digital_twin.record_interaction(
                    str(payload.get("kind", "chat")),
                    str(payload.get("content", "")),
                    project=project,
                    context=payload.get("context") if isinstance(payload.get("context"), dict) else None,
                )
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/twin/predict":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                context_text = str(payload.get("context", "")).strip()
                self.send_json(digital_twin.predict_intent(context_text, project=project))
                return
            if parsed.path == "/api/twin/goals":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                try:
                    goal = digital_twin.create_goal(
                        str(payload.get("title", "")).strip(),
                        project=project,
                        description=str(payload.get("description", "")),
                        target_date=str(payload.get("target_date", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"goal": goal}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/twin/consolidate":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                self.send_json(digital_twin.consolidate_memory(project=project))
                return

            # ---- Vault POST routes ----
            if parsed.path == "/api/vault/store":
                payload = self.read_json()
                passphrase = str(payload.get("passphrase", "")).strip()
                secret = str(payload.get("secret", "")).strip()
                label = str(payload.get("label", "")).strip()
                project = str(payload.get("project", "default")).strip() or "default"
                if not passphrase or not secret or not label:
                    self.send_json({"error": "label, secret, and passphrase are required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    entry = vault.store_secret(
                        label, secret, passphrase,
                        project=project,
                        category=str(payload.get("category", "secret")),
                        tags=[str(t) for t in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else None,
                        notes=str(payload.get("notes", "")),
                    )
                except (ValueError, PermissionError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"entry": entry}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/vault/retrieve":
                payload = self.read_json()
                entry_id = str(payload.get("entry_id", "")).strip()
                passphrase = str(payload.get("passphrase", "")).strip()
                project = str(payload.get("project", "default")).strip() or "default"
                if not entry_id or not passphrase:
                    self.send_json({"error": "entry_id and passphrase are required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    secret_data = vault.retrieve_secret(entry_id, passphrase, project=project)
                except (KeyError, ValueError, PermissionError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"secret": secret_data})
                return
            if parsed.path == "/api/vault/rotate-key":
                payload = self.read_json()
                old_pp = str(payload.get("old_passphrase", "")).strip()
                new_pp = str(payload.get("new_passphrase", "")).strip()
                project = str(payload.get("project", "default")).strip() or "default"
                if not old_pp or not new_pp:
                    self.send_json({"error": "old_passphrase and new_passphrase are required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    result = vault.rotate_key(old_pp, new_pp, project=project)
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/vault/export":
                payload = self.read_json()
                passphrase = str(payload.get("passphrase", "")).strip()
                project = str(payload.get("project", "default")).strip() or "default"
                if not passphrase:
                    self.send_json({"error": "passphrase required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(vault.export_vault(passphrase, project=project))
                return

            # ---- Scraper POST routes ----
            if parsed.path == "/api/scraper/scrape":
                payload = self.read_json()
                url = str(payload.get("url", "")).strip()
                if not url:
                    self.send_json({"error": "url is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                project = str(payload.get("project", "default")).strip() or "default"
                try:
                    result = scraper.scrape_url(
                        url, project=project,
                        respect_robots=parse_bool(payload.get("respect_robots"), default=True),
                        ingest_to_knowledge=parse_bool(payload.get("ingest"), default=True),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/scraper/schedule":
                payload = self.read_json()
                url = str(payload.get("url", "")).strip()
                if not url:
                    self.send_json({"error": "url is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                project = str(payload.get("project", "default")).strip() or "default"
                try:
                    job = scraper.schedule_scrape(
                        url, project=project,
                        interval_hours=float(payload.get("interval_hours", 24.0)),
                        label=str(payload.get("label", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"job": job}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/scraper/run-due":
                payload = self.read_json()
                project = str(payload.get("project", "default")).strip() or "default"
                self.send_json(scraper.run_due_jobs(project=project))
                return

            # ---- Fine-tuning POST routes ----
            if parsed.path == "/api/finetune/datasets":
                payload = self.read_json()
                try:
                    ds = fine_tuning.create_dataset(
                        str(payload.get("name", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        description=str(payload.get("description", "")),
                        format=str(payload.get("format", "jsonl")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"dataset": ds}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/finetune/examples":
                payload = self.read_json()
                dataset_id = str(payload.get("dataset_id", "")).strip()
                if not dataset_id:
                    self.send_json({"error": "dataset_id is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    ex = fine_tuning.add_example(
                        dataset_id,
                        str(payload.get("user_input", "")).strip(),
                        str(payload.get("assistant_output", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        system_prompt=str(payload.get("system_prompt", "")),
                        quality_score=float(payload.get("quality_score", 1.0)),
                    )
                except (ValueError, KeyError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"example": ex}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/finetune/jobs":
                payload = self.read_json()
                try:
                    job = fine_tuning.create_job(
                        str(payload.get("dataset_id", "")).strip(),
                        str(payload.get("base_model", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        method=str(payload.get("method", "lora")),
                        config=payload.get("config") if isinstance(payload.get("config"), dict) else None,
                        template_id=str(payload.get("template_id", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"job": job}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/finetune/adapters":
                payload = self.read_json()
                try:
                    adapter = fine_tuning.register_adapter(
                        str(payload.get("name", "")).strip(),
                        str(payload.get("base_model", "")).strip(),
                        project=str(payload.get("project", "default")).strip() or "default",
                        description=str(payload.get("description", "")),
                        rank=int(payload.get("rank", 16)),
                        alpha=float(payload.get("alpha", 16.0)),
                        target_modules=[str(m) for m in payload.get("target_modules", [])] if isinstance(payload.get("target_modules"), list) else None,
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"adapter": adapter}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/finetune/adapters/activate":
                payload = self.read_json()
                adapter_id = str(payload.get("adapter_id", "")).strip()
                if not adapter_id:
                    self.send_json({"error": "adapter_id is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    adapter = fine_tuning.activate_adapter(
                        adapter_id,
                        project=str(payload.get("project", "default")).strip() or "default",
                    )
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json({"adapter": adapter})
                return

            # ---- Auth POST routes ----
            if parsed.path == "/api/auth/register":
                payload = self.read_json()
                try:
                    result = auth.register(
                        str(payload.get("username", "")),
                        str(payload.get("email", "")),
                        str(payload.get("password", "")),
                        display_name=str(payload.get("display_name", "")),
                    )
                except (ValueError, AuthError) as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"user": result}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/auth/login":
                payload = self.read_json()
                try:
                    result = auth.login(
                        str(payload.get("username_or_email", "")),
                        str(payload.get("password", "")),
                        otp_code=str(payload.get("otp_code", "")),
                        ip=self.client_address[0] if self.client_address else "",
                        user_agent=self.headers.get("User-Agent", ""),
                    )
                except AuthError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.UNAUTHORIZED)
                    return
                self.send_json(result)
                return
            if parsed.path.startswith("/api/local-ai/models/") and parsed.path.endswith("/dependencies"):
                model_id = unquote(parsed.path.removeprefix("/api/local-ai/models/").removesuffix("/dependencies")).strip()
                payload = self.read_json()
                try:
                    result = local_ai.add_model_dependency(
                        model_id, str(payload.get("dependent_kind", "")), str(payload.get("dependent_id", ""))
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"dependency": result}, status=HTTPStatus.CREATED)
                return
            if parsed.path.startswith("/api/local-ai/models/") and parsed.path.endswith("/tools"):
                model_id = unquote(parsed.path.removeprefix("/api/local-ai/models/").removesuffix("/tools")).strip()
                payload = self.read_json()
                try:
                    result = local_ai.assign_tool(
                        model_id, str(payload.get("tool_id", "")),
                        enabled=parse_bool(payload.get("enabled"), default=True),
                        config=payload.get("config") if isinstance(payload.get("config"), dict) else {},
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"assignment": result}, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/auth/refresh":
                payload = self.read_json()
                try:
                    result = auth.refresh_session(str(payload.get("refresh_token", "")))
                except AuthError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.UNAUTHORIZED)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/auth/logout":
                self.send_json(auth.logout(self._bearer_token()))
                return
            if parsed.path == "/api/auth/sessions/revoke":
                user = self._require_auth()
                if user is None:
                    return
                payload = self.read_json()
                try:
                    result = auth.revoke_session(user["user_id"], str(payload.get("session_id", "")))
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/auth/2fa/setup":
                user = self._require_auth()
                if user is None:
                    return
                self.send_json(auth.setup_two_factor(user["user_id"]))
                return
            if parsed.path == "/api/auth/2fa/verify":
                user = self._require_auth()
                if user is None:
                    return
                payload = self.read_json()
                try:
                    result = auth.verify_two_factor(user["user_id"], str(payload.get("otp_code", "")))
                except AuthError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/auth/2fa/disable":
                user = self._require_auth()
                if user is None:
                    return
                payload = self.read_json()
                try:
                    result = auth.disable_two_factor(user["user_id"], str(payload.get("otp_code", "")))
                except AuthError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/auth/api-tokens":
                user = self._require_auth()
                if user is None:
                    return
                payload = self.read_json()
                try:
                    result = auth.create_api_token(
                        user["user_id"],
                        str(payload.get("name", "")),
                        scopes=payload.get("scopes") if isinstance(payload.get("scopes"), list) else None,
                        expires_at=str(payload.get("expires_at", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return

            # ---- Editor document + draft POST routes ----
            if parsed.path == "/api/document":
                payload = self.read_json()
                try:
                    result = documents.create_document(
                        str(payload.get("title", "")),
                        str(payload.get("content", "")),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._document_write_owner(payload),
                        doc_format=str(payload.get("format", "markdown")).strip() or "markdown",
                        tags=[str(t) for t in payload.get("tags", [])] if isinstance(payload.get("tags"), list) else None,
                        folder=str(payload.get("folder", "")),
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json({"document": result}, status=HTTPStatus.CREATED)
                return
            if parsed.path.startswith("/api/document/") and parsed.path.endswith("/archive"):
                doc_id = unquote(parsed.path.removeprefix("/api/document/").removesuffix("/archive")).strip()
                payload = self.read_json()
                archived = parse_bool(payload.get("archived"), default=True)
                try:
                    result = documents.archive_document(doc_id, archived, owner=self._auth_scope_owner())
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
                return
            if parsed.path.startswith("/api/document/") and "/restore/" in parsed.path:
                remainder = parsed.path.removeprefix("/api/document/")
                doc_id, _, version_text = remainder.partition("/restore/")
                doc_id = unquote(doc_id).strip()
                try:
                    version_number = int(version_text)
                except ValueError:
                    self.send_json({"error": "invalid version number"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    result = documents.restore_version(doc_id, version_number, owner=self._auth_scope_owner())
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
                return
            if parsed.path == "/api/editor-drafts":
                payload = self.read_json()
                doc_id = str(payload.get("doc_id", "")).strip()
                try:
                    result = documents.create_draft(
                        doc_id,
                        str(payload.get("content", "")),
                        base_version=int(payload.get("base_version", 0) or 0),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._document_write_owner(payload),
                        scope_owner=self._auth_scope_owner(),
                    )
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return

            # ---- Calendar external-sync POST routes ----
            if parsed.path == "/api/calendar/providers/register":
                payload = self.read_json()
                try:
                    result = calendar_sync.register_provider(
                        str(payload.get("provider", "")),
                        project=str(payload.get("project", "default")).strip() or "default",
                        owner=self._payload_owner(payload),
                        base_url=str(payload.get("base_url", "")),
                        account_ref=str(payload.get("account_ref", "")),
                        calendar_id=str(payload.get("calendar_id", "")),
                        config=payload.get("config") if isinstance(payload.get("config"), dict) else None,
                    )
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result, status=HTTPStatus.CREATED)
                return
            if parsed.path == "/api/calendar/sync":
                payload = self.read_json()
                try:
                    result = calendar_sync.run_sync(
                        str(payload.get("provider_id", "")),
                        direction=str(payload.get("direction", "pull")),
                        conflict_policy=str(payload.get("conflict_policy", "latest_update")),
                        owner=self._auth_scope_owner(),
                    )
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result, status=HTTPStatus.ACCEPTED)
                return
            if parsed.path.startswith("/api/calendar/conflicts/") and parsed.path.endswith("/resolve"):
                conflict_id = unquote(parsed.path.removeprefix("/api/calendar/conflicts/").removesuffix("/resolve")).strip()
                payload = self.read_json()
                try:
                    result = calendar_sync.resolve_conflict(
                        conflict_id,
                        str(payload.get("resolution", "")),
                        owner=self._auth_scope_owner(),
                    )
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return

            # ---- Communication resilience POST routes ----
            if parsed.path == "/api/communication/sync-resilient":
                payload = self.read_json()
                result = communication.sync_connector_resilient(
                    str(payload.get("platform", "")).strip(),
                    project=str(payload.get("project", "default")).strip() or "default",
                    owner=self._payload_owner(payload),
                    payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                    allow_external=parse_bool(payload.get("allow_external"), default=False),
                    max_attempts=parse_int(str(payload.get("max_attempts", "")), default=3),
                )
                ok_status = HTTPStatus.OK if result.get("status") == "dead_lettered" else HTTPStatus.CREATED
                self.send_json(result, status=ok_status)
                return

            # ---- Model-ops lifecycle POST routes ----
            if parsed.path.startswith("/api/local-ai/models/") and parsed.path.endswith("/transition"):
                model_id = unquote(parsed.path.removeprefix("/api/local-ai/models/").removesuffix("/transition")).strip()
                payload = self.read_json()
                try:
                    result = local_ai.transition_model_lifecycle(
                        model_id,
                        str(payload.get("state", "")),
                        reason=str(payload.get("reason", "")),
                        detail=payload.get("detail") if isinstance(payload.get("detail"), dict) else None,
                    )
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self.send_json(result)
                return

            self.send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_PUT(self) -> None:
            parsed = urlparse(self.path)
            if not self._enforce_auth_gate(parsed.path, "PUT"):
                return
            if parsed.path.startswith("/api/document/"):
                doc_id = unquote(parsed.path.removeprefix("/api/document/")).strip()
                payload = self.read_json()
                try:
                    result = documents.update_document(
                        doc_id,
                        title=payload["title"] if "title" in payload else None,
                        content=payload["content"] if "content" in payload else None,
                        if_version=int(payload["if_version"]) if "if_version" in payload else None,
                        owner=self._auth_scope_owner(),
                    )
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                except DocumentError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
                    return
                self.send_json({"document": result})
                return
            if parsed.path.startswith("/api/editor-drafts/"):
                draft_id = unquote(parsed.path.removeprefix("/api/editor-drafts/")).strip()
                payload = self.read_json()
                try:
                    result = documents.update_draft(
                        draft_id,
                        str(payload.get("content", "")),
                        base_version=int(payload["base_version"]) if "base_version" in payload else None,
                        owner=self._auth_scope_owner(),
                    )
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json({"draft": result})
                return
            self.send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_PATCH(self) -> None:
            parsed = urlparse(self.path)
            if not self._enforce_auth_gate(parsed.path, "PATCH"):
                return
            phd_route = handle_phd_patch(
                parsed.path,
                phd_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if phd_route is not None:
                self.send_json(phd_route.payload, status=phd_route.status)
                return
            contacts_route = handle_contacts_patch(
                parsed.path,
                contacts_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if contacts_route is not None:
                self.send_json(contacts_route.payload, status=contacts_route.status)
                return
            scheduler_route = handle_scheduler_patch(
                parsed.path,
                scheduler_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if scheduler_route is not None:
                self.send_json(scheduler_route.payload, status=scheduler_route.status)
                return
            conversation_route = handle_conversation_patch(
                parsed.path,
                conversation_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if conversation_route is not None:
                self.send_json(conversation_route.payload, status=conversation_route.status)
                return
            mcp_runtime_route = handle_mcp_runtime_patch(
                parsed.path,
                mcp_runtime_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if mcp_runtime_route is not None:
                self.send_json(mcp_runtime_route.payload, status=mcp_runtime_route.status)
                return
            if parsed.path.startswith("/api/document/"):
                doc_id = unquote(parsed.path.removeprefix("/api/document/")).strip()
                payload = self.read_json()
                try:
                    result = documents.patch_document(
                        doc_id,
                        tags=[str(t) for t in payload["tags"]] if isinstance(payload.get("tags"), list) else None,
                        folder=payload["folder"] if "folder" in payload else None,
                        title=payload["title"] if "title" in payload else None,
                        owner=self._auth_scope_owner(),
                    )
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json({"document": result})
                return
            self.send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            if not self._enforce_auth_gate(parsed.path, "DELETE"):
                return
            phd_route = handle_phd_delete(
                parsed.path,
                phd_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if phd_route is not None:
                self.send_json(phd_route.payload, status=phd_route.status)
                return
            identity_route = handle_identity_delete(parsed.path, identity_route_dependencies(self))
            if identity_route is not None:
                self.send_json(identity_route.payload, status=identity_route.status)
                return
            contacts_route = handle_contacts_delete(
                parsed.path,
                contacts_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if contacts_route is not None:
                self.send_json(contacts_route.payload, status=contacts_route.status)
                return
            conversation_route = handle_conversation_delete(
                parsed.path,
                conversation_route_dependencies(self.read_json, self._query_owner, self._payload_owner),
            )
            if conversation_route is not None:
                self.send_json(conversation_route.payload, status=conversation_route.status)
                return
            if parsed.path.startswith("/api/document/"):
                doc_id = unquote(parsed.path.removeprefix("/api/document/")).strip()
                try:
                    result = documents.delete_document(doc_id, owner=self._auth_scope_owner())
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
                return
            if parsed.path.startswith("/api/editor-drafts/"):
                draft_id = unquote(parsed.path.removeprefix("/api/editor-drafts/")).strip()
                try:
                    result = documents.delete_draft(draft_id, owner=self._auth_scope_owner())
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
                return
            if parsed.path.startswith("/api/auth/api-tokens/"):
                user = self._require_auth()
                if user is None:
                    return
                token_id = unquote(parsed.path.removeprefix("/api/auth/api-tokens/")).strip()
                try:
                    result = auth.revoke_api_token(user["user_id"], token_id)
                except KeyError as exc:
                    self.send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self.send_json(result)
                return
            self.send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def serve_static(self, request_path: str) -> None:
            if request_path in {"/favicon.ico", "/icons/icon-192.png", "/icons/icon-512.png"}:
                size = 512 if "512" in request_path else 192
                if request_path == "/favicon.ico":
                    size = 32
                body = _generated_icon_png(size)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.end_headers()
                self.wfile.write(body)
                return
            relative = "index.html" if request_path in {"", "/"} else unquote(request_path).lstrip("/")
            candidate = (root / relative).resolve()
            if not str(candidate).startswith(str(root)) or not candidate.is_file():
                self.send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            body = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if candidate.name in {"index.html", "app.js", "styles.css", "sw.js"}:
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict[str, object]:
            length = parse_int(self.headers.get("Content-Length"), default=0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}

        def _bearer_token(self) -> str:
            header = self.headers.get("Authorization", "")
            if header.lower().startswith("bearer "):
                return header[7:].strip()
            return ""

        def _require_auth(self) -> dict[str, object] | None:
            user = auth.resolve_bearer(self._bearer_token())
            if user is None:
                self.send_json({"error": "authentication required"}, status=HTTPStatus.UNAUTHORIZED)
                return None
            self._auth_user = user
            return user

        def _auth_scope_owner(self) -> str:
            """Return the authenticated local owner id when auth enforcement is active.

            In the default single-user/local mode this returns an empty string,
            preserving the existing project-scoped behavior. When
            OPENAGENT_AUTH_ENFORCED=1 is active, document and draft routes use
            this as an owner filter so cross-user ids resolve as not found.
            """
            if not auth_enforcement_enabled():
                return ""
            user = getattr(self, "_auth_user", None)
            if user is None:
                user = auth.resolve_bearer(self._bearer_token())
                if user is not None:
                    self._auth_user = user
            return str(user.get("user_id", "")).strip() if user else ""

        def _document_write_owner(self, payload: dict[str, object]) -> str:
            scoped_owner = self._auth_scope_owner()
            if scoped_owner:
                return scoped_owner
            return str(payload.get("owner", "local")).strip() or "local"

        def _query_owner(self, params: dict[str, list[str]]) -> str:
            scoped_owner = self._auth_scope_owner()
            if scoped_owner:
                return scoped_owner
            return first(params, "owner") or ""

        def _payload_owner(self, payload: dict[str, object], default: str = "local") -> str:
            scoped_owner = self._auth_scope_owner()
            if scoped_owner:
                return scoped_owner
            return str(payload.get("owner", default)).strip() or default

        def _enforce_auth_gate(self, path: str, method: str) -> bool:
            """Return True if the request may proceed, else it has already sent 401/403.

            Only active when OPENAGENT_AUTH_ENFORCED is enabled (see
            config.auth_enforcement_enabled), so default/existing behavior is
            unchanged until an operator opts in. When enabled, gates every
            route group listed in auth.PROTECTED_ROUTE_SCOPES: GET requires
            the read scope, other verbs require the write/admin scope.
            """
            if not auth_enforcement_enabled():
                return True
            scope = required_scope_for_path(path, method)
            if scope is None:
                return True
            user = auth.resolve_bearer(self._bearer_token())
            if user is None:
                self.send_json({"error": "authentication required"}, status=HTTPStatus.UNAUTHORIZED)
                return False
            if scope not in user.get("scopes", []):
                self.send_json({"error": "insufficient scope"}, status=HTTPStatus.FORBIDDEN)
                return False
            self._auth_user = user
            return True

        def _handle_file_upload(self) -> None:
            """Analyze an uploaded document or image and add readable evidence to RAG."""
            content_type = self.headers.get("Content-Type", "")
            content_length = parse_int(self.headers.get("Content-Length"), default=0)
            if "multipart/form-data" not in content_type:
                self.send_json({"error": "multipart/form-data required"}, status=HTTPStatus.BAD_REQUEST)
                return
            if content_length <= 0 or content_length > 50 * 1024 * 1024:
                self.send_json({"error": "Content-Length missing or too large (max 50 MB)"}, status=HTTPStatus.BAD_REQUEST)
                return
            raw = self.rfile.read(content_length)
            # Extract boundary from content-type header
            boundary = b""
            for part in content_type.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[9:].strip('"').encode("utf-8")
                    break
            if not boundary:
                self.send_json({"error": "No boundary in multipart form"}, status=HTTPStatus.BAD_REQUEST)
                return
            parts = _parse_multipart_parts(raw, boundary)
            file_part = next(
                (part for part in parts if str(part.get("filename", "")).strip() and isinstance(part.get("content"), bytes)),
                None,
            )
            if file_part is None:
                self.send_json({"error": "No file was included in the upload."}, status=HTTPStatus.BAD_REQUEST)
                return
            project_part = next((part for part in parts if part.get("name") == "project"), None)
            project = (
                bytes(project_part.get("content", b"")).decode("utf-8", errors="replace").strip()
                if project_part else "default"
            ) or "default"
            filename = str(file_part.get("filename", "")).strip() or "attachment.bin"
            content = bytes(file_part.get("content", b""))
            content_type_hint = str(file_part.get("content_type", "")).strip()
            asset = media.ingest_bytes(
                content,
                filename=filename,
                project=project,
                source="chat-attachment",
                source_path=filename,
                metadata={"mime_type": content_type_hint} if content_type_hint else {},
            )
            analysis = asset.get("analysis") if isinstance(asset.get("analysis"), dict) else {}
            text_content = str(analysis.get("extracted_text", "")).strip()
            if not text_content and Path(filename).suffix.lower() in {".docx", ".odt", ".rtf", ".html", ".htm", ".csv", ".md"}:
                from ..document_processing import extract_office_text

                text_content = extract_office_text(filename, content).strip()
            document = None
            if text_content:
                document = knowledge.ingest_text(
                    title=filename,
                    content=text_content,
                    project=project,
                    source="chat-attachment",
                    source_path=filename,
                    metadata={"asset_id": asset.get("id", ""), "mime_type": asset.get("mime_type", "")},
                )
            attachment = {
                "id": asset.get("id", ""),
                "name": filename,
                "filename": filename,
                "media_type": asset.get("media_type", "document"),
                "mime_type": asset.get("mime_type", content_type_hint),
                "size_bytes": asset.get("size_bytes", len(content)),
                "summary": analysis.get("summary", ""),
                "extracted_text": text_content[:50_000],
                "findings": analysis.get("findings", {}),
                "document_id": (document or {}).get("id", "") if isinstance(document, dict) else "",
            }
            self.send_json(
                {"attachment": attachment, "asset": asset, "document": document, "filename": filename, "chars_extracted": len(text_content)},
                status=HTTPStatus.CREATED,
            )

        def _handle_voice_upload(self) -> None:
            """Handle browser audio uploads from current and older workbench builds."""

            content_type = self.headers.get("Content-Type", "")
            content_length = parse_int(self.headers.get("Content-Length"), default=0)
            if content_length <= 0 or content_length > 50 * 1024 * 1024:
                self.send_json({"error": "Content-Length missing or too large (max 50 MB)"}, status=HTTPStatus.BAD_REQUEST)
                return

            fields: dict[str, str] = {}
            audio_data = b""
            filename = "recording.webm"
            if "multipart/form-data" in content_type:
                raw = self.rfile.read(content_length)
                boundary = b""
                for part in content_type.split(";"):
                    part = part.strip()
                    if part.startswith("boundary="):
                        boundary = part[9:].strip('"').encode("utf-8")
                        break
                if not boundary:
                    self.send_json({"error": "No boundary in multipart form"}, status=HTTPStatus.BAD_REQUEST)
                    return
                for part in _parse_multipart_parts(raw, boundary):
                    name = str(part.get("name", ""))
                    part_filename = str(part.get("filename", ""))
                    content = part.get("content", b"")
                    if not isinstance(content, bytes):
                        continue
                    if part_filename or name in {"audio", "file", "recording", "upload"}:
                        audio_data = content
                        filename = part_filename or filename
                    elif name:
                        fields[name] = content.decode("utf-8", errors="replace").strip()
            elif "application/json" in content_type:
                payload = self.read_json()
                fields = {str(key): str(value) for key, value in payload.items() if not isinstance(value, (dict, list))}
                try:
                    audio_data = base64.b64decode(str(payload.get("audio_base64", "")).encode("ascii"), validate=True)
                except (ValueError, binascii.Error):
                    self.send_json({"error": "audio_base64 must contain valid base64 audio data"}, status=HTTPStatus.BAD_REQUEST)
                    return
                filename = str(payload.get("filename", filename)).strip() or filename
            else:
                audio_data = self.rfile.read(content_length)
                filename = self.headers.get("X-OpenAgent-Filename", filename)

            if not audio_data:
                self.send_json({"error": "No audio data uploaded."}, status=HTTPStatus.BAD_REQUEST)
                return

            project = fields.get("project", "default").strip() or "default"
            session_id = fields.get("session_id", "").strip()
            if not session_id:
                session = voice.create_session(
                    project=project,
                    mode=fields.get("mode", "push_to_talk").strip() or "push_to_talk",
                    language=fields.get("language", "en").strip() or "en",
                )
                session_id = str(session["id"])

            try:
                transcription = voice.transcribe_audio(
                    session_id,
                    audio_data=audio_data,
                    filename=filename,
                    speaker=fields.get("speaker", "user").strip() or "user",
                    transcript_hint=fields.get("transcript_hint", "").strip(),
                    language=fields.get("language", "en").strip() or "en",
                    stt_engine=fields.get("stt_engine", "").strip(),
                    allow_fallback_stt=parse_bool(fields.get("allow_fallback_stt"), default=False),
                    metadata={
                        "client_route": "voice-upload",
                        "content_type": content_type,
                        "size_bytes": len(audio_data),
                    },
                )
            except ValueError as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"transcription": transcription, "session_id": session_id}, status=HTTPStatus.CREATED)

        def send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
            if not self._wants_legacy_json():
                payload = self._envelope(payload, status)
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

        def _wants_legacy_json(self) -> bool:
            """Return True only when a caller explicitly asks for raw legacy JSON.

            The v2 response envelope is now the default contract. During the
            desktop/web migration period, successful responses keep the legacy
            top-level keys beside the envelope fields so current clients remain
            compatible. A caller can still request the old raw shape with
            X-Response-Envelope: legacy or ?envelope=legacy for diagnostics.
            """
            header = self.headers.get("X-Response-Envelope", "").strip().lower()
            if header in {"legacy", "raw", "off", "0", "false"}:
                return True
            query = parse_qs(urlparse(self.path).query)
            return (first(query, "envelope") or "").strip().lower() in {"legacy", "raw", "off", "0", "false"}

        def _envelope(self, payload: object, status: HTTPStatus) -> dict[str, object]:
            meta = {"request_id": uuid4().hex[:16], "ts": utc_now()}
            if int(status) >= 400:
                message = payload.get("error", "Error") if isinstance(payload, dict) else str(payload)
                code = status.name.lower() if hasattr(status, "name") else str(int(status))
                return {"success": False, "data": None, "error": {"code": code, "message": message}, "meta": meta}
            data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
            envelope = {"success": True, "data": data, "error": None, "meta": meta}
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if key not in envelope:
                        envelope[key] = value
            return envelope

        def send_event_stream(self, events: object) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for event in events:
                    body = f"data: {json.dumps(event, sort_keys=True)}\n\n".encode("utf-8")
                    self.wfile.write(body)
                    self.wfile.flush()
            except (ProviderConfigurationError, ValueError) as exc:
                body = f"data: {json.dumps({'type': 'error', 'error': str(exc)}, sort_keys=True)}\n\n"
                self.wfile.write(body.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: object) -> None:
            return

    class OpenAgentHTTPServer(ThreadingHTTPServer):
        # Model, research, and transcription requests may outlive the desktop
        # window briefly. They must never keep an otherwise closed workbench
        # process alive or retain the single-instance lock.
        daemon_threads = True
        block_on_close = False
        allow_reuse_address = True

        def server_close(self) -> None:
            super().server_close()
            for name, closer, timeout_seconds in (
                ("model-runtime", ultra_model.close, 4.0),
                ("hardware-runtime", adaptive_engine.close, 8.0),
                ("retrieval-runtime", advanced_rag.close, 3.0),
            ):
                completed = threading.Event()

                def close_resource(callback=closer, event=completed) -> None:
                    try:
                        callback()
                    finally:
                        event.set()

                thread = threading.Thread(
                    target=close_resource,
                    name=f"openagent-close-{name}",
                    daemon=True,
                )
                thread.start()
                completed.wait(timeout=timeout_seconds)

    server = OpenAgentHTTPServer((host, port), OpenAgentHandler)
    server.openvino_warm_event = openvino_warm_event  # type: ignore[attr-defined]
    server.openvino_warm_thread = openvino_warm_thread  # type: ignore[attr-defined]
    return server


def run_server(host: str, port: int, data_dir: str | Path) -> None:
    store = OpenAgentStore(data_dir)
    server = build_http_server(host, port, store)
    address = f"http://{host}:{server.server_address[1]}"
    print(f"OpenAgent local core running at {address}")
    print(f"Data directory: {store.data_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping OpenAgent.")
    finally:
        server.server_close()


def first(params: dict[str, list[str]], key: str) -> str | None:
    value = params.get(key, [None])[0]
    return value or None


def parse_int(value: str | None, *, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def parse_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
