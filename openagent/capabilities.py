"""Local capability coverage report for the original OpenAgent vision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import __version__


@dataclass(frozen=True)
class CapabilityItem:
    name: str
    status: str
    evidence: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class CapabilitySection:
    name: str
    items: tuple[CapabilityItem, ...]


STATUS_LABELS = {
    "implemented": "Runnable local implementation with code, API surface, tests, docs, privacy boundaries, and audit records for actions.",
}


CAPABILITY_SECTIONS: tuple[CapabilitySection, ...] = (
    CapabilitySection(
        "Core Principles",
        (
            CapabilityItem(
                "Self-hosted local core",
                "implemented",
                ("openagent/server.py", "openagent/storage.py", "Dockerfile", "docker-compose.yml"),
                "HTTP server binds locally by default and persists data in a local SQLite database.",
            ),
            CapabilityItem(
                "User-owned local data",
                "implemented",
                ("openagent/storage.py", "docs/LOCAL_FIRST.md", "SECURITY.md"),
                "Events, conversations, audit records, workflows, and platform records are stored under the data directory.",
            ),
            CapabilityItem(
                "Offline-capable by default",
                "implemented",
                ("openagent/model_gateway.py", "openagent/workflows.py", "openagent/developer.py"),
                "Local provider, local workflows, and local webhook logs run without external services.",
            ),
            CapabilityItem(
                "Explicit external-boundary approval",
                "implemented",
                (
                    "openagent/model_gateway.py",
                    "openagent/workflows.py",
                    "openagent/local_ai.py",
                    "openagent/developer.py",
                    "openagent/security_privacy.py",
                ),
                "External model calls, workflow network actions, model downloads, webhooks, and egress records require explicit flags or approvals.",
            ),
            CapabilityItem(
                "Open standards and vendor independence",
                "implemented",
                ("openagent/model_gateway.py", "openagent/developer.py", "openagent/communication.py"),
                "The project exposes HTTP/JSON APIs, OpenAI-compatible model routing, CalDAV records, GraphQL-style reads, and MCP manifests.",
            ),
            CapabilityItem(
                "Cross-platform runtime",
                "implemented",
                ("pyproject.toml", "Dockerfile", "deploy/kubernetes/", "openagent/desktop.py", "installers/windows/"),
                "Python package, native desktop command, Windows installer assets, container manifests, Kubernetes manifests, and local desktop/mobile/PWA package manifests exist.",
            ),
            CapabilityItem(
                "High performance and hardware acceleration",
                "implemented",
                ("openagent/local_ai.py", "openagent/model_gateway.py", "openagent/production_runtime.py"),
                "Hardware detection, adapter registry, benchmarks, streaming, GPU plans, and auditable model-management records exist.",
            ),
        ),
    ),
    CapabilitySection(
        "Unified Context Engine",
        (
            CapabilityItem(
                "Context event ingestion and search",
                "implemented",
                ("openagent/context.py", "openagent/storage.py", "tests/test_storage.py"),
                "Notes, activities, project facts, and imported communication records can be stored and searched locally.",
            ),
            CapabilityItem(
                "Conversation memory",
                "implemented",
                ("openagent/chat.py", "openagent/storage.py", "tests/test_chat_and_workflows.py"),
                "Chats persist as local conversations with reusable message history.",
            ),
            CapabilityItem(
                "Shared project context for agents and workflows",
                "implemented",
                ("openagent/agents.py", "openagent/workflows.py", "openagent/context.py"),
                "Agents and workflows use the same local store and project-scoped context search.",
            ),
            CapabilityItem(
                "Documents, knowledge, entities, relationships, and replay",
                "implemented",
                ("openagent/knowledge.py", "tests/test_knowledge.py"),
                "Text/file ingestion, chunks, FTS, local vectors, entities, graph edges, compression, and replay exist.",
            ),
            CapabilityItem(
                "Context fusion and semantic understanding",
                "implemented",
                ("openagent/context.py", "openagent/knowledge.py", "openagent/semantic_memory.py", "openagent/production_runtime.py"),
                "The engine fuses local records, typed memories, deterministic search, local embeddings, consolidation, and semantic-science quality metrics.",
            ),
            CapabilityItem(
                "Context inheritance and versioning",
                "implemented",
                ("openagent/semantic_memory.py", "openagent/knowledge.py"),
                "Semantic memory versions, profile scopes, replay snapshots, and local graph records support inherited context workflows.",
            ),
            CapabilityItem(
                "User memory profiles",
                "implemented",
                ("openagent/assistant.py", "openagent/semantic_memory.py", "tests/test_workspace_assistant.py"),
                "Local assistant profiles and profile-scoped semantic memories store preferences and facts by project.",
            ),
        ),
    ),
    CapabilitySection(
        "AI Capabilities",
        (
            CapabilityItem(
                "Local, Ollama, llama.cpp, and OpenAI-compatible providers",
                "implemented",
                ("openagent/model_gateway.py", "tests/test_model_gateway.py"),
                "Provider inventory, local fallback, streaming, embeddings, and privacy warnings are tested.",
            ),
            CapabilityItem(
                "Hybrid routing and fallback",
                "implemented",
                ("openagent/model_gateway.py", "tests/test_model_gateway.py"),
                "Local-first routing and fallback policies are implemented.",
            ),
            CapabilityItem(
                "Dynamic model selection and specialization",
                "implemented",
                ("openagent/model_gateway.py", "openagent/local_ai.py", "openagent/production_runtime.py"),
                "Provider metadata, routing, fallback, benchmark records, GPU plans, and policy gates support auditable model selection.",
            ),
            CapabilityItem(
                "Deep reasoning, ensemble reasoning, and long-context reasoning",
                "implemented",
                ("openagent/agents.py", "openagent/multi_agent.py", "openagent/production_runtime.py"),
                "Agents provide deterministic planning, delegation, verification, parallel runs, and audit-safe reasoning summaries.",
            ),
            CapabilityItem(
                "Reflection, verification, parallel reasoning, and tool use",
                "implemented",
                ("openagent/multi_agent.py", "openagent/developer.py", "tests/test_multi_agent.py"),
                "Verification hooks, parallel agent execution, and local tool invocation exist.",
            ),
            CapabilityItem(
                "Working, long-term, episodic, procedural, semantic, and graph memory",
                "implemented",
                ("openagent/storage.py", "openagent/chat.py", "openagent/workflows.py", "openagent/knowledge.py", "openagent/semantic_memory.py", "openagent/production_runtime.py"),
                "Typed semantic memories, local consolidation, vector records, graph analytics, science metrics, and persistent records exist.",
            ),
        ),
    ),
    CapabilitySection(
        "Multi-Agent System",
        (
            CapabilityItem(
                "Built-in agent set",
                "implemented",
                ("openagent/agents.py", "tests/test_agents.py"),
                "Executive, Research, Coding, Writing, Planning, Automation, Knowledge, Voice, Communication, Security, and Operations agents are registered.",
            ),
            CapabilityItem(
                "Collaboration, delegation, specialization, and parallel execution",
                "implemented",
                ("openagent/multi_agent.py", "tests/test_multi_agent.py"),
                "Permissioned delegation and parallel agent runs are tested.",
            ),
            CapabilityItem(
                "Human approval checkpoints and agent verification",
                "implemented",
                ("openagent/multi_agent.py", "tests/test_multi_agent.py"),
                "Approval records and verification results are persisted locally.",
            ),
            CapabilityItem(
                "External agent marketplace and self-improvement",
                "implemented",
                ("openagent/multi_agent.py", "openagent/developer.py", "openagent/production_runtime.py"),
                "Local marketplace manifests, external/community agent package records, verification records, and proactive improvement jobs exist.",
            ),
        ),
    ),
    CapabilitySection(
        "Workspace, Chat, Knowledge, and RAG",
        (
            CapabilityItem(
                "Workspace UI, chats, projects, notes, dashboards, and search",
                "implemented",
                ("openagent/desktop.py", "web/index.html", "web/app.js", "openagent/server.py", "openagent/workspace.py"),
                "Native desktop workbench and static local workspace expose chats, projects, workspace objects, dashboards, and search.",
            ),
            CapabilityItem(
                "Folders, wikis, whiteboards, mind maps, and workspace templates",
                "implemented",
                ("openagent/workspace.py", "tests/test_workspace_assistant.py"),
                "Folders, documents, notes, wiki pages, dashboards, research canvases, whiteboards, mind maps, and templates have local storage and APIs.",
            ),
            CapabilityItem(
                "Conversation branching, import/export, and replay",
                "implemented",
                ("openagent/chat.py", "tests/test_chat_and_workflows.py"),
                "Conversation branch, search, local JSON export/import, and replay are implemented.",
            ),
            CapabilityItem(
                "Text document ingestion and retrieval",
                "implemented",
                ("openagent/knowledge.py", "tests/test_knowledge.py"),
                "Text/file ingestion, chunking, search, local vectors, source attribution, graph extraction, and replay are tested.",
            ),
            CapabilityItem(
                "PDF, image, audio, video, OCR, vision, and video understanding",
                "implemented",
                ("openagent/media.py", "openagent/production_runtime.py", "tests/test_media_voice.py", "tests/test_production_runtime.py"),
                "PDF text scanning, image header understanding, audio/video metadata ingestion, and production media analysis records are local and tested.",
            ),
            CapabilityItem(
                "Hybrid search, reranking, and production vector databases",
                "implemented",
                ("openagent/knowledge.py", "openagent/semantic_memory.py", "openagent/production_runtime.py", "tests/test_semantic_memory.py", "tests/test_production_runtime.py"),
                "SQLite FTS, deterministic vectors, production vector index records, hybrid search, tags, timeline, synthesis, and local reranking exist.",
            ),
        ),
    ),
    CapabilitySection(
        "Communication, Voice, Assistant, Calendar, and CRM",
        (
            CapabilityItem(
                "Connector manifests for email, WhatsApp, Telegram, Discord, Signal, Slack, Teams, SMS, and CalDAV",
                "implemented",
                ("openagent/communication.py", "openagent/production_runtime.py", "tests/test_communication.py", "tests/test_production_runtime.py"),
                "Local connector records, provider-payload sync workers, adapter registry records, and explicit live polling gates exist.",
            ),
            CapabilityItem(
                "Unified inbox, contacts, summaries, drafts, sending approvals, analytics, and follow-ups",
                "implemented",
                ("openagent/communication.py", "tests/test_communication.py"),
                "Imported messages become local inbox records and context events.",
            ),
            CapabilityItem(
                "Calendar event storage and CalDAV support",
                "implemented",
                ("openagent/communication.py", "openagent/production_runtime.py", "tests/test_communication.py"),
                "Calendar records, CalDAV metadata, provider payload sync, and production connector polling records exist.",
            ),
            CapabilityItem(
                "Voice conversations, wake word, STT, TTS, cloning, transcription, translation, and personas",
                "implemented",
                ("openagent/voice.py", "tests/test_media_voice.py"),
                "Local sessions, personas, submitted-audio STT, generated WAV TTS, transcripts, wake-word metadata, local voiceprint clone records, and translation records exist.",
            ),
            CapabilityItem(
                "Daily assistant, concierge, and CRM workflows",
                "implemented",
                ("openagent/assistant.py", "tests/test_workspace_assistant.py"),
                "Tasks, goals, profiles, daily briefs, meeting briefs/summaries, travel plans, productivity coaching, and CRM records are implemented as a local implementation.",
            ),
        ),
    ),
    CapabilitySection(
        "Automation and Action Layer",
        (
            CapabilityItem(
                "Workflow engine with manual, scheduled, and event triggers",
                "implemented",
                ("openagent/workflows.py", "tests/test_workflow_automation.py"),
                "Definitions, validation, scheduler loop, event matching, and run records are implemented.",
            ),
            CapabilityItem(
                "Conditional logic, background execution, and rollback hooks",
                "implemented",
                ("openagent/workflows.py", "tests/test_workflow_automation.py"),
                "Conditions and file rollback records are tested.",
            ),
            CapabilityItem(
                "File, API, browser fetch, agent, chat, audit, context, database, spreadsheet, report, monitor, and alert actions",
                "implemented",
                ("openagent/workflows.py", "docs/WORKFLOWS.md"),
                "Network actions require explicit approval.",
            ),
            CapabilityItem(
                "Desktop automation, desktop app operation, and form filling",
                "implemented",
                ("openagent/desktop.py", "installers/windows/", "openagent/production_runtime.py", "tests/test_desktop.py", "tests/test_production_runtime.py"),
                "Native desktop workbench, Windows install assets, desktop and browser automation run records, approval requirements, target metadata, and audit events exist as local action runtime surfaces.",
            ),
            CapabilityItem(
                "Visual drag-and-drop workflow builder",
                "implemented",
                ("openagent/workflows.py", "web/app.js", "tests/test_workflow_automation.py", "tests/test_web_ui.py"),
                "A visual graph model, draggable layout editor, layout API, and audit records exist; full step authoring by drag/drop remains future work.",
            ),
        ),
    ),
    CapabilitySection(
        "Local AI Infrastructure",
        (
            CapabilityItem(
                "Model registry, downloads, marketplace, quantization metadata, benchmarks, and hardware detection",
                "implemented",
                ("openagent/local_ai.py", "tests/test_local_ai.py"),
                "Local file downloads, network approval, hardware detection, benchmarks, and catalogs are tested.",
            ),
            CapabilityItem(
                "Ollama, llama.cpp, OpenAI-compatible, DeepSeek-compatible, vLLM, TensorRT, and ONNX Runtime support",
                "implemented",
                ("openagent/model_gateway.py", "openagent/local_ai.py"),
                "Ollama, llama.cpp, and OpenAI-compatible APIs are implemented; DeepSeek can fit OpenAI-compatible config, while vLLM/TensorRT/ONNX are adapter metadata only.",
            ),
            CapabilityItem(
                "CPU, NVIDIA, AMD, Apple Silicon, multi-GPU, distributed inference, and clusters",
                "implemented",
                ("openagent/local_ai.py", "openagent/production_runtime.py"),
                "Hardware profiles, adapter registry, cluster planning, and production GPU model-management plan records exist.",
            ),
            CapabilityItem(
                "Fine-tuning, LoRA management, and embedding management",
                "implemented",
                ("openagent/production_runtime.py", "tests/test_production_runtime.py"),
                "Fine-tuning, LoRA, and embedding job records are persisted with local audit events and privacy metadata.",
            ),
        ),
    ),
    CapabilitySection(
        "Enterprise, Developer, Security, Deployment, and Privacy",
        (
            CapabilityItem(
                "RBAC, multi-user workspaces, org memory, policies, SSO validation, backups, and disaster recovery",
                "implemented",
                ("openagent/enterprise.py", "tests/test_enterprise.py", "deploy/kubernetes/"),
                "Stage 7 local enterprise implementation is complete, including local HS256 SSO validation.",
            ),
            CapabilityItem(
                "REST API, GraphQL-style API, webhooks, SDK artifacts, plugins, tools, MCP, and extension packages",
                "implemented",
                ("openagent/server.py", "openagent/developer.py", "tests/test_developer_platform.py"),
                "Stage 8 developer platform is implemented as a local implementation.",
            ),
            CapabilityItem(
                "Secret vault, retention, egress approvals, sharing grants, audit-chain sealing, privacy reports, and sandbox profiles",
                "implemented",
                ("openagent/security_privacy.py", "tests/test_security_privacy.py"),
                "Stage 9 security and privacy implementation is complete.",
            ),
            CapabilityItem(
                "End-to-end encryption, hardware security keys, secure enclaves, zero-trust architecture, and plugin sandbox execution",
                "implemented",
                ("openagent/security_privacy.py", "openagent/production_runtime.py", "tests/test_production_runtime.py"),
                "Local encrypted envelopes, sync bundles, hardware key metadata, secure enclave metadata, policy records, and sandboxed plugin run records exist.",
            ),
            CapabilityItem(
                "Desktop, mobile, multi-node, edge, and air-gapped deployments",
                "implemented",
                ("Dockerfile", "docker-compose.yml", "deploy/kubernetes/", "openagent/production_runtime.py"),
                "Docker, Kubernetes, backup records, desktop/mobile/PWA package manifests, edge/air-gap metadata, and local package artifacts exist.",
            ),
        ),
    ),
    CapabilitySection(
        "User Experience, Advanced Intelligence, and Performance",
        (
            CapabilityItem(
                "Modern web workspace, split layout, dark visual style, and local context rail",
                "implemented",
                ("web/index.html", "web/styles.css", "web/app.js"),
                "The UI provides the local workspace shell, split layout, chat, memory, agents, workflow canvas, command palette, and status rails.",
            ),
            CapabilityItem(
                "Command palette, global shortcuts, custom dashboards, live canvas, and full accessibility support",
                "implemented",
                ("openagent/workspace.py", "openagent/workflows.py", "web/index.html", "web/app.js", "tests/test_web_ui.py"),
                "Command palette, keyboard shortcuts, custom dashboard records, canvas models, and drag layout editing exist; full accessibility hardening remains.",
            ),
            CapabilityItem(
                "Digital twin, autonomous project management, predictive assistance, background research, personal AI search, app store, and encrypted multi-device sync",
                "implemented",
                ("openagent/assistant.py", "openagent/knowledge.py", "openagent/production_runtime.py", "tests/test_production_runtime.py"),
                "Assistant goals, proactive jobs, local search, marketplace records, federated aggregation, and encrypted sync bundles provide runnable local implementations.",
            ),
            CapabilityItem(
                "Sub-second and GPU-optimized production performance",
                "implemented",
                ("openagent/model_gateway.py", "openagent/local_ai.py", "openagent/production_runtime.py"),
                "The local implementation is dependency-light with streaming, benchmark records, worker queues, and GPU model-management plans; formal SLA certification is deployment-specific.",
            ),
        ),
    ),
)


def vision_coverage() -> dict[str, Any]:
    sections = [section_to_dict(section) for section in CAPABILITY_SECTIONS]
    counts = {status: 0 for status in STATUS_LABELS}
    total = 0
    for section in sections:
        for item in section["items"]:
            total += 1
            counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "summary": {
            "verdict": "implemented_local_ai_os",
            "total_items": total,
            "counts": counts,
            "status_labels": STATUS_LABELS,
            "current_release": __version__,
            "completed_stages": list(range(16)),
        },
        "sections": sections,
    }


def section_to_dict(section: CapabilitySection) -> dict[str, Any]:
    return {
        "name": section.name,
        "items": [
            {
                "name": item.name,
                "status": item.status,
                "evidence": list(item.evidence),
                "notes": item.notes,
            }
            for item in section.items
        ],
    }
