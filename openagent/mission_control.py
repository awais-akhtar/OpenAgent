"""Mission control feature matrix and local end-to-end checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .agents import AgentRegistry
from .assistant import AssistantManager
from .calendar_system import CalendarSystem
from .capabilities import CAPABILITY_SECTIONS
from .chat import ChatService
from .communication import CommunicationHub
from .context import ContextEngine
from .developer import DeveloperPlatform
from .digital_twin import DigitalTwin
from .fine_tuning import FineTuningManager
from .knowledge import KnowledgeEngine
from .local_ai import LocalAIInfrastructure
from .media import MediaManager
from .model_gateway import ModelGateway
from .models import AuditEvent, utc_now
from .multi_agent import MultiAgentRuntime
from .production_runtime import ProductionRuntime
from .research import ResearchEngine
from .scraper import WebScraper
from .security_privacy import SecurityPrivacyManager
from .semantic_memory import SemanticMemoryManager
from .storage import OpenAgentStore
from .vault import EncryptedVault
from .voice import VoiceRuntime
from .workspace import WorkspaceManager
from .workflows import WorkflowEngine


MISSION_PRIVACY_BOUNDARY = (
    "Mission Control runs local-only smoke checks against the configured data "
    "directory. It creates auditable demo records and does not call live external "
    "providers, messaging services, calendars, browsers, or model downloads."
)


class MissionControl:
    """Visible feature inventory plus runnable local verification."""

    def __init__(self, store: OpenAgentStore, gateway: ModelGateway | None = None) -> None:
        self.store = store
        self.gateway = gateway or ModelGateway()

    def feature_matrix(self, *, project: str = "default") -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for section in CAPABILITY_SECTIONS:
            for item in section.items:
                rows.append(
                    {
                        "section": section.name,
                        "name": item.name,
                        "status": item.status,
                        "evidence": list(item.evidence),
                        "notes": item.notes,
                        "privacy_boundary": MISSION_PRIVACY_BOUNDARY,
                    }
                )
        return {
            "project": project,
            "status": "ready",
            "feature_count": len(rows),
            "sections": [
                {"name": section.name, "count": len(section.items)}
                for section in CAPABILITY_SECTIONS
            ],
            "features": rows,
            "privacy_boundary": MISSION_PRIVACY_BOUNDARY,
        }

    def run_full_check(self, *, project: str = "default") -> dict[str, Any]:
        run_id = f"mission-{utc_now()}"
        started_at = utc_now()
        self.store.add_audit_event(
            AuditEvent(
                actor="mission-control",
                action="mission.check.start",
                resource=run_id,
                project=project,
                metadata={"feature_count": len(self.feature_matrix(project=project)["features"])},
            )
        )

        checks: list[dict[str, Any]] = []
        scenarios: list[tuple[str, str, Callable[[], dict[str, Any]]]] = [
            ("Unified Context Engine", "Context fusion, search, history, and project memory", lambda: self._check_context(project)),
            ("AI Workspace and Model Gateway", "Local chat, provider routing, embeddings, and fallback", lambda: self._check_chat_models(project)),
            ("Multi-Agent Platform", "Built-in agents, collaboration, verification, and parallel runs", lambda: self._check_agents(project)),
            ("Workspace System", "Notes, folders, wiki pages, dashboards, templates, and canvases", lambda: self._check_workspace(project)),
            ("Knowledge and RAG", "Ingestion, chunking, hybrid retrieval, graph, synthesis, replay", lambda: self._check_knowledge(project)),
            ("Automation Platform", "Workflow builder, visual runs, schedules, rollback-aware execution", lambda: self._check_workflows(project)),
            ("Personal Assistant", "Goals, tasks, meeting summaries, travel plans, CRM, daily brief", lambda: self._check_assistant(project)),
            ("Communication Hub", "Unified inbox, contacts, drafts, follow-ups, calendar sync records", lambda: self._check_communications(project)),
            ("Voice and Media", "Voice sessions, browser audio STT, TTS, translation, local media analysis", lambda: self._check_voice_media(project)),
            ("Calendar System", "Multiple calendars, events, free slots, reminders, iCal export", lambda: self._check_calendar(project)),
            ("Digital Memory and Twin", "Profile, interactions, patterns, prediction, proactive suggestions", lambda: self._check_twin(project)),
            ("Encrypted Vault", "Encrypted local secrets, retrieval, access logs, key rotation path", lambda: self._check_vault(project)),
            ("Data Processing and Scraper", "Scheduled scraping records and knowledge-ingest boundary", lambda: self._check_scraper(project)),
            ("Local AI Infrastructure", "Model registry, hardware detection, marketplace, benchmark records", lambda: self._check_local_ai(project)),
            ("Fine-Tuning and LoRA", "Datasets, examples, jobs, adapters, activation, templates", lambda: self._check_finetuning(project)),
            ("Developer Platform", "REST/GraphQL schema, API keys, SDK, plugins, tools, MCP, webhooks", lambda: self._check_developer(project)),
            ("Security and Privacy", "Privacy report, sandbox profiles, retention, egress, audit chain", lambda: self._check_security(project)),
            ("Advanced Production Layer", "Vector DB, crypto, packages, marketplace, automation, sync, federation", lambda: self._check_production(project)),
        ]

        for name, description, fn in scenarios:
            checks.append(self._run_check(name, description, fn))

        passed = sum(1 for check in checks if check["status"] == "passed")
        failed = len(checks) - passed
        status = "passed" if failed == 0 else "failed"
        completed_at = utc_now()
        self.store.add_audit_event(
            AuditEvent(
                actor="mission-control",
                action="mission.check.complete",
                resource=run_id,
                project=project,
                status="allowed" if status == "passed" else "error",
                metadata={"passed": passed, "failed": failed, "total": len(checks)},
            )
        )
        return {
            "id": run_id,
            "project": project,
            "status": status,
            "passed": passed,
            "failed": failed,
            "total": len(checks),
            "checks": checks,
            "started_at": started_at,
            "completed_at": completed_at,
            "privacy_boundary": MISSION_PRIVACY_BOUNDARY,
        }

    def _run_check(self, name: str, description: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            evidence = fn()
        except Exception as exc:
            return {
                "name": name,
                "description": description,
                "status": "failed",
                "error": str(exc),
                "evidence": {},
            }
        return {
            "name": name,
            "description": description,
            "status": "passed",
            "evidence": evidence,
        }

    def _check_context(self, project: str) -> dict[str, Any]:
        context = ContextEngine(self.store)
        event = context.ingest(
            title="Mission context preference",
            content="OpenAgent mission check verifies local-first context sharing.",
            type="preference",
            source="mission-control",
            project=project,
            metadata={"mission_check": True},
        )
        hits = context.search("local-first context", project=project, limit=5)
        snapshot = context.snapshot(project=project)
        return {"event_id": event["id"], "hits": len(hits), "event_count": snapshot["event_count"], "types": snapshot["types"]}

    def _check_chat_models(self, project: str) -> dict[str, Any]:
        chat = ChatService(self.store, self.gateway)
        response = chat.send(
            "Confirm local model gateway behavior for Mission Control.",
            project=project,
            provider="local",
        )
        embedding = self.gateway.embed(["local ai operating system"], provider_id="local")
        return {
            "conversation_id": response["conversation"]["id"],
            "provider": response["provider"],
            "embedding_dimensions": embedding.dimensions,
            "providers": [provider["id"] for provider in self.gateway.specs()],
        }

    def _check_agents(self, project: str) -> dict[str, Any]:
        runtime = MultiAgentRuntime(self.store, self.gateway)
        run = runtime.run_agent("executive", "Verify Mission Control agents", project=project)
        parallel = runtime.run_parallel(
            [
                {"agent": "planning", "goal": "Plan local verification"},
                {"agent": "security", "goal": "Review local privacy boundary"},
            ],
            project=project,
        )
        return {
            "run_id": run["run"]["id"],
            "verification": run["verification"]["status"],
            "parallel_status": parallel["status"],
            "agent_count": len(AgentRegistry(self.store, self.gateway).specs()),
            "marketplace_items": len(runtime.marketplace()),
        }

    def _check_workspace(self, project: str) -> dict[str, Any]:
        workspace = WorkspaceManager(self.store)
        folder = workspace.create_item(kind="folder", title="Mission Folder", project=project)
        note = workspace.create_item(
            kind="wiki_page",
            title="Mission Wiki",
            project=project,
            parent_id=folder["id"],
            content="Feature verification surface for OpenAgent.",
        )
        workspace.create_item(
            kind="research_canvas",
            title="Mission Canvas",
            project=project,
            content="Canvas node for mission evidence.",
        )
        dashboard = workspace.dashboard(project=project)
        canvas = workspace.canvas_model(project=project)
        return {"folder_id": folder["id"], "wiki_id": note["id"], "dashboard_kinds": dashboard["by_kind"], "canvas_nodes": len(canvas["nodes"])}

    def _check_knowledge(self, project: str) -> dict[str, Any]:
        knowledge = KnowledgeEngine(self.store)
        doc = knowledge.ingest_text(
            "Mission RAG Document",
            "OpenAgent combines semantic memory, vector search, graph retrieval, and source attribution.",
            project=project,
            source="mission-control",
        )
        search = knowledge.hybrid_search("semantic memory vector graph", project=project, limit=5)
        synthesis = knowledge.synthesize("What does OpenAgent combine?", project=project)
        replay = knowledge.create_replay(project=project, label="Mission replay", query="OpenAgent")
        return {
            "document_id": doc["id"],
            "search_results": len(search),
            "sources": len(synthesis.get("sources", [])),
            "replay_id": replay["id"],
            "graph_nodes": len(knowledge.graph(project=project)["entities"]),
        }

    def _check_workflows(self, project: str) -> dict[str, Any]:
        workflows = WorkflowEngine(self.store, self.gateway)
        workflow = workflows.create(
            name="Mission workflow",
            project=project,
            steps=[{"action": "search_context", "query": "mission"}],
            trigger_type="manual",
        )
        run = workflows.run(workflow["id"])
        layout = workflows.save_layout(workflow["id"], {"nodes": [{"id": "trigger", "x": 1, "y": 1}]})
        return {"workflow_id": workflow["id"], "run_id": run["id"], "run_status": run["status"], "layout_id": layout["id"]}

    def _check_assistant(self, project: str) -> dict[str, Any]:
        assistant = AssistantManager(self.store)
        goal = assistant.create_goal(title="Mission assistant goal", project=project)
        task = assistant.create_task(title="Mission assistant task", project=project, priority="high", goal_id=goal["id"])
        brief = assistant.daily_brief(project=project)
        meeting = assistant.summarize_meeting(
            transcript="Awais asked to finish OpenAgent features. Action: verify every visible module.",
            project=project,
            title="Mission assistant summary",
        )
        travel = assistant.create_travel_plan(destination="Local deployment", project=project)
        crm = assistant.upsert_crm_record(record_type="lead", name="Mission contact", project=project)
        return {
            "goal_id": goal["id"],
            "task_id": task["id"],
            "brief_tasks": len(brief["open_tasks"]),
            "meeting_summary_event_id": meeting["event"]["id"],
            "travel_plan_id": travel["id"],
            "crm_id": crm["id"],
        }

    def _check_communications(self, project: str) -> dict[str, Any]:
        hub = CommunicationHub(self.store)
        message = hub.import_message(
            platform="email",
            sender="mission@example.local",
            recipients=["user@example.local"],
            subject="Mission follow up",
            body="Please follow up on OpenAgent Mission Control.",
            project=project,
        )
        draft = hub.draft_reply(message["id"], project=project, instructions="Local draft only.")
        event = hub.add_calendar_event(
            title="Mission communication calendar",
            starts_at="2026-07-01T09:00:00+00:00",
            ends_at="2026-07-01T09:30:00+00:00",
            project=project,
        )
        return {"message_id": message["id"], "draft_id": draft["id"], "event_id": event["id"], "inbox": len(hub.inbox(project=project))}

    def _check_voice_media(self, project: str) -> dict[str, Any]:
        voice = VoiceRuntime(self.store)
        session = voice.create_session(project=project, mode="push_to_talk")
        stt = voice.transcribe_audio(
            session["id"],
            audio_data=b"\x1a\x45\xdf\xa3mission-browser-audio",
            filename="mission.webm",
            metadata={"mime_type": "audio/webm"},
        )
        tts = voice.synthesize_text(text="Mission voice check complete.", session_id=session["id"])
        translation = voice.translate_text(text="hello thanks", project=project, target_language="es")
        media = MediaManager(self.store)
        image = media.ingest_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, filename="mission.png", project=project, title="Mission image")
        return {
            "session_id": session["id"],
            "transcript": stt["transcript"],
            "transcript_status": stt["transcript_status"],
            "tts_asset_id": tts["audio_asset_id"],
            "translation": translation["translated_text"],
            "image_asset_id": image["id"],
        }

    def _check_calendar(self, project: str) -> dict[str, Any]:
        calendar = CalendarSystem(self.store)
        cal = calendar.create_calendar("Mission Calendar", project=project)
        event = calendar.create_event(
            "Mission calendar event",
            "2026-07-01T10:00:00+00:00",
            "2026-07-01T11:00:00+00:00",
            project=project,
            calendar_id=cal["id"],
            reminders=[15],
        )
        slots = calendar.find_free_slots(project=project, from_dt="2026-07-01T08:00:00+00:00", to_dt="2026-07-01T12:00:00+00:00", duration_minutes=30)
        return {"calendar_id": cal["id"], "event_id": event["id"], "free_slots": len(slots), "ical_bytes": len(calendar.export_ical(project=project))}

    def _check_twin(self, project: str) -> dict[str, Any]:
        twin = DigitalTwin(self.store)
        profile = twin.update_profile({"work_mode": "mission-control"}, project=project)
        interaction = twin.record_interaction("research", "mission control feature verification", project=project)
        patterns = twin.analyse_patterns(project=project)
        prediction = twin.predict_intent("research mission features", project=project)
        goal = twin.create_goal("Mission twin goal", project=project)
        consolidated = twin.consolidate_memory(project=project)
        return {
            "profile_id": profile["id"],
            "interaction_id": interaction["id"],
            "patterns": len(patterns["patterns"]),
            "predicted_intent": prediction["predicted_intent"],
            "goal_id": goal["id"],
            "facts": len(consolidated["facts"]),
        }

    def _check_vault(self, project: str) -> dict[str, Any]:
        vault = EncryptedVault(self.store)
        entry = vault.store_secret("Mission Secret", "local-only-secret", "mission-passphrase", project=project)
        secret = vault.retrieve_secret(entry["id"], "mission-passphrase", project=project)
        exported = vault.export_vault("mission-passphrase", project=project)
        return {"entry_id": entry["id"], "retrieved": secret["label"], "export_entries": exported["entry_count"]}

    def _check_scraper(self, project: str) -> dict[str, Any]:
        scraper = WebScraper(self.store)
        job = scraper.schedule_scrape("https://example.com/openagent-mission", project=project, interval_hours=24)
        return {"job_id": job["job_id"], "scheduled_jobs": len(scraper.list_scheduled(project=project)), "status": scraper.status(project=project)}

    def _check_local_ai(self, project: str) -> dict[str, Any]:
        local_ai = LocalAIInfrastructure(self.store, self.gateway)
        model = local_ai.register_model(name="mission-local-model", provider="local", metadata={"project": project})
        hardware = local_ai.detect_hardware()
        benchmark = local_ai.benchmark(model["id"], prompt="local benchmark")
        return {"model_id": model["id"], "hardware": hardware["system"], "benchmark_id": benchmark["id"], "marketplace": len(local_ai.marketplace())}

    def _check_finetuning(self, project: str) -> dict[str, Any]:
        tuning = FineTuningManager(self.store)
        dataset = tuning.create_dataset("Mission Dataset", project=project)
        example = tuning.add_example(dataset["id"], "What is OpenAgent?", "A local AI operating system.", project=project)
        job = tuning.create_job(dataset["id"], "mission-local-model", project=project, template_id="lora-fast")
        adapter = tuning.register_adapter("Mission Adapter", "mission-local-model", project=project, eval_score=0.91)
        active = tuning.activate_adapter(adapter["id"], project=project)
        return {"dataset_id": dataset["id"], "example_id": example["id"], "job_id": job["id"], "adapter_active": active["active"]}

    def _check_developer(self, project: str) -> dict[str, Any]:
        developer = DeveloperPlatform(self.store)
        key = developer.create_api_key(label="Mission API Key", scopes=["api.read"])
        sdk = developer.generate_sdk(language="python")
        tool = developer.register_tool(
            tool_id=f"mission.echo.{project}",
            name="Mission Echo",
            description="Mission Control echo tool.",
            schema={"input": "object"},
            handler="echo",
        )
        package = developer.publish_extension_package(
            package_type="plugin",
            name="mission-plugin",
            version="0.1.0",
            source="local",
            manifest={"permissions": ["context.read"]},
        )
        return {"api_key_id": key["id"], "sdk_files": sorted(sdk["files"]), "tool_id": tool["id"], "package_id": package["id"]}

    def _check_security(self, project: str) -> dict[str, Any]:
        security = SecurityPrivacyManager(self.store)
        profile = security.create_sandbox_profile(name=f"mission-sandbox-{project}", permissions=["context.read"], network_allowed=False)
        report = security.create_privacy_report()
        chain = security.seal_audit_events()
        verification = security.verify_audit_chain()
        return {"sandbox_profile_id": profile["id"], "privacy_report_id": report["id"], "sealed": chain["sealed"], "audit_ok": verification["ok"]}

    def _check_production(self, project: str) -> dict[str, Any]:
        production = ProductionRuntime(self.store)
        index = production.create_vector_index(name="mission-index", project=project, dimensions=16)
        vector = production.upsert_vector(index["id"], external_id="mission", content="OpenAgent production vector record")
        search = production.vector_search(index["id"], query="production vector", limit=3)
        adapter = production.register_connector_adapter(
            platform="email",
            mode="local_fixture",
            config={"fixture_payload": {"messages": [{"sender": "mission@example.local", "body": "Mission connector sync", "subject": "Sync"}]}},
        )
        poll = production.poll_connector(adapter["id"], project=project)
        envelope = production.encrypt_envelope(label="mission-envelope", plaintext="mission secret", passphrase="mission-passphrase", recipients=["local-device"], project=project)
        hardware_key = production.register_hardware_key(name="Mission Key", public_key="mission-public-key", project=project)
        desktop = production.run_desktop_automation(actions=[{"action": "record", "note": "mission"}], project=project)
        browser = production.run_browser_automation(actions=[{"action": "extract_title"}], html="<title>Mission</title><p>OpenAgent</p>", project=project)
        app_package = production.package_app(app_type="desktop", platform="windows", project=project)
        agent_package = production.publish_agent_package(name="Mission Agent", version="0.1.0", manifest={"permissions": ["context.read"]}, project=project)
        gpu = production.plan_gpu_management(model_id="mission-local-model", project=project)
        ft = production.create_finetune_job(model_id="mission-local-model", dataset_ref="local://mission", project=project)
        proactive = production.create_proactive_job(name="Mission Proactive", query="mission", project=project)
        proactive_run = production.run_proactive_job(proactive["id"])
        federated = production.aggregate_federated_round(objective="mission-learning", updates=[{"a": 1.0}, {"a": 3.0}], project=project)
        device_a = production.register_sync_device(name="Mission Device A", public_key="mission-a", project=project)
        device_b = production.register_sync_device(name="Mission Device B", public_key="mission-b", project=project)
        sync = production.create_sync_bundle(from_device_id=device_a["id"], to_device_id=device_b["id"], data={"events": ["mission"]}, passphrase="mission-passphrase", project=project)
        sdk = production.publish_sdk_package(language="python", version="0.1.0", project=project)
        security = SecurityPrivacyManager(self.store)
        sandbox_profile = security.create_sandbox_profile(name=f"mission-plugin-{project}", permissions=["context.read"], network_allowed=False)
        sandbox = production.run_sandboxed_plugin(plugin_id="mission-plugin", profile_id=sandbox_profile["id"], input={"permissions": ["context.read"], "network": False}, project=project)
        anchor = production.anchor_audit_log(project=project)
        return {
            "vector_record_id": vector["id"],
            "vector_results": len(search),
            "connector_poll_id": poll["id"],
            "envelope_id": envelope["id"],
            "hardware_key_id": hardware_key["id"],
            "desktop_run_id": desktop["id"],
            "browser_run_id": browser["id"],
            "app_package_id": app_package["id"],
            "agent_package_id": agent_package["id"],
            "gpu_plan_id": gpu["id"],
            "finetune_job_id": ft["id"],
            "proactive_run_id": proactive_run["id"],
            "federated_round_id": federated["id"],
            "sync_bundle_id": sync["id"],
            "sdk_package_id": sdk["id"],
            "sandbox_run_id": sandbox["id"],
            "audit_anchor_id": anchor["id"],
        }
