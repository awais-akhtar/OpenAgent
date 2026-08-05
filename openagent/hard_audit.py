"""Executable hard audit for the OpenAgent vision checklist."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Callable

from .advanced_rag import AdvancedRAGEngine
from .adaptive_engine.compiler import AdaptiveCompiler
from .adaptive_engine.hardware import DeviceProfile, HardwareProfile
from .adaptive_engine.ir import ModelIR, OpCode, build_transformer_graph, lower_graph_to_tensor_ir
from .assistant import AssistantManager
from .capabilities import vision_coverage
from .chat import ChatService
from .communication import CommunicationHub
from .desktop import desktop_status, windows_installation_manifest
from .developer import DeveloperPlatform
from .enterprise import EnterpriseManager
from .knowledge import KnowledgeEngine
from .local_ai import LocalAIInfrastructure
from .media import MediaManager
from .model_gateway import ModelGateway
from .multi_agent import MultiAgentRuntime
from .production_runtime import ProductionRuntime
from .security_privacy import SecurityPrivacyManager
from .semantic_memory import SemanticMemoryManager
from .storage import OpenAgentStore
from .ultra_model_engine import UltraModelEngine
from .voice import VoiceRuntime, synthesize_tone_wav
from .vector_store import VectorStore
from .workflows import WorkflowEngine
from .workspace import WorkspaceManager


CheckFn = Callable[[], dict[str, Any]]


def run_hard_audit(store: OpenAgentStore, *, project: str = "hard-audit") -> dict[str, Any]:
    """Exercise the local product systems and return pass/fail evidence."""

    checks: list[dict[str, Any]] = []

    def check(name: str, area: str, fn: CheckFn) -> None:
        try:
            evidence = fn()
            checks.append({"name": name, "area": area, "status": "passed", "evidence": evidence})
        except Exception as exc:  # pragma: no cover - failure path is surfaced in the response
            checks.append({"name": name, "area": area, "status": "failed", "error": str(exc)})

    gateway = ModelGateway()
    chat = ChatService(store, gateway)
    workflows = WorkflowEngine(store, gateway)
    vector_store = VectorStore(store)
    knowledge = KnowledgeEngine(store, vector_store=vector_store)
    advanced_rag = AdvancedRAGEngine(
        store,
        knowledge=knowledge,
        vector_store=vector_store,
        gateway=gateway,
    )
    semantic = SemanticMemoryManager(store)
    communication = CommunicationHub(store)
    voice = VoiceRuntime(store)
    assistant = AssistantManager(store)
    workspace = WorkspaceManager(store)
    local_ai = LocalAIInfrastructure(store, gateway)
    ultra_model = UltraModelEngine(store)
    enterprise = EnterpriseManager(store)
    developer = DeveloperPlatform(store)
    security = SecurityPrivacyManager(store)
    production = ProductionRuntime(store, gateway)
    multi_agent = MultiAgentRuntime(store, gateway)

    check(
        "capability-verdict",
        "coverage",
        lambda: require(
            vision_coverage()["summary"]["counts"] == {"implemented": 53},
            vision_coverage()["summary"],
        ),
    )
    check(
        "local-model-routing-streaming-embedding",
        "ai",
        lambda: {
            "completion_provider": gateway.complete([{"role": "user", "content": "stay local"}]).provider,
            "stream_chunks": len(list(gateway.stream_complete([{"role": "user", "content": "stream"}]))),
            "embedding_dimensions": gateway.embed(["private memory"]).dimensions,
        },
    )
    check(
        "chat-context-memory",
        "workspace",
        lambda: chat.send("Remember the hard audit context.", project=project)["conversation"],
    )
    check(
        "knowledge-rag",
        "knowledge",
        lambda: knowledge.ingest_text(
            title="Hard audit document",
            content="OpenAgent performs local RAG, graph extraction, tags, and synthesis.",
            project=project,
        ),
    )
    check(
        "advanced-rag-vector-cpu-engine",
        "knowledge",
        lambda: advanced_rag_probe(advanced_rag, gateway, project),
    )
    check(
        "semantic-memory-vector-graph",
        "memory",
        lambda: semantic.add_memory(
            title="Hard audit preference",
            content="Prefer local private models.",
            project=project,
            memory_type="preference",
            tags=["privacy"],
        ),
    )
    check(
        "multi-agent-delegation-verification",
        "agents",
        lambda: {
            "run": multi_agent.run_agent("planning", "Plan the hard audit", project=project),
            "parallel": multi_agent.run_parallel(
                [{"agent": "research", "goal": "Check docs"}, {"agent": "coding", "goal": "Check tests"}],
                project=project,
            ),
        },
    )
    check(
        "workflow-actions",
        "automation",
        lambda: workflows.run(
            workflows.create(
                name="Hard audit workflow",
                project=project,
                steps=[
                    {"action": "add_event", "title": "Hard audit event", "content": "workflow event"},
                    {"action": "file_write", "path": "audit/result.txt", "content": "ok"},
                    {"action": "file_read", "path": "audit/result.txt"},
                ],
            )["id"]
        ),
    )
    check(
        "communication-calendar-crm",
        "assistant",
        lambda: {
            "message": communication.import_message(
                platform="email",
                sender="owner@example.com",
                subject="Hard audit",
                body="Please follow up on the hard audit.",
                project=project,
            ),
            "calendar": communication.add_calendar_event(
                title="Hard audit review",
                starts_at="2026-06-26T10:00:00",
                ends_at="2026-06-26T10:30:00",
                project=project,
            ),
            "task": assistant.create_task(title="Finish hard audit", project=project),
            "crm": assistant.upsert_crm_record(record_type="relationship", name="Owner", project=project),
        },
    )
    check(
        "voice-stt-tts-clone",
        "voice",
        lambda: voice_probe(voice, project),
    )
    check(
        "workspace-objects",
        "workspace",
        lambda: {
            "folder": workspace.create_item(kind="folder", title="Hard Audit", project=project),
            "dashboard": workspace.create_item(kind="dashboard", title="Audit Dashboard", project=project),
            "template": workspace.create_item(kind="template", title="Audit Template", content="Hard audit template", project=project),
        },
    )
    check(
        "media-understanding",
        "knowledge",
        lambda: MediaManager(store).ingest_bytes(b"%PDF-1.4\nBT (Hard audit PDF text) Tj ET\n%%EOF", filename="audit.pdf", project=project),
    )
    check(
        "local-ai-model-management",
        "local-ai",
        lambda: {
            "hardware": local_ai.detect_hardware(),
            "model": local_ai.register_model(name="hard-audit-local", provider="local", metadata={"audit": True}),
            "benchmark": local_ai.benchmark(local_ai.register_model(name="hard-audit-bench", provider="local", metadata={"audit": True})["id"]),
        },
    )
    check(
        "ultra-scale-cpu-planning-and-verification",
        "local-ai",
        lambda: ultra_model_probe(ultra_model, project),
    )
    check(
        "adaptive-heterogeneous-compiler",
        "local-ai",
        adaptive_compiler_probe,
    )
    check(
        "enterprise-security-developer",
        "platform",
        lambda: platform_probe(enterprise, developer, security, project),
    )
    check(
        "production-runtime",
        "production",
        lambda: production_probe(production, security, project),
    )
    check(
        "windows-native-desktop-install",
        "deployment",
        lambda: desktop_probe(store, project),
    )

    ultra_model.close()
    advanced_rag.close()
    passed = sum(1 for item in checks if item["status"] == "passed")
    failed = len(checks) - passed
    return {
        "ok": failed == 0,
        "project": project,
        "passed": passed,
        "failed": failed,
        "total": len(checks),
        "checks": checks,
        "privacy_boundary": "Hard audit uses only local runtime operations and writes evidence to the configured OpenAgent data directory.",
    }


def advanced_rag_probe(
    rag: AdvancedRAGEngine,
    gateway: ModelGateway,
    project: str,
) -> dict[str, Any]:
    result = rag.query("How does OpenAgent perform local RAG?", project=project, generate=False)
    status = rag.status(project=project)
    benchmark = gateway.benchmark_inference_engine(iterations=25)
    return require(
        bool(result["sources"]),
        {
            "algorithm": result["retrieval"]["algorithm"],
            "sources": len(result["sources"]),
            "confidence": result["confidence"],
            "active_accelerators": status["vector_store"]["active_accelerators"],
            "inference_planning_mean_ms": benchmark["planning_ms"]["mean"],
            "inference_cache_lookup_mean_ms": benchmark["cache_lookup_mean_ms"],
        },
    )


def ultra_model_probe(engine: UltraModelEngine, project: str) -> dict[str, Any]:
    plan = engine.plan(
        project=project,
        specification={
            "name": "Hard audit 200B planning profile",
            "architecture": "dense",
            "parameter_count_b": 200,
            "quantization_bits": 2,
        },
        context_tokens=2048,
        hardware={
            "total_ram_gb": 32,
            "available_ram_gb": 24,
            "memory_bandwidth_gbps": 30,
            "disk_bandwidth_gbps": 2,
            "cpu_threads": 16,
        },
    )
    verification = engine.verify_output(
        "OpenAgent keeps model planning local [S1].",
        evidence=["OpenAgent keeps model planning local."],
        project=project,
    )
    return require(
        plan["mode"] == "planning-only" and verification["status"] == "verified",
        {
            "engine": engine.status(project=project)["engine"],
            "plan": plan,
            "verification": verification,
        },
    )


def adaptive_compiler_probe() -> dict[str, Any]:
    operations = tuple(item.value for item in OpCode)
    cpu = DeviceProfile(
        id="cpu:0",
        kind="cpu",
        name="Hard audit CPU",
        vendor="generic",
        memory_gb=16,
        unified_memory=True,
        compute_tops=2,
        memory_bandwidth_gbps=30,
        precisions=("fp32", "fp16", "int8", "int4"),
        supported_ops=operations,
        backends=("openvino",),
    )
    gpu = DeviceProfile(
        id="gpu:0",
        kind="gpu",
        name="Hard audit GPU",
        vendor="generic",
        memory_gb=12,
        unified_memory=True,
        compute_tops=12,
        memory_bandwidth_gbps=80,
        precisions=("fp32", "fp16", "int8", "int4"),
        supported_ops=tuple(value for value in operations if value not in {"tokenization", "sampling"}),
        backends=("openvino",),
    )
    hardware = HardwareProfile(
        id="hard-audit-hardware",
        platform="local-test",
        total_ram_gb=32,
        available_ram_gb=24,
        cpu_threads=16,
        devices=(cpu, gpu),
        runtimes={"openvino": {"available": True}},
        power={"ac_connected": True},
        discovered_at=time.time(),
    )
    graph = build_transformer_graph(
        block_count=2,
        weight_bytes=2_000_000_000,
        parameter_count=1_000_000_000,
        embedding_length=2048,
    )
    model = ModelIR(
        id="hard-audit-model",
        name="Hard audit model",
        source_path="hard-audit.onnx",
        source_format="onnx",
        fingerprint="hard-audit-model-fingerprint",
        architecture="transformer",
        parameter_count=1_000_000_000,
        weight_bytes=2_000_000_000,
        quantization_bits=16,
        context_length=4096,
        embedding_length=2048,
        block_count=2,
        graph_nodes=graph,
        tensor_nodes=lower_graph_to_tensor_ir(graph),
    )
    package = AdaptiveCompiler().compile(
        model,
        hardware,
        prompt_tokens=128,
        context_tokens=2048,
        memory_budget_gb=4,
        preferred_device_id="gpu:0",
    )
    compute_devices = {
        item.device_id
        for item in package.decode.placements
        if item.op not in {"tokenization", "sampling"}
    }
    return require(
        package.status == "planned"
        and package.prefill.transfer_count <= 2
        and package.decode.transfer_count <= 2
        and len(compute_devices) == 1
        and compute_devices == {"gpu:0"}
        and package.constraints["measured_preferred_device"] == "gpu:0"
        and bool(package.kernel_plan["decode"]),
        {
            "planner": package.planner,
            "prefill_regions": package.prefill.device_regions,
            "decode_regions": package.decode.device_regions,
            "kernel_regions": len(package.kernel_plan["decode"]),
            "memory_plan": package.memory_plan,
            "measured_preferred_device": package.constraints["measured_preferred_device"],
        },
    )


def voice_probe(voice: VoiceRuntime, project: str) -> dict[str, Any]:
    session = voice.create_session(project=project, mode="push_to_talk")
    clone = voice.clone_voice_from_audio(
        name="Hard Audit Voice",
        audio_data=synthesize_tone_wav("voice clone sample"),
        project=project,
        consent="Hard audit submitted this local sample.",
    )
    stt = voice.transcribe_audio(
        session["id"],
        audio_data=synthesize_tone_wav("speech sample"),
        transcript_hint="Hard audit speech transcribed.",
    )
    tts = voice.synthesize_text(session_id=session["id"], text="Hard audit audio ready.")
    translation = voice.translate_text(text="hello thanks", project=project, target_language="es")
    return {"session": session["id"], "clone": clone["persona"]["id"], "stt": stt["transcript"], "tts": tts["audio_asset_id"], "translation": translation["translated_text"]}


def desktop_probe(store: OpenAgentStore, project: str) -> dict[str, Any]:
    status = desktop_status(store.data_dir, project=project)
    installer = windows_installation_manifest()
    module_ids = {module["id"] for module in status["desktop_modules"]}
    missing = [relative for relative, file_status in installer["files"].items() if not file_status["exists"]]
    require(status["browser_required"] is False, status)
    require(status["windows_installable"] is True, status)
    require(status["entrypoints"]["package_script"] == "openagent-desktop", status)
    require({"workspace", "knowledge", "assistant", "models", "voice_media", "automation", "security_developer"}.issubset(module_ids), status)
    require(not missing, {"missing": missing, "installer": installer})
    return {
        "status": status,
        "installer": installer,
        "privacy_boundary": status["privacy_boundary"],
    }


def platform_probe(
    enterprise: EnterpriseManager,
    developer: DeveloperPlatform,
    security: SecurityPrivacyManager,
    project: str,
) -> dict[str, Any]:
    user = enterprise.upsert_user(email="audit@example.com", name="Audit Owner")
    workspace = enterprise.create_workspace(name="Audit Workspace", owner_user_id=user["id"])
    enterprise.add_member(workspace_id=workspace["id"], user_id=user["id"], role_id="admin")
    key = developer.create_api_key(label="hard-audit", scopes=["api.read"])
    sdk = developer.generate_sdk(language="python")
    secret = security.store_secret(name="audit-secret", value="private", passphrase="passphrase")
    profile = security.create_sandbox_profile(name="audit-sandbox", permissions=["context.read"])
    return {"user": user["id"], "workspace": workspace["id"], "api_key": key["id"], "sdk_files": sorted(sdk["files"]), "secret": secret["id"], "sandbox_profile": profile["id"], "project": project}


def production_probe(
    production: ProductionRuntime,
    security: SecurityPrivacyManager,
    project: str,
) -> dict[str, Any]:
    policy = production.upsert_policy_rule(scope="external.model", effect="deny", conditions={"unless": "allow_external"}, project=project)
    index = production.create_vector_index(name="hard-audit", project=project)
    vector = production.upsert_vector(index["id"], external_id="audit-doc", content="hard audit private vector")
    browser = production.run_browser_automation(
        project=project,
        html='<title>Hard Audit</title><form><input name="email"></form>',
        actions=[{"action": "fill", "selector": "email", "value": "audit@example.com"}, {"action": "assert_text", "text": "Hard Audit"}],
    )
    desktop = production.run_desktop_automation(
        project=project,
        actions=[{"action": "write_file", "path": "audit.txt", "content": "ok"}, {"action": "read_file", "path": "audit.txt"}],
    )
    package = production.package_app(app_type="pwa", platform="web", project=project)
    proactive = production.create_proactive_job(name="Hard Audit Monitor", query="hard audit", project=project)
    federated = production.aggregate_federated_round(objective="audit", updates=[{"score": 1.0}, {"score": 3.0}], project=project)
    device_a = production.register_sync_device(name="A", public_key="pub-a", project=project)
    device_b = production.register_sync_device(name="B", public_key="pub-b", project=project)
    bundle = production.create_sync_bundle(from_device_id=device_a["id"], to_device_id=device_b["id"], data={"audit": True}, passphrase="passphrase", project=project)
    profile = security.create_sandbox_profile(name="hard-audit-plugin", permissions=["context.read"])
    sandbox = production.run_sandboxed_plugin(plugin_id="audit-plugin", profile_id=profile["id"], input={"permissions": ["context.read"]}, project=project)
    token = make_hs256_jwt({"iss": "https://idp.audit", "aud": "openagent", "sub": "audit@example.com", "exp": int(time.time()) + 600}, b"audit-secret")
    oidc = production.validate_oidc_jwks(token=token, jwks={"keys": [{"kty": "oct", "kid": "audit", "alg": "HS256", "k": b64url(b"audit-secret")}]}, issuer="https://idp.audit", audience="openagent", project=project)
    return {"policy": policy["id"], "vector": vector["id"], "browser": browser["id"], "desktop": desktop["id"], "package": package["id"], "proactive": proactive["id"], "federated": federated["id"], "sync": bundle["id"], "sandbox": sandbox["id"], "oidc_valid": oidc["valid"]}


def require(condition: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    if not condition:
        raise AssertionError(json.dumps(evidence, sort_keys=True))
    return evidence


def make_hs256_jwt(claims: dict[str, object], secret: bytes) -> str:
    header = {"alg": "HS256", "kid": "audit", "typ": "JWT"}
    signing_input = ".".join([b64url_json(header), b64url_json(claims)])
    signature = hmac.new(secret, signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{b64url(signature)}"


def b64url_json(payload: dict[str, object]) -> str:
    return b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
