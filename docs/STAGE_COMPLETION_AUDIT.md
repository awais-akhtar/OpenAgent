# Stage Completion Audit

This audit tracks the roadmap stages against the repository's local code, tests,
documentation, and trust boundaries. A stage marked as a local implementation
means the corresponding code path and contract exist here. It does not mean
that every optional dependency, provider credential, accelerator, installer,
or external service is available on every machine.

## Current Result

Stages 0 through 15 have local code or local control surfaces in this tree.
Some later stages are deliberately record-based: they describe policy,
planning, packaging, or deployment work that still needs the target provider,
hardware, credentials, or operating system before it can be certified. Stage 13
adds local product behavior for submitted-audio STT, audio generation,
app-package records, browser/form boundaries, and desktop file actions. Stage
14 adds a native Windows desktop install path. Stage 15 exposes the current
local feature surface in the desktop workbench; it is not a claim that every
external integration is complete.

For the item-by-item check against the full original vision, see `docs/VISION_COVERAGE.md` or the local `/api/capabilities` endpoint.

## Stage Checklist

| Stage | Status | Main code | Tests | Documentation |
| --- | --- | --- | --- | --- |
| Stage 0: Local Core implementation | Local implementation | `openagent/storage.py`, `openagent/context.py`, `openagent/server.py`, `web/` | `tests/test_storage.py`, `tests/test_server.py` | `docs/ARCHITECTURE.md`, `docs/LOCAL_FIRST.md`, `README.md` |
| Stage 1: Model Gateway | Local implementation | `openagent/model_gateway.py`, chat/model API routes | `tests/test_model_gateway.py`, `tests/test_server.py` | `docs/MODEL_GATEWAY.md`, `docs/PRIVACY_SECURITY.md` |
| Stage 2: Memory and Knowledge | Local implementation | `openagent/knowledge.py`, context replay and knowledge API routes | `tests/test_knowledge.py`, `tests/test_server.py` | `docs/KNOWLEDGE_ENGINE.md` |
| Stage 3: Multi-Agent Runtime | Local implementation | `openagent/agents.py`, `openagent/multi_agent.py` | `tests/test_agents.py`, `tests/test_multi_agent.py`, `tests/test_server.py` | `docs/AGENTS.md`, `docs/MULTI_AGENT_RUNTIME.md` |
| Stage 4: Workflow Automation | Local implementation | `openagent/workflows.py`, workflow API routes | `tests/test_workflow_automation.py`, `tests/test_chat_and_workflows.py`, `tests/test_server.py` | `docs/WORKFLOWS.md` |
| Stage 5: Communication Hub | Local implementation | `openagent/communication.py`, communication API routes | `tests/test_communication.py`, `tests/test_server.py` | `docs/COMMUNICATION_HUB.md` |
| Stage 6: Local AI Infrastructure | Local implementation | `openagent/local_ai.py`, local AI API routes | `tests/test_local_ai.py`, `tests/test_server.py` | `docs/LOCAL_AI_INFRASTRUCTURE.md` |
| Stage 7: Enterprise and Team Deployment | Local controls and manifests | `openagent/enterprise.py`, enterprise API routes, `deploy/kubernetes/` | `tests/test_enterprise.py`, `tests/test_server.py` | `docs/ENTERPRISE_DEPLOYMENT.md`, `SECURITY.md` |
| Stage 8: Developer Platform and Extensibility | Local implementation | `openagent/developer.py`, developer and GraphQL API routes | `tests/test_developer_platform.py`, `tests/test_server.py` | `docs/DEVELOPER_PLATFORM.md` |
| Stage 9: Security and Privacy Hardening | Local controls | `openagent/security_privacy.py`, security/privacy API routes | `tests/test_security_privacy.py`, `tests/test_server.py` | `docs/SECURITY_PRIVACY_HARDENING.md`, `SECURITY.md` |
| Stage 10: Workspace and Personal Assistant Layer | Local implementation | `openagent/workspace.py`, `openagent/assistant.py`, chat, knowledge, workflow, and API route expansions | `tests/test_workspace_assistant.py`, `tests/test_chat_and_workflows.py`, `tests/test_knowledge.py`, `tests/test_workflow_automation.py`, `tests/test_server.py` | `README.md`, `docs/ROADMAP.md`, `docs/VISION_COVERAGE.md` |
| Stage 11: Semantic, Media, Voice, and UI Runtime | Local implementation | `openagent/semantic_memory.py`, `openagent/media.py`, `openagent/voice.py`, communication sync, workflow layout, API route expansions, `web/` | `tests/test_semantic_memory.py`, `tests/test_media_voice.py`, `tests/test_communication.py`, `tests/test_workflow_automation.py`, `tests/test_web_ui.py`, `tests/test_server.py` | `docs/SEMANTIC_MEDIA_VOICE.md`, `docs/ROADMAP.md`, `docs/VISION_COVERAGE.md` |
| Stage 12: Production Runtime and Final Coverage | Local controls and records | `openagent/production_runtime.py`, production API routes | `tests/test_production_runtime.py`, `tests/test_server.py` | `docs/PRODUCTION_RUNTIME.md`, `docs/FEATURE_STATUS.md`, `docs/VISION_COVERAGE.md` |
| Stage 13: Full Local Product Runtime | Local implementation and package surfaces | `openagent/production_runtime.py`, `openagent/voice.py`, generated app package directories, browser/form automation, desktop local file automation | `tests/test_production_runtime.py`, `tests/test_media_voice.py`, `tests/test_server.py`, `tests/test_capabilities.py` | `README.md`, `docs/ROADMAP.md`, `docs/VISION_COVERAGE.md`, `docs/STAGE_COMPLETION_AUDIT.md` |
| Stage 14: Native Windows Desktop Installability | Local Windows implementation | `openagent/desktop.py`, `installers/windows/`, `MANIFEST.in`, `openagent/production_runtime.py` | `tests/test_desktop.py`, `tests/test_hard_audit.py`, `tests/test_production_runtime.py`, `tests/test_capabilities.py` | `docs/WINDOWS_DESKTOP.md`, `README.md`, `docs/ROADMAP.md`, `docs/STAGE_COMPLETION_AUDIT.md` |
| Stage 15: Native Desktop Workbench | Local workbench implementation | `openagent/desktop.py`, `openagent/workspace.py`, `openagent/knowledge.py`, `openagent/assistant.py`, `openagent/communication.py`, `openagent/local_ai.py`, `openagent/voice.py`, `openagent/production_runtime.py`, `openagent/security_privacy.py`, `openagent/developer.py` | `tests/test_desktop.py`, `tests/test_hard_audit.py`, `tests/test_capabilities.py` | `docs/WINDOWS_DESKTOP.md`, `README.md`, `docs/ROADMAP.md`, `docs/STAGE_COMPLETION_AUDIT.md` |

The hard audit is executable through `POST /api/capabilities/hard-audit`, `openagent-hard-check`, `openagent-desktop --hard-audit-json`, and `openagent/hard_audit.py`. It checks the local contracts for model routing, chat, knowledge/RAG, semantic memory, agents, workflows, communication/calendar/assistant, voice records and local engines, workspace objects, media, local AI, enterprise/developer/security, production-control behavior, and the native Windows desktop path. It does not certify an external provider, a missing model server, a hardware backend, or a production deployment. The desktop feature smoke is executable through `openagent-desktop --feature-smoke-json`.

## Definition-of-Done Review

- Tests for critical behavior: covered by the standard-library suite with stage-specific tests.
- Documentation for configuration and trust boundaries: covered in stage docs, security docs, and this audit.
- No hidden network access: external model, workflow, connector, and download actions require explicit configuration or approval flags.
- Migration path for stored data: SQLite tables are initialized idempotently with `CREATE TABLE IF NOT EXISTS`; future breaking schema work should add versioned migrations.
- Clear user controls for permissions: agent permissions, workflow approvals, communication send approvals, network approval flags, Stage 7 RBAC/policy records, Stage 8 developer tool approvals, Stage 9 egress/share/sandbox controls, and Stage 12 production policy gates are available.
- Auditable actions: desktop launch, desktop feature smoke, chat, agent, workflow, communication, workspace, semantic memory, media, voice, production runtime, local AI, enterprise admin, developer platform, and security/privacy operations write local records or audit events.

## Deployment-Specific Integrations

Stage 15 closes the local desktop feature-discoverability gap in this roadmap. Deployments may still need external provider polling, hardware security modules, certified cryptography, signed desktop installers, mobile store builds, or enterprise IdP-specific policies. Those dependencies are intentionally kept visible rather than being represented as completed by local metadata alone.
