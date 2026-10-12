"""Ordered SQLite schema migrations for the local OpenAgent store."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3
from typing import Iterable
from uuid import uuid4

from .models import utc_now


class MigrationError(RuntimeError):
    """A schema migration cannot be applied or verified."""


@dataclass(frozen=True, slots=True)
class AddColumn:
    table: str
    definition: str

    @property
    def name(self) -> str:
        return self.definition.split(None, 1)[0].strip('"[]`')

    @property
    def canonical(self) -> str:
        return f"ALTER TABLE {self.table} ADD COLUMN {self.definition.strip()}"


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]
    add_columns: tuple[AddColumn, ...] = ()
    destructive: bool = False
    legacy_checksum: str = ""

    @property
    def checksum(self) -> str:
        if self.legacy_checksum:
            return self.legacy_checksum
        canonical = "\n".join(statement.strip() for statement in self.statements)
        if self.add_columns:
            additions = "\n".join(column.canonical for column in self.add_columns)
            canonical = f"{canonical}\n{additions}" if canonical else additions
        return hashlib.sha256(f"{self.version}:{self.name}\n{canonical}".encode("utf-8")).hexdigest()


CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
 id TEXT PRIMARY KEY, type TEXT NOT NULL, source TEXT NOT NULL, project TEXT NOT NULL,
 title TEXT NOT NULL, content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_project_created ON events(project, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
CREATE TABLE IF NOT EXISTS agent_runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT NOT NULL, goal TEXT NOT NULL,
 response_json TEXT NOT NULL, project TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_project_created ON agent_runs(project, created_at DESC);
CREATE TABLE IF NOT EXISTS conversations (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_project_updated ON conversations(project, updated_at DESC);
CREATE TABLE IF NOT EXISTS chat_messages (
 id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
 provider TEXT NOT NULL, model TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
 FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_created ON chat_messages(conversation_id, created_at ASC);
CREATE TABLE IF NOT EXISTS audit_events (
 id TEXT PRIMARY KEY, actor TEXT NOT NULL, action TEXT NOT NULL, resource TEXT NOT NULL,
 project TEXT NOT NULL, status TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_project_created ON audit_events(project, created_at DESC);
CREATE TABLE IF NOT EXISTS workflows (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
 trigger_type TEXT NOT NULL DEFAULT 'manual', steps_json TEXT NOT NULL DEFAULT '[]', enabled INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workflows_project_updated ON workflows(project, updated_at DESC);
CREATE TABLE IF NOT EXISTS workflow_runs (
 id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, project TEXT NOT NULL, status TEXT NOT NULL,
 result_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, completed_at TEXT NOT NULL,
 FOREIGN KEY(workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_workflow_runs_project_created ON workflow_runs(project, created_at DESC);
CREATE TABLE IF NOT EXISTS job_queue (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, job_type TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}',
 state TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
 max_attempts INTEGER NOT NULL DEFAULT 3, run_after TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
 progress_message TEXT NOT NULL DEFAULT '', result_json TEXT NOT NULL DEFAULT '{}', last_error TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT NOT NULL DEFAULT '', cancelled_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_job_queue_ready ON job_queue(state, run_after, priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_job_queue_project_updated ON job_queue(project, updated_at DESC);
CREATE TABLE IF NOT EXISTS job_attempts (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL, worker_id TEXT NOT NULL, state TEXT NOT NULL,
 started_at TEXT NOT NULL, finished_at TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
 result_json TEXT NOT NULL DEFAULT '{}', FOREIGN KEY(job_id) REFERENCES job_queue(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_job_attempts_job ON job_attempts(job_id, started_at DESC);
CREATE TABLE IF NOT EXISTS job_locks (
 job_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL, lease_until TEXT NOT NULL, acquired_at TEXT NOT NULL,
 heartbeat_at TEXT NOT NULL, FOREIGN KEY(job_id) REFERENCES job_queue(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_job_locks_lease_until ON job_locks(lease_until);
CREATE TABLE IF NOT EXISTS job_dead_letters (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL, project TEXT NOT NULL, job_type TEXT NOT NULL,
 payload_json TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '', attempts INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL, FOREIGN KEY(job_id) REFERENCES job_queue(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_job_dead_letters_project_created ON job_dead_letters(project, created_at DESC);
"""


VECTOR_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS vector_store_collections (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL,
 adapter TEXT NOT NULL DEFAULT 'sqlite', dimensions INTEGER NOT NULL,
 distance TEXT NOT NULL DEFAULT 'cosine', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_vector_store_collections_project_name
 ON vector_store_collections(project, name);
CREATE TABLE IF NOT EXISTS vector_store_chunks (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, collection_id TEXT NOT NULL,
 source_id TEXT NOT NULL, chunk_id TEXT NOT NULL, content_hash TEXT NOT NULL,
 content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
 embedding_provider TEXT NOT NULL, embedding_model TEXT NOT NULL,
 dimensions INTEGER NOT NULL, vector_json TEXT NOT NULL,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 FOREIGN KEY(collection_id) REFERENCES vector_store_collections(id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_vector_store_chunks_lineage
 ON vector_store_chunks(collection_id, source_id, chunk_id);
CREATE INDEX IF NOT EXISTS idx_vector_store_chunks_project_collection
 ON vector_store_chunks(project, collection_id, updated_at DESC);
"""


CALENDAR_SYNC_SCHEMA = """
CREATE TABLE IF NOT EXISTS calendar_sync_providers (
 id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default', provider TEXT NOT NULL,
 base_url TEXT NOT NULL DEFAULT '', account_ref TEXT NOT NULL DEFAULT '',
 calendar_id TEXT NOT NULL DEFAULT '', config_json TEXT NOT NULL DEFAULT '{}',
 status TEXT NOT NULL DEFAULT 'connected', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calendar_sync_providers_project
 ON calendar_sync_providers(project);
CREATE TABLE IF NOT EXISTS calendar_sync_jobs (
 id TEXT PRIMARY KEY, provider_id TEXT NOT NULL, project TEXT NOT NULL DEFAULT 'default',
 direction TEXT NOT NULL DEFAULT 'pull', conflict_policy TEXT NOT NULL DEFAULT 'latest_update',
 state TEXT NOT NULL DEFAULT 'queued', pulled INTEGER NOT NULL DEFAULT 0,
 pushed INTEGER NOT NULL DEFAULT 0, conflicts INTEGER NOT NULL DEFAULT 0,
 error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
 completed_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_calendar_sync_jobs_provider
 ON calendar_sync_jobs(provider_id, created_at DESC);
CREATE TABLE IF NOT EXISTS calendar_sync_uid_map (
 id TEXT PRIMARY KEY, provider_id TEXT NOT NULL, local_event_id TEXT NOT NULL,
 remote_uid TEXT NOT NULL, remote_hash TEXT NOT NULL DEFAULT '',
 local_hash TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,
 UNIQUE(provider_id, remote_uid)
);
CREATE TABLE IF NOT EXISTS calendar_sync_conflicts (
 id TEXT PRIMARY KEY, provider_id TEXT NOT NULL, project TEXT NOT NULL DEFAULT 'default',
 local_event_id TEXT NOT NULL DEFAULT '', remote_uid TEXT NOT NULL DEFAULT '',
 local_payload_json TEXT NOT NULL DEFAULT '{}', remote_payload_json TEXT NOT NULL DEFAULT '{}',
 local_updated_at TEXT NOT NULL DEFAULT '', remote_seen_at TEXT NOT NULL DEFAULT '',
 status TEXT NOT NULL DEFAULT 'open', resolution TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL, resolved_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_calendar_sync_conflicts_project
 ON calendar_sync_conflicts(project, status);
"""


INTEGRATION_HUB_SCHEMA = """
CREATE TABLE IF NOT EXISTS integration_pairing_codes (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, label TEXT NOT NULL,
 code_hash TEXT NOT NULL UNIQUE, scopes_json TEXT NOT NULL DEFAULT '[]',
 status TEXT NOT NULL, expires_at TEXT NOT NULL, created_at TEXT NOT NULL,
 redeemed_at TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_integration_pairing_project_created
 ON integration_pairing_codes(project, created_at DESC);
CREATE TABLE IF NOT EXISTS integration_preferences (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, scope TEXT NOT NULL,
 key TEXT NOT NULL, value_json TEXT NOT NULL, updated_at TEXT NOT NULL,
 UNIQUE(project, scope, key)
);
CREATE INDEX IF NOT EXISTS idx_integration_preferences_project_scope
 ON integration_preferences(project, scope);
CREATE TABLE IF NOT EXISTS integration_diagnostics (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, report_json TEXT NOT NULL,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_integration_diagnostics_project_created
 ON integration_diagnostics(project, created_at DESC);
CREATE TABLE IF NOT EXISTS integration_safe_command_runs (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, command_key TEXT NOT NULL,
 status TEXT NOT NULL, output TEXT NOT NULL DEFAULT '',
 error TEXT NOT NULL DEFAULT '', return_code INTEGER NOT NULL DEFAULT 0,
 duration_ms REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_integration_safe_runs_project_created
 ON integration_safe_command_runs(project, created_at DESC);
"""


CONTACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL,
 handles_json TEXT NOT NULL DEFAULT '{}', notes TEXT NOT NULL DEFAULT '',
 metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contacts_project_name ON contacts(project, name);
"""


WORKSPACE_SCHEMA = """
CREATE TABLE IF NOT EXISTS workspace_items (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, kind TEXT NOT NULL,
 title TEXT NOT NULL, parent_id TEXT NOT NULL DEFAULT '',
 content TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workspace_items_project_kind
 ON workspace_items(project, kind, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_workspace_items_parent
 ON workspace_items(project, parent_id);
"""


ASSISTANT_SCHEMA = """
CREATE TABLE IF NOT EXISTS assistant_profiles (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, user_id TEXT NOT NULL,
 preferences_json TEXT NOT NULL DEFAULT '{}', facts_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assistant_goals (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL,
 description TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'active',
 horizon TEXT NOT NULL DEFAULT 'short', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assistant_tasks (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'open', priority TEXT NOT NULL DEFAULT 'normal',
 due_at TEXT NOT NULL DEFAULT '', goal_id TEXT NOT NULL DEFAULT '',
 contact_id TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assistant_tasks_project_status
 ON assistant_tasks(project, status, due_at);
CREATE TABLE IF NOT EXISTS assistant_meeting_briefs (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL,
 starts_at TEXT NOT NULL DEFAULT '', attendees_json TEXT NOT NULL DEFAULT '[]',
 notes TEXT NOT NULL DEFAULT '', brief_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assistant_travel_plans (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, destination TEXT NOT NULL,
 starts_at TEXT NOT NULL DEFAULT '', ends_at TEXT NOT NULL DEFAULT '',
 plan_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crm_records (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, record_type TEXT NOT NULL,
 name TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
 value REAL NOT NULL DEFAULT 0, contact TEXT NOT NULL DEFAULT '',
 notes TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_crm_records_project_type
 ON crm_records(project, record_type, status);
"""


MCP_SCHEMA = """
CREATE TABLE IF NOT EXISTS mcp_servers (
 id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
 owner TEXT NOT NULL DEFAULT 'local', name TEXT NOT NULL,
 transport TEXT NOT NULL DEFAULT 'stdio', command_json TEXT NOT NULL DEFAULT '[]',
 endpoint TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'configured',
 metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_project_owner
 ON mcp_servers(project, owner, updated_at DESC);
CREATE TABLE IF NOT EXISTS mcp_tools (
 id TEXT PRIMARY KEY, server_id TEXT NOT NULL,
 project TEXT NOT NULL DEFAULT 'default', owner TEXT NOT NULL DEFAULT 'local',
 name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
 enabled INTEGER NOT NULL DEFAULT 1, approval_required INTEGER NOT NULL DEFAULT 1,
 schema_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL,
 FOREIGN KEY(server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_server ON mcp_tools(server_id, name);
CREATE TABLE IF NOT EXISTS mcp_tool_permissions (
 id TEXT PRIMARY KEY, tool_id TEXT NOT NULL,
 project TEXT NOT NULL DEFAULT 'default', owner TEXT NOT NULL DEFAULT 'local',
 enabled INTEGER NOT NULL DEFAULT 1, approval_required INTEGER NOT NULL DEFAULT 1,
 reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
 FOREIGN KEY(tool_id) REFERENCES mcp_tools(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_permissions_tool
 ON mcp_tool_permissions(tool_id, created_at DESC);
CREATE TABLE IF NOT EXISTS mcp_connection_events (
 id TEXT PRIMARY KEY, server_id TEXT NOT NULL,
 event_type TEXT NOT NULL, detail_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL,
 FOREIGN KEY(server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
);
"""


TWIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS twin_profiles (
 id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
 profile_json TEXT NOT NULL DEFAULT '{}', version INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_twin_profile_project ON twin_profiles(project);
CREATE TABLE IF NOT EXISTS twin_interactions (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, kind TEXT NOT NULL,
 content TEXT NOT NULL DEFAULT '', topics_json TEXT NOT NULL DEFAULT '[]',
 hour_of_day INTEGER NOT NULL DEFAULT 12, day_of_week TEXT NOT NULL DEFAULT '',
 context_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_twin_interactions_project
 ON twin_interactions(project, created_at DESC);
CREATE TABLE IF NOT EXISTS twin_preferences (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, category TEXT NOT NULL,
 key TEXT NOT NULL, value TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
 source TEXT NOT NULL DEFAULT 'inferred', created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_twin_pref_key
 ON twin_preferences(project, category, key);
CREATE TABLE IF NOT EXISTS twin_patterns (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, pattern_type TEXT NOT NULL,
 data_json TEXT NOT NULL DEFAULT '{}', confidence REAL NOT NULL DEFAULT 0.5,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS twin_predictions (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, prediction_type TEXT NOT NULL,
 prediction TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 0.5,
 context_json TEXT NOT NULL DEFAULT '{}', acted_on INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS twin_goals (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL,
 description TEXT NOT NULL DEFAULT '', target_date TEXT NOT NULL DEFAULT '',
 status TEXT NOT NULL DEFAULT 'active', progress REAL NOT NULL DEFAULT 0.0,
 metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
"""


SEMANTIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_memories (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, memory_type TEXT NOT NULL,
 title TEXT NOT NULL, content TEXT NOT NULL, profile_id TEXT NOT NULL DEFAULT '',
 tags_json TEXT NOT NULL DEFAULT '[]', source TEXT NOT NULL DEFAULT 'manual',
 confidence REAL NOT NULL DEFAULT 1.0, metadata_json TEXT NOT NULL DEFAULT '{}',
 embedding_json TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_semantic_memories_project_type
 ON semantic_memories(project, memory_type, updated_at DESC);
CREATE TABLE IF NOT EXISTS semantic_memory_versions (
 id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, project TEXT NOT NULL,
 version INTEGER NOT NULL, content_hash TEXT NOT NULL, snapshot_json TEXT NOT NULL,
 created_at TEXT NOT NULL,
 FOREIGN KEY(memory_id) REFERENCES semantic_memories(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS vector_collections (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL,
 dimensions INTEGER NOT NULL, distance TEXT NOT NULL DEFAULT 'cosine',
 metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vector_collections_project
 ON vector_collections(project, name);
CREATE TABLE IF NOT EXISTS vector_records (
 id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, project TEXT NOT NULL,
 external_id TEXT NOT NULL DEFAULT '', content TEXT NOT NULL,
 content_hash TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
 vector_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 FOREIGN KEY(collection_id) REFERENCES vector_collections(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_vector_records_collection
 ON vector_records(collection_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS semantic_rerank_runs (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, query TEXT NOT NULL,
 input_count INTEGER NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_analytics_snapshots (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, summary_json TEXT NOT NULL,
 created_at TEXT NOT NULL
);
"""

VAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS vault_entries (
 id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default', category TEXT NOT NULL,
 label TEXT NOT NULL, tags_json TEXT NOT NULL DEFAULT '[]', ciphertext TEXT NOT NULL,
 key_fingerprint TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '',
 access_count INTEGER NOT NULL DEFAULT 0, last_accessed TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vault_project_cat ON vault_entries(project, category);
CREATE TABLE IF NOT EXISTS vault_access_log (
 id TEXT PRIMARY KEY, entry_id TEXT NOT NULL, project TEXT NOT NULL,
 action TEXT NOT NULL, success INTEGER NOT NULL DEFAULT 1,
 ip_hint TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vault_key_registry (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, fingerprint TEXT NOT NULL,
 label TEXT NOT NULL DEFAULT 'default', created_at TEXT NOT NULL
);
"""

FINETUNE_SCHEMA = """
CREATE TABLE IF NOT EXISTS finetune_datasets (
 id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default', name TEXT NOT NULL,
 description TEXT NOT NULL DEFAULT '', format TEXT NOT NULL DEFAULT 'jsonl',
 num_examples INTEGER NOT NULL DEFAULT 0, split_train REAL NOT NULL DEFAULT 0.9,
 split_val REAL NOT NULL DEFAULT 0.1, stored_path TEXT NOT NULL DEFAULT '',
 metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS finetune_examples (
 id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, project TEXT NOT NULL,
 system_prompt TEXT NOT NULL DEFAULT '', user_input TEXT NOT NULL,
 assistant_output TEXT NOT NULL, split TEXT NOT NULL DEFAULT 'train',
 quality_score REAL NOT NULL DEFAULT 1.0, metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL,
 FOREIGN KEY(dataset_id) REFERENCES finetune_datasets(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_ft_examples_dataset ON finetune_examples(dataset_id, split);
CREATE TABLE IF NOT EXISTS finetune_jobs (
 id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default', dataset_id TEXT NOT NULL,
 base_model TEXT NOT NULL, method TEXT NOT NULL DEFAULT 'lora',
 status TEXT NOT NULL DEFAULT 'queued', config_json TEXT NOT NULL DEFAULT '{}',
 progress REAL NOT NULL DEFAULT 0.0, current_loss REAL NOT NULL DEFAULT 0.0,
 output_adapter_id TEXT NOT NULL DEFAULT '', error_message TEXT NOT NULL DEFAULT '',
 started_at TEXT NOT NULL DEFAULT '', completed_at TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ft_jobs_project ON finetune_jobs(project, status);
CREATE TABLE IF NOT EXISTS lora_adapters (
 id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default', name TEXT NOT NULL,
 base_model TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
 rank INTEGER NOT NULL DEFAULT 16, alpha REAL NOT NULL DEFAULT 16.0,
 target_modules_json TEXT NOT NULL DEFAULT '[]', stored_path TEXT NOT NULL DEFAULT '',
 size_bytes INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 0,
 eval_score REAL NOT NULL DEFAULT 0.0, version INTEGER NOT NULL DEFAULT 1,
 metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lora_project_model ON lora_adapters(project, base_model);
CREATE TABLE IF NOT EXISTS finetune_templates (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
 method TEXT NOT NULL DEFAULT 'lora', config_json TEXT NOT NULL DEFAULT '{}',
 use_case TEXT NOT NULL DEFAULT 'general', created_at TEXT NOT NULL
);
"""

SCRAPER_SCHEMA = """
CREATE TABLE IF NOT EXISTS scrape_jobs (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, url TEXT NOT NULL,
 schedule_cron TEXT NOT NULL DEFAULT '', interval_hours REAL NOT NULL DEFAULT 24.0,
 last_run TEXT NOT NULL DEFAULT '', next_run TEXT NOT NULL DEFAULT '',
 status TEXT NOT NULL DEFAULT 'pending', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scrape_jobs_project ON scrape_jobs(project, next_run);
CREATE TABLE IF NOT EXISTS scrape_results (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL, project TEXT NOT NULL, url TEXT NOT NULL,
 status_code INTEGER NOT NULL DEFAULT 200, title TEXT NOT NULL DEFAULT '',
 content_hash TEXT NOT NULL DEFAULT '', word_count INTEGER NOT NULL DEFAULT 0,
 changed INTEGER NOT NULL DEFAULT 1, knowledge_doc_id TEXT NOT NULL DEFAULT '',
 error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scrape_results_job ON scrape_results(job_id, created_at DESC);
"""

DOCUMENT_PROCESSING_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_artifacts (
 id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, project TEXT NOT NULL,
 kind TEXT NOT NULL, local_path TEXT NOT NULL, sha256 TEXT NOT NULL,
 metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_document_artifacts_doc
 ON document_artifacts(doc_id, created_at DESC);
CREATE TABLE IF NOT EXISTS document_annotations (
 id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, page INTEGER NOT NULL,
 kind TEXT NOT NULL, geometry_json TEXT NOT NULL DEFAULT '{}',
 text TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_document_annotations_doc
 ON document_annotations(doc_id, page, created_at);
"""

MEDIA_SCHEMA = """
CREATE TABLE IF NOT EXISTS media_assets (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL,
 media_type TEXT NOT NULL, mime_type TEXT NOT NULL, source TEXT NOT NULL,
 source_path TEXT NOT NULL DEFAULT '', stored_path TEXT NOT NULL DEFAULT '',
 size_bytes INTEGER NOT NULL, content_hash TEXT NOT NULL,
 metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_media_assets_project_created
 ON media_assets(project, created_at DESC);
CREATE TABLE IF NOT EXISTS media_analyses (
 id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, project TEXT NOT NULL,
 analyzer TEXT NOT NULL, summary TEXT NOT NULL,
 extracted_text TEXT NOT NULL DEFAULT '', findings_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL,
 FOREIGN KEY(asset_id) REFERENCES media_assets(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_media_analyses_asset
 ON media_analyses(asset_id, created_at DESC);
"""

VOICE_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_personas (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL,
 language TEXT NOT NULL DEFAULT 'en', style TEXT NOT NULL DEFAULT 'clear',
 voiceprint_asset_id TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voice_personas_project ON voice_personas(project, name);
CREATE TABLE IF NOT EXISTS voice_sessions (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, mode TEXT NOT NULL,
 persona_id TEXT NOT NULL DEFAULT '', language TEXT NOT NULL DEFAULT 'en',
 wake_word TEXT NOT NULL DEFAULT '', always_listening INTEGER NOT NULL DEFAULT 0,
 status TEXT NOT NULL DEFAULT 'active', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voice_sessions_project ON voice_sessions(project, updated_at DESC);
CREATE TABLE IF NOT EXISTS voice_turns (
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL, project TEXT NOT NULL,
 speaker TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL,
 audio_asset_id TEXT NOT NULL DEFAULT '', is_final INTEGER NOT NULL DEFAULT 1,
 metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
 FOREIGN KEY(session_id) REFERENCES voice_sessions(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_voice_turns_session ON voice_turns(session_id, created_at ASC);
CREATE TABLE IF NOT EXISTS voice_translations (
 id TEXT PRIMARY KEY, project TEXT NOT NULL, source_language TEXT NOT NULL,
 target_language TEXT NOT NULL, source_text TEXT NOT NULL, translated_text TEXT NOT NULL,
 provider TEXT NOT NULL DEFAULT 'local-rule', metadata_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL
);
"""

AGENT_BRIDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_bridge_tasks (
 id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
 owner TEXT NOT NULL DEFAULT 'local', agent TEXT NOT NULL DEFAULT 'assistant',
 goal TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
 payload_json TEXT NOT NULL DEFAULT '{}', result_json TEXT NOT NULL DEFAULT '{}',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_bridge_tasks_project_owner
 ON agent_bridge_tasks(project, owner, updated_at DESC);
"""


def _statements(script: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in script.split(";") if part.strip())


RESEARCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    parent_id TEXT NOT NULL DEFAULT '',
    branch_label TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT 'auto',
    model TEXT NOT NULL DEFAULT '',
    allow_external INTEGER NOT NULL DEFAULT 0,
    max_iterations INTEGER NOT NULL DEFAULT 3,
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    job_id TEXT NOT NULL DEFAULT '',
    plan TEXT NOT NULL DEFAULT '',
    report TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_research_sessions_project
    ON research_sessions(project, archived, updated_at DESC);
CREATE TABLE IF NOT EXISTS research_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress INTEGER NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_checkpoints_session
    ON research_checkpoints(session_id, id);
CREATE TABLE IF NOT EXISTS research_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    snippet TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_sources_session
    ON research_sources(session_id, id);
"""


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "core-sqlite-baseline", (), legacy_checksum="openagent-core-v1"),
    Migration(2, "centralize-core-schema", _statements(CORE_SCHEMA)),
    Migration(3, "centralize-vector-store-schema", _statements(VECTOR_STORE_SCHEMA)),
    Migration(
        4,
        "add-vector-store-collection-revision",
        (),
        add_columns=(
            AddColumn("vector_store_collections", "revision INTEGER NOT NULL DEFAULT 0"),
        ),
    ),
    Migration(5, "centralize-calendar-sync-schema", _statements(CALENDAR_SYNC_SCHEMA)),
    Migration(
        6,
        "upgrade-calendar-sync-lineage",
        (),
        add_columns=(
            AddColumn("calendar_sync_providers", "owner TEXT NOT NULL DEFAULT 'local'"),
            AddColumn("calendar_sync_providers", "sync_token TEXT NOT NULL DEFAULT ''"),
            AddColumn("calendar_sync_uid_map", "remote_etag TEXT NOT NULL DEFAULT ''"),
            AddColumn("calendar_sync_uid_map", "remote_href TEXT NOT NULL DEFAULT ''"),
            AddColumn("calendar_sync_uid_map", "deleted_at TEXT NOT NULL DEFAULT ''"),
        ),
    ),
    Migration(
        7,
        "index-calendar-sync-owner",
        _statements(
            """
            CREATE INDEX IF NOT EXISTS idx_calendar_sync_providers_owner_project
             ON calendar_sync_providers(owner, project);
            """
        ),
    ),
    Migration(8, "centralize-integration-hub-schema", _statements(INTEGRATION_HUB_SCHEMA)),
    Migration(
        9, "centralize-contacts-schema", _statements(CONTACTS_SCHEMA),
        add_columns=(AddColumn("contacts", "owner TEXT NOT NULL DEFAULT 'local'"),),
    ),
    Migration(
        10, "index-contacts-owner", _statements(
            "CREATE INDEX IF NOT EXISTS idx_contacts_owner_project_name ON contacts(owner, project, name);"
        ),
    ),
    Migration(11, "centralize-workspace-schema", _statements(WORKSPACE_SCHEMA)),
    Migration(12, "centralize-assistant-schema", _statements(ASSISTANT_SCHEMA)),
    Migration(13, "centralize-mcp-schema", _statements(MCP_SCHEMA)),
    Migration(14, "centralize-twin-schema", _statements(TWIN_SCHEMA)),
    Migration(15, "centralize-semantic-schema", _statements(SEMANTIC_SCHEMA)),
    Migration(16, "centralize-vault-schema", _statements(VAULT_SCHEMA)),
    Migration(17, "centralize-finetune-schema", _statements(FINETUNE_SCHEMA)),
    Migration(18, "centralize-scraper-schema", _statements(SCRAPER_SCHEMA)),
    Migration(19, "centralize-document-processing-schema", _statements(DOCUMENT_PROCESSING_SCHEMA)),
    Migration(20, "centralize-media-schema", _statements(MEDIA_SCHEMA)),
    Migration(21, "centralize-voice-schema", _statements(VOICE_SCHEMA)),
    Migration(22, "centralize-agent-bridge-schema", _statements(AGENT_BRIDGE_SCHEMA)),
    Migration(23, "centralize-research-schema", _statements(RESEARCH_SCHEMA)),
)


class MigrationRunner:
    """Apply and verify ordered migrations on one SQLite database."""

    def __init__(self, connection: sqlite3.Connection, migrations: Iterable[Migration] = MIGRATIONS) -> None:
        self.connection = connection
        self.migrations = tuple(sorted(migrations, key=lambda item: item.version))
        self._validate_registry()

    def run(self) -> list[int]:
        self._bootstrap_registry()
        owner = uuid4().hex
        applied_now: list[int] = []
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "INSERT OR REPLACE INTO schema_migration_lock (id, owner, acquired_at) VALUES (1, ?, ?)",
                (owner, utc_now()),
            )
            applied_rows = self.connection.execute(
                "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
            ).fetchall()
            self._validate_applied_history(applied_rows)
            applied = {
                int(row["version"]): (str(row["name"]), str(row["checksum"]))
                for row in applied_rows
            }
            for migration in self.migrations:
                existing = applied.get(migration.version)
                if existing:
                    if existing != (migration.name, migration.checksum):
                        raise MigrationError(
                            f"Migration {migration.version} does not match the recorded name and checksum."
                        )
                    continue
                if migration.destructive:
                    raise MigrationError(
                        f"Migration {migration.version} is destructive and requires a verified backup before execution."
                    )
                for statement in migration.statements:
                    self.connection.execute(statement)
                for column in migration.add_columns:
                    if not self._column_exists(column.table, column.name):
                        self.connection.execute(column.canonical)
                self.connection.execute(
                    "INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
                    (migration.version, migration.name, migration.checksum, utc_now()),
                )
                applied_now.append(migration.version)
            integrity = str(self.connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity.casefold() != "ok":
                raise MigrationError(f"SQLite integrity check failed after migration: {integrity}")
            self.connection.execute("DELETE FROM schema_migration_lock WHERE id = 1 AND owner = ?", (owner,))
            self.connection.commit()
            return applied_now
        except Exception:
            self.connection.rollback()
            raise

    def _validate_registry(self) -> None:
        versions = [migration.version for migration in self.migrations]
        if any(version < 1 for version in versions):
            raise MigrationError("Migration versions must be positive integers.")
        if len(versions) != len(set(versions)):
            raise MigrationError("Migration versions must be unique.")
        names = [migration.name for migration in self.migrations]
        if any(not name.strip() for name in names):
            raise MigrationError("Migration names cannot be empty.")
        if len(names) != len(set(names)):
            raise MigrationError("Migration names must be unique.")

    def _validate_applied_history(self, rows: list[sqlite3.Row]) -> None:
        applied_versions = [int(row["version"]) for row in rows]
        expected_versions = [migration.version for migration in self.migrations]
        if applied_versions != expected_versions[: len(applied_versions)]:
            raise MigrationError(
                "Recorded migrations are not a valid ordered prefix of the migration registry."
            )

    def _column_exists(self, table: str, column: str) -> bool:
        rows = self.connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        if not rows:
            raise MigrationError(f"Cannot add column {column}: table {table} does not exist.")
        return any(str(row[1]) == column for row in rows)

    def _bootstrap_registry(self) -> None:
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, applied_at TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS schema_migration_lock (
                id INTEGER PRIMARY KEY CHECK(id = 1), owner TEXT NOT NULL, acquired_at TEXT NOT NULL
            )"""
        )
        self.connection.commit()
