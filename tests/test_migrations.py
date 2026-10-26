from __future__ import annotations

import sqlite3
import threading
from tempfile import TemporaryDirectory
import unittest

from openagent.migrations import AddColumn, MIGRATIONS, Migration, MigrationError, MigrationRunner
from openagent.calendar_sync import CalendarSyncManager
from openagent.calendar_system import CalendarSystem
from openagent.storage import OpenAgentStore
from openagent.vector_store import VectorStore


class MigrationTests(unittest.TestCase):
    def test_migration_runner_executes_multistatement_blocks_atomically(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        migrations = (
            Migration(1, "multi-statement-schema", (
                "CREATE TABLE first_table (value TEXT); CREATE INDEX idx_first_value ON first_table(value);",
            )),
        )
        try:
            self.assertEqual(MigrationRunner(connection, migrations).run(), [1])
            objects = {
                (str(row["type"]), str(row["name"]))
                for row in connection.execute(
                    "SELECT type, name FROM sqlite_master WHERE name IN ('first_table', 'idx_first_value')"
                )
            }
            self.assertEqual(objects, {("table", "first_table"), ("index", "idx_first_value")})
            self.assertEqual(MigrationRunner(connection, migrations).run(), [])
        finally:
            connection.close()

    def test_new_database_records_ordered_checksummed_migrations(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            status = store.migration_status()
            self.assertEqual(status["current_version"], MIGRATIONS[-1].version)
            self.assertEqual(status["integrity"], "ok")
            self.assertEqual(
                [item["checksum"] for item in status["migrations"]],
                [migration.checksum for migration in MIGRATIONS],
            )

    def test_legacy_version_one_database_upgrades_without_data_loss(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            connection.execute(
                "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT, checksum TEXT, applied_at TEXT)"
            )
            connection.execute(
                "INSERT INTO schema_migrations VALUES (1, 'core-sqlite-baseline', 'openagent-core-v1', '2026-01-01')"
            )
            connection.commit()
            connection.close()
            store = OpenAgentStore(path)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)
            with store.session() as upgraded:
                self.assertIsNotNone(
                    upgraded.execute("SELECT name FROM sqlite_master WHERE name='events'").fetchone()
                )

    def test_legacy_agent_runs_remain_unowned_after_owner_binding_migration(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:23]).run()
            connection.execute(
                "INSERT INTO agent_runs(agent, goal, response_json, project, created_at) VALUES (?, ?, ?, ?, ?)",
                ("legacy", "legacy goal", '{"response":"legacy result"}', "legacy-project", "2020-01-01T00:00:00+00:00"),
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                row = upgraded.execute("SELECT owner FROM agent_runs WHERE project = 'legacy-project'").fetchone()
                project = upgraded.execute(
                    "SELECT owner FROM agent_run_projects WHERE project = 'legacy-project'"
                ).fetchone()
            self.assertEqual(row["owner"], "")
            self.assertIsNone(project)

    def test_checksum_drift_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            with store.session() as connection:
                connection.execute(
                    "UPDATE schema_migrations SET checksum='changed' WHERE version=?",
                    (MIGRATIONS[-1].version,),
                )
            connection = store.connect()
            try:
                with self.assertRaises(MigrationError):
                    MigrationRunner(connection).run()
            finally:
                connection.close()

    def test_concurrent_initialization_serializes_migration_runners(self) -> None:
        with TemporaryDirectory() as tmp:
            failures: list[Exception] = []

            def open_store() -> None:
                try:
                    OpenAgentStore(tmp)
                except Exception as exc:
                    failures.append(exc)

            threads = [threading.Thread(target=open_store) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertEqual(failures, [])
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(
                OpenAgentStore(tmp).migration_status()["current_version"],
                MIGRATIONS[-1].version,
            )

    def test_vector_store_schema_upgrades_from_service_owned_shape_without_data_loss(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:2]).run()
            connection.executescript(
                """
                CREATE TABLE vector_store_collections (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL,
                    adapter TEXT NOT NULL DEFAULT 'sqlite', dimensions INTEGER NOT NULL,
                    distance TEXT NOT NULL DEFAULT 'cosine',
                    metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE vector_store_chunks (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, collection_id TEXT NOT NULL,
                    source_id TEXT NOT NULL, chunk_id TEXT NOT NULL, content_hash TEXT NOT NULL,
                    content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding_provider TEXT NOT NULL, embedding_model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL, vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(collection_id) REFERENCES vector_store_collections(id) ON DELETE CASCADE
                );
                INSERT INTO vector_store_collections
                    (id, project, name, dimensions, created_at, updated_at)
                VALUES ('collection-1', 'default', 'legacy', 8, '2026-01-01', '2026-01-01');
                INSERT INTO vector_store_chunks
                    (id, project, collection_id, source_id, chunk_id, content_hash, content,
                     embedding_provider, embedding_model, dimensions, vector_json, created_at, updated_at)
                VALUES ('chunk-1', 'default', 'collection-1', 'source-1', '0', 'hash', 'kept',
                        'local-hash', 'legacy', 8, '[0,0,0,0,0,0,0,0]', '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                columns = {
                    str(row["name"])
                    for row in upgraded.execute("PRAGMA table_info(vector_store_collections)")
                }
                collection = upgraded.execute(
                    "SELECT name, revision FROM vector_store_collections WHERE id='collection-1'"
                ).fetchone()
                chunk = upgraded.execute(
                    "SELECT content FROM vector_store_chunks WHERE id='chunk-1'"
                ).fetchone()
            self.assertIn("revision", columns)
            self.assertEqual(dict(collection), {"name": "legacy", "revision": 0})
            self.assertEqual(chunk["content"], "kept")
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_existing_current_vector_schema_is_adopted_without_duplicate_column(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:3]).run()
            connection.execute(
                "ALTER TABLE vector_store_collections ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
            )
            connection.commit()
            applied = MigrationRunner(connection).run()
            second_pass = MigrationRunner(connection).run()
            columns = [
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(vector_store_collections)")
            ]
            connection.close()
            self.assertEqual(applied, [migration.version for migration in MIGRATIONS[3:]])
            self.assertEqual(second_pass, [])
            self.assertEqual(columns.count("revision"), 1)

    def test_vector_store_constructor_uses_central_schema_registry(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            vectors = VectorStore(store)
            try:
                collection = vectors.ensure_collection("constructor-check", dimensions=8)
            finally:
                vectors.close()
            self.assertEqual(collection["revision"], 0)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_registry_rejects_duplicate_versions_before_mutating_database(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        migrations = (
            Migration(1, "first", ("CREATE TABLE first_table (id INTEGER)",)),
            Migration(1, "second", ("CREATE TABLE second_table (id INTEGER)",)),
        )
        with self.assertRaisesRegex(MigrationError, "versions must be unique"):
            MigrationRunner(connection, migrations)
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        connection.close()
        self.assertEqual(tables, [])

    def test_runner_rejects_out_of_order_recorded_history(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute(
            "CREATE TABLE schema_migrations "
            "(version INTEGER PRIMARY KEY, name TEXT, checksum TEXT, applied_at TEXT)"
        )
        connection.execute(
            "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
            (2, MIGRATIONS[1].name, MIGRATIONS[1].checksum, "2026-01-01"),
        )
        connection.commit()
        with self.assertRaisesRegex(MigrationError, "valid ordered prefix"):
            MigrationRunner(connection).run()
        connection.close()

    def test_runner_rejects_database_from_a_newer_unknown_registry(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        MigrationRunner(connection).run()
        connection.execute(
            "INSERT INTO schema_migrations VALUES (99, 'future', 'future-checksum', '2026-01-01')"
        )
        connection.commit()
        with self.assertRaisesRegex(MigrationError, "valid ordered prefix"):
            MigrationRunner(connection).run()
        connection.close()

    def test_add_column_definition_is_part_of_migration_checksum(self) -> None:
        first = Migration(
            9,
            "column-check",
            (),
            add_columns=(AddColumn("example", "revision INTEGER NOT NULL DEFAULT 0"),),
        )
        changed = Migration(
            9,
            "column-check",
            (),
            add_columns=(AddColumn("example", "revision INTEGER NOT NULL DEFAULT 1"),),
        )
        self.assertNotEqual(first.checksum, changed.checksum)

    def test_calendar_sync_schema_upgrades_from_service_owned_shape_without_data_loss(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:4]).run()
            connection.executescript(
                """
                CREATE TABLE calendar_sync_providers (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
                    provider TEXT NOT NULL, base_url TEXT NOT NULL DEFAULT '',
                    account_ref TEXT NOT NULL DEFAULT '', calendar_id TEXT NOT NULL DEFAULT '',
                    config_json TEXT NOT NULL DEFAULT '{}', status TEXT NOT NULL DEFAULT 'connected',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE calendar_sync_jobs (
                    id TEXT PRIMARY KEY, provider_id TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default', direction TEXT NOT NULL DEFAULT 'pull',
                    conflict_policy TEXT NOT NULL DEFAULT 'latest_update',
                    state TEXT NOT NULL DEFAULT 'queued', pulled INTEGER NOT NULL DEFAULT 0,
                    pushed INTEGER NOT NULL DEFAULT 0, conflicts INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE calendar_sync_uid_map (
                    id TEXT PRIMARY KEY, provider_id TEXT NOT NULL, local_event_id TEXT NOT NULL,
                    remote_uid TEXT NOT NULL, remote_hash TEXT NOT NULL DEFAULT '',
                    local_hash TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,
                    UNIQUE(provider_id, remote_uid)
                );
                CREATE TABLE calendar_sync_conflicts (
                    id TEXT PRIMARY KEY, provider_id TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default', local_event_id TEXT NOT NULL DEFAULT '',
                    remote_uid TEXT NOT NULL DEFAULT '', local_payload_json TEXT NOT NULL DEFAULT '{}',
                    remote_payload_json TEXT NOT NULL DEFAULT '{}', local_updated_at TEXT NOT NULL DEFAULT '',
                    remote_seen_at TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'open',
                    resolution TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL DEFAULT ''
                );
                INSERT INTO calendar_sync_providers
                    (id, project, provider, calendar_id, created_at, updated_at)
                VALUES ('provider-1', 'science', 'manual', 'calendar-1', '2026-01-01', '2026-01-01');
                INSERT INTO calendar_sync_uid_map
                    (id, provider_id, local_event_id, remote_uid, remote_hash, local_hash, updated_at)
                VALUES ('map-1', 'provider-1', 'event-1', 'remote-1', 'remote-hash', 'local-hash', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                provider = upgraded.execute(
                    "SELECT owner, sync_token FROM calendar_sync_providers WHERE id='provider-1'"
                ).fetchone()
                mapping = upgraded.execute(
                    "SELECT remote_uid, remote_etag, remote_href, deleted_at "
                    "FROM calendar_sync_uid_map WHERE id='map-1'"
                ).fetchone()
                indexes = {
                    str(row["name"])
                    for row in upgraded.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' "
                        "AND name LIKE 'idx_calendar_sync_%'"
                    )
                }
            self.assertEqual(dict(provider), {"owner": "local", "sync_token": ""})
            self.assertEqual(
                dict(mapping),
                {"remote_uid": "remote-1", "remote_etag": "", "remote_href": "", "deleted_at": ""},
            )
            self.assertIn("idx_calendar_sync_providers_owner_project", indexes)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_existing_current_calendar_sync_schema_is_adopted_idempotently(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:5]).run()
            for addition in MIGRATIONS[5].add_columns:
                connection.execute(addition.canonical)
            connection.execute(
                "CREATE INDEX idx_calendar_sync_providers_owner_project "
                "ON calendar_sync_providers(owner, project)"
            )
            connection.commit()
            applied = MigrationRunner(connection).run()
            second_pass = MigrationRunner(connection).run()
            connection.close()
            self.assertEqual(applied, [migration.version for migration in MIGRATIONS[5:]])
            self.assertEqual(second_pass, [])

    def test_calendar_sync_constructor_uses_central_schema_registry(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            calendar = CalendarSystem(store)
            sync = CalendarSyncManager(store, calendar)
            provider = sync.register_provider("manual", project="science")
            self.assertEqual(provider["owner"], "local")
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_integration_hub_schema_adopts_existing_records(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:7]).run()
            connection.executescript(
                """
                CREATE TABLE integration_pairing_codes (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, label TEXT NOT NULL,
                    code_hash TEXT NOT NULL UNIQUE, scopes_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL, expires_at TEXT NOT NULL, created_at TEXT NOT NULL,
                    redeemed_at TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE integration_preferences (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, scope TEXT NOT NULL,
                    key TEXT NOT NULL, value_json TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(project, scope, key)
                );
                CREATE TABLE integration_diagnostics (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE integration_safe_command_runs (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, command_key TEXT NOT NULL,
                    status TEXT NOT NULL, output TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '', return_code INTEGER NOT NULL DEFAULT 0,
                    duration_ms REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL
                );
                INSERT INTO integration_pairing_codes
                    (id, project, label, code_hash, status, expires_at, created_at)
                VALUES ('pair-1', 'core', 'Laptop', 'hash-1', 'active', '2027-01-01', '2026-01-01');
                INSERT INTO integration_preferences
                    (id, project, scope, key, value_json, updated_at)
                VALUES ('pref-1', 'core', 'ui', 'theme', '"dark"', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                pair = upgraded.execute(
                    "SELECT label FROM integration_pairing_codes WHERE id='pair-1'"
                ).fetchone()
                preference = upgraded.execute(
                    "SELECT value_json FROM integration_preferences WHERE id='pref-1'"
                ).fetchone()
                indexes = {
                    row["name"] for row in upgraded.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' "
                        "AND name LIKE 'idx_integration_%'"
                    )
                }
            self.assertEqual(pair["label"], "Laptop")
            self.assertEqual(preference["value_json"], '"dark"')
            self.assertIn("idx_integration_safe_runs_project_created", indexes)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_contacts_schema_upgrades_ownerless_records_and_indexes(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:8]).run()
            connection.executescript(
                """
                CREATE TABLE contacts (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL,
                    handles_json TEXT NOT NULL DEFAULT '{}', notes TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO contacts (id, project, name, created_at, updated_at)
                VALUES ('person-1', 'core', 'Ada', '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                row = upgraded.execute("SELECT name, owner FROM contacts WHERE id='person-1'").fetchone()
                indexes = {
                    item["name"] for item in upgraded.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_contacts_%'"
                    )
                }
            self.assertEqual(dict(row), {"name": "Ada", "owner": "local"})
            self.assertIn("idx_contacts_owner_project_name", indexes)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_existing_current_contacts_schema_is_adopted_idempotently(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:8]).run()
            connection.execute(
                "CREATE TABLE contacts (id TEXT PRIMARY KEY, project TEXT NOT NULL, "
                "owner TEXT NOT NULL DEFAULT 'local', name TEXT NOT NULL, "
                "handles_json TEXT NOT NULL DEFAULT '{}', notes TEXT NOT NULL DEFAULT '', "
                "metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            connection.commit()
            self.assertEqual(MigrationRunner(connection).run(), [migration.version for migration in MIGRATIONS[8:]])
            self.assertEqual(MigrationRunner(connection).run(), [])
            columns = [row["name"] for row in connection.execute("PRAGMA table_info(contacts)")]
            connection.close()
            self.assertEqual(columns.count("owner"), 1)

    def test_workspace_schema_adopts_existing_items(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:10]).run()
            connection.executescript(
                """
                CREATE TABLE workspace_items (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, kind TEXT NOT NULL,
                    title TEXT NOT NULL, parent_id TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO workspace_items
                    (id, project, kind, title, content, created_at, updated_at)
                VALUES ('item-1', 'core', 'note', 'Preserved', 'existing text', '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                item = upgraded.execute(
                    "SELECT title, content FROM workspace_items WHERE id='item-1'"
                ).fetchone()
                indexes = {
                    row["name"] for row in upgraded.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_workspace_items_%'"
                    )
                }
            self.assertEqual(dict(item), {"title": "Preserved", "content": "existing text"})
            self.assertIn("idx_workspace_items_parent", indexes)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_assistant_schema_adopts_existing_task_and_profile(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:11]).run()
            connection.executescript(
                """
                CREATE TABLE assistant_profiles (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, user_id TEXT NOT NULL,
                    preferences_json TEXT NOT NULL DEFAULT '{}', facts_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO assistant_profiles (id, project, user_id, created_at, updated_at)
                VALUES ('profile-1', 'core', 'ada', '2026-01-01', '2026-01-01');
                CREATE TABLE assistant_tasks (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open', priority TEXT NOT NULL DEFAULT 'normal',
                    due_at TEXT NOT NULL DEFAULT '', goal_id TEXT NOT NULL DEFAULT '',
                    contact_id TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO assistant_tasks (id, project, title, created_at, updated_at)
                VALUES ('task-1', 'core', 'Preserved task', '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                task = upgraded.execute("SELECT title FROM assistant_tasks WHERE id='task-1'").fetchone()
                profile = upgraded.execute("SELECT user_id FROM assistant_profiles WHERE id='profile-1'").fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_assistant_tasks_project_status'"
                ).fetchone()
            self.assertEqual(task["title"], "Preserved task")
            self.assertEqual(profile["user_id"], "ada")
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_mcp_schema_adopts_existing_server_and_tool(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:12]).run()
            connection.executescript(
                """
                CREATE TABLE mcp_servers (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local', name TEXT NOT NULL,
                    transport TEXT NOT NULL DEFAULT 'stdio', command_json TEXT NOT NULL DEFAULT '[]',
                    endpoint TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'configured',
                    metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE mcp_tools (
                    id TEXT PRIMARY KEY, server_id TEXT NOT NULL,
                    project TEXT NOT NULL DEFAULT 'default', owner TEXT NOT NULL DEFAULT 'local',
                    name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1, approval_required INTEGER NOT NULL DEFAULT 1,
                    schema_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE
                );
                INSERT INTO mcp_servers (id, name, created_at, updated_at)
                VALUES ('server-1', 'Preserved', '2026-01-01', '2026-01-01');
                INSERT INTO mcp_tools (id, server_id, name, created_at, updated_at)
                VALUES ('tool-1', 'server-1', 'echo', '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                server = upgraded.execute("SELECT name FROM mcp_servers WHERE id='server-1'").fetchone()
                tool = upgraded.execute("SELECT name FROM mcp_tools WHERE id='tool-1'").fetchone()
                indexes = {
                    row["name"] for row in upgraded.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_mcp_%'"
                    )
                }
            self.assertEqual(server["name"], "Preserved")
            self.assertEqual(tool["name"], "echo")
            self.assertIn("idx_mcp_tools_server", indexes)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_twin_schema_adopts_existing_profile_and_interaction(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:13]).run()
            connection.executescript(
                """
                CREATE TABLE twin_profiles (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
                    profile_json TEXT NOT NULL DEFAULT '{}', version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE twin_interactions (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, kind TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '', topics_json TEXT NOT NULL DEFAULT '[]',
                    hour_of_day INTEGER NOT NULL DEFAULT 12,
                    day_of_week TEXT NOT NULL DEFAULT '',
                    context_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                );
                INSERT INTO twin_profiles (id, project, profile_json, created_at, updated_at)
                VALUES ('profile-1', 'core', '{"name":"Ada"}', '2026-01-01', '2026-01-01');
                INSERT INTO twin_interactions (id, project, kind, content, created_at)
                VALUES ('event-1', 'core', 'chat', 'Preserved event', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                profile = upgraded.execute("SELECT profile_json FROM twin_profiles WHERE id='profile-1'").fetchone()
                event = upgraded.execute("SELECT content FROM twin_interactions WHERE id='event-1'").fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_twin_profile_project'"
                ).fetchone()
            self.assertEqual(profile["profile_json"], '{"name":"Ada"}')
            self.assertEqual(event["content"], "Preserved event")
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_semantic_schema_adopts_existing_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:14]).run()
            connection.executescript(
                """
                CREATE TABLE semantic_memories (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, memory_type TEXT NOT NULL,
                    title TEXT NOT NULL, content TEXT NOT NULL,
                    profile_id TEXT NOT NULL DEFAULT '', tags_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT 'manual', confidence REAL NOT NULL DEFAULT 1.0,
                    metadata_json TEXT NOT NULL DEFAULT '{}', embedding_json TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO semantic_memories (
                    id, project, memory_type, title, content, embedding_json, created_at, updated_at
                ) VALUES (
                    'memory-1', 'core', 'semantic', 'Preserved', 'Existing data', '[]',
                    '2026-01-01', '2026-01-01'
                );
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                memory = upgraded.execute(
                    "SELECT title, content FROM semantic_memories WHERE id='memory-1'"
                ).fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_semantic_memories_project_type'"
                ).fetchone()
            self.assertEqual(dict(memory), {"title": "Preserved", "content": "Existing data"})
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_vault_schema_adopts_existing_encrypted_entry(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:15]).run()
            connection.executescript(
                """
                CREATE TABLE vault_entries (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
                    category TEXT NOT NULL, label TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]', ciphertext TEXT NOT NULL,
                    key_fingerprint TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO vault_entries (
                    id, category, label, ciphertext, key_fingerprint, created_at, updated_at
                ) VALUES ('entry-1', 'secret', 'Preserved', 'opaque-ciphertext', 'fingerprint',
                          '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                entry = upgraded.execute(
                    "SELECT label, ciphertext FROM vault_entries WHERE id='entry-1'"
                ).fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_vault_project_cat'"
                ).fetchone()
            self.assertEqual(dict(entry), {"label": "Preserved", "ciphertext": "opaque-ciphertext"})
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_finetune_schema_adopts_existing_dataset(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:16]).run()
            connection.executescript(
                """
                CREATE TABLE finetune_datasets (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
                    name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                    format TEXT NOT NULL DEFAULT 'jsonl', num_examples INTEGER NOT NULL DEFAULT 0,
                    split_train REAL NOT NULL DEFAULT 0.9, split_val REAL NOT NULL DEFAULT 0.1,
                    stored_path TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO finetune_datasets (id, project, name, created_at, updated_at)
                VALUES ('dataset-1', 'core', 'Preserved', '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                row = upgraded.execute(
                    "SELECT name FROM finetune_datasets WHERE id='dataset-1'"
                ).fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_ft_examples_dataset'"
                ).fetchone()
            self.assertEqual(row["name"], "Preserved")
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_scraper_schema_adopts_existing_job(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:17]).run()
            connection.executescript(
                """
                CREATE TABLE scrape_jobs (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, url TEXT NOT NULL,
                    schedule_cron TEXT NOT NULL DEFAULT '', interval_hours REAL NOT NULL DEFAULT 24.0,
                    last_run TEXT NOT NULL DEFAULT '', next_run TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending', metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO scrape_jobs (id, project, url, created_at, updated_at)
                VALUES ('job-1', 'core', 'https://example.org', '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                job = upgraded.execute("SELECT url FROM scrape_jobs WHERE id='job-1'").fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_scrape_results_job'"
                ).fetchone()
            self.assertEqual(job["url"], "https://example.org")
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_document_processing_schema_adopts_existing_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:18]).run()
            connection.executescript(
                """
                CREATE TABLE document_artifacts (
                    id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, project TEXT NOT NULL,
                    kind TEXT NOT NULL, local_path TEXT NOT NULL, sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                );
                INSERT INTO document_artifacts (
                    id, doc_id, project, kind, local_path, sha256, created_at
                ) VALUES ('artifact-1', 'doc-1', 'core', 'source-pdf', 'source.pdf', 'hash', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                artifact = upgraded.execute(
                    "SELECT local_path FROM document_artifacts WHERE id='artifact-1'"
                ).fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_document_annotations_doc'"
                ).fetchone()
            self.assertEqual(artifact["local_path"], "source.pdf")
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_media_schema_adopts_existing_asset(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:19]).run()
            connection.executescript(
                """
                CREATE TABLE media_assets (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL,
                    media_type TEXT NOT NULL, mime_type TEXT NOT NULL, source TEXT NOT NULL,
                    source_path TEXT NOT NULL DEFAULT '', stored_path TEXT NOT NULL DEFAULT '',
                    size_bytes INTEGER NOT NULL, content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
                );
                INSERT INTO media_assets (
                    id, project, title, media_type, mime_type, source,
                    size_bytes, content_hash, created_at
                ) VALUES (
                    'asset-1', 'core', 'Preserved', 'document', 'text/plain', 'file',
                    4, 'hash', '2026-01-01'
                );
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                asset = upgraded.execute(
                    "SELECT title FROM media_assets WHERE id='asset-1'"
                ).fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_media_analyses_asset'"
                ).fetchone()
            self.assertEqual(asset["title"], "Preserved")
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_voice_schema_adopts_existing_session(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:20]).run()
            connection.executescript(
                """
                CREATE TABLE voice_sessions (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, mode TEXT NOT NULL,
                    persona_id TEXT NOT NULL DEFAULT '', language TEXT NOT NULL DEFAULT 'en',
                    wake_word TEXT NOT NULL DEFAULT '', always_listening INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active', metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO voice_sessions (id, project, mode, created_at, updated_at)
                VALUES ('session-1', 'core', 'push_to_talk', '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                session = upgraded.execute(
                    "SELECT mode FROM voice_sessions WHERE id='session-1'"
                ).fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_voice_turns_session'"
                ).fetchone()
            self.assertEqual(session["mode"], "push_to_talk")
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)

    def test_agent_bridge_schema_adopts_existing_task(self) -> None:
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:21]).run()
            connection.executescript(
                """
                CREATE TABLE agent_bridge_tasks (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local', agent TEXT NOT NULL DEFAULT 'assistant',
                    goal TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    payload_json TEXT NOT NULL DEFAULT '{}', result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO agent_bridge_tasks (id, project, goal, created_at, updated_at)
                VALUES ('task-1', 'core', 'Preserved task', '2026-01-01', '2026-01-01');
                """
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            with store.session() as upgraded:
                task = upgraded.execute(
                    "SELECT goal, status FROM agent_bridge_tasks WHERE id='task-1'"
                ).fetchone()
                index = upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' "
                    "AND name='idx_agent_bridge_tasks_project_owner'"
                ).fetchone()
            self.assertEqual(dict(task), {"goal": "Preserved task", "status": "queued"})
            self.assertIsNotNone(index)
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)


class ResearchMigrationTests(unittest.TestCase):
    def test_legacy_research_records_survive_registry_adoption(self) -> None:
        from openagent.migrations import RESEARCH_SCHEMA
        from openagent.research_lifecycle import ResearchLifecycle

        with TemporaryDirectory() as tmp:
            path = f"{tmp}/openagent.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            MigrationRunner(connection, MIGRATIONS[:22]).run()
            connection.executescript(RESEARCH_SCHEMA)
            connection.execute(
                "INSERT INTO research_sessions "
                "(id, project, query, status, report, created_at, updated_at) "
                "VALUES ('legacy', 'science', 'Heat transfer', 'completed', "
                "'Preserved report', '2026-01-01', '2026-01-01')"
            )
            connection.execute(
                "INSERT INTO research_checkpoints "
                "(session_id, stage, progress, message, created_at) "
                "VALUES ('legacy', 'completed', 100, 'Preserved checkpoint', '2026-01-01')"
            )
            connection.execute(
                "INSERT INTO research_sources "
                "(session_id, source_id, title, source_type, snippet, created_at) "
                "VALUES ('legacy', 'paper-1', 'Preserved source', 'paper', 'Evidence', '2026-01-01')"
            )
            connection.commit()
            connection.close()

            store = OpenAgentStore(path)
            lifecycle = ResearchLifecycle(store)
            record = lifecycle.get("legacy")
            self.assertEqual(record["report"], "Preserved report")
            self.assertEqual(record["checkpoints"][0]["message"], "Preserved checkpoint")
            self.assertEqual(record["sources"][0]["id"], "paper-1")
            self.assertIsNone(lifecycle.get("legacy", owner="authenticated-owner"))
            with store.session() as connection:
                self.assertEqual(connection.execute(
                    "SELECT owner FROM research_sessions WHERE id = 'legacy'"
                ).fetchone()["owner"], "")
            lifecycle.initialize()
            self.assertEqual(lifecycle.get("legacy"), record)
            with store.session() as connection:
                self.assertEqual(MigrationRunner(connection).run(), [])
            self.assertEqual(store.migration_status()["current_version"], MIGRATIONS[-1].version)


if __name__ == "__main__":
    unittest.main()
