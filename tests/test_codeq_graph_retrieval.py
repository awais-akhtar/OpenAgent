"""Phase 5 retrieval quality: graph-grounded, fresh and budgeted context."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from openagent.codebase.repository import RepositoryIntelligence
from openagent.codebase.service import CodebaseService
from openagent.codebase.work_graph import build_code_knowledge_graph
from openagent.inference_accelerator import estimate_messages_tokens
from openagent.storage import OpenAgentStore

RETRIEVAL_CORPUS_VERSION = "codeq-retrieval-v1"


class GraphGuidedRetrievalTests(unittest.TestCase):
    def _repo(self, root: Path) -> RepositoryIntelligence:
        (root / "app.py").write_text(
            "from helpers import normalize\n\n\ndef handle_email(value):\n    return normalize(value)\n",
            encoding="utf-8",
        )
        (root / "helpers.py").write_text(
            "def normalize(value):\n    return value.strip().lower()\n",
            encoding="utf-8",
        )
        (root / "tests").mkdir()
        (root / "tests" / "test_app.py").write_text(
            "from app import handle_email\n\ndef test_handle_email():\n    assert handle_email(' A ') == 'a'\n",
            encoding="utf-8",
        )
        (root / "irrelevant.py").write_text("def unrelated():\n    return 'nothing here'\n", encoding="utf-8")
        return RepositoryIntelligence(root)

    def test_declared_task_file_is_retrieved_without_lexical_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self._repo(Path(directory))
            graph = build_code_knowledge_graph(repo.scan())

            rows, stats = repo.select_task_context(
                "a phrase with no matching identifier", graph, seed_paths=["app.py"],
            )

            self.assertEqual("app.py", rows[0]["path"])
            self.assertEqual("task_file", rows[0]["retrieval_source"])
            self.assertEqual(hashlib.sha256((Path(directory) / "app.py").read_bytes()).hexdigest(), rows[0]["content_sha256"])
            self.assertEqual("app.py", stats["manifest"][0]["path"])

    def test_graph_neighbors_retrieve_imported_module_and_candidate_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self._repo(Path(directory))
            graph = build_code_knowledge_graph(repo.scan())

            rows, stats = repo.select_task_context("unmatched_query_zz", graph, seed_paths=["app.py"])

            by_path = {row["path"]: row for row in rows}
            self.assertIn("helpers.py", by_path)
            self.assertEqual("graph", by_path["helpers.py"]["retrieval_source"])
            self.assertEqual("imported-by-selected-file", by_path["helpers.py"]["graph_relation"])
            self.assertIn("tests/test_app.py", by_path)
            self.assertTrue(any(
                relation["confidence"] == "candidate"
                and relation["relation"] == "possible-test-for-selected-file"
                for relation in by_path["tests/test_app.py"]["graph_relations"]
            ))
            self.assertEqual([], stats["stale_graph_paths"])

    def test_graph_retrieval_finds_verified_python_callers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.py").write_text("def normalize(value):\n    return value.strip()\n", encoding="utf-8")
            (root / "route.py").write_text(
                "from service import normalize\n\ndef endpoint(value):\n    return normalize(value)\n",
                encoding="utf-8",
            )
            repo = RepositoryIntelligence(root)
            graph = build_code_knowledge_graph(repo.scan())

            rows, _ = repo.select_task_context("unrelated search phrase", graph, seed_paths=["service.py"])

            caller = next(row for row in rows if row["path"] == "route.py")
            self.assertEqual("graph", caller["retrieval_source"])
            self.assertEqual("calls-selected-file", caller["graph_relation"])
            self.assertIn(caller["graph_confidence"], {"unique-symbol-name", "syntax-call-unique-symbol-name"})

    def test_graph_retrieval_improves_fixture_recall_without_adding_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = self._repo(Path(directory))
            graph = build_code_knowledge_graph(repo.scan())
            query = "handle_email"
            relevant = {"app.py", "helpers.py", "tests/test_app.py"}
            baseline = {item["path"] for item in repo.select_context(query, limit=12)}

            hybrid, _ = repo.select_task_context(query, graph, seed_paths=["app.py"])
            selected = {item["path"] for item in hybrid}
            baseline_recall = len(baseline & relevant) / len(relevant)
            hybrid_recall = len(selected & relevant) / len(relevant)
            baseline_noise = len(baseline - relevant)
            hybrid_noise = len(selected - relevant)

            self.assertGreater(hybrid_recall, baseline_recall)
            self.assertLessEqual(hybrid_noise, baseline_noise)
            self.assertNotIn("irrelevant.py", selected)

    def test_changed_graph_neighbor_is_rejected_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = self._repo(root)
            graph = build_code_knowledge_graph(repo.scan())
            (root / "helpers.py").write_text("def normalize(value):\n    return value\n", encoding="utf-8")

            rows, stats = repo.select_task_context("no lexical match xyz", graph, seed_paths=["app.py"])

            self.assertNotIn("helpers.py", {row["path"] for row in rows})
            self.assertIn("helpers.py", stats["stale_graph_paths"])

    def test_changed_graph_seed_cannot_drive_stale_neighbor_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = self._repo(root)
            graph = build_code_knowledge_graph(repo.scan())
            (root / "app.py").write_text("def unrelated():\n    return 1\n", encoding="utf-8")

            rows, stats = repo.select_task_context("no lexical match xyz", graph, seed_paths=["app.py"])

            self.assertIn("app.py", {row["path"] for row in rows})
            self.assertNotIn("helpers.py", {row["path"] for row in rows})
            self.assertIn("app.py", stats["stale_graph_paths"])

    def test_direct_symbol_retrieval_requires_fresh_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = self._repo(root)
            graph = build_code_knowledge_graph(repo.scan())
            fresh_rows, _ = repo.select_task_context("normalize", graph)
            helpers = next(row for row in fresh_rows if row["path"] == "helpers.py")
            self.assertEqual("symbol", helpers["retrieval_source"])

            (root / "helpers.py").write_text("def scrub(value):\n    return value.strip()\n", encoding="utf-8")
            stale_rows, stale_stats = repo.select_task_context("normalize", graph)
            stale_helpers = [row for row in stale_rows if row["path"] == "helpers.py"]
            self.assertFalse(any(row["retrieval_source"] == "symbol" for row in stale_helpers))
            self.assertIn("helpers.py", stale_stats["stale_graph_paths"])

    def test_route_aware_retrieval_matches_method_and_path_and_checks_freshness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "contacts.py"
            source.write_text("def create_contact():\n    return {'ok': True}\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            repo = RepositoryIntelligence(root)
            graph = build_code_knowledge_graph({
                "root": str(root.resolve()),
                "file_inventory": [{"path": "contacts.py", "content_sha256": digest, "index_status": "indexed"}],
                "routes": [{"file": "contacts.py", "method": "POST", "path": "/api/contacts", "provenance": "test-route-index"}],
            })
            rows, stats = repo.select_task_context("Implement POST /api/contacts endpoint", graph)
            row = next(item for item in rows if item["path"] == "contacts.py")
            self.assertEqual("route", row["retrieval_source"])
            self.assertEqual(1, stats["route_candidates"])

            source.write_text("def unrelated():\n    return None\n", encoding="utf-8")
            stale_rows, stale_stats = repo.select_task_context("Implement POST /api/contacts endpoint", graph)
            stale_row = next(item for item in stale_rows if item["path"] == "contacts.py")
            self.assertEqual("lexical", stale_row["retrieval_source"])
            self.assertIn("contacts.py", stale_stats["stale_graph_paths"])

    def test_fts_index_is_incremental_persistent_and_owner_project_workspace_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "billing.py").write_text("def settle_invoice(): return 'ledgerkeyword'\n", encoding="utf-8")
            (workspace / "notes.md").write_text("ledgerkeyword billing receipt\n", encoding="utf-8")
            store = OpenAgentStore(root / "data")
            service = CodebaseService(store, object())  # type: ignore[arg-type]
            if not service._fts_available:
                self.skipTest("SQLite build does not include FTS5")
            workspace_path = str(workspace.resolve())
            project_map = RepositoryIntelligence(workspace_path).scan()
            sync = service._refresh_retrieval_fts("p1", "alice", workspace_path, project_map)
            self.assertEqual("ready", sync["status"])
            self.assertEqual(2, sync["indexed_files"])
            candidates = service._search_retrieval_fts("p1", "alice", workspace_path, "ledgerkeyword", limit=10)
            self.assertEqual({"billing.py", "notes.md"}, {row["path"] for row in candidates})
            self.assertEqual([], service._search_retrieval_fts("p1", "bob", workspace_path, "ledgerkeyword"))
            self.assertEqual([], service._search_retrieval_fts("p2", "alice", workspace_path, "ledgerkeyword"))

            (workspace / "notes.md").unlink()
            changed_map = RepositoryIntelligence(workspace_path).scan(previous=project_map)
            updated = service._refresh_retrieval_fts("p1", "alice", workspace_path, changed_map)
            self.assertEqual(1, updated["removed"])
            self.assertEqual(["billing.py"], [row["path"] for row in service._search_retrieval_fts("p1", "alice", workspace_path, "ledgerkeyword")])

            restarted = CodebaseService(OpenAgentStore(root / "data"), object())  # type: ignore[arg-type]
            self.assertEqual(["billing.py"], [row["path"] for row in restarted._search_retrieval_fts("p1", "alice", workspace_path, "ledgerkeyword")])
            other_workspace = root / "other-workspace"
            other_workspace.mkdir()
            (other_workspace / "other.py").write_text("differentvalue = True\n", encoding="utf-8")
            restarted._refresh_retrieval_fts(
                "p1", "alice", str(other_workspace), RepositoryIntelligence(other_workspace).scan(),
            )
            self.assertEqual([], restarted._search_retrieval_fts("p1", "alice", workspace_path, "ledgerkeyword"))
            self.assertEqual(["other.py"], [row["path"] for row in restarted._search_retrieval_fts("p1", "alice", str(other_workspace), "differentvalue")])

    def test_versioned_four_project_retrieval_corpus_measures_hybrid_recall_noise_and_context(self) -> None:
        self.assertEqual("codeq-retrieval-v1", RETRIEVAL_CORPUS_VERSION)
        corpus = [
            {
                "query": "update account contact normalization",
                "files": {
                    "src/account.py": "from text_utils import normalize_contact\ndef update_account(value): return normalize_contact(value)\n",
                    "src/text_utils.py": "def normalize_contact(value): return value.strip().lower()\n",
                    "tests/test_account.py": "from src.account import update_account\ndef test_update_account(): assert update_account(' A ') == 'a'\n",
                    "docs/colors.md": "Use the approved colors for the account overview.\n",
                },
                "relevant": {"src/account.py", "src/text_utils.py", "tests/test_account.py"},
            },
            {
                "query": "profile render selected display formatter",
                "files": {
                    "src/profile.ts": "import { formatValue } from './formatters';\nexport function renderProfile(value: string) { return formatValue(value); }\n",
                    "src/formatters.ts": "export function formatValue(value: string) { return value.trim(); }\n",
                    "tests/profile.test.ts": "import { renderProfile } from '../src/profile';\ntest('profile', () => renderProfile(' a '));\n",
                    "docs/unrelated.md": "Deployment color palette.\n",
                },
                "relevant": {"src/profile.ts", "src/formatters.ts", "tests/profile.test.ts"},
            },
            {
                "query": "POST /api/invoices parse amount validation",
                "files": {
                    "billing/routes.py": "from parsing import parse_amount\n@router.post('/api/invoices')\ndef create_invoice(value): return parse_amount(value)\n",
                    "billing/parsing.py": "def parse_amount(value): return float(value)\n",
                    "tests/test_routes.py": "def test_create_invoice(): assert True\n",
                    "assets/logo.svg": "<svg/>\n",
                },
                "relevant": {"billing/routes.py", "billing/parsing.py", "tests/test_routes.py"},
            },
            {
                "query": "search indexed workspace knowledge record",
                "files": {
                    "knowledge/search.py": "def find_record(text): return lookup_index(text)\n",
                    "knowledge/index.py": "def lookup_index(text): return text\n",
                    "tests/test_search.py": "def test_find_record(): assert True\n",
                    "notes/branding.md": "Unrelated brand colors\n",
                },
                "relevant": {"knowledge/search.py", "knowledge/index.py", "tests/test_search.py"},
            },
        ]
        total = {"lexical_recall_at_3": 0.0, "hybrid_recall_at_3": 0.0, "hybrid_without_fts_recall_at_3": 0.0, "lexical_noise_at_3": 0, "hybrid_noise_at_3": 0, "hybrid_without_fts_noise_at_3": 0, "lexical_context_chars": 0, "hybrid_context_chars": 0, "route_hits": 0, "fts_hits": 0}
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            service = CodebaseService(OpenAgentStore(data_root), object())  # type: ignore[arg-type]
            if not service._fts_available:
                self.skipTest("SQLite build does not include FTS5")
            for index, fixture in enumerate(corpus):
                workspace = Path(directory) / f"project-{index}"
                workspace.mkdir()
                for relative, content in fixture["files"].items():
                    path = workspace / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                repo = RepositoryIntelligence(workspace)
                project_map = repo.scan()
                graph = build_code_knowledge_graph(project_map)
                owner, project_id = "fixture-owner", f"fixture-{index}"
                service._refresh_retrieval_fts(project_id, owner, str(workspace), project_map)
                fts_candidates = service._search_retrieval_fts(project_id, owner, str(workspace), fixture["query"], limit=30)
                lexical_rows = repo.select_context(fixture["query"], limit=3)
                hybrid_rows, stats = repo.select_task_context(
                    fixture["query"], graph, limit=3, graph_limit=3, fts_candidates_input=fts_candidates,
                )
                # Ablation: the same hybrid retrieval with the FTS5 candidates removed, to decide FTS5 on measurement.
                nofts_rows, _ = repo.select_task_context(
                    fixture["query"], graph, limit=3, graph_limit=3, fts_candidates_input=[],
                )
                relevant = fixture["relevant"]
                lexical_paths = {row["path"] for row in lexical_rows[:3]}
                hybrid_paths = {row["path"] for row in hybrid_rows[:3]}
                nofts_paths = {row["path"] for row in nofts_rows[:3]}
                total["lexical_recall_at_3"] += len(lexical_paths & relevant) / len(relevant)
                total["hybrid_recall_at_3"] += len(hybrid_paths & relevant) / len(relevant)
                total["hybrid_without_fts_recall_at_3"] += len(nofts_paths & relevant) / len(relevant)
                total["lexical_noise_at_3"] += len(lexical_paths - relevant)
                total["hybrid_noise_at_3"] += len(hybrid_paths - relevant)
                total["hybrid_without_fts_noise_at_3"] += len(nofts_paths - relevant)
                total["lexical_context_chars"] += sum(len(row["content"]) for row in lexical_rows[:3])
                total["hybrid_context_chars"] += sum(len(row["content"]) for row in hybrid_rows[:3])
                total["route_hits"] += sum(row["retrieval_source"] == "route" for row in hybrid_rows)
                total["fts_hits"] += stats["fts_candidates"]
        count = len(corpus)
        total["lexical_recall_at_3"] /= count
        total["hybrid_recall_at_3"] /= count
        total["hybrid_without_fts_recall_at_3"] /= count
        total["corpus_projects"] = count
        total["corpus_version"] = RETRIEVAL_CORPUS_VERSION
        print("RETRIEVAL_CORPUS_V1 " + json.dumps(total, sort_keys=True), flush=True)
        self.assertGreaterEqual(total["hybrid_recall_at_3"], total["lexical_recall_at_3"])
        self.assertLessEqual(total["hybrid_noise_at_3"], total["lexical_noise_at_3"])
        self.assertGreaterEqual(total["route_hits"], 1)
        self.assertGreater(total["fts_hits"], 0)

    def test_stale_fts_candidate_is_rejected_before_prompt_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old.py"
            source.write_text("oldneedle = True\n", encoding="utf-8")
            old_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            graph = build_code_knowledge_graph(RepositoryIntelligence(root).scan())
            source.write_text("new_value = True\n", encoding="utf-8")
            rows, stats = RepositoryIntelligence(root).select_task_context(
                "oldneedle", graph, fts_candidates_input=[{"path": "old.py", "content_sha256": old_hash, "score": 5}],
            )
            self.assertNotIn("old.py", {row["path"] for row in rows})
            self.assertIn("old.py", stats["stale_fts_paths"])

    def test_context_budget_bounds_complete_json_and_preserves_fresh_location_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = self._repo(root)
            (root / "large_a.py").write_text("\n".join(f"needle_value_{index} = {index}" for index in range(400)), encoding="utf-8")
            (root / "large_b.py").write_text("\n".join(f"needle_other_{index} = {index}" for index in range(400)), encoding="utf-8")
            graph = build_code_knowledge_graph(repo.scan())

            rows, stats = repo.select_task_context(
                "needle_value needle_other", graph, limit=12, max_chars=2_400, max_file_chars=1_500,
            )

            serialized = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
            self.assertLessEqual(len(serialized), 2_400)
            self.assertLessEqual(stats["context_chars"], stats["context_budget_chars"])
            self.assertTrue(stats["truncated_to_budget"])
            self.assertTrue(all(row["content_sha256"] and row["line_start"] >= 1 and row["line_end"] >= row["line_start"] for row in rows))
            self.assertIn("large_a.py", {row["path"] for row in rows})
            self.assertNotIn("irrelevant.py", {row["path"] for row in rows})

    def test_fixed_multirepository_task_corpus_recalls_imports_and_tests(self) -> None:
        corpus = [
            {
                "language": "python",
                "goal": "adjust contact formatting in account service",
                "seed": "src/accounts.py",
                "files": {
                    "src/accounts.py": "from text_utils import clean\n\ndef update_account(value):\n    return clean(value)\n",
                    "src/text_utils.py": "def clean(value):\n    return value.strip()\n",
                    "tests/test_accounts.py": "from src.accounts import update_account\n\ndef test_update_account():\n    assert update_account(' x ') == 'x'\n",
                    "docs/colors.md": "visual colors and typography only\n",
                },
                "relevant": {"src/accounts.py", "src/text_utils.py", "tests/test_accounts.py"},
            },
            {
                "language": "typescript",
                "goal": "update profile contact display behavior",
                "seed": "src/profile.ts",
                "files": {
                    "src/profile.ts": "import { formatValue } from './formatters';\nexport function renderProfile(value: string) { return formatValue(value); }\n",
                    "src/formatters.ts": "export function formatValue(value: string) { return value.trim(); }\n",
                    "tests/profile.test.ts": "import { renderProfile } from '../src/profile';\ntest('profile', () => renderProfile(' a '));\n",
                    "docs/unrelated.md": "deployment color palette\n",
                },
                "relevant": {"src/profile.ts", "src/formatters.ts", "tests/profile.test.ts"},
            },
            {
                "language": "python",
                "goal": "repair invoice amount parsing",
                "seed": "billing/invoice.py",
                "files": {
                    "billing/invoice.py": "from parsing import parse_amount\n\ndef invoice_total(value):\n    return parse_amount(value)\n",
                    "billing/parsing.py": "def parse_amount(value):\n    return float(value)\n",
                    "tests/test_invoice.py": "from billing.invoice import invoice_total\n\ndef test_invoice_total():\n    assert invoice_total('2') == 2.0\n",
                    "assets/logo.svg": "<svg/>\n",
                },
                "relevant": {"billing/invoice.py", "billing/parsing.py", "tests/test_invoice.py"},
            },
        ]
        total_baseline_recall = total_hybrid_recall = 0.0
        total_baseline_noise = total_hybrid_noise = 0
        for project in corpus:
            with self.subTest(language=project["language"], seed=project["seed"]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for relative, content in project["files"].items():
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                repo = RepositoryIntelligence(root)
                graph = build_code_knowledge_graph(repo.scan())
                baseline = {row["path"] for row in repo.select_context(project["goal"], limit=16)}
                hybrid_rows, _stats = repo.select_task_context(
                    project["goal"], graph, seed_paths=[project["seed"]],
                )
                hybrid = {row["path"] for row in hybrid_rows}
                relevant = project["relevant"]
                total_baseline_recall += len(baseline & relevant) / len(relevant)
                total_hybrid_recall += len(hybrid & relevant) / len(relevant)
                total_baseline_noise += len(baseline - relevant)
                total_hybrid_noise += len(hybrid - relevant)

        self.assertGreater(total_hybrid_recall, total_baseline_recall)
        self.assertLessEqual(total_hybrid_noise, total_baseline_noise)

    def test_large_distractor_corpus_retrieves_graph_neighbors_across_languages(self) -> None:
        """Measure retrieval on larger repositories where lexical noise dominates.

        Each independent project has 80 similarly-worded but unrelated documents.
        The task seed and parser-grounded call edge must still put the implementation
        file into the bounded model context. This exercises the real scanner, graph,
        freshness checks and retriever rather than a hand-built graph fixture.
        """
        from openagent.codebase.tree_sitter_index import TreeSitterIndexer

        indexer = TreeSitterIndexer()
        fixtures = {
            "python": (
                "src/invoice.py", "src/fees.py",
                "from fees import calculate_fee\ndef update_invoice(): return calculate_fee()\n",
                "def calculate_fee(): return 1\n",
            ),
            "typescript": (
                "src/invoice.ts", "src/fees.ts",
                "import { calculateFee } from './fees';\nexport function updateInvoice(){ return calculateFee(); }\n",
                "export function calculateFee(){ return 1; }\n",
            ),
            "go": (
                "src/invoice.go", "src/fees.go",
                "package billing\nfunc UpdateInvoice() int { return CalculateFee() }\n",
                "package billing\nfunc CalculateFee() int { return 1 }\n",
            ),
            "java": (
                "src/InvoiceService.java", "src/BillingMath.java",
                "class InvoiceService { int updateInvoice(){ return calculateFee(); } }\n",
                "class BillingMath { static int calculateFee(){ return 1; } }\n",
            ),
        }
        missing = sorted(set(fixtures) - set(indexer.loaded_languages))
        if missing:
            self.skipTest(f"requires locally installed Tree-sitter grammars: {missing}")

        baseline_recall = 0
        hybrid_recall = 0
        total_relevant = 0
        for language, (source_rel, helper_rel, source, helper) in fixtures.items():
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for relative, content in ((source_rel, source), (helper_rel, helper)):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                for index in range(80):
                    note = root / "docs" / f"invoice-fees-{index:03}.md"
                    note.parent.mkdir(parents=True, exist_ok=True)
                    note.write_text(
                        f"Invoice fees and calculation policies for unrelated operations note {index}.\n",
                        encoding="utf-8",
                    )

                repository = RepositoryIntelligence(root)
                project_map = repository.scan()
                graph = build_code_knowledge_graph(project_map)
                query = "fix invoice fee calculation and normalization"
                lexical = repository.select_context(query, limit=8)
                hybrid, stats = repository.select_task_context(
                    query, graph, seed_paths=[source_rel], limit=8, max_chars=12_000,
                )
                relevant = {source_rel, helper_rel}
                baseline_recall += len({row["path"] for row in lexical} & relevant)
                hybrid_recall += len({row["path"] for row in hybrid} & relevant)
                total_relevant += len(relevant)
                self.assertTrue(relevant <= {row["path"] for row in hybrid}, (language, stats["manifest"]))
                helper_row = next(row for row in hybrid if row["path"] == helper_rel)
                self.assertIn(helper_row["retrieval_source"], {"graph", "symbol"})
                self.assertTrue(helper_row["content_sha256"])
                self.assertGreaterEqual(helper_row["line_start"], 1)

        print({
            "corpus": "codeq-retrieval-large-v1",
            "projects": len(fixtures),
            "files_per_project": 82,
            "queries": len(fixtures),
            "lexical_recall": f"{baseline_recall}/{total_relevant}",
            "graph_hybrid_recall": f"{hybrid_recall}/{total_relevant}",
        })
        self.assertGreater(hybrid_recall, baseline_recall)

    def test_generic_task_words_do_not_create_false_symbol_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "codebase/scip_index.py": "def read_scip_index():\n    return True\n",
                "commands.py": "def build_parser():\n    return True\n",
                "evidence.py": "def source_fingerprint():\n    return True\n\ndef file_digests():\n    return []\n",
            }
            inventory = []
            symbols = []
            for relative, source in sources.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                inventory.append({"path": relative, "content_sha256": digest, "index_status": "indexed"})
                for line, name in enumerate(
                    [line.split("def ", 1)[1].split("(", 1)[0] for line in source.splitlines() if line.startswith("def ")],
                    1,
                ):
                    symbols.append({
                        "path": relative, "name": name, "kind": "function_definition", "line": line,
                        "content_sha256": digest, "provenance": "test-symbol-index",
                    })

            repository = RepositoryIntelligence(root)
            graph = build_code_knowledge_graph({
                "root": str(root.resolve()), "file_inventory": inventory, "symbols": symbols,
                "imports": {}, "routes": [], "test_files": [],
            })
            _rows, stats = repository.select_task_context(
                "SCIP index source range validation current file hash safe path parser freshness",
                graph, seed_paths=["codebase/scip_index.py"], limit=8,
            )

            self.assertEqual(0, stats["symbol_candidates"])

    def test_action_prompt_fits_input_budget_and_prioritizes_task_files(self) -> None:
        context_rows = [
            {"path": "src/required.py", "retrieval_source": "task_file", "content_sha256": "a" * 64,
             "line_start": 1, "line_end": 150, "content": "\n".join(f"required_{i} = {i}" for i in range(150))},
            {"path": "src/neighbor.py", "retrieval_source": "graph", "content_sha256": "b" * 64,
             "line_start": 1, "line_end": 150, "content": "\n".join(f"neighbor_{i} = {i}" for i in range(150))},
        ]
        sections = [
            ("Overall goal", "Keep the account email normalized", True),
            ("Current task", "Update the account service", True),
            ("Agent role", "programmer", True),
            ("Task evidence", "{}", False),
            ("Project memory", json.dumps({"large": "x" * 8000}), False),
            ("Recent project sessions", "old session " * 100, False),
            ("Relevant repository context", json.dumps(context_rows, separators=(",", ":")), False),
            ("Available tools", "read_file, patch_file, run_tests", True),
        ]

        messages, budget = CodebaseService._budget_action_messages(
            "system instructions", sections, input_token_budget=700,
        )

        self.assertTrue(budget["within_budget"])
        self.assertLessEqual(estimate_messages_tokens(messages), 700)
        user_prompt = messages[-1]["content"]
        self.assertIn("Overall goal: Keep the account email normalized", user_prompt)
        self.assertIn("Current task: Update the account service", user_prompt)
        self.assertIn("Available tools:", user_prompt)
        self.assertIn("src/required.py", user_prompt)
        self.assertNotIn("src/neighbor.py", user_prompt)
        self.assertIn("Relevant repository context excerpts", budget["trimmed_sections"])

    def test_action_prompt_reports_unfit_required_goal_instead_of_claiming_budget_pass(self) -> None:
        messages, budget = CodebaseService._budget_action_messages(
            "system instructions",
            [("Overall goal", "x" * 2_000, True), ("Available tools", "read_file", True)],
            input_token_budget=128,
        )
        self.assertFalse(budget["within_budget"])
        self.assertGreater(estimate_messages_tokens(messages), 128)


if __name__ == "__main__":
    unittest.main()
