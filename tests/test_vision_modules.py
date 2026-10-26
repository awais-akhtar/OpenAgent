"""Tests for all new vision modules: calendar, digital_twin, vault, scraper, fine_tuning."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
import json
from unittest.mock import patch


def make_store():
    from openagent.storage import OpenAgentStore
    tmp = tempfile.mkdtemp()
    return OpenAgentStore(data_dir=Path(tmp))


# ===========================================================================
# Calendar System
# ===========================================================================

class TestCalendarSystem(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        from openagent.calendar_system import CalendarSystem
        self.cal = CalendarSystem(self.store)

    def test_default_calendars_seeded(self):
        calendars = self.cal.list_calendars(project="default")
        names = [c["name"] for c in calendars]
        self.assertIn("Personal", names)
        self.assertIn("Work", names)

    def test_create_calendar(self):
        cal = self.cal.create_calendar("Test Calendar", project="test")
        self.assertEqual(cal["name"], "Test Calendar")
        self.assertEqual(cal["project"], "test")

    def test_create_and_retrieve_event(self):
        ev = self.cal.create_event(
            "Team standup",
            "2026-07-01T09:00:00",
            "2026-07-01T09:30:00",
            project="default",
        )
        self.assertEqual(ev["title"], "Team standup")
        retrieved = self.cal.get_event(ev["id"])
        self.assertEqual(retrieved["id"], ev["id"])

    def test_list_events_by_date_range(self):
        self.cal.create_event("July meeting", "2026-07-10T10:00:00", "2026-07-10T11:00:00", project="default")
        events = self.cal.list_events(project="default", from_dt="2026-07-01", to_dt="2026-07-31")
        self.assertTrue(any(e["title"] == "July meeting" for e in events))

    def test_update_event(self):
        ev = self.cal.create_event("Old title", "2026-07-15T12:00:00", "2026-07-15T13:00:00", project="default")
        updated = self.cal.update_event(ev["id"], updates={"title": "New title"})
        self.assertEqual(updated["title"], "New title")

    def test_delete_event(self):
        ev = self.cal.create_event("Temp event", "2026-07-20T08:00:00", "2026-07-20T09:00:00", project="default")
        deleted = self.cal.delete_event(ev["id"])
        self.assertTrue(deleted)

    def test_find_free_slots(self):
        slots = self.cal.find_free_slots(
            project="default",
            from_dt="2026-07-01T09:00:00",
            to_dt="2026-07-01T17:00:00",
            duration_minutes=30,
        )
        self.assertIsInstance(slots, list)
        for slot in slots:
            self.assertIn("starts_at", slot)
            self.assertIn("ends_at", slot)

    def test_ical_import_export(self):
        ical = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:test-123@openagent
SUMMARY:Imported Meeting
DTSTART:20260801T100000Z
DTEND:20260801T110000Z
DESCRIPTION:Imported from iCal
END:VEVENT
END:VCALENDAR"""
        result = self.cal.import_ical(ical, project="default")
        self.assertEqual(result["imported"], 1)
        exported = self.cal.export_ical(project="default")
        self.assertIn("BEGIN:VCALENDAR", exported)

    def test_optimize_schedule(self):
        result = self.cal.optimize_schedule(project="default")
        self.assertIn("total_events", result)
        self.assertIn("suggestions", result)

    def test_status(self):
        status = self.cal.status(project="default")
        self.assertIn("calendars", status)
        self.assertIn("events", status)
        self.assertIn("features", status)


# ===========================================================================
# Digital Twin
# ===========================================================================

class TestDigitalTwin(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        from openagent.digital_twin import DigitalTwin
        self.twin = DigitalTwin(self.store)

    def test_get_profile_creates_default(self):
        profile = self.twin.get_profile(project="default")
        self.assertIn("profile", profile)
        self.assertIn("preferred_language", profile["profile"])

    def test_update_profile(self):
        updated = self.twin.update_profile({"name": "Alice"}, project="default")
        self.assertEqual(updated["profile"]["name"], "Alice")

    def test_record_interaction(self):
        result = self.twin.record_interaction("chat", "Tell me about machine learning", project="default")
        self.assertIn("id", result)
        self.assertIn("topics", result)
        self.assertIsInstance(result["topics"], list)

    def test_source_metadata_explains_inferred_preference_signals_without_content(self):
        recorded = self.twin.record_interaction(
            "research", "Please help me investigate materials science", project="alpha",
            context={"private_path": "do-not-return"},
        )
        self.twin.record_interaction("chat", "materials science results", project="beta")

        source = next(item for item in self.twin.list_sources(project="alpha") if item["id"] == recorded["id"])
        self.assertTrue(source["preference_signals"])
        self.assertEqual(source["preference_signals"][0]["relationship"], "topic signal supplied to the preference learner")
        self.assertTrue({signal["key"] for signal in source["preference_signals"]}.issubset(set(source["topics"])))
        self.assertNotIn("content", source)
        self.assertNotIn("context", source)
        self.assertNotIn("private_path", json.dumps(source))
        self.assertEqual({item["id"] for item in self.twin.list_sources(project="alpha")}, {recorded["id"]})

    def test_analyse_patterns(self):
        self.twin.record_interaction("chat", "research Python code debugging", project="default")
        self.twin.record_interaction("search", "python debugging tips tools", project="default")
        patterns = self.twin.analyse_patterns(project="default")
        self.assertIn("patterns", patterns)
        self.assertIn("top_topics", patterns)

    def test_predict_intent(self):
        result = self.twin.predict_intent("I need to research machine learning", project="default")
        self.assertIn("predicted_intent", result)
        self.assertIn("confidence", result)
        self.assertIn("all_scores", result)
        self.assertGreater(result["confidence"], 0)

    def test_proactive_suggestions(self):
        suggestions = self.twin.generate_proactive_suggestions(project="default")
        self.assertIsInstance(suggestions, list)

    def test_create_and_list_goals(self):
        goal = self.twin.create_goal("Learn Python", project="default", description="Master Python")
        self.assertEqual(goal["title"], "Learn Python")
        goals = self.twin.list_goals(project="default")
        self.assertTrue(any(g["id"] == goal["id"] for g in goals))

    def test_update_goal_progress(self):
        goal = self.twin.create_goal("Write tests", project="default")
        updated = self.twin.update_goal_progress(goal["id"], 0.5)
        self.assertAlmostEqual(updated["progress"], 0.5)

    def test_consolidate_memory(self):
        result = self.twin.consolidate_memory(project="default")
        self.assertIn("consolidated", result)
        self.assertIn("facts", result)

    def test_status(self):
        status = self.twin.status(project="default")
        self.assertIn("interactions_recorded", status)
        self.assertIn("active_goals", status)
        self.assertIn("features", status)


# ===========================================================================
# Encrypted Vault
# ===========================================================================

class TestEncryptedVault(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        from openagent.vault import EncryptedVault
        self.vault = EncryptedVault(self.store)

    def test_store_and_retrieve_secret(self):
        entry = self.vault.store_secret(
            "GitHub API Key", "ghp_test_secret_1234", "my-secure-passphrase",
            project="default", category="api_key",
        )
        self.assertIn("id", entry)
        self.assertNotIn("ciphertext", entry)  # meta only

        retrieved = self.vault.retrieve_secret(entry["id"], "my-secure-passphrase", project="default")
        self.assertEqual(retrieved["secret"], "ghp_test_secret_1234")
        self.assertEqual(retrieved["label"], "GitHub API Key")

    def test_wrong_passphrase_raises(self):
        entry = self.vault.store_secret("Test", "secret", "correct-pass", project="default")
        with self.assertRaises(ValueError):
            self.vault.retrieve_secret(entry["id"], "wrong-pass", project="default")

    def test_list_entries_no_plaintext(self):
        self.vault.store_secret("Key 1", "val1", "pass123", project="default")
        self.vault.store_secret("Key 2", "val2", "pass123", project="default", category="token")
        entries = self.vault.list_entries(project="default")
        self.assertGreaterEqual(len(entries), 2)
        for e in entries:
            self.assertNotIn("ciphertext", e)
            self.assertNotIn("secret", e)

    def test_update_entry(self):
        entry = self.vault.store_secret("My note", "secret note", "pass", project="default", category="private_note")
        updated = self.vault.update_entry(entry["id"], "pass", project="default", new_label="Updated note")
        self.assertEqual(updated["label"], "Updated note")

    def test_delete_entry(self):
        entry = self.vault.store_secret("Temp", "temp secret", "pass", project="default")
        deleted = self.vault.delete_entry(entry["id"], "pass", project="default")
        self.assertTrue(deleted)

    def test_key_rotation(self):
        entry = self.vault.store_secret("Rotate me", "sensitive value", "old-pass", project="default")
        result = self.vault.rotate_key("old-pass", "new-pass", project="default")
        self.assertGreater(result["rotated"], 0)
        # Can now retrieve with new passphrase
        retrieved = self.vault.retrieve_secret(entry["id"], "new-pass", project="default")
        self.assertEqual(retrieved["secret"], "sensitive value")

    def test_export_vault(self):
        self.vault.store_secret("Export test", "val", "pass", project="default")
        bundle = self.vault.export_vault("pass", project="default")
        self.assertIn("encrypted_bundle", bundle)
        self.assertIn("format", bundle)
        self.assertGreater(bundle["entry_count"], 0)

    def test_status(self):
        status = self.vault.status(project="default")
        self.assertIn("total_entries", status)
        self.assertIn("encryption", status)
        self.assertIn("features", status)

    def test_category_filtering(self):
        self.vault.store_secret("PW", "pass", "pp", project="default", category="password")
        self.vault.store_secret("Key", "key", "pp", project="default", category="api_key")
        passwords = self.vault.list_entries(project="default", category="password")
        self.assertTrue(all(e["category"] == "password" for e in passwords))


# ===========================================================================
# Web Scraper
# ===========================================================================

class TestWebScraper(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        from openagent.scraper import WebScraper
        self.scraper = WebScraper(self.store)

    def test_invalid_scheme_rejected(self):
        from openagent.scraper import _validate_url
        with self.assertRaises(ValueError):
            _validate_url("ftp://example.com/file")
        with self.assertRaises(ValueError):
            _validate_url("file:///etc/passwd")

    def test_ssrf_protection_localhost(self):
        from openagent.scraper import _validate_url
        with self.assertRaises(ValueError):
            _validate_url("http://127.0.0.1/admin")
        with self.assertRaises(ValueError):
            _validate_url("http://192.168.1.1/config")

    def test_schedule_scrape(self):
        # Using a well-formed public URL (won't actually fetch in tests)
        job = self.scraper.schedule_scrape(
            "https://example.com", project="default", interval_hours=12.0
        )
        self.assertIn("job_id", job)
        self.assertEqual(job["interval_hours"], 12.0)

    def test_list_scheduled(self):
        self.scraper.schedule_scrape("https://example.com/page", project="default")
        jobs = self.scraper.list_scheduled(project="default")
        self.assertGreaterEqual(len(jobs), 1)

    def test_schedule_is_offline_and_keeps_fetch_time_ssrf_checks(self):
        with patch("openagent.egress.socket.getaddrinfo", side_effect=AssertionError("scheduling must not resolve DNS")):
            job = self.scraper.schedule_scrape("https://example.com/offline", project="default")
        self.assertEqual(job["status"], "awaiting_approval")
        for url in ("http://127.0.0.1/admin", "http://localhost/private", "ftp://example.com/file"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.scraper.schedule_scrape(url, project="default")

    def test_due_scrape_dns_failure_is_recorded_instead_of_crashing_worker(self):
        job = self.scraper.schedule_scrape(
            "https://unresolvable.example/offline", project="default", allow_external=True,
        )
        with self.store.session() as connection:
            connection.execute("UPDATE scrape_jobs SET next_run = '2000-01-01T00:00:00+00:00' WHERE id = ?", (job["job_id"],))
        with patch.object(self.scraper, "scrape_url", side_effect=ValueError("Outbound hostname could not be resolved.")):
            result = self.scraper.run_due_jobs(project="default")
        self.assertEqual(result, {"ran": 0, "errors": 1, "total_due": 1})
        history = self.scraper.get_scrape_history(project="default", limit=1)
        self.assertEqual(history[0]["error"], "Outbound hostname could not be resolved.")

    def test_cancel_pending_schedule_is_project_scoped_and_keeps_history(self):
        job = self.scraper.schedule_scrape(
            "https://example.com/cancellable", project="owner-project", allow_external=True,
        )
        self.assertIsNone(self.scraper.cancel_scheduled(job["job_id"], project="other-project"))
        cancelled = self.scraper.cancel_scheduled(job["job_id"], project="owner-project")
        self.assertEqual(cancelled["status"], "cancelled")
        rows = self.scraper.list_scheduled(project="owner-project")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], job["job_id"])
        self.assertEqual(rows[0]["status"], "cancelled")

    def test_cancel_awaiting_approval_and_running_schedules_is_truthful(self):
        pending = self.scraper.schedule_scrape("https://example.com/pending", project="default")
        self.assertEqual(pending["status"], "awaiting_approval")
        cancelled = self.scraper.cancel_scheduled(pending["job_id"], project="default")
        self.assertEqual(cancelled["status"], "cancelled")

        running = self.scraper.schedule_scrape(
            "https://example.com/running", project="default", allow_external=True,
        )
        with self.store.session() as connection:
            connection.execute("UPDATE scrape_jobs SET status='running' WHERE id=?", (running["job_id"],))
        result = self.scraper.cancel_scheduled(running["job_id"], project="default")
        self.assertEqual(result["status"], "running")

    def test_cancelled_schedule_is_not_processed_and_restart_recovers_claimed_job(self):
        job = self.scraper.schedule_scrape(
            "https://example.com/restart", project="default", allow_external=True,
        )
        with self.store.session() as connection:
            connection.execute("UPDATE scrape_jobs SET next_run='2000-01-01T00:00:00+00:00' WHERE id=?", (job["job_id"],))
        self.scraper.cancel_scheduled(job["job_id"], project="default")
        with patch.object(self.scraper, "scrape_url") as fetch:
            result = self.scraper.run_due_jobs(project="default")
        self.assertEqual(result, {"ran": 0, "errors": 0, "total_due": 0})
        fetch.assert_not_called()

        with self.store.session() as connection:
            connection.execute("UPDATE scrape_jobs SET status='running' WHERE id=?", (job["job_id"],))
        self.assertEqual(self.scraper.recover_interrupted_schedules(), 1)
        recovered = self.scraper.list_scheduled(project="default")
        self.assertEqual(recovered[0]["status"], "scheduled")

    def test_cancel_racing_due_claim_does_not_claim_running_fetch_was_stopped(self):
        job = self.scraper.schedule_scrape(
            "https://example.com/race", project="default", allow_external=True,
        )
        with self.store.session() as connection:
            connection.execute("UPDATE scrape_jobs SET next_run='2000-01-01T00:00:00+00:00' WHERE id=?", (job["job_id"],))
        started = threading.Event()
        release = threading.Event()
        results = []

        def fetch(*args, **kwargs):
            started.set()
            if not release.wait(timeout=5):
                raise AssertionError("test fetch was not released")
            return {"url": job["url"], "status": "ok", "status_code": 200}

        with patch.object(self.scraper, "scrape_url", side_effect=fetch):
            worker = threading.Thread(target=lambda: results.append(self.scraper.run_due_jobs(project="default")))
            worker.start()
            self.assertTrue(started.wait(timeout=5), "due job did not reach its fetch")
            during = self.scraper.cancel_scheduled(job["job_id"], project="default")
            self.assertEqual(during["status"], "running")
            release.set()
            worker.join(timeout=5)

        self.assertFalse(worker.is_alive(), "due job did not finish")
        self.assertEqual(results, [{"ran": 1, "errors": 0, "total_due": 1}])
        self.assertEqual(self.scraper.list_scheduled(project="default")[0]["status"], "scheduled")

    def test_status(self):
        status = self.scraper.status(project="default")
        self.assertIn("scheduled_jobs", status)
        self.assertIn("features", status)

    def test_html_text_extraction(self):
        from openagent.scraper import _TextExtractor
        html = """<html><head><title>Test Page</title>
        <meta name="description" content="A test page">
        </head><body><h1>Hello World</h1><p>This is a test.</p></body></html>"""
        extractor = _TextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        self.assertIn("Hello World", text)
        self.assertIn("This is a test", text)
        self.assertEqual(extractor.title, "Test Page")
        self.assertEqual(extractor.description, "A test page")


# ===========================================================================
# Fine-Tuning Manager
# ===========================================================================

class TestFineTuningManager(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        from openagent.fine_tuning import FineTuningManager
        self.ft = FineTuningManager(self.store)

    def test_templates_seeded(self):
        templates = self.ft.list_templates()
        names = [t["name"] for t in templates]
        self.assertIn("LoRA Fast", names)
        self.assertIn("QLoRA 4-bit", names)

    def test_create_dataset(self):
        ds = self.ft.create_dataset("My Dataset", project="default", description="Test dataset")
        self.assertEqual(ds["name"], "My Dataset")
        self.assertEqual(ds["num_examples"], 0)

    def test_add_and_count_examples(self):
        ds = self.ft.create_dataset("Training set", project="default")
        self.ft.add_example(ds["id"], "What is Python?", "Python is a programming language.", project="default")
        self.ft.add_example(ds["id"], "What is AI?", "AI is artificial intelligence.", project="default")
        refreshed = self.ft.get_dataset(ds["id"])
        self.assertEqual(refreshed["num_examples"], 2)

    def test_add_example_rejects_dataset_from_another_project(self):
        ds = self.ft.create_dataset("Private dataset", project="owner-project")
        with self.assertRaisesRegex(KeyError, "Dataset not found in project"):
            self.ft.add_example(ds["id"], "prompt", "response", project="other-project")
        self.assertEqual(self.ft.get_dataset(ds["id"])["num_examples"], 0)

    def test_export_dataset_jsonl(self):
        ds = self.ft.create_dataset("Export test", project="default")
        self.ft.add_example(ds["id"], "User question", "Assistant answer", project="default")
        jsonl = self.ft.export_dataset_jsonl(ds["id"])
        # May be empty if example landed in val split; check type at least
        self.assertIsInstance(jsonl, str)

    def test_create_job(self):
        ds = self.ft.create_dataset("Job dataset", project="default")
        job = self.ft.create_job(ds["id"], "llama3.1:8b", project="default", method="lora")
        self.assertEqual(job["base_model"], "llama3.1:8b")
        self.assertEqual(job["method"], "lora")
        self.assertEqual(job["status"], "queued")

    def test_create_job_rejects_dataset_from_another_project(self):
        ds = self.ft.create_dataset("Private training", project="owner-project")
        with self.assertRaisesRegex(KeyError, "Dataset not found in project"):
            self.ft.create_job(ds["id"], "local-model", project="other-project")
        self.assertEqual(self.ft.list_jobs(project="other-project"), [])

    def test_update_job_progress(self):
        ds = self.ft.create_dataset("Progress dataset", project="default")
        job = self.ft.create_job(ds["id"], "llama3.1", project="default")
        updated = self.ft.update_job_progress(job["id"], 0.5, loss=1.23, status="running")
        self.assertAlmostEqual(updated["progress"], 0.5)
        self.assertEqual(updated["status"], "running")

    def test_complete_job(self):
        ds = self.ft.create_dataset("Complete dataset", project="default")
        job = self.ft.create_job(ds["id"], "mistral", project="default")
        completed = self.ft.complete_job(job["id"])
        self.assertEqual(completed["status"], "completed")
        self.assertAlmostEqual(completed["progress"], 1.0)

    def test_cancel_queued_job_is_project_scoped_and_terminal(self):
        ds = self.ft.create_dataset("Cancelable dataset", project="research")
        job = self.ft.create_job(ds["id"], "qwen", project="research")
        cancelled = self.ft.cancel_queued_job(job["id"], project="research")
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertTrue(cancelled["completed_at"])
        with self.assertRaises(KeyError):
            self.ft.cancel_queued_job(job["id"], project="other")
        with self.assertRaisesRegex(RuntimeError, "Only queued or running jobs"):
            self.ft.cancel_queued_job(job["id"], project="research")

    def test_running_job_stays_cancelling_until_worker_acknowledges(self):
        ds = self.ft.create_dataset("Running dataset", project="default")
        job = self.ft.create_job(ds["id"], "qwen", project="default")
        self.ft.update_job_progress(job["id"], 0.25, status="running")
        requested = self.ft.cancel_queued_job(job["id"], project="default")
        self.assertEqual(requested["status"], "cancelling")
        # Late progress/completion callbacks must not overwrite the request.
        self.assertEqual(self.ft.update_job_progress(job["id"], 0.9, status="running")["status"], "cancelling")
        self.assertEqual(self.ft.complete_job(job["id"])["status"], "cancelling")
        self.assertEqual(self.ft._finish_job_cancelled(job["id"], "default")["status"], "cancelled")

    def test_register_and_activate_adapter(self):
        adapter = self.ft.register_adapter(
            "my-lora", "llama3.1:8b", project="default",
            rank=16, alpha=32.0, eval_score=0.85,
        )
        self.assertEqual(adapter["name"], "my-lora")
        self.assertEqual(adapter["rank"], 16)
        activated = self.ft.activate_adapter(adapter["id"], project="default")
        self.assertTrue(activated["active"])

    def test_list_adapters(self):
        self.ft.register_adapter("adapter-a", "llama3.1", project="default")
        self.ft.register_adapter("adapter-b", "llama3.1", project="default")
        adapters = self.ft.list_adapters(project="default", base_model="llama3.1")
        self.assertGreaterEqual(len(adapters), 2)

    def test_status(self):
        status = self.ft.status(project="default")
        self.assertIn("datasets", status)
        self.assertIn("jobs", status)
        self.assertIn("adapters", status)
        self.assertIn("features", status)


if __name__ == "__main__":
    unittest.main()
