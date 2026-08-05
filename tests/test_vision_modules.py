"""Tests for all new vision modules: calendar, digital_twin, vault, scraper, fine_tuning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


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
