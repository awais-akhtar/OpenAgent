"""A serving plan must never be reported as a running model.

L09 of the gap board asks for the launch boundary to stay truthful on a host
without a local serving runtime, which is the case here. Two paths used to
claim more than was true:

* ``start_serve`` caught ``FileNotFoundError`` from the missing runtime and set
  the session to ``ready`` with a pid of ``0``, calling it "mock mode", so the
  screen showed a model ready to answer while no process existed.
* ``UltraModelService.create_session`` downloaded the model when it was not
  cached, so one create click started a multi-gigabyte transfer, and it then
  recorded the plan-only session as ``ready``.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openagent.model_serve_manager import ModelServeManager


class MissingRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.models_dir = Path(self._tmp.name)
        self.manager = ModelServeManager(models_dir=str(self.models_dir))
        self.model_path = self.models_dir / "model.gguf"
        self.model_path.write_bytes(b"not a real model, only a file that exists")

    def _plan(self, project: str = "p", owner: str = "owner-a") -> str:
        plan = self.manager.create_serve_plan(
            model_id="test-model", model_path=str(self.model_path), project=project, owner=owner
        )
        self.assertEqual(plan["status"], "planned")
        return plan["session_id"]

    def _start_without_runtime(self, session_id: str, **scope: str) -> dict:
        """Start with the runtime absent, which is this host's real state."""
        import openagent.model_serve_manager as module

        def refuse(*_args, **_kwargs):
            raise FileNotFoundError("llama-server")

        original = module.subprocess.Popen
        module.subprocess.Popen = refuse
        self.addCleanup(lambda: setattr(module.subprocess, "Popen", original))
        return self.manager.start_serve(session_id, **scope)

    def test_a_missing_runtime_reports_unavailable_not_ready(self) -> None:
        result = self._start_without_runtime(self._plan(), project="p", owner="owner-a")
        self.assertEqual(result["status"], "unavailable")

    def test_a_missing_runtime_claims_no_process(self) -> None:
        result = self._start_without_runtime(self._plan(), project="p", owner="owner-a")
        self.assertIsNone(result.get("pid"))
        self.assertFalse(result.get("process_identity"))

    def test_the_logs_say_why_nothing_is_serving(self) -> None:
        result = self._start_without_runtime(self._plan(), project="p", owner="owner-a")
        self.assertTrue(
            any("not installed" in line for line in result["logs"]),
            result["logs"],
        )
        self.assertFalse(any("mock" in line.lower() for line in result["logs"]), result["logs"])

    def test_a_plan_alone_is_not_running(self) -> None:
        session_id = self._plan()
        session = self.manager.get_session(session_id, project="p", owner="owner-a")
        self.assertEqual(session["status"], "planned")
        self.assertIsNone(session.get("pid"))

    def test_a_missing_model_file_is_refused_before_any_launch(self) -> None:
        self.model_path.unlink()
        result = self.manager.start_serve(self._plan(), project="p", owner="owner-a")
        self.assertIn("Model file not found", result["error"])

    def test_another_owner_cannot_start_the_session(self) -> None:
        result = self.manager.start_serve(self._plan(), project="p", owner="owner-b")
        self.assertIn("does not belong", result["error"])

    def test_another_project_cannot_stop_the_session(self) -> None:
        result = self.manager.stop_serve(self._plan(), project="other", owner="owner-a")
        self.assertIn("does not belong", result["error"])

    def test_stopping_an_unavailable_session_does_not_fail(self) -> None:
        session_id = self._plan()
        self._start_without_runtime(session_id, project="p", owner="owner-a")
        stopped = self.manager.stop_serve(session_id, project="p", owner="owner-a")
        self.assertEqual(stopped["status"], "stopped")

    def test_the_state_file_stays_beside_the_models(self) -> None:
        self._plan()
        self.assertEqual(self.manager.state_file.parent, self.models_dir)


class UltraSessionDownloadBoundaryTests(unittest.TestCase):
    """Creating a session must not fetch a model."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        from openagent.ultra_model_service import UltraModelService

        self.service = UltraModelService(models_dir=str(Path(self._tmp.name) / "models"))
        self.downloads: list[str] = []

        def record_download(model_id: str):
            self.downloads.append(model_id)
            raise AssertionError("create_session must not download")

        self.service.downloader.download_model = record_download  # type: ignore[method-assign]

    def _offer(self, model_id: str, cached: bool) -> None:
        self.service.downloader.get_model_info = lambda requested: (  # type: ignore[method-assign]
            {"id": requested, "cached": cached} if requested == model_id else None
        )

    def test_an_uncached_model_is_refused_without_downloading(self) -> None:
        self._offer("big-model", cached=False)
        result = self.service.create_session("big-model", project="p", owner="owner-a")
        self.assertTrue(result.get("requires_download"))
        self.assertIn("not downloaded", result["error"])
        self.assertEqual(self.downloads, [])

    def test_an_unknown_model_is_refused(self) -> None:
        self._offer("big-model", cached=False)
        result = self.service.create_session("no-such-model", project="p", owner="owner-a")
        self.assertIn("Unknown model", result["error"])
        self.assertEqual(self.downloads, [])

    def test_a_cached_model_yields_a_planned_session(self) -> None:
        model_path = Path(self._tmp.name) / "cached.gguf"
        model_path.write_bytes(b"cached model file")
        self._offer("cached-model", cached=True)
        self.service.downloader.get_model_path = lambda _model_id: model_path  # type: ignore[method-assign]
        result = self.service.create_session("cached-model", project="p", owner="owner-a")
        self.assertNotIn("error", result)
        # A plan, not a process.
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["serve_plan"]["status"], "planned")
        self.assertEqual(self.downloads, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class CookbookReadinessHonestyTests(unittest.TestCase):
    """Readiness must describe this host, not the shipped mock."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        from openagent.model_cookbook import ModelCookbook
        from openagent.storage import OpenAgentStore

        self.cookbook = ModelCookbook(OpenAgentStore(self._tmp.name))

    def _with_executable(self, path: str | None) -> dict:
        from openagent import model_cookbook

        original = model_cookbook.shutil.which
        model_cookbook.shutil.which = lambda name: path if name == "llama-server" else original(name)
        self.addCleanup(lambda: setattr(model_cookbook.shutil, "which", original))
        return self.cookbook.get_readiness_status(project="p")

    def test_no_installed_runtime_is_blocked(self) -> None:
        readiness = self._with_executable(None)
        self.assertFalse(readiness["llama_server_available"])
        self.assertEqual(readiness["status"], "blocked")
        self.assertIn("not installed", readiness["message"])

    def test_the_shipped_mock_never_counts_as_availability(self) -> None:
        readiness = self._with_executable(None)
        # mock_llama_server ships with the package, so it imports on every
        # host. Deciding availability by importing it made this answer "ready"
        # everywhere, including a host with no serving runtime at all.
        self.assertTrue(readiness["mock_server_available"])
        self.assertFalse(readiness["llama_server_available"])

    def test_an_installed_runtime_is_ready_and_named(self) -> None:
        readiness = self._with_executable("C:/tools/llama-server.exe")
        self.assertTrue(readiness["llama_server_available"])
        self.assertEqual(readiness["status"], "ready")
        self.assertEqual(readiness["llama_server_path"], "C:/tools/llama-server.exe")

    def test_a_cookbook_session_is_planned_with_no_process(self) -> None:
        plan = self.cookbook.create_serve_plan(
            project="p", model_name="test-model", model_source="local",
            serve_type="llama-server", config={"host": "127.0.0.1", "port": 9100},
        )
        session = self.cookbook.create_session(project="p", serve_plan_id=plan["id"])
        self.assertEqual(session["status"], "planned")
        self.assertNotIn("pid", session)


class UltraReadinessHonestyTests(unittest.TestCase):
    """The ultra readiness report must say whether a model can be served."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        from openagent.ultra_model_service import UltraModelService

        self.service = UltraModelService(models_dir=str(Path(self._tmp.name) / "models"))

    def _readiness(self, executable: str | None, cached: bool) -> dict:
        from openagent import ultra_model_service as module

        original = module.shutil.which
        module.shutil.which = lambda name: executable if name == "llama-server" else original(name)
        self.addCleanup(lambda: setattr(module.shutil, "which", original))
        self.service.get_available_models = lambda: {"m": {"id": "m", "cached": cached}}  # type: ignore[method-assign]
        return self.service.get_readiness_status(project="p", owner="owner-a")

    def test_no_runtime_is_blocked_and_says_why(self) -> None:
        readiness = self._readiness(None, cached=True)
        self.assertEqual(readiness["status"], "blocked")
        self.assertFalse(readiness["llama_server_available"])
        self.assertIn("not installed", readiness["message"])

    def test_a_runtime_with_no_downloaded_model_is_blocked(self) -> None:
        readiness = self._readiness("C:/tools/llama-server.exe", cached=False)
        self.assertEqual(readiness["status"], "blocked")
        self.assertTrue(readiness["llama_server_available"])
        self.assertIn("no model is downloaded", readiness["message"])

    def test_a_runtime_and_a_downloaded_model_is_ready(self) -> None:
        readiness = self._readiness("C:/tools/llama-server.exe", cached=True)
        self.assertEqual(readiness["status"], "ready")
        self.assertEqual(readiness["llama_server_path"], "C:/tools/llama-server.exe")

    def test_the_manager_flags_alone_never_decide_readiness(self) -> None:
        readiness = self._readiness(None, cached=True)
        # Both of these are true on any host, which is why they used to make
        # every host look ready.
        self.assertTrue(readiness["downloader_ready"])
        self.assertTrue(readiness["serve_manager_ready"])
        self.assertEqual(readiness["status"], "blocked")
