"""The native folder chooser must not block a thread or open on a stranger's screen.

`POST /api/codebase/folder-picker` opens a modal Tk directory dialog on the
desktop of the machine running the server. Two things followed from that being
unconditional and unbounded, both found by sending the route an empty body:

* the request never returned -- the probe timed out after 20 seconds -- because
  a modal dialog waits for a person, so the HTTP thread serving it was held for
  as long as the window stood open;
* in any deployment mode other than a local single-user install, the window
  appears on the server's desktop rather than the caller's, so a remote caller
  could raise a dialog on the owner's screen and hold a thread with it.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from openagent.codebase.service import CodebaseService
from openagent.storage import OpenAgentStore


@contextmanager
def deployment(mode: str):
    previous = os.environ.get("OPENAGENT_DEPLOYMENT_MODE")
    os.environ["OPENAGENT_DEPLOYMENT_MODE"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENAGENT_DEPLOYMENT_MODE", None)
        else:
            os.environ["OPENAGENT_DEPLOYMENT_MODE"] = previous


class FolderPickerBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.service = CodebaseService(OpenAgentStore(self._tmp.name), None)  # type: ignore[arg-type]

    def test_a_remote_deployment_is_refused_without_opening_anything(self) -> None:
        with deployment("personal_remote"):
            with self.assertRaises(ValueError) as raised:
                self.service.pick_source_folder()
        message = str(raised.exception)
        self.assertIn("single_user", message)
        # The refusal has to tell the caller what to do instead.
        self.assertIn("source_path", message)

    def test_a_team_deployment_is_refused(self) -> None:
        with deployment("team"):
            with self.assertRaises(ValueError):
                self.service.pick_source_folder()

    def test_an_enterprise_deployment_is_refused(self) -> None:
        with deployment("enterprise"):
            with self.assertRaises(ValueError):
                self.service.pick_source_folder()

    def test_the_wait_is_bounded(self) -> None:
        # Tk offers no timeout of its own, so the bound is the service's and it
        # has to be a real, finite number of seconds.
        timeout = CodebaseService.FOLDER_PICKER_TIMEOUT_SECONDS
        self.assertIsInstance(timeout, int)
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, 600)

    def test_a_local_install_schedules_that_bound_before_waiting(self) -> None:
        """The dialog is never entered without the timer that releases it."""
        import openagent.codebase.service as module

        order: list[str] = []

        class FakeRoot:
            def withdraw(self) -> None:
                order.append("withdraw")

            def attributes(self, *_args: object) -> None:
                order.append("attributes")

            def after(self, milliseconds: int, _callback: object) -> None:
                order.append(f"after:{milliseconds}")

            def quit(self) -> None:  # noqa: A003 - the Tk name.
                order.append("quit")

            def destroy(self) -> None:
                order.append("destroy")

        class FakeDialog:
            @staticmethod
            def askdirectory(**_kwargs: object) -> str:
                order.append("askdirectory")
                return ""

        fake_tk = type("FakeTk", (), {"Tk": staticmethod(lambda: FakeRoot())})
        real_import = module.__builtins__["__import__"] if isinstance(module.__builtins__, dict) else __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
            if name == "tkinter" and fromlist and "filedialog" in fromlist:
                return type("FakeTkinter", (), {"filedialog": FakeDialog})
            if name == "tkinter":
                return fake_tk
            return real_import(name, globals, locals, fromlist, level)

        import builtins

        original = builtins.__import__
        builtins.__import__ = fake_import
        try:
            with deployment("single_user"):
                self.assertEqual(self.service.pick_source_folder(), "")
        finally:
            builtins.__import__ = original

        self.assertIn(f"after:{CodebaseService.FOLDER_PICKER_TIMEOUT_SECONDS * 1000}", order)
        self.assertLess(
            order.index(f"after:{CodebaseService.FOLDER_PICKER_TIMEOUT_SECONDS * 1000}"),
            order.index("askdirectory"),
            f"the timer must be armed before the modal dialog is entered: {order}",
        )
        self.assertEqual(order[-1], "destroy")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
