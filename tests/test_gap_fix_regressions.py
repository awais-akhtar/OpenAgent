"""Focused regressions for confirmed frontend/backend boundary failures."""
import json
import threading
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
from urllib.request import Request, urlopen

from openagent.storage import OpenAgentStore
from openagent.personal_os import PersonalOS, PersonalOSError
from openagent.vault import EncryptedVault
from openagent.multi_agent import MultiAgentRuntime
from openagent.routes.runtime import build_http_server
from openagent.scheduler import Scheduler


class GapFixRegressions(unittest.TestCase):
    def test_preset_update_delete_are_id_stable_and_scoped(self):
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            service = PersonalOS(store)
            preset = service.upsert_preset(name="Original", project="p", owner="alice")
            for project, owner in [("other", "alice"), ("p", "bob")]:
                with self.assertRaises(PersonalOSError):
                    service.upsert_preset(preset_id=preset["id"], name="Forged", project=project, owner=owner)
                self.assertFalse(service.delete_preset(preset["id"], project=project, owner=owner))
            changed = service.upsert_preset(preset_id=preset["id"], name="Renamed", project="p", owner="alice")
            self.assertEqual(changed["id"], preset["id"])
            self.assertEqual(changed["created_at"], preset["created_at"])
            self.assertEqual(PersonalOS(OpenAgentStore(tmp)).presets(project="p", owner="alice")[0]["name"], "Renamed")
            self.assertTrue(service.delete_preset(preset["id"], project="p", owner="alice"))
            self.assertEqual(service.presets(project="p", owner="alice"), [])

    def test_vault_requests_validate_entries_duration_and_transitions(self):
        with TemporaryDirectory() as tmp:
            vault = EncryptedVault(OpenAgentStore(tmp))
            entry = vault.store_secret("Synthetic", "noncredential-example", "test-only-passphrase", project="p")
            for ids, duration, project in [(["missing"],60,"p"), ([entry["id"]],0,"p"),
                    ([entry["id"]],1441,"p"), ([entry["id"]],True,"p"), ([entry["id"]],60,"other"),
                    ([7],60,"p")]:
                with self.assertRaises(ValueError):
                    vault.create_access_request("Review", ids, project=project, duration_minutes=duration)
            request = vault.create_access_request("Review", [entry["id"]], project="p")
            self.assertEqual(EncryptedVault(OpenAgentStore(tmp)).list_access_requests(project="p")[0]["id"], request["id"])
            with self.assertRaises(KeyError):
                vault.approve_access_request(request["id"], project="other")
            self.assertEqual(vault.approve_access_request(request["id"], project="p")["status"], "approved")
            with self.assertRaises(ValueError):
                vault.deny_access_request(request["id"], project="p")
            request2 = vault.create_access_request("Review", [entry["id"]], project="p")
            self.assertEqual(vault.deny_access_request(request2["id"], project="p")["status"], "denied")

    def test_real_http_preset_crud_and_sse_object_frames(self):
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {"OPENAGENT_AUTH_ENFORCED":"0"}):
            store = OpenAgentStore(tmp)
            server = build_http_server("127.0.0.1", 0, store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = "http://127.0.0.1:" + str(server.server_address[1])
            def request(method, path, body):
                with urlopen(Request(base+path, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}, method=method), timeout=5) as response:
                    return json.load(response)
            try:
                preset=request("POST", "/api/personal-os/presets", {"name":"Original"})["preset"]
                changed=request("PUT", "/api/personal-os/presets/"+preset["id"], {"name":"Renamed"})["preset"]
                self.assertEqual(changed["id"],preset["id"])
                self.assertEqual(changed["name"],"Renamed")
                self.assertEqual(request("DELETE", "/api/personal-os/presets/"+preset["id"], {})["deleted"],preset["id"])
                runtime=MultiAgentRuntime(store)
                with patch.object(runtime, "_execute_job", lambda *_: None):
                    job=runtime.start_job("executive", "Synthetic only", project="default", owner="local")
                with urlopen(base+"/api/agents/jobs/"+job["id"]+"/events/stream",timeout=5) as response:
                    frames=[json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith("data: ")]
                self.assertTrue(frames)
                self.assertTrue(all(isinstance(frame,dict) for frame in frames))
                self.assertEqual(frames[0]["event"], "stream_started")
                ended = next(frame for frame in frames if frame.get("event") == "stream_ended")
                self.assertFalse(ended["has_more"])
                workflow=store.create_workflow(name="HTTP snapshot", steps=[], project="default")
                run=store.save_workflow_run(workflow_id=workflow["id"], project="default", status="completed", result={})
                with urlopen(base+"/api/workflows/runs/"+run["id"]+"/events/stream?project=default",timeout=5) as response:
                    workflow_frames=[json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith("data: ")]
                self.assertTrue(all(isinstance(frame,dict) for frame in workflow_frames))
                self.assertFalse(next(frame for frame in workflow_frames if frame.get("event") == "stream_ended")["has_more"])

            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)

    def test_workflow_snapshot_honors_cursor_and_project(self):
        with TemporaryDirectory() as tmp:
            store=OpenAgentStore(tmp)
            # Create workflow first to satisfy foreign key
            workflow = store.create_workflow(name="synthetic", steps=[], project="p")
            run=store.save_workflow_run(workflow_id=workflow["id"],project="p",status="completed",result={"output":"Synthetic"})
            first=store.get_workflow_run_events(run["id"],project="p")
            self.assertEqual(len(first["events"]),1)
            self.assertEqual(store.get_workflow_run_events(run["id"],project="p",after=first["cursor"])["events"],[])
            with self.assertRaises(KeyError):
                store.get_workflow_run_events(run["id"],project="other")

    def test_cookbook_never_stops_an_unverified_or_reused_pid(self):
        from types import SimpleNamespace
        class TimeoutExpired(Exception): pass
        class NoSuchProcess(Exception): pass
        class AccessDenied(Exception): pass
        psutil = SimpleNamespace(Process=Mock(), TimeoutExpired=TimeoutExpired, NoSuchProcess=NoSuchProcess, AccessDenied=AccessDenied)
        from openagent.model_cookbook import ModelCookbook
        with TemporaryDirectory() as tmp:
            cookbook=ModelCookbook(OpenAgentStore(tmp))
            session=cookbook.serve(model_ref="synthetic",runner="python",args=["--version"])
            with cookbook.store.session() as connection:
                connection.execute("UPDATE model_serve_sessions SET mode='local-process',status='running',pid=123,process_created_at=10 WHERE id=?",(session["id"],))
            process=Mock()
            process.create_time.return_value=20.0
            with patch.dict("sys.modules", {"psutil": psutil}), patch.object(psutil,"Process",return_value=process):
                with self.assertRaises(ValueError): cookbook.stop(session["id"])
            process.terminate.assert_not_called()
            self.assertEqual(cookbook.session(session["id"])["status"],"running")
            process.create_time.return_value=10.0
            process.wait.side_effect=[psutil.TimeoutExpired(5),0]
            with patch.dict("sys.modules", {"psutil": psutil}), patch.object(psutil,"Process",return_value=process):
                self.assertEqual(cookbook.stop(session["id"])["status"],"stopped")
            process.kill.assert_called_once()
            self.assertEqual(process.wait.call_count,2)

    def test_scheduler_delivery_requires_and_forwards_preferences(self):
        with TemporaryDirectory() as tmp:
            scheduler=Scheduler(OpenAgentStore(tmp))
            calendar=Mock()
            scheduler.run_calendar_due_notifications(calendar_system=calendar)
            calendar.deliver_due_notifications.assert_not_called()
            preferences=Mock()
            calendar.deliver_due_notifications.return_value={"delivered":[],"notifications_enabled":False}
            result=scheduler.run_calendar_due_notifications(project="p",calendar_system=calendar,integration_hub=preferences)
            calendar.deliver_due_notifications.assert_called_once_with(project="p",integration_hub=preferences)
            self.assertFalse(result["notifications_enabled"])

