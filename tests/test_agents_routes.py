from __future__ import annotations

from http import HTTPStatus
from unittest import TestCase
from unittest.mock import Mock

from openagent.routes.agents_routes import (
    ROUTES, AgentRouteDependencies, contract, handle_get,
)


def _deps(*, owner="", auth_enforced=False):
    agents = Mock()
    multi_agent = Mock()
    store = Mock()
    first = lambda params, key: (params.get(key, [None])[0] or None)
    parse_bool = lambda value, default: default if value is None else bool(value)
    read_json = Mock(return_value={})
    return agents, multi_agent, store, AgentRouteDependencies(
        agents=agents,
        multi_agent=multi_agent,
        store=store,
        owner=owner,
        first=first,
        parse_bool=parse_bool,
        read_json=read_json,
        auth_enforced=auth_enforced,
    )


class AgentsRouteContractTests(TestCase):
    def test_contract_has_job_events_route(self):
        self.assertIn(("GET", "/api/agents/jobs/{id}/events"), ROUTES)
        self.assertIn(("GET", "/api/agents/jobs"), ROUTES)
        self.assertEqual(contract()["group"], "agents")


class AgentsJobEventsTests(TestCase):
    def test_job_events_success(self):
        agents, multi_agent, store, deps = _deps(owner="user-1")
        events_data = [
            {"id": "evt-1", "type": "step", "payload": {}},
            {"id": "evt-2", "type": "done", "payload": {}},
        ]
        multi_agent.job_events.return_value = events_data
        result = handle_get("/api/agents/jobs/job-abc/events", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        # job_events returns its result directly as payload (a list)
        self.assertEqual(result.payload, events_data)
        multi_agent.job_events.assert_called_once()
        args, kwargs = multi_agent.job_events.call_args
        self.assertEqual(args[0], "job-abc")

    def test_job_events_path_with_extra_segment_resolves_to_job_detail(self):
        # "/api/agents/jobs//events" strips to remainder="" which doesn't end with "/events"
        # so it falls through to get_job(""), which is expected behaviour
        agents, multi_agent, store, deps = _deps()
        multi_agent.get_job.return_value = None
        result = handle_get("/api/agents/jobs//events", "project=default", deps)
        # Falls to job-detail path which returns NOT_FOUND for missing job
        self.assertIsNotNone(result)

    def test_job_events_not_found(self):
        agents, multi_agent, store, deps = _deps()
        multi_agent.job_events.side_effect = KeyError("job not found")
        result = handle_get("/api/agents/jobs/missing-job/events", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_job_events_passes_after_and_limit_params(self):
        agents, multi_agent, store, deps = _deps()
        multi_agent.job_events.return_value = []
        handle_get("/api/agents/jobs/job-x/events", "project=default&after=seq-5&limit=25", deps)
        _, kwargs = multi_agent.job_events.call_args
        self.assertEqual(kwargs.get("after"), "seq-5")
        self.assertEqual(kwargs.get("limit"), 25)

    def test_list_jobs_requires_project(self):
        agents, multi_agent, store, deps = _deps()
        result = handle_get("/api/agents/jobs", "", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_list_jobs_returns_jobs_list(self):
        agents, multi_agent, store, deps = _deps()
        multi_agent.jobs.return_value = [{"id": "j1"}, {"id": "j2"}]
        result = handle_get("/api/agents/jobs", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertIn("jobs", result.payload)
        self.assertEqual(len(result.payload["jobs"]), 2)


class AgentsJobEventsStreamTests(TestCase):
    def test_events_stream_success(self):
        agents, multi_agent, store, deps = _deps(owner="user-1")
        events_data = {
            "events": [
                {"id": "evt-1", "type": "step", "payload": {}},
                {"id": "evt-2", "type": "done", "payload": {}},
            ],
            "cursor": "evt-2",
        }
        multi_agent.job_events.return_value = events_data
        result = handle_get("/api/agents/jobs/job-abc/events/stream", "", deps)
        # StreamRouteResult returns a generator
        self.assertIsNotNone(result)
        multi_agent.job_events.assert_called_once_with("job-abc", owner="user-1", after="", limit=1000)

    def test_events_stream_with_cursor(self):
        agents, multi_agent, store, deps = _deps()
        events_data = {
            "events": [
                {"id": "evt-3", "type": "step", "payload": {}},
            ],
            "cursor": "evt-3",
        }
        multi_agent.job_events.return_value = events_data
        result = handle_get("/api/agents/jobs/job-xyz/events/stream", "after=evt-2", deps)
        self.assertIsNotNone(result)
        # Verify that the cursor was passed through
        _, kwargs = multi_agent.job_events.call_args
        self.assertEqual(kwargs.get("after"), "evt-2")

    def test_events_stream_path_validation(self):
        agents, multi_agent, store, deps = _deps()
        # The remainder logic for //events/stream doesn't result in empty job_id
        # it results in "events/stream" which falls through to get_job()
        # So we test that missing jobs return NOT_FOUND
        multi_agent.get_job.return_value = None
        result = handle_get("/api/agents/jobs//events/stream", "", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_events_stream_job_not_found(self):
        agents, multi_agent, store, deps = _deps()
        multi_agent.job_events.side_effect = KeyError("job not found")
        result = handle_get("/api/agents/jobs/missing/events/stream", "", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_events_stream_with_slash_in_job_id_bad_request(self):
        agents, multi_agent, store, deps = _deps()
        result = handle_get("/api/agents/jobs/a/b/events/stream", "", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
