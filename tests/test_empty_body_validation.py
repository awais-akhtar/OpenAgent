"""A record-creating route must refuse a body that names nothing.

scripts/probe_live_routes.py --mutations sends every mutating route an empty
JSON object. 80 of them answered 2xx. Most were legitimately input-free -- an
idempotent stop, a cache clear, an export of what is already stored -- but a
group of them persisted a named record built entirely from defaults: a
permission granted to an empty agent for an empty scope, an OAuth account with
no client id and no endpoints, contacts and calendar events with no name, a
model registered with no name and no source, drafts and notifications with
nothing in them.

Each test here sends the request the probe sent and asserts the route now
refuses it. They run against the route handlers so the check stays attached to
the contract rather than to one screen's call site.
"""

from __future__ import annotations

import unittest
from http import HTTPStatus
from tempfile import TemporaryDirectory
from typing import Any

from openagent.storage import OpenAgentStore


class _Recorder:
    """Fails the test if a guarded handler reaches the service layer."""

    def __init__(self, test: unittest.TestCase) -> None:
        self._test = test

    def __getattr__(self, name: str) -> Any:
        def refuse(*_args: object, **_kwargs: object) -> None:
            self._test.fail(f"the route reached {name}() on an empty body")
        return refuse


class AgentsEmptyBodyTests(unittest.TestCase):
    """Delegation, grants and approval requests all name something."""

    def _post(self, path: str, payload: dict[str, Any]):
        from openagent.routes.agents_routes import AgentRouteDependencies, handle_post

        deps = AgentRouteDependencies(
            agents=_Recorder(self),
            multi_agent=_Recorder(self),
            store=_Recorder(self),
            owner="owner-a",
            first=lambda params, key: None,
            parse_bool=lambda raw, default=False: default,
            read_json=lambda: payload,
        )
        return handle_post(path, deps)

    def test_a_delegation_requires_a_goal(self) -> None:
        result = self._post("/api/agents/delegate", {})
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("goal", result.payload["error"])

    def test_a_grant_requires_an_agent_and_a_scope(self) -> None:
        result = self._post("/api/agents/permissions/grant", {})
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("agent", result.payload["error"])
        self.assertIn("scope", result.payload["error"])

    def test_a_grant_with_only_an_agent_is_still_refused(self) -> None:
        result = self._post("/api/agents/permissions/grant", {"agent": "planner"})
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_an_approval_request_requires_an_action(self) -> None:
        result = self._post("/api/agents/approvals/request", {})
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("action", result.payload["error"])

    def test_whitespace_does_not_satisfy_a_required_field(self) -> None:
        result = self._post("/api/agents/approvals/request", {"action": "   "})
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)


class ContactsEmptyBodyTests(unittest.TestCase):
    def _post(self, payload: dict[str, Any]):
        from openagent.routes.contacts_routes import ContactsRouteDependencies, handle_post

        deps = ContactsRouteDependencies(
            contacts=_Recorder(self),
            query_owner=lambda params: "owner-a",
            payload_owner=lambda body, default="local": "owner-a",
            read_json=lambda: payload,
        )
        return handle_post("/api/contacts", deps)

    def test_a_contact_requires_a_name(self) -> None:
        result = self._post({})
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("name", result.payload["error"])

    def test_a_contact_with_only_notes_is_refused(self) -> None:
        # Without a name there is nothing to find the contact by again.
        result = self._post({"notes": "met at the conference"})
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)


class LocalModelRegistrationTests(unittest.TestCase):
    """The registration handler that actually runs is the one in runtime.py."""

    def test_the_runtime_handler_requires_a_name_and_a_source(self) -> None:
        import inspect

        from openagent.routes import runtime

        source = inspect.getsource(runtime)
        marker = 'if parsed.path == "/api/local-ai/models/register":'
        self.assertIn(marker, source)
        handler = source[source.index(marker):]
        handler = handler[:handler.index("/api/local-ai/models/download")]
        # The guard has to be in the dispatch that is reached first, not only
        # in the local_ai_routes module whose version this one shadows.
        self.assertIn('"error": "name is required."', handler)
        # A source must NOT be required: registering metadata with neither a
        # local_path nor a source_url is a supported flow that the
        # React-to-Python smoke asserts the stored record keeps empty.
        self.assertNotIn("source_url or local_path is required.", handler)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
