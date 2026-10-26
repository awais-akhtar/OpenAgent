from __future__ import annotations

from http import HTTPStatus
from unittest import TestCase
from unittest.mock import Mock

from openagent.routes.communication_routes import (
    ROUTES, CommunicationRouteDependencies, contract, handle_delete, handle_get, handle_post,
)


def _deps(payload=None, *, owner=""):
    comm = Mock()
    read_json = Mock(return_value=payload if payload is not None else {})
    first = lambda params, key: (params.get(key, [None])[0] or None)
    query_owner = lambda params: owner
    payload_owner = lambda p, default: owner or default
    parse_bool = lambda value, default: default if value is None else bool(value)
    parse_int = lambda value, default: default if value is None else int(value)
    return comm, CommunicationRouteDependencies(
        communication=comm,
        read_json=read_json,
        first=first,
        parse_bool=parse_bool,
        parse_int=parse_int,
        query_owner=query_owner,
        payload_owner=payload_owner,
    )


class CommunicationRouteContractTests(TestCase):
    def test_contract_has_delete_message_route(self):
        self.assertIn(("DELETE", "/api/communication/messages/{message_id}"), ROUTES)
        self.assertEqual(contract()["group"], "communication")

    def test_contract_includes_key_post_routes(self):
        methods = [m for m, _ in ROUTES]
        self.assertIn("POST", methods)
        self.assertIn("GET", methods)
        self.assertIn("DELETE", methods)


class CommunicationDeleteMessageTests(TestCase):
    def test_delete_message_success(self):
        comm, deps = _deps()
        comm.delete_message.return_value = {"deleted": True, "id": "msg-1"}
        result = handle_delete("/api/communication/messages/msg-1", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.OK)
        self.assertTrue(result.payload["deleted"])
        comm.delete_message.assert_called_once()
        args, kwargs = comm.delete_message.call_args
        self.assertEqual(args[0], "msg-1")

    def test_delete_message_not_found(self):
        from openagent.communication import CommunicationMessageNotFoundError
        comm, deps = _deps()
        comm.delete_message.side_effect = CommunicationMessageNotFoundError("not found")
        result = handle_delete("/api/communication/messages/missing-id", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_delete_message_empty_id_returns_bad_request(self):
        comm, deps = _deps()
        result = handle_delete("/api/communication/messages/", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_delete_message_unknown_path_returns_none(self):
        comm, deps = _deps()
        result = handle_delete("/api/communication/other/path", "project=default", deps)
        self.assertIsNone(result)

    def test_delete_message_with_slash_in_id_returns_bad_request(self):
        comm, deps = _deps()
        result = handle_delete("/api/communication/messages/a/b", "project=default", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_delete_message_passes_owner_from_query_owner(self):
        comm, deps = _deps(owner="user-x")
        comm.delete_message.return_value = {"deleted": True, "id": "msg-5"}
        handle_delete("/api/communication/messages/msg-5", "project=default", deps)
        _, kwargs = comm.delete_message.call_args
        self.assertEqual(kwargs.get("owner"), "user-x")

    def test_delete_message_passes_project_from_query(self):
        comm, deps = _deps()
        comm.delete_message.return_value = {"deleted": True, "id": "msg-7"}
        handle_delete("/api/communication/messages/msg-7", "project=myproj", deps)
        _, kwargs = comm.delete_message.call_args
        self.assertEqual(kwargs.get("project"), "myproj")


class CommunicationSendMessageTests(TestCase):
    def test_send_message_success(self):
        payload = {
            "project": "default",
            "recipients": ["alice@example.com"],
            "subject": "Test Subject",
            "body": "Test body content"
        }
        comm, deps = _deps(payload=payload, owner="user-1")
        comm.send_message.return_value = {
            "id": "sent-1",
            "message_id": "msg-1",
            "status": "sent",
            "sent_at": "2026-10-25T10:00:00",
        }
        result = handle_post("/api/communication/messages/msg-1/send", deps)
        self.assertEqual(result.status, HTTPStatus.CREATED)
        self.assertEqual(result.payload["message"]["status"], "sent")
        comm.send_message.assert_called_once()
        args, kwargs = comm.send_message.call_args
        self.assertEqual(args[0], "msg-1")
        self.assertEqual(kwargs.get("project"), "default")
        self.assertEqual(kwargs.get("recipients"), ["alice@example.com"])

    def test_send_message_missing_recipients(self):
        payload = {
            "project": "default",
            "subject": "Test Subject",
            "body": "Test body content"
        }
        comm, deps = _deps(payload=payload)
        comm.send_message.side_effect = ValueError("recipients are required.")
        result = handle_post("/api/communication/messages/msg-1/send", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("error", result.payload)

    def test_send_message_missing_subject(self):
        payload = {
            "project": "default",
            "recipients": ["alice@example.com"],
            "body": "Test body content"
        }
        comm, deps = _deps(payload=payload)
        comm.send_message.side_effect = ValueError("subject is required.")
        result = handle_post("/api/communication/messages/msg-1/send", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_send_message_missing_body(self):
        payload = {
            "project": "default",
            "recipients": ["alice@example.com"],
            "subject": "Test Subject"
        }
        comm, deps = _deps(payload=payload)
        comm.send_message.side_effect = ValueError("body is required.")
        result = handle_post("/api/communication/messages/msg-1/send", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_send_message_not_found(self):
        from openagent.communication import CommunicationMessageNotFoundError
        payload = {
            "project": "default",
            "recipients": ["alice@example.com"],
            "subject": "Test Subject",
            "body": "Test body content"
        }
        comm, deps = _deps(payload=payload)
        comm.send_message.side_effect = CommunicationMessageNotFoundError("Message not found.")
        result = handle_post("/api/communication/messages/missing-id/send", deps)
        self.assertEqual(result.status, HTTPStatus.NOT_FOUND)

    def test_send_message_empty_id_returns_bad_request(self):
        payload = {
            "project": "default",
            "recipients": ["alice@example.com"],
            "subject": "Test Subject",
            "body": "Test body content"
        }
        comm, deps = _deps(payload=payload)
        result = handle_post("/api/communication/messages//send", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_send_message_with_slash_in_id_returns_bad_request(self):
        payload = {
            "project": "default",
            "recipients": ["alice@example.com"],
            "subject": "Test Subject",
            "body": "Test body content"
        }
        comm, deps = _deps(payload=payload)
        result = handle_post("/api/communication/messages/a/b/send", deps)
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)

    def test_send_message_with_owner(self):
        payload = {
            "project": "default",
            "recipients": ["alice@example.com"],
            "subject": "Test Subject",
            "body": "Test body content"
        }
        comm, deps = _deps(payload=payload, owner="user-2")
        comm.send_message.return_value = {
            "id": "sent-1",
            "message_id": "msg-1",
            "status": "sent",
        }
        handle_post("/api/communication/messages/msg-1/send", deps)
        _, kwargs = comm.send_message.call_args
        self.assertEqual(kwargs.get("owner"), "user-2")
