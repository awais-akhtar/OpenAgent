"""Call every registered read route against a real server and record what happens.

The route inventory is generated from declarations: a path appears there because
a module lists it, not because anything behind it works. The React-to-Python
smoke proves 51 workflows end to end, which is the strongest evidence this
project has, but it covers a small share of the registry. Between those two
sits a question nothing answered: of the routes this product declares, how many
actually return a usable answer on a clean install?

This probe answers that for every GET in the registry. It boots the real HTTP
server over a temporary store, mints one credential holding every scope in the
catalog, calls each route, and classifies the response. It is read-only: POST,
PUT, PATCH and DELETE are listed as not probed rather than invoked, because
exercising a mutation needs a workflow, which is what the smoke is for.

A clean install has no records, so an empty list is the correct answer for most
routes and is reported as such. What this is looking for is the rest: a 500, a
route that answers 404 because nothing dispatches it, an unhandled parameter
reported as a client error, or a handler that raises.

    python scripts/probe_live_routes.py [--json PATH]

The exit status is 1 when any route failed outright, so this can gate a run.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


# Values substituted for a path parameter. A probe cannot know a real id, so it
# sends one that certainly does not exist: the answer being looked for is a
# clean "not found", not a crash.
ABSENT_ID = "probe-absent-identifier"

# Query parameters a handler needs before it can do anything. Without these it
# answers 400, which is correct behaviour and not worth reporting as a finding.
REQUIRED_QUERY: dict[str, dict[str, str]] = {
    "/api/g25/graph/query": {"q": "probe"},
    "/api/g25/graph/paths": {"start": "probe-a", "end": "probe-b"},
    "/api/g25/rag/context": {"q": "probe"},
    "/api/g25/memory/history": {"session_id": ABSENT_ID},
    "/api/fact-provenance/search": {"q": "probe"},
    "/api/voice/websocket/status": {"stream_id": ABSENT_ID},
    "/api/semantic/search": {"q": "probe"},
    "/api/knowledge/search": {"q": "probe"},
    "/api/search": {"q": "probe"},
    "/api/messages": {"conversation_id": ABSENT_ID},
    "/api/codebase/files": {"path": "README.md"},
    "/api/conversation/messages": {"conversation_id": ABSENT_ID},
    "/api/chat/export": {"conversation_id": ABSENT_ID},
    "/api/chat/replay": {"conversation_id": ABSENT_ID},
    "/api/workflows/layout": {"workflow_id": ABSENT_ID},
    "/api/voice/turns": {"session_id": ABSENT_ID},
    "/api/enterprise/workspace/members": {"workspace_id": ABSENT_ID},
    "/api/enterprise/org-memory": {"workspace_id": ABSENT_ID},
    "/api/calendar/events/{id}/occurrences": {"from": "2026-01-01T00:00:00Z", "until": "2026-12-31T00:00:00Z"},
    "/api/calendar/free-slots": {"from": "2026-01-01T00:00:00Z", "to": "2026-01-02T00:00:00Z"},
    "/api/document/{doc_id}/version/{num}": {},
}

# Routes where the absent id this probe sends is itself the reason for a 404 or
# a 400. The answer is correct, so it is recorded without being called a
# finding; what would be a finding is a crash or an empty success.
EXPECTS_ABSENT: frozenset[str] = frozenset({
    "/api/g25/memory/history",
    "/api/voice/websocket/status",
    "/api/conversation/messages",
    "/api/chat/export",
    "/api/chat/replay",
    "/api/workflows/layout",
    "/api/voice/turns",
    "/api/enterprise/workspace/members",
    "/api/enterprise/org-memory",
    "/api/messages",
    "/api/document/{doc_id}/version/{num}",
    # An agent-run project exists for an owner only once a run claims it, so a
    # clean install answers 404 for the default project. That is the ownership
    # boundary L11 describes, and OperationsScreens reads the 404 as an empty
    # list rather than an error.
    "/api/agent-runs",
})

# Routes deliberately left alone: they block, stream without end, or act on the
# host rather than reading from the store.
SKIP_PATHS: frozenset[str] = frozenset({
    "/api/codebase/runs/{id}/events",  # Server-sent events; blocks by design.
    "/api/chat/stream",
    "/api/runtime-tools/browser-test",  # Launches a browser on the host.
})



# The server allows 360 general requests per 60 seconds and offers no override,
# which is correct for a security control: the probe paces itself instead. A
# GET-only run fits inside one window and never waits; a run that also sends
# mutations exceeds it and sleeps just enough to stay under. Without this the
# probe recorded its own 429s as route failures.
_REQUEST_WINDOW_SECONDS = 60.0
_REQUEST_BUDGET = 340  # Margin under the server's 360.
_recent_requests: deque[float] = deque()


def pace() -> None:
    """Block until another request fits inside the server's window."""
    now = time.monotonic()
    while _recent_requests and _recent_requests[0] <= now - _REQUEST_WINDOW_SECONDS:
        _recent_requests.popleft()
    if len(_recent_requests) >= _REQUEST_BUDGET:
        wait = _REQUEST_WINDOW_SECONDS - (now - _recent_requests[0]) + 0.05
        if wait > 0:
            time.sleep(wait)
        return pace()
    _recent_requests.append(time.monotonic())


def retry_after_seconds(headers: Any) -> float:
    try:
        return min(65.0, max(1.0, float(headers.get("Retry-After", 1))))
    except (TypeError, ValueError):
        return 1.0


# One keep-alive connection for the whole run. urllib opens a fresh socket per
# request, so probing every route left several hundred sockets in TIME_WAIT; in
# a full pytest run that pressure surfaced elsewhere as
# `ConnectionAbortedError: [WinError 10053]` in an unrelated HTTP test. Reusing
# one connection keeps the probe to a single socket and makes it faster.
_connection: http.client.HTTPConnection | None = None
_connection_origin = ""


def request(base_url: str, method: str, path: str, token: str, body: bytes | None = None) -> tuple[int, str, Any]:
    """Send one request on the shared connection, reconnecting if it drops."""
    global _connection, _connection_origin

    origin = base_url.removeprefix("http://")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"

    for attempt in range(2):
        if _connection is None or _connection_origin != origin:
            close_connection()
            _connection = http.client.HTTPConnection(origin, timeout=30)
            _connection_origin = origin
        try:
            _connection.request(method, path, body=body, headers=headers)
            response = _connection.getresponse()
            raw = response.read().decode("utf-8", "replace")
            return response.status, raw, response.getheaders()
        except (http.client.HTTPException, OSError):
            # A dropped keep-alive is expected now and then; reopen once.
            close_connection()
            if attempt:
                raise
    raise RuntimeError("unreachable")


def close_connection() -> None:
    global _connection, _connection_origin
    if _connection is not None:
        try:
            _connection.close()
        except Exception:  # noqa: BLE001 - closing must never fail a run.
            pass
    _connection = None
    _connection_origin = ""


def classify(status: int, payload: Any, raw: str) -> tuple[str, str]:
    """Return a (verdict, note) pair for one response."""
    if status >= 500:
        return "server_error", f"HTTP {status}: {raw[:200]}"
    if status == 404:
        # A registered route answering 404 for an absent id is correct. A
        # registered route answering 404 for its own bare path is not
        # dispatched by anything.
        return "not_found", f"HTTP 404: {raw[:200]}"
    if status in {401, 403}:
        return "refused", f"HTTP {status}"
    if status >= 400:
        return "client_error", f"HTTP {status}: {raw[:200]}"

    body = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
    if isinstance(body, dict) and body.get("error"):
        return "error_in_body", f"HTTP {status} carrying error: {str(body['error'])[:160]}"
    if body is None or body == {} or body == []:
        return "empty", f"HTTP {status}, no content"
    if isinstance(body, dict):
        lists = [key for key, value in body.items() if isinstance(value, list)]
        if lists and all(not body[key] for key in lists) and len(body) <= len(lists) + 2:
            return "empty_collection", f"HTTP {status}, empty {', '.join(sorted(lists))}"
    return "answered", f"HTTP {status}"


def probe(base_url: str, token: str, method: str, path: str) -> dict[str, Any]:
    """Call one route and classify the answer."""
    concrete = path
    for opening, closing in (("{", "}"),):
        while opening in concrete:
            start = concrete.index(opening)
            end = concrete.index(closing, start)
            concrete = concrete[:start] + ABSENT_ID + concrete[end + 1:]

    query = REQUIRED_QUERY.get(path, {})
    parameters = {"project": "default", **query}
    suffix = "&".join(f"{key}={value}" for key, value in parameters.items())
    url = f"{base_url}{concrete}{'&' if '?' in concrete else '?'}{suffix}"

    record: dict[str, Any] = {"method": method, "path": path, "url": concrete}
    target = url.removeprefix(base_url)
    for attempt in range(3):
        pace()
        try:
            status, raw, headers = request(base_url, method, target, token)
        except Exception as error:  # noqa: BLE001 - the probe reports, never raises.
            record.update(verdict="transport_error", note=f"{type(error).__name__}: {error}", status=0)
            return record
        if status == 429 and attempt < 2:
            # The probe outran the window, not a fault in the route.
            time.sleep(retry_after_seconds(dict(headers)))
            continue
        break

    try:
        payload = json.loads(raw)
    except ValueError:
        payload = None

    verdict, note = classify(status, payload, raw)
    record.update(verdict=verdict, note=note, status=status)
    return record



# ---------------------------------------------------------------------------
# Mutation probing
#
# Every mutating route is sent an empty JSON body. That is both the safest
# request a probe can make and the most informative one: a handler that
# requires any field cannot act on it, so nothing is created, sent or launched,
# and the correct answer is a 4xx naming what is missing. What this looks for
# is the rest -- a handler that crashes on an empty body, one that nothing
# dispatches, and one that answers 2xx, meaning it created something out of
# nothing and validates nothing.
# ---------------------------------------------------------------------------

# Routes this probe will not call even with an empty body, because they can act
# with no input, reach outside the process, or outlive the probe. Each entry is
# a reason, not a convenience: a route is here because calling it could have an
# effect the probe cannot undo, never because it was inconvenient to classify.
MUTATION_DENY: dict[str, str] = {
    # Downloads. The owner's standing instruction is that no model is fetched.
    "/api/local-ai/models/download": "downloads a model",
    "/api/local-ai/ollama/pull": "downloads a model",
    "/api/ultra-models/models/{id}/download": "downloads a model",
    "/api/skills/import-url": "fetches a remote URL",
    "/api/skills/marketplace/{id}/install": "fetches and installs",
    # Command execution on the host.
    "/api/integrations/safe-shell": "executes a shell command",
    "/api/production/sandbox/run": "executes sandboxed code",
    "/api/terminal/runs": "starts a terminal process",
    "/api/skills/{id}/sandbox": "executes a skill",
    "/api/skills/{id}/test": "executes a skill",
    "/api/runtime-tools/browser-test": "launches a browser",
    # Outbound network.
    "/api/developer/webhooks/dispatch": "sends an outbound request",
    "/api/communication/send": "sends a message",
    "/api/communication/send/request-approval": "sends a message",
    "/api/communication/send/resolve": "sends a message",
    "/api/communication/messages/{id}/send": "sends a message",
    "/api/calendar/reminders/deliver": "delivers notifications",
    "/api/security/egress/request": "opens an egress decision",
    "/api/security/egress/resolve": "resolves an egress decision",
    "/api/scraper/scrape": "fetches a remote URL",
    "/api/scraper/run-due": "fetches remote URLs",
    "/api/scraper/schedule": "schedules remote fetches",
    # Work that outlives the probe or loads a model.
    "/api/scheduler/autonomous/start": "starts a background worker",
    "/api/scheduler/autonomous/execute-once": "runs scheduled work",
    "/api/tasks/run-due": "runs due work",
    "/api/production/queue/run-next": "runs queued work",
    "/api/production/proactive/run": "runs proactive work",
    "/api/inference/benchmark": "loads a model and benchmarks",
    "/api/inference/warmup": "loads a model",
    "/api/local-ai/benchmark": "loads a model and benchmarks",
    "/api/rag/benchmark": "loads a model and benchmarks",
    "/api/adaptive-engine/benchmark": "loads a model and benchmarks",
    "/api/adaptive-engine/direct/benchmark": "loads a model and benchmarks",
    # Model-gateway invocations. These are what the React-to-Python smoke
    # exercises against its fixture provider, which is the right instrument
    # for them; here they would call a provider that is not configured.
    "/api/agents/run": "invokes the model gateway",
    "/api/agents/run-checked": "invokes the model gateway",
    "/api/research/run": "invokes the model gateway",
    "/api/science/run": "invokes the model gateway",
    "/api/science/execute": "invokes the model gateway",
    "/api/codebase/runs": "invokes the model gateway",
    "/api/mission/run-check": "invokes the model gateway",
    "/api/semantic/consolidate": "invokes the model gateway",
    "/api/twin/consolidate": "invokes the model gateway",
    "/api/twin/predict": "invokes the model gateway",
    # Changes the probe server's own behaviour part-way through a run.
    "/api/settings/runtime": "changes runtime settings mid-probe",
    # Restores over existing state.
    "/api/backups/restore": "restores over the store",
    # Opens a modal folder chooser on the host desktop. It is bounded now, but
    # a probe should not raise a window on the owner's screen.
    "/api/codebase/folder-picker": "opens a dialog on the host desktop",
    "/api/document/{doc_id}/restore/{num}": "restores a document version",
}


def probe_mutation(base_url: str, token: str, method: str, path: str) -> dict[str, Any]:
    """Send one mutating route an empty JSON body and classify the answer."""
    concrete = path
    while "{" in concrete:
        start = concrete.index("{")
        end = concrete.index("}", start)
        concrete = concrete[:start] + ABSENT_ID + concrete[end + 1:]

    record: dict[str, Any] = {"method": method, "path": path, "url": concrete}
    target = f"{concrete}?project=default"
    for attempt in range(3):
        pace()
        try:
            status, raw, headers = request(base_url, method, target, token, body=b"{}")
        except Exception as error:  # noqa: BLE001 - the probe reports, never raises.
            record.update(verdict="transport_error", note=f"{type(error).__name__}: {error}", status=0)
            return record
        if status == 429 and attempt < 2:
            time.sleep(retry_after_seconds(dict(headers)))
            continue
        break

    if status >= 500:
        verdict = "server_error"
    elif status == 404:
        # An empty body names no record, so "not found" is a refusal of this
        # request rather than evidence that nothing dispatches the route. Which
        # routes are dispatched is answered by the GET probe above and by
        # tests/test_route_contracts.py.
        verdict = "absent_record_refused"
    elif status in {401, 403}:
        # This credential holds every scope, so a refusal is either an
        # authentication endpoint rejecting empty credentials or an explicit
        # approval gate, both of which are the behaviour wanted.
        verdict = "refused_without_input"
    elif status >= 400:
        verdict = "validated"
    else:
        verdict = "accepted_empty_body"
    record.update(verdict=verdict, note=f"HTTP {status}: {raw[:200]}", status=status)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write the full record set here.")
    parser.add_argument("--mutations", action="store_true",
                        help="Also send every mutating route an empty JSON body (see MUTATION_DENY).")
    arguments = parser.parse_args(argv)

    temporary = tempfile.TemporaryDirectory(prefix="openagent-route-probe-")
    root = Path(temporary.name)
    # These are read from the process environment, so leaving them set would
    # change every later caller in the same process. Running this as a test did
    # exactly that: OPENAGENT_AUTH_ENFORCED stayed on and 33 route tests that
    # expect unauthenticated access started answering 401.
    probe_environment = {
        "OPENAGENT_DATA_DIR": str(root / "data"),
        "OPENAGENT_AUTH_ENFORCED": "1",
        "OPENAGENT_BOOTSTRAP_TOKEN": "route-probe-bootstrap-secret-32-characters",
        "OPENAGENT_MODEL_ROUTING": os.environ.get("OPENAGENT_MODEL_ROUTING", "cpu-baseline"),
    }
    restore_environment = {key: os.environ.get(key) for key in probe_environment}
    os.environ.update(probe_environment)
    try:
        return _run(arguments, root, restore_environment)
    finally:
        for key, value in restore_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        temporary.cleanup()


def _run(arguments: argparse.Namespace, root: Path, _restore: dict[str, str | None]) -> int:
    """The probe itself, with the environment already in place."""

    # Each run gets a fresh server with its own limiter, so the pacing window
    # must not carry over from a previous run in the same process.
    _recent_requests.clear()

    from openagent.auth import SCOPE_CATALOG, AuthManager
    from openagent.routes import route_contracts
    from openagent.server import build_http_server
    from openagent.storage import OpenAgentStore

    store = OpenAgentStore(root / "data")
    auth = AuthManager(store)
    owner = auth.register(
        "probe-owner", "probe-owner@example.test", "temporary-password-1",
        bootstrap_token=os.environ["OPENAGENT_BOOTSTRAP_TOKEN"],
    )
    # One credential holding every scope: this probe asks what a route does when
    # it is allowed to run, not whether the scope gate works. Gate coverage is
    # tests/test_auth_enforcement.py and the smoke's denied-scope checks.
    token = auth.create_api_token(
        owner["user_id"], "route-probe",
        scopes=[entry["scope"] for entry in SCOPE_CATALOG],
    )["token"]

    server = build_http_server("127.0.0.1", 0, store)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, name="route-probe-http", daemon=True)
    thread.start()

    records: list[dict[str, Any]] = []
    mutation_records: list[dict[str, Any]] = []
    not_probed: list[dict[str, str]] = []
    try:
        for group in route_contracts():
            for route in group["routes"]:
                method, path = route["method"], route["path"]
                if path in SKIP_PATHS:
                    not_probed.append({"group": group["group"], "method": method, "path": path, "reason": "skipped"})
                    continue
                if method != "GET":
                    if not arguments.mutations:
                        not_probed.append({"group": group["group"], "method": method, "path": path, "reason": "mutating"})
                        continue
                    reason = MUTATION_DENY.get(path)
                    if reason:
                        not_probed.append({"group": group["group"], "method": method, "path": path,
                                           "reason": f"denied: {reason}"})
                        continue
                    record = probe_mutation(base_url, token, method, path)
                    record["group"] = group["group"]
                    mutation_records.append(record)
                    continue
                record = probe(base_url, token, method, path)
                record["group"] = group["group"]
                records.append(record)
    finally:
        close_connection()
        server.shutdown()
        server.server_close()

    # A route told to look for an id that does not exist, and answering that it
    # does not exist, has behaved correctly. Separate those from real findings
    # so the count means something.
    for record in records:
        if record["path"] in EXPECTS_ABSENT and record["verdict"] in {"not_found", "client_error"}:
            record["verdict"] = "absent_id_refused"

    counts = Counter(record["verdict"] for record in records)
    failing = [r for r in records if r["verdict"] in {"server_error", "transport_error", "error_in_body", "client_error"}]
    not_dispatched = [
        r for r in records
        if r["verdict"] == "not_found" and "{" not in r["path"] and r["path"] not in EXPECTS_ABSENT
    ]

    print(f"Probed {len(records)} GET routes across {len({r['group'] for r in records})} groups.")
    print(f"Not probed: {len(not_probed)} ({sum(1 for r in not_probed if r['reason'] == 'mutating')} mutating,"
          f" {sum(1 for r in not_probed if r['reason'] == 'skipped')} skipped).")
    print()
    for verdict, count in counts.most_common():
        print(f"  {verdict:<18} {count}")

    if not_dispatched:
        print(f"\nRegistered but answering 404 on their own path ({len(not_dispatched)}):")
        for record in not_dispatched:
            print(f"  {record['group']:<22} {record['path']}")

    if failing:
        print(f"\nFailed outright ({len(failing)}):")
        for record in sorted(failing, key=lambda r: (r["verdict"], r["path"])):
            print(f"  [{record['verdict']}] {record['group']:<20} {record['path']}")
            print(f"      {record['note']}")

    mutation_findings: list[dict[str, Any]] = []
    if arguments.mutations:
        mutation_counts = Counter(record["verdict"] for record in mutation_records)
        mutation_findings = [
            r for r in mutation_records
            # accepted_empty_body is reported separately: a handler that acts
            # with no input may be a validation gap or may be a legitimately
            # input-free operation, and only reading it tells you which.
            if r["verdict"] in {"server_error", "transport_error"}
        ]
        denied = [r for r in not_probed if r["reason"].startswith("denied")]
        print(f"\nSent an empty JSON body to {len(mutation_records)} mutating routes."
              f" {len(denied)} were not called by policy.")
        print()
        for verdict, count in mutation_counts.most_common():
            print(f"  {verdict:<24} {count}")

        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in mutation_findings:
            grouped.setdefault(record["verdict"], []).append(record)
        for verdict in sorted(grouped):
            entries = grouped[verdict]
            print(f"\n{verdict} ({len(entries)}):")
            for record in sorted(entries, key=lambda r: r["path"]):
                print(f"  {record['method']:<7} {record['group']:<22} {record['path']}")
                print(f"      {record['note'][:200]}")

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps({
            "probed": records,
            "mutations": mutation_records,
            "not_probed": not_probed,
            "counts": dict(counts),
        }, indent=2), encoding="utf-8")
        print(f"\nFull records written to {arguments.json}")

    return 1 if failing or not_dispatched or mutation_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
