"""Local JSONL serving: one command per stdin line, responses/events on stdout.

Command ``id`` values correlate responses; ``request_id`` values correlate
generation events. Sessions persist for the lifetime of this process, not disk.
No network listener is created. Decoding belongs exclusively to OlmoeService.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import threading
from typing import TextIO

from .failure_reporting import add_failure_note, failure_summary
from .olmoe_service import ChatMessage, OlmoeChatDecoder, OlmoeService


class JsonlServer:
    def __init__(self, service: OlmoeService, output: TextIO, *, max_requests: int = 64,
                 max_line_chars: int = 1024 * 1024) -> None:
        if max_requests <= 0 or max_line_chars <= 0:
            raise ValueError("limits must be positive")
        self.service = service
        self.output = output
        self.max_requests = max_requests
        self.max_line_chars = max_line_chars
        self._output_lock = threading.Lock()
        self._readers: dict[str, threading.Thread] = {}

    def emit(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=True, allow_nan=False)
        with self._output_lock:
            self.output.write(line + "\n")
            self.output.flush()

    def _events(self, handle) -> None:
        try:
            for event in handle.events():
                self.emit({"type": "event", **asdict(event)})
        except BaseException as primary:
            # A failed JSONL consumer (for example, a closed stdout pipe)
            # must stop generation instead of leaving a request producing
            # events that can no longer be delivered.
            try:
                handle.cancel()
            except BaseException as cleanup:
                if hasattr(primary, "add_note"):
                    add_failure_note(primary, "request cancellation after event output failure failed: " + failure_summary(cleanup))
            raise

    def _reap(self) -> None:
        for rid, reader in list(self._readers.items()):
            if not reader.is_alive():
                reader.join()
                del self._readers[rid]

    def dispatch(self, command: dict) -> bool:
        """Handle a command; return False after shutdown acknowledgement."""
        self._reap()
        action = command.get("action")
        response = {"type": "response", "id": command.get("id"), "action": action}
        try:
            if action == "submit":
                if len(self._readers) >= self.max_requests:
                    raise ValueError("request capacity reached")
                options = {key: command[key] for key in
                           ("session_id", "request_id", "priority", "timeout", "max_tokens")
                           if key in command}
                for key in ("session_id", "request_id"):
                    if key in options:
                        _text(options[key], key)
                for key in ("priority", "max_tokens"):
                    if key in options and type(options[key]) is not int:
                        raise ValueError(f"{key} must be an integer")
                if "timeout" in options:
                    value = options["timeout"]
                    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                        raise ValueError("timeout must be finite and positive")
                if options.get("request_id") in self._readers:
                    raise ValueError("duplicate request id")
                handle = self.service.submit(command.get("prompt"), **options)
                reader = threading.Thread(target=self._events, args=(handle,),
                                          name=f"olmoe-events-{handle.request_id}")
                self._readers[handle.request_id] = reader
                try:
                    # Acknowledge before starting the event reader so the
                    # response is ordered ahead of request events. If the
                    # output fails (or the reader cannot start), the already
                    # submitted request must not remain generating without a
                    # consumer.
                    self.emit({**response, "ok": True, "request_id": handle.request_id,
                               "session_id": handle.session_id})
                    reader.start()
                except BaseException as primary:
                    self._readers.pop(handle.request_id, None)
                    try:
                        handle.cancel()
                    except BaseException as cleanup:
                        if hasattr(primary, "add_note"):
                            add_failure_note(primary, "request cancellation after submit acknowledgement failure failed: " + failure_summary(cleanup))
                    raise
                return True
            if action == "cancel":
                result = self.service.cancel(_text(command.get("request_id"), "request_id"))
            elif action == "session":
                if command.get("create", False):
                    messages = command.get("messages", [])
                    if not isinstance(messages, list):
                        raise ValueError("messages must be an array")
                    sid = command.get("session_id")
                    if sid is not None:
                        _text(sid, "session_id")
                    result = asdict(self.service.create_session(
                        sid, [ChatMessage(**message) for message in messages]))
                else:
                    result = asdict(self.service.session(_text(command.get("session_id"), "session_id")))
            elif action == "clear":
                result = self.service.clear_session(_text(command.get("session_id"), "session_id"))
            elif action == "status":
                result = {**self.service.telemetry(), "event_readers": len(self._readers),
                          "request_capacity": self.max_requests}
            elif action == "shutdown":
                self.service.shutdown(wait=False, cancel_pending=True)
                result = "shutting_down"
            else:
                raise ValueError("unknown action")
            self.emit({**response, "ok": True, "result": result})
            return action != "shutdown"
        except (ValueError, TypeError, RuntimeError, KeyError) as exc:
            self.emit({**response, "ok": False, "error": {
                "code": getattr(exc, "code", "invalid_request"), "message": str(exc)}})
            return True

    def run(self, source: TextIO) -> None:
        try:
            while True:
                line = source.readline(self.max_line_chars + 1)
                if not line:
                    break
                try:
                    if len(line) > self.max_line_chars:
                        while line and not line.endswith("\n"):
                            line = source.readline(self.max_line_chars + 1)
                        raise ValueError("line exceeds character limit")
                    command = json.loads(line)
                    if not isinstance(command, dict):
                        raise ValueError("command must be an object")
                except ValueError as exc:
                    self.emit({"type": "response", "id": None, "ok": False,
                               "error": {"code": "invalid_json", "message": str(exc)}})
                    continue
                if not self.dispatch(command):
                    break
        finally:
            self.service.shutdown(wait=True, cancel_pending=True)
            for reader in self._readers.values():
                if reader.ident is not None:
                    reader.join()
            self._readers.clear()


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain text")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-gb", type=float, default=4.0)
    parser.add_argument("--io-mode", choices=("buffered", "mmap", "direct", "uring"), default="mmap")
    parser.add_argument("--kernel", default=None, help="registered expert kernel backend")
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--max-line-chars", type=int, default=1024 * 1024)
    parser.add_argument("--continuous", action="store_true", help="batch independent sessions at token boundaries")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--active-kv-mb", type=int, default=2048)
    parser.add_argument("--prompt-chunk-tokens", type=int, default=32)
    parser.add_argument("--blas-threads", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.cache_gb) or args.cache_gb <= 0:
        parser.error("--cache-gb must be finite and positive")
    if args.max_requests <= 0 or args.max_line_chars <= 0:
        parser.error("limits must be positive")
    if min(args.batch_size, args.active_kv_mb, args.prompt_chunk_tokens) <= 0:
        parser.error("batch and memory limits must be positive")
    # Heavy runtime imports and checkpoint loading occur only after CLI parsing.
    from .expert_runtime import SparseExpertManifest
    from .olmoe import OlmoeModel

    try:
        with ExitStack() as resources:
            from .cpu_threads import blas_thread_scope

            resources.enter_context(blas_thread_scope(args.blas_threads))
            kernels, client = None, None
            if args.kernel is not None:
                from .olmoe_cli import _kernel_resources

                kernels, client = _kernel_resources(args.model, args.kernel, args.cache_gb)
                if client is not None:
                    resources.callback(client.close)
                if kernels is not None:
                    resources.callback(kernels.close)
            model = resources.enter_context(OlmoeModel(args.model, SparseExpertManifest.read(args.manifest),
                        expert_cache_bytes=int(args.cache_gb * 1024**3),
                        io_mode=args.io_mode, kernels=kernels, preferred_kernel=args.kernel))
            if args.continuous:
                from .olmoe_continuous import ContinuousOlmoeService

                service = ContinuousOlmoeService(
                    OlmoeChatDecoder(model), queue_capacity=args.max_requests,
                    microbatch_size=args.batch_size,
                    active_kv_capacity_bytes=args.active_kv_mb * 1024**2,
                    prompt_chunk_tokens=args.prompt_chunk_tokens)
            else:
                service = OlmoeService(OlmoeChatDecoder(model), queue_capacity=args.max_requests)
            JsonlServer(service, sys.stdout, max_requests=args.max_requests,
                        max_line_chars=args.max_line_chars).run(sys.stdin)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
