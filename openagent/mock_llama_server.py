"""Mock llama-server for testing Ultra Model readiness without external dependencies."""

from __future__ import annotations

import json
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

logger = logging.getLogger(__name__)


class MockLlamaServerHandler(BaseHTTPRequestHandler):
    """HTTP handler for mock llama-server."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "ok",
                "uptime": time.time() - getattr(self.server, "start_time", time.time()),
                "model_loaded": True,
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        if self.path == "/v1/chat/completions":
            # Mock chat completion
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            try:
                request_data = json.loads(body) if body else {}
                messages = request_data.get("messages", [])
                model = request_data.get("model", "mock-model")

                response = {
                    "id": "chatcmpl-mock-123",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Mock response from llama-server. This is a test response.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                }
                self.wfile.write(json.dumps(response).encode())
            except Exception as exc:
                self.send_response(500)
                self.wfile.write(json.dumps({"error": str(exc)}).encode())

        elif self.path == "/v1/models":
            # List available models
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            response = {
                "object": "list",
                "data": [
                    {
                        "id": "mock-model",
                        "object": "model",
                        "owned_by": "mock-llama-server",
                        "permission": [],
                    }
                ],
            }
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass


def run_mock_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the mock llama-server."""
    server = HTTPServer((host, port), MockLlamaServerHandler)
    server.start_time = time.time()

    logger.info(f"Mock llama-server running at http://{host}:{port}")
    print(f"Mock llama-server running at http://{host}:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Mock llama-server stopping")
        server.shutdown()


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    logging.basicConfig(level=logging.INFO)
    run_mock_server(host, port)
