"""Serve the built React UI and the real OpenAgent Python backend for GAP-011 chat checks.

Nothing is mocked: chat requests run through openagent.routes.runtime against a temporary database.
Prints CHAT_ORIGIN=http://127.0.0.1:<port> once ready.
"""

from __future__ import annotations

import argparse
import tempfile
import threading
from pathlib import Path

from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui-root", required=True, type=Path)
    parser.add_argument("--port", default=0, type=int)
    args = parser.parse_args()
    data_dir = Path(tempfile.mkdtemp(prefix="gap011-chat-"))
    server = build_http_server("127.0.0.1", args.port, OpenAgentStore(data_dir), web_root=args.ui_root.resolve())
    print(f"CHAT_ORIGIN=http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
