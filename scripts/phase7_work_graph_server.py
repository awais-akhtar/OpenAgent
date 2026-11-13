"""Serve the desktop React build against a supplied local OpenAgent data directory."""

from __future__ import annotations

import argparse
from ipaddress import ip_address
from pathlib import Path
from http.server import ThreadingHTTPServer

from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--ui-root", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=0, type=int)
    args = parser.parse_args()
    try:
        if not ip_address(args.host).is_loopback:
            parser.error("Phase 7 integration server must bind to a loopback address")
    except ValueError:
        parser.error("--host must be a loopback IP address")
    data_dir = args.data_dir.expanduser().resolve()
    ui_root = args.ui_root.expanduser().resolve()
    if not (data_dir / "openagent.sqlite3").is_file():
        parser.error(f"OpenAgent database not found: {data_dir / 'openagent.sqlite3'}")
    if not (ui_root / "index.html").is_file():
        parser.error(f"built React UI not found: {ui_root / 'index.html'}")

    server: ThreadingHTTPServer = build_http_server(
        args.host,
        args.port,
        OpenAgentStore(data_dir),
        web_root=ui_root,
    )
    print(f"PHASE7_ORIGIN=http://{args.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
