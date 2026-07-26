#!/usr/bin/env python3
"""Range-capable static file server for the Phase 0 spike.

`python -m http.server` ignores `Range:` headers and answers `200` with the whole file,
which would make a 2 GB COPC download in full before anything renders — it cannot be used
to evaluate COPC streaming. This server implements what plan decision D5 requires of the
real asset endpoint, and nothing more:

* `Range: bytes=a-b` -> `206 Partial Content` with `Content-Range`
* `Accept-Ranges: bytes` advertised on `HEAD`/`GET`
* CORS for the Vite dev origin, exposing the headers copc.js reads
* per-request logging of the served byte ranges to a JSONL file, so "bytes ranged" and
  "request count" are measured at the server rather than trusted from the client

Usage:
    python range_server.py --root /path/to/artifacts --port 8123 --log /tmp/ranges.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socketserver
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
CHUNK = 1 << 16


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus single-range support and a request log."""

    server_version = "Phase0RangeServer/1.0"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, log_path: Path | None = None, **kwargs):
        self._log_path = log_path
        super().__init__(*args, **kwargs)

    # --- logging ---------------------------------------------------------------

    _log_lock = threading.Lock()

    def _record(self, **fields) -> None:
        if not self._log_path:
            return
        fields["t"] = time.time()
        with self._log_lock:
            with self._log_path.open("a") as fh:
                fh.write(json.dumps(fields) + "\n")

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        if self.command != "GET" or "range" not in fmt.lower():
            return

    # --- CORS ------------------------------------------------------------------

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Expose-Headers",
            "Accept-Ranges, Content-Range, Content-Length, Content-Encoding",
        )
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib naming
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # --- GET with ranges -------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        range_header = self.headers.get("Range")
        if not range_header:
            self._record(event="full", path=self.path)
            super().do_GET()
            return

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return

        match = RANGE_RE.match(range_header.strip())
        if not match:
            self.send_error(416, "Unsupported range")
            return

        size = os.path.getsize(path)
        start_s, end_s = match.groups()
        if start_s == "":  # suffix range: bytes=-N
            length = int(end_s or 0)
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        length = end - start + 1
        self._record(
            event="range", path=self.path, start=start, end=end, length=length, size=size
        )

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()

        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                buf = fh.read(min(CHUNK, remaining))
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):
                    # The client aborted - expected when the viewer cancels offscreen
                    # requests. Record it: cancellation is a plan requirement, so it
                    # should be observable here.
                    self._record(event="aborted", path=self.path, start=start, end=end)
                    return
                remaining -= len(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="directory to serve")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--log", type=Path, help="JSONL request log path")
    args = ap.parse_args()

    if args.log and args.log.exists():
        args.log.unlink()

    handler = partial(
        RangeRequestHandler, directory=str(args.root.resolve()), log_path=args.log
    )

    socketserver.TCPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving {args.root.resolve()} at http://127.0.0.1:{args.port} (ranges enabled)")
        if args.log:
            print(f"request log: {args.log}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
