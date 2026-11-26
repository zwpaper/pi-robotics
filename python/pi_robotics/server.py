"""Loopback HTTP request handler for the pi-robotics control server.

Kept separate from :mod:`pi_robotics.control` so the routing table and CORS
policy are one small, readable unit. ``make_handler`` closes over the
``ControlServer`` instance (passed as ``outer``) and only calls its public
helpers, so there is no import cycle.
"""

from __future__ import annotations

import http.server
import json
import urllib.parse
from typing import Any


def loopback_origin(origin: str | None) -> str | None:
    """Return ``origin`` only when it is a loopback web origin.

    The control API is loopback-only, and the only browser that reaches it is
    the camera page served from the same loopback port. Reflecting just loopback
    origins (instead of ``*``) keeps a page on another host from reading
    responses if the port is ever exposed through a port-forward.
    """
    if not origin:
        return None
    try:
        parsed = urllib.parse.urlparse(origin)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.hostname in ("127.0.0.1", "localhost", "::1"):
        return origin
    return None


def make_handler(outer: Any) -> type[http.server.BaseHTTPRequestHandler]:
    """Builds the request handler class for a ``ControlServer`` instance."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # noqa: D102
            return

        def _send(self, code: int, payload: Any) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            origin = loopback_origin(self.headers.get("Origin"))
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(200, {"ok": True})

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def do_GET(self) -> None:  # noqa: N802
            path = urllib.parse.urlparse(self.path).path
            if path in ("/status", "/health"):
                self._send(200, outer._state.status())
            elif path == "/state":
                self._send(200, outer._state.snapshot())
            elif path in ("/scenes", "/list-scenes"):
                from pi_robotics import menagerie, scenes

                names = sorted(set(scenes.list_scenes()) | set(menagerie.MODELS))
                self._send(200, {"scenes": names})
            elif path == "/scene":
                self._send(
                    200,
                    {
                        "scene": outer._state.scene_name,
                        "source": outer._state.status().get("scene_source"),
                        "xml": outer._state.scene_xml(),
                    },
                )
            elif path == "/cameras":
                self._send(200, {"cameras": outer.list_cameras()})
            elif path == "/workspace":
                self._send(200, outer.workspace_status())
            elif path == "/camera":
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                page = outer.camera_page(query)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif path == "/stream.mjpg":
                outer.stream_mjpeg(self)
            else:
                self._send(404, {"error": f"unknown endpoint {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = urllib.parse.urlparse(self.path).path
            body = self._body()
            try:
                result = outer._handle_post(path, body)
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"ok": False, "error": str(exc)})
                return
            self._send(200, result)

    return Handler
