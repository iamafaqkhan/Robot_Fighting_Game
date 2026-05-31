"""
Vercel serverless function — POST /api/predict

Accepts live game state JSON, returns { "action": "<name>" }.
Uses module-level model caching in rl.inference for fast warm requests.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

# Project root on sys.path (Vercel runs from repo root)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rl.inference import predict_action  # noqa: E402


class handler(BaseHTTPRequestHandler):
    """Vercel Python entrypoint (class name must be `handler`)."""

    def do_OPTIONS(self) -> None:
        self._respond(204, b"")

    def do_GET(self) -> None:
        """Lightweight health check for cold-start probes."""
        try:
            from rl.inference import MODEL_PATH, get_agent

            get_agent()
            body = json.dumps(
                {"status": "ok", "model": MODEL_PATH.name, "loaded": True}
            ).encode("utf-8")
            self._respond(200, body)
        except Exception as exc:  # noqa: BLE001
            body = json.dumps({"status": "error", "detail": str(exc)}).encode("utf-8")
            self._respond(503, body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            action = predict_action(payload)
            self._respond(200, json.dumps({"action": action}).encode("utf-8"))
        except json.JSONDecodeError:
            self._respond(400, json.dumps({"action": "step_left", "error": "invalid_json"}).encode())
        except Exception as exc:  # noqa: BLE001
            self._respond(
                500,
                json.dumps({"action": "step_left", "error": str(exc)}).encode(),
            )

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return  # keep serverless logs minimal
