from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

try:
    from .evaluate_suite import ALL_PROFILES
    from .fetch_model import fetch
    from .policy_arena import PolicyArena
except ImportError:
    from evaluate_suite import ALL_PROFILES
    from fetch_model import fetch
    from policy_arena import PolicyArena


DEFAULT_MODEL = "universal-search-dagger-2026-08-30"
PAGE = Path(__file__).with_name("policy_arena.html")


def handler_for(arena: PolicyArena) -> type[BaseHTTPRequestHandler]:
    page = PAGE.read_bytes()

    class ArenaHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._respond(HTTPStatus.OK, "text/html; charset=utf-8", page)
                return
            if self.path == "/api/state":
                self._json(HTTPStatus.OK, arena.state())
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                payload = self._payload()
                if self.path == "/api/action":
                    state = arena.act(
                        int(payload["index"]),
                        int(payload["revision"]),
                    )
                elif self.path == "/api/reset":
                    state = arena.reset(int(payload["seed"]))
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                self._json(HTTPStatus.OK, state)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.CONFLICT, {"error": str(error)})

        def _payload(self) -> dict[str, object]:
            length = int(self.headers.get("content-length", "0"))
            if length < 1 or length > 65536:
                raise ValueError("request body must contain at most 65536 bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            self._respond(
                status,
                "application/json; charset=utf-8",
                json.dumps(payload, separators=(",", ":")).encode(),
            )

        def _respond(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("x-content-type-options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, message_format: str, *arguments: object) -> None:
            print(f"{self.address_string()} {message_format % arguments}")

    return ArenaHandler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--profile", choices=ALL_PROFILES, default="classic_generic_2022")
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--action-limit", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    arguments = parser.parse_args()
    checkpoint = arguments.checkpoint or fetch(DEFAULT_MODEL, None)
    arena = PolicyArena(
        checkpoint,
        arguments.profile,
        arguments.width,
        arguments.height,
        arguments.seed,
        arguments.action_limit,
        arguments.device,
    )
    server = ThreadingHTTPServer((arguments.host, arguments.port), handler_for(arena))
    url = f"http://{arguments.host}:{server.server_port}"
    print(f"Neural policy arena: {url}")
    print(f"Checkpoint: {checkpoint}")
    if not arguments.no_browser:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
