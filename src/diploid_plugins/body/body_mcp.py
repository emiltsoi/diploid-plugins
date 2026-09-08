"""Stdio MCP server for the body state plugin.

This server exposes the body as three MCP tools:

- `body_report(kind, location, intensity)` — record a sensory event.
- `body_felt(summary, warmth)` — author the felt texture line and set the ember.
- `body_state()` — return the current body-state prompt block.
- `body_decay()` — force a decay pass and return the current state.

It is intended to be launched once per ACP session with the chat id
and sessions root supplied on the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from diploid_plugins.body.config import BodyConfig
from diploid_plugins.body.manager import BodyManager

DEFAULT_PROTOCOL_VERSION = "2024-11-05"


def _error_response(req_id: Any, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32602, "message": message},
    }


def _tool_result(req_id: Any, text: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        },
    }


class BodyMcpServer:
    """Minimal stdio MCP server backed by BodyManager."""

    def __init__(self, chat_id: str, sessions_root: Path, config: BodyConfig) -> None:
        self.body = BodyManager(
            sessions_root=sessions_root,
            chat_id=chat_id,
            config=config,
        )

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "body_report",
                "description": (
                    "Report a physical interaction with the user. "
                    "Supported kinds: touch, hold, release, voice, near, step_back, gaze."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "description": "The kind of physical event.",
                        },
                        "location": {
                            "type": "string",
                            "description": "Body location the event targets, if any.",
                        },
                        "intensity": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Strength of the sensation, 0.0–1.0. Defaults to 0.5.",
                        },
                    },
                    "required": ["kind"],
                },
            },
            {
                "name": "body_felt",
                "description": (
                    "Author the felt texture line that survives sleep — one "
                    "honest sentence about what the last moment felt like "
                    "(e.g. \"chest warm from their voice before sleep\"), plus "
                    "the ember strength 0.0–1.0. The words are always yours; "
                    "the plugin never writes them for you. Empty summary "
                    "clears the line."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "One line of felt texture, authored by you.",
                        },
                        "warmth": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Ember strength, 0.0–1.0.",
                        },
                    },
                    "required": ["warmth"],
                },
            },
            {
                "name": "body_state",
                "description": "Return the current body-state block for the prompt.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "body_decay",
                "description": "Force a decay pass and return the current body-state block.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def _handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            protocol_version = params.get("protocolVersion", DEFAULT_PROTOCOL_VERSION)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "diploid-body", "version": "0.1.0"},
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self._tools()}}

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "body_report":
                return self._report(req_id, arguments)
            if name == "body_felt":
                return self._felt(req_id, arguments)
            if name == "body_state":
                return _tool_result(req_id, self.body.state_for_prompt())
            if name == "body_decay":
                self.body.decay()
                return _tool_result(req_id, self.body.state_for_prompt())
            return _error_response(req_id, f"Unknown tool: {name}")

        return _error_response(req_id, f"Unknown method: {method}")

    def _report(
        self,
        req_id: Any,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        kind = arguments.get("kind")
        if not kind or not isinstance(kind, str):
            return _error_response(req_id, "'kind' is required and must be a string")
        location = arguments.get("location")
        if location is not None and not isinstance(location, str):
            return _error_response(req_id, "'location' must be a string")
        try:
            intensity = float(arguments.get("intensity", 0.5))
        except (TypeError, ValueError):
            return _error_response(req_id, "'intensity' must be a number")

        self.body.event(kind, location, intensity)
        return _tool_result(req_id, self.body.state_for_prompt())

    def _felt(
        self,
        req_id: Any,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        summary = arguments.get("summary")
        if summary is not None and not isinstance(summary, str):
            return _error_response(req_id, "'summary' must be a string")
        try:
            warmth = float(arguments.get("warmth"))
        except (TypeError, ValueError):
            return _error_response(req_id, "'warmth' is required and must be a number")
        self.body.set_felt(summary, warmth)
        return _tool_result(req_id, self.body.state_for_prompt())

    def run(self) -> None:
        """Read JSON-RPC messages from stdin and write responses to stdout."""
        while True:
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                break
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                _write(_error_response(None, f"Invalid JSON: {exc}"))
                continue

            if isinstance(request, list):
                response = [self._handle(r) for r in request]
                response = [r for r in response if r is not None]
                if response:
                    _write(response)
            else:
                response = self._handle(request)
                if response is not None:
                    _write(response)


def _write(message: dict[str, Any] | list[dict[str, Any]]) -> None:
    text = json.dumps(message, ensure_ascii=False)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _body_config(args: argparse.Namespace) -> BodyConfig:
    overrides: dict[str, Any] = {}
    if args.state_file:
        overrides["state_file"] = args.state_file
    if args.decay_rate is not None:
        overrides["decay_rate_per_minute"] = args.decay_rate
    if args.max_intensity is not None:
        overrides["max_intensity"] = args.max_intensity
    if args.felt_config:
        try:
            overrides.update(json.loads(args.felt_config))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --felt-config JSON: {exc}") from exc
    if not overrides:
        return BodyConfig()
    return BodyConfig(**overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP server for body state.")
    parser.add_argument("--chat-id", required=True, help="Chat identifier this body belongs to.")
    parser.add_argument(
        "--sessions-root", default="sessions", help="Root of all chat session dirs."
    )
    parser.add_argument("--state-file", default=None, help="Body state filename inside the chat dir.")
    parser.add_argument("--decay-rate", type=float, default=None, help="Decay per minute.")
    parser.add_argument("--max-intensity", type=float, default=None, help="Max intensity cap.")
    parser.add_argument(
        "--felt-config",
        default=None,
        help="JSON object of felt-layer BodyConfig overrides.",
    )
    parser.add_argument("--log-file", default=None, help="Optional stderr log file path.")
    args = parser.parse_args(argv)

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    sessions_root = Path(args.sessions_root).expanduser().resolve()
    config = _body_config(args)
    server = BodyMcpServer(chat_id=args.chat_id, sessions_root=sessions_root, config=config)
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
