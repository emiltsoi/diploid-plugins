"""Stdio MCP server for harness-native background tasks."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from diploid_agent.mcp_stdio import StdioMcpServer, _error_response, _tool_result

SELF_WAKE_REASON = "self_wake"
_MAX_REASON_CHARS = 160
_MAX_MESSAGE_CHARS = 2000


class HarnessMcpServer(StdioMcpServer):
    """MCP server that exposes harness tools to the ACP child.

    The ACP child can use `harness_subagent` to start a long-running background
    task that runs in a separate AcpEngine. When the task completes, the harness
    starts a new turn for the chat and posts the result to Telegram.
    """

    def __init__(
        self,
        chat_id: str,
        harness_url: str,
    ) -> None:
        super().__init__("diploid-harness", "0.1.0")
        self.chat_id = chat_id
        self.harness_url = harness_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.harness_url,
            timeout=60.0,
        )
        self._api_key = os.environ.get("HARNESS_API_KEY")

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            if method == "GET":
                resp = self._client.get(path, headers=self._headers())
            else:
                resp = self._client.post(path, json=body, headers=self._headers())
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            detail: object = exc.response.text
            try:
                detail = exc.response.json().get("detail", detail)
            except Exception:  # noqa: BLE001, S110
                pass
            return {"error": f"HTTP {exc.response.status_code}: {detail}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "harness_subagent",
                "description": (
                    "Start a background subagent that runs in a separate process. "
                    "The harness will continue the chat with the result when it finishes. "
                    "Use this for long tasks that might exceed the parent turn timeout."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to send to the subagent.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional short context shown in the continuation.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional model override for the subagent.",
                        },
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "harness_subagent_status",
                "description": (
                    "Check the status of background subagents for the current chat, "
                    "including whether they are running, completed, or failed."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "harness_self_wake",
                "description": (
                    "Arm a self-wake: the harness starts a turn for this chat at the "
                    "requested time, delivering your message back to you as if you had "
                    "sent it to yourself. Provide exactly one of delay_seconds or at "
                    "(epoch seconds or ISO-8601). The harness rate-limits self-wakes "
                    "(max pending, min interval between arms, max horizon) — use "
                    "harness_self_wake_list to inspect armed wakes and "
                    "harness_self_wake_cancel to retract one."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Short why — recorded on the wake event and shown to your future self.",
                        },
                        "delay_seconds": {
                            "type": "number",
                            "description": "Wake this many seconds from now.",
                        },
                        "at": {
                            "type": ["string", "number"],
                            "description": "Wake at this time — epoch seconds or ISO-8601.",
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "The instruction your future self receives as the turn's "
                                "message. Defaults to the reason."
                            ),
                        },
                        "deliver": {
                            "type": "boolean",
                            "description": (
                                "If true, the wake turn's reply is delivered to the user. "
                                "Default false — the wake runs silently."
                            ),
                            "default": False,
                        },
                        "priority": {
                            "type": "integer",
                            "description": "Wake priority (higher fires first). Default 1.",
                            "default": 1,
                        },
                    },
                    "required": ["reason"],
                },
            },
            {
                "name": "harness_self_wake_list",
                "description": "List this chat's currently armed self-wakes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "harness_self_wake_cancel",
                "description": "Retract an armed self-wake by event id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "event_id": {
                            "type": "string",
                            "description": "The wake event id returned by harness_self_wake.",
                        },
                    },
                    "required": ["event_id"],
                },
            },
            {
                "name": "harness_cron_list",
                "description": (
                    "List this service's cron jobs: id, schedule or trigger, "
                    "next due time, last status and summary, whether one is "
                    "running, and today's turn-delivery count. Manual firing "
                    "is the operator's door (POST /cron/<id>/run)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "harness_restart",
                "description": (
                    "Schedule a graceful restart of a harness service unit — "
                    "default: your own service. The unit drains in-flight turns, "
                    "flushes plugin state, and systemd restarts it after a short "
                    "delay; your current reply is delivered before the restart "
                    "fires. IMPORTANT: write your <self_state> handoff BEFORE "
                    "restarting yourself. Requires the authorship restart_enabled "
                    "toggle; only allowlisted units are accepted; a reason is "
                    "required and is recorded for the operator."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": (
                                "Target systemd unit. Defaults to your own service "
                                "(DIPLOID_SERVICE_NAME)."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Why the restart is needed — recorded and shown to the operator."
                            ),
                        },
                    },
                    "required": ["reason"],
                },
            },
        ]

    @staticmethod
    def _parse_at(value: str | float) -> float:
        """Parse an ``at`` argument: epoch seconds or ISO-8601."""
        if isinstance(value, (int, float)):
            return float(value)
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        dt = datetime.fromisoformat(
            text.removesuffix("Z") + ("+00:00" if text.endswith("Z") else "")
        )
        return dt.timestamp()

    def _self_wake(self, arguments: dict[str, Any], req_id: Any) -> dict[str, Any]:
        reason = str(arguments.get("reason", "")).strip()
        if not reason:
            return _error_response(req_id, "Missing required 'reason' argument.")
        reason = reason[:_MAX_REASON_CHARS]

        delay = arguments.get("delay_seconds")
        at = arguments.get("at")
        if (delay is None) == (at is None):
            return _error_response(req_id, "Provide exactly one of 'delay_seconds' or 'at'.")
        try:
            if delay is not None:
                scheduled_at = time.time() + float(delay)
            else:
                scheduled_at = self._parse_at(at)
        except (TypeError, ValueError) as exc:
            return _error_response(req_id, f"Invalid wake time: {exc}")

        message = str(arguments.get("message", "")).strip()[:_MAX_MESSAGE_CHARS]
        user_message = f"[self-wake: {reason}]"
        if message:
            user_message += f"\n{message}"

        deliver = bool(arguments.get("deliver", False))
        body = {
            "chat_id": self.chat_id,
            "reason": SELF_WAKE_REASON,
            "scheduled_at": scheduled_at,
            "payload": {
                "user_message": user_message,
                "agent_reason": reason,
                "notify": deliver,
            },
            "silent": not deliver,
            "priority": int(arguments.get("priority", 1)),
        }
        result = self._request("POST", "/timer", body)
        if result.get("error"):
            return _error_response(req_id, result["error"])
        return _tool_result(
            req_id,
            json.dumps(
                {
                    "event_id": result.get("event_id"),
                    "scheduled_at": scheduled_at,
                    "fires_in_seconds": round(max(0.0, scheduled_at - time.time()), 1),
                },
                ensure_ascii=False,
            ),
        )

    def _restart(self, arguments: dict[str, Any], req_id: Any) -> dict[str, Any]:
        reason = str(arguments.get("reason", "")).strip()
        if not reason:
            return _error_response(req_id, "Missing required 'reason' argument.")
        reason = reason[:_MAX_REASON_CHARS]
        service = str(arguments.get("service", "")).strip() or os.environ.get(
            "DIPLOID_SERVICE_NAME", ""
        )
        if not service:
            return _error_response(req_id, "No target service known.")
        # Always dial the CALLER's own listener so the requester's gate
        # (restart_enabled + restart_allowed_units) evaluates the request.
        # Keying the fallback on the *target* would be a confused deputy:
        # a request for a peer unit would land on the peer's socket and be
        # judged by the peer's allowlist instead.
        sock_path = os.environ.get("DIPLOID_CONTROL_SOCKET", "")
        if not sock_path:
            own = os.environ.get("DIPLOID_SERVICE_NAME", "")
            if own:
                sock_path = f"/tmp/diploid-ctl-{own}/control.sock"
        if not sock_path:
            return _error_response(req_id, "No harness control socket in this environment.")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(10.0)
                s.connect(sock_path)
                s.sendall(
                    json.dumps(
                        {
                            "action": "restart_service",
                            "service": service,
                            "command": "restart",
                            "reason": reason,
                            "chat_id": self.chat_id,
                            "token": os.environ.get("DIPLOID_CONTROL_TOKEN", ""),
                        }
                    ).encode("utf-8")
                )
                ack = json.loads(s.recv(4096).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return _error_response(req_id, f"Control socket request failed: {exc}")
        status = str(ack.get("status", "unknown"))
        if status not in {"ok", "scheduled"}:
            return _error_response(req_id, f"Restart not scheduled: {status}")
        return _tool_result(
            req_id,
            json.dumps({"service": service, "status": status}, ensure_ascii=False),
        )

    def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        req_id: Any,
    ) -> dict[str, Any]:
        if name == "harness_subagent_status":
            result = self._request("GET", f"/subagents/{self.chat_id}")
            if result.get("error"):
                return _error_response(req_id, result["error"])
            return _tool_result(
                req_id,
                json.dumps(result, ensure_ascii=False, default=str),
            )

        if name == "harness_self_wake":
            return self._self_wake(arguments, req_id)

        if name == "harness_self_wake_list":
            qs = urlencode({"chat_id": self.chat_id, "reason": SELF_WAKE_REASON})
            result = self._request("GET", f"/timer/pending?{qs}")
            if result.get("error"):
                return _error_response(req_id, result["error"])
            return _tool_result(
                req_id,
                json.dumps(result.get("events", []), ensure_ascii=False, default=str),
            )

        if name == "harness_cron_list":
            result = self._request("GET", "/cron")
            if result.get("error"):
                return _error_response(req_id, result["error"])
            jobs = [
                {
                    "id": job.get("id"),
                    "enabled": job.get("enabled"),
                    "source": job.get("source"),
                    "delivery": job.get("delivery"),
                    "call_type": job.get("call_type"),
                    "schedule": job.get("schedule"),
                    "trigger": job.get("trigger"),
                    "trigger_state": job.get("trigger_state"),
                    "next_due_at": job.get("next_due_at"),
                    "last_run_at": job.get("last_run_at"),
                    "last_status": job.get("last_status"),
                    "last_summary": job.get("last_summary"),
                    "running": job.get("running"),
                    "turns_today": job.get("turns_today"),
                }
                for job in result.get("jobs", [])
            ]
            return _tool_result(
                req_id,
                json.dumps(
                    {
                        "enabled": result.get("enabled"),
                        "jobs": jobs,
                        "warnings": result.get("warnings", []),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            )

        if name == "harness_restart":
            return self._restart(arguments, req_id)

        if name == "harness_self_wake_cancel":
            event_id = str(arguments.get("event_id", "")).strip()
            if not event_id:
                return _error_response(req_id, "Missing required 'event_id' argument.")
            result = self._request(
                "POST", "/timer/cancel", {"chat_id": self.chat_id, "event_id": event_id}
            )
            if result.get("error"):
                return _error_response(req_id, result["error"])
            return _tool_result(req_id, json.dumps(result, ensure_ascii=False))

        if name != "harness_subagent":
            return _error_response(req_id, f"Unknown tool: {name}")

        prompt = arguments.get("prompt", "")
        if not prompt:
            return _error_response(req_id, "Missing required 'prompt' argument.")

        body: dict[str, Any] = {"chat_id": self.chat_id, "prompt": prompt}
        if arguments.get("context"):
            body["context"] = arguments["context"]
        if arguments.get("model"):
            body["model"] = arguments["model"]

        result = self._request("POST", "/subagent", body)
        if result.get("error"):
            return _error_response(req_id, result["error"])

        return _tool_result(
            req_id,
            json.dumps(
                {
                    "dispatch_id": result.get("dispatch_id"),
                    "reply": result.get("reply"),
                    "notice": result.get("notice"),
                },
                ensure_ascii=False,
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="diploid-harness MCP server")
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--harness-url", default="http://127.0.0.1:4003")
    parser.add_argument(
        "--sessions-root", default=str(Path.home() / ".local" / "share" / "diploid")
    )
    args = parser.parse_args()

    server = HarnessMcpServer(
        chat_id=args.chat_id,
        harness_url=args.harness_url,
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
