"""Stdio MCP server for harness-native background tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx
from diploid_agent.mcp_stdio import StdioMcpServer, _error_response, _tool_result


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
        ]

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
    parser.add_argument("--sessions-root", default=str(Path.home() / ".local" / "share" / "diploid"))
    args = parser.parse_args()

    server = HarnessMcpServer(
        chat_id=args.chat_id,
        harness_url=args.harness_url,
    )
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
