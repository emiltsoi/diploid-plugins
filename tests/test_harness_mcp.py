"""Tests for the diploid-harness MCP server tools."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from diploid_plugins.harness.harness_mcp import HarnessMcpServer


def _server(captured: list[dict[str, Any]]) -> HarnessMcpServer:
    server = HarnessMcpServer(chat_id="chat-1", harness_url="http://127.0.0.1:1")

    def fake_request(method: str, path: str, body: dict[str, Any] | None = None):
        captured.append({"method": method, "path": path, "body": body})
        return {"event_id": "wake-abc", "ok": True, "events": []}

    server._request = fake_request  # type: ignore[method-assign]
    return server


def _tool_text(resp: dict[str, Any]) -> Any:
    return json.loads(resp["result"]["content"][0]["text"])


def test_tools_listed() -> None:
    server = HarnessMcpServer(chat_id="chat-1", harness_url="http://127.0.0.1:1")
    names = {t["name"] for t in server._tools()}
    assert "harness_self_wake" in names
    assert "harness_self_wake_list" in names
    assert "harness_self_wake_cancel" in names


def test_self_wake_posts_timer_event() -> None:
    captured: list[dict[str, Any]] = []
    server = _server(captured)
    resp = server._call_tool(
        "harness_self_wake",
        {"reason": "check the build", "delay_seconds": 600},
        req_id=1,
    )
    out = _tool_text(resp)
    assert out["event_id"] == "wake-abc"
    assert out["fires_in_seconds"] > 590

    assert captured[0]["method"] == "POST"
    assert captured[0]["path"] == "/timer"
    body = captured[0]["body"]
    assert body["reason"] == "self_wake"
    assert body["chat_id"] == "chat-1"
    assert body["silent"] is True
    assert body["payload"]["notify"] is False
    assert body["payload"]["user_message"].startswith("[self-wake: check the build]")
    assert body["payload"]["agent_reason"] == "check the build"
    assert abs(body["scheduled_at"] - (time.time() + 600)) < 5


def test_self_wake_deliver_and_message() -> None:
    captured: list[dict[str, Any]] = []
    server = _server(captured)
    server._call_tool(
        "harness_self_wake",
        {
            "reason": "report render",
            "delay_seconds": 60,
            "message": "Tell Emil the render finished.",
            "deliver": True,
        },
        req_id=1,
    )
    body = captured[0]["body"]
    assert body["silent"] is False
    assert body["payload"]["notify"] is True
    assert "Tell Emil the render finished." in body["payload"]["user_message"]


def test_self_wake_at_epoch_and_iso() -> None:
    captured: list[dict[str, Any]] = []
    server = _server(captured)
    epoch = time.time() + 1200
    server._call_tool(
        "harness_self_wake", {"reason": "r", "at": epoch}, req_id=1
    )
    assert abs(captured[0]["body"]["scheduled_at"] - epoch) < 1

    server._call_tool(
        "harness_self_wake", {"reason": "r", "at": "2030-01-01T00:00:00Z"}, req_id=2
    )
    expected = datetime(2030, 1, 1, tzinfo=UTC).timestamp()
    assert abs(captured[1]["body"]["scheduled_at"] - expected) < 1


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"reason": "r"},
        {"reason": "r", "delay_seconds": 10, "at": 123.0},
        {"reason": "r", "at": "not a time"},
    ],
)
def test_self_wake_rejects_bad_arguments(arguments: dict[str, Any]) -> None:
    server = _server([])
    resp = server._call_tool("harness_self_wake", arguments, req_id=1)
    assert "error" in resp


def test_self_wake_list_filters_reason() -> None:
    captured: list[dict[str, Any]] = []
    server = _server(captured)
    resp = server._call_tool("harness_self_wake_list", {}, req_id=1)
    assert _tool_text(resp) == []
    assert captured[0]["method"] == "GET"
    assert "chat_id=chat-1" in captured[0]["path"]
    assert "reason=self_wake" in captured[0]["path"]


def test_self_wake_cancel() -> None:
    captured: list[dict[str, Any]] = []
    server = _server(captured)
    resp = server._call_tool(
        "harness_self_wake_cancel", {"event_id": "wake-abc"}, req_id=1
    )
    assert _tool_text(resp)["ok"] is True
    assert captured[0]["method"] == "POST"
    assert captured[0]["path"] == "/timer/cancel"
    assert captured[0]["body"] == {"chat_id": "chat-1", "event_id": "wake-abc"}


def test_self_wake_cancel_requires_event_id() -> None:
    server = _server([])
    resp = server._call_tool("harness_self_wake_cancel", {}, req_id=1)
    assert "error" in resp
