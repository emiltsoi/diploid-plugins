"""Tests for the body state plugin."""

from __future__ import annotations

import json
import time
from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.plugins.base import TurnInfo, WakeContext

from diploid_plugins.body.body import BodyPlugin
from diploid_plugins.body.body_mcp import BodyMcpServer
from diploid_plugins.body.config import BodyConfig
from diploid_plugins.body.manager import BodyManager


def _make_config() -> PluginConfig:
    return PluginConfig(
        name="body",
        enabled=True,
        state_file="chat_body_state.json",
        prompt_slot="body",
        prompt_order=60,
        max_prompt_chars=1024,
        config={"decay_rate_per_minute": 0.05, "max_intensity": 1.0},
    )


def _wake_context() -> WakeContext:
    return WakeContext(
        chat_id="chat-1",
        record=None,
        now=time.time(),
        instance_id="harness-123",
        instance_started_at=time.time(),
        previous_turn_at=None,
        pending_dispatches=[],
    )


def _turn_info() -> TurnInfo:
    return TurnInfo(
        chat_id="chat-1",
        session_id="s1",
        session_number=1,
        turn_number=1,
        updated_at=time.time(),
        last_stop_reason="completed",
        user_message="hello",
        reply="hi",
    )


def test_body_manager_defaults(tmp_path: Path) -> None:
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    assert mgr.state.skin_warmth == 0.0
    assert mgr.state.posture == "standing"
    assert mgr.state.gaze == "away"


def test_body_manager_event_saves(tmp_path: Path) -> None:
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.event("touch", "hand", 0.8)
    assert mgr.state.skin_warmth > 0.0
    state_path = tmp_path / "chat-1" / "chat_body_state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert data["last_event"] == "touch hand"


def test_body_manager_decay(tmp_path: Path) -> None:
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig(decay_rate_per_minute=10.0))
    mgr.event("touch", "hand", 1.0)
    before = mgr.state.skin_warmth
    # Simulate a minute of silence by back-dating updated_at.
    mgr.state.updated_at = time.time() - 60.0
    mgr.decay()
    assert mgr.state.skin_warmth < before


def test_body_manager_refresh(tmp_path: Path) -> None:
    mgr1 = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr1.event("touch", "hand", 1.0)

    # A second manager sees the saved state without needing to reload.
    mgr2 = BodyManager(tmp_path, "chat-1", BodyConfig())
    assert mgr2.state.skin_warmth == 1.0

    # Simulate an external write and refresh.
    external = {"skin_warmth": 0.2, "updated_at": time.time(), "gaze": "at you"}
    state_path = tmp_path / "chat-1" / "chat_body_state.json"
    state_path.write_text(json.dumps(external))
    mgr2.refresh()
    assert mgr2.state.skin_warmth == 0.2


def test_body_plugin_prompt_block(tmp_path: Path) -> None:
    plugin = BodyPlugin(_make_config(), "chat-1", tmp_path)
    block = plugin.prompt_block()
    assert block is not None
    assert "Current body state" in block


def test_body_plugin_on_waking_refreshes_external_state(tmp_path: Path) -> None:
    plugin = BodyPlugin(_make_config(), "chat-1", tmp_path)

    # Simulate the harness restoring a snapshot from a previous session.
    external = {
        "skin_warmth": 0.9,
        "chest_warmth": 0.5,
        "updated_at": time.time() - 120.0,
        "gaze": "at you",
    }
    state_path = tmp_path / "chat-1" / "chat_body_state.json"
    state_path.write_text(json.dumps(external))

    plugin.on_waking(_wake_context())
    assert plugin._body.state.gaze == "at you"
    assert plugin._body.state.skin_warmth <= 0.9


def test_body_plugin_event_command(tmp_path: Path) -> None:
    plugin = BodyPlugin(_make_config(), "chat-1", tmp_path)
    reply = plugin.event(event="hold", location="hand", intensity="0.7")
    assert "Felt: hold hand (0.7)" in reply
    assert plugin._body.state.hand_held == "hand"
    assert plugin._body.state.skin_warmth == 0.7


def test_body_plugin_on_turn_end_clears_events(tmp_path: Path) -> None:
    plugin = BodyPlugin(_make_config(), "chat-1", tmp_path)
    plugin._body.event("touch", "hand", 0.5)
    assert len(plugin._body._events) == 1
    plugin.on_turn_end(_turn_info())
    assert len(plugin._body._events) == 0


def test_body_plugin_memory_items(tmp_path: Path) -> None:
    before = time.time()
    plugin = BodyPlugin(_make_config(), "chat-1", tmp_path)
    plugin._body.event("touch", "hand", 0.5)
    items = plugin.memory_items(since=before)
    assert len(items) == 1
    assert items[0].metadata["kind"] == "touch"
    assert "hand" in items[0].content


def test_body_mcp_server_report(tmp_path: Path) -> None:
    server = BodyMcpServer("chat-1", tmp_path, BodyConfig())
    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "body_report",
                "arguments": {"kind": "touch", "location": "cheek", "intensity": 0.6},
            },
        }
    )
    assert resp is not None
    assert "result" in resp
    text = resp["result"]["content"][0]["text"]
    assert "Current body state" in text
    assert server.body.state.skin_warmth > 0.0


def test_body_mcp_server_decay(tmp_path: Path) -> None:
    server = BodyMcpServer("chat-1", tmp_path, BodyConfig(decay_rate_per_minute=10.0))
    server.body.event("touch", "hand", 1.0)
    server.body.state.updated_at = time.time() - 60.0
    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "body_decay"},
        }
    )
    assert resp is not None
    assert "result" in resp
    assert server.body.state.skin_warmth < 1.0
