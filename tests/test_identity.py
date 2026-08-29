"""Tests for the identity/self-narrative plugin and MCP server."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.plugins import PluginManager
from diploid_agent.plugins.base import TurnInfo

from diploid_plugins.identity.identity import IdentityPlugin
from diploid_plugins.identity.identity_mcp import IdentityMcpServer


def _make_config(tmp_path: Path, **overrides: object) -> PluginConfig:
    return PluginConfig(
        name="identity",
        enabled=True,
        module="diploid_plugins.identity",
        prompt_slot="self_narrative",
        first_prompt_only=True,
        prompt_order=5,
        state_file="chat_SELF.md",
        max_prompt_chars=2048,
        skill="identity",
        **overrides,  # type: ignore[arg-type]
    )


def test_empty_prompt_block_is_none(tmp_path: Path) -> None:
    plugin = IdentityPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    assert plugin.prompt_block() is None
    assert plugin.event(event="state") == "No self-narrative set."


def test_update_and_prompt_block(tmp_path: Path) -> None:
    plugin = IdentityPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    reply = plugin.event(event="update", content="I am helpful and concise.")
    assert "updated" in reply.lower()

    block = plugin.prompt_block()
    assert block is not None
    assert "## Who I am right now" in block
    assert "I am helpful and concise." in block

    path = tmp_path / "chat-1" / "chat_SELF.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "<!-- identity-file: chat_SELF.md -->" in text
    assert "persona SOUL.md and MEMORY.md stay canonical" in text
    assert "manual @" in text

    history_path = tmp_path / "chat-1" / "chat_SELF_history.jsonl"
    assert history_path.exists()
    records = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["reason"] == "manual"
    assert "old_text" in records[0]
    assert "new_text" in records[0]


def test_update_with_reason(tmp_path: Path) -> None:
    plugin = IdentityPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    plugin.event(event="update", content="I am focused.", reason="audit")

    path = tmp_path / "chat-1" / "chat_SELF.md"
    text = path.read_text(encoding="utf-8")
    assert "audit @" in text

    records = [
        json.loads(line)
        for line in (tmp_path / "chat-1" / "chat_SELF_history.jsonl").read_text().splitlines()
    ]
    assert records[0]["reason"] == "audit"


def test_event_update_via_raw_args(tmp_path: Path) -> None:
    plugin = IdentityPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    reply = plugin.event(event="update", raw_args="I am direct.")
    assert "updated" in reply.lower()
    assert "I am direct." in plugin.prompt_block()


def test_clear(tmp_path: Path) -> None:
    plugin = IdentityPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    plugin.event(event="update", content="I am present.")
    assert plugin.prompt_block() is not None

    reply = plugin.event(event="clear")
    assert "cleared" in reply.lower()
    assert plugin.prompt_block() is None

    path = tmp_path / "chat-1" / "chat_SELF.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!-- identity-file:")
    assert "\n\n" not in text.strip().split("\n\n", 1)[0]  # header only

    records = [
        json.loads(line)
        for line in (tmp_path / "chat-1" / "chat_SELF_history.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2


def test_prompt_block_respects_max_chars(tmp_path: Path) -> None:
    plugin = IdentityPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    plugin.event(event="update", content="word " * 100)
    block = plugin.prompt_block(max_chars=60)
    assert block is not None
    assert len(block) <= 60
    assert "## Who I am right now" in block


def test_history_returns_records(tmp_path: Path) -> None:
    plugin = IdentityPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    plugin._update("first", "manual")
    plugin._update("second", "manual")
    plugin._update("third", "manual")

    records = plugin._history(limit=2)
    assert len(records) == 2
    assert records[0]["new_text"].endswith("second")
    assert records[1]["new_text"].endswith("third")


def test_mcp_server_handle_update_and_state(tmp_path: Path) -> None:
    server = IdentityMcpServer("chat-1", tmp_path, "chat_SELF.md")
    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "identity_update",
                "arguments": {
                    "content": "I am the MCP server.",
                    "reason": "mcp-test",
                },
            },
        }
    )
    assert resp is not None
    assert not resp["result"]["isError"]
    assert "I am the MCP server." in resp["result"]["content"][0]["text"]

    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "identity_state", "arguments": {}},
        }
    )
    assert resp is not None
    assert not resp["result"]["isError"]
    assert "I am the MCP server." in resp["result"]["content"][0]["text"]

    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "identity_history", "arguments": {"limit": 5}},
        }
    )
    assert resp is not None
    assert not resp["result"]["isError"]
    history = json.loads(resp["result"]["content"][0]["text"])
    assert len(history) == 1
    assert history[0]["reason"] == "mcp-test"

    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "identity_clear", "arguments": {}},
        }
    )
    assert resp is not None
    assert not resp["result"]["isError"]

    state = json.loads(
        (tmp_path / "chat-1" / "chat_SELF_history.jsonl").read_text().splitlines()[-1]
    )
    assert state["reason"] == "manual"


def test_mcp_server_run_lists_tools(tmp_path: Path, monkeypatch) -> None:
    server = IdentityMcpServer("chat-1", tmp_path, "chat_SELF.md")
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    stdin = io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    server.run()
    stdout.seek(0)
    responses = [json.loads(line) for line in stdout if line.strip()]
    assert len(responses) == 2
    assert responses[1]["result"]["tools"]
    names = {t["name"] for t in responses[1]["result"]["tools"]}
    assert names == {"identity_update", "identity_state", "identity_history", "identity_clear"}


def test_plugin_manager_fill_prompt_slots_places_self_narrative(tmp_path: Path) -> None:
    mgr = PluginManager(
        plugins=[_make_config(tmp_path)],
        sessions_root=tmp_path,
        instance_id="test-instance",
        instance_started_at=0.0,
    )
    mgr.event("chat-1", "identity", event="update", content="Stay grounded.")

    slots: dict[str, list[str]] = {"self_narrative": []}
    filled = mgr.fill_prompt_slots("chat-1", slots, is_first=True)
    assert "self_narrative" in filled
    assert len(filled["self_narrative"]) == 1
    assert "Stay grounded." in filled["self_narrative"][0]


def test_auto_rewrite_off_by_default(tmp_path: Path) -> None:
    plugin = IdentityPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    turn = TurnInfo(
        chat_id="chat-1",
        session_id="s-1",
        session_number=1,
        turn_number=1,
        updated_at=time.time(),
        last_stop_reason=None,
        user_message="Update yourself.",
        reply="Some text\n```identity\nI am the auto narrative.\n```\nmore text",
    )
    plugin.on_turn_end(turn)
    assert plugin.prompt_block() is None
    assert not (tmp_path / "chat-1" / "chat_SELF.md").exists()


def test_auto_rewrite_applies_when_enabled(tmp_path: Path) -> None:
    plugin = IdentityPlugin(
        _make_config(tmp_path, config={"auto_rewrite": True}), "chat-1", tmp_path
    )
    turn = TurnInfo(
        chat_id="chat-1",
        session_id="s-1",
        session_number=1,
        turn_number=1,
        updated_at=time.time(),
        last_stop_reason=None,
        user_message="Update yourself.",
        reply="First block\n```identity\nI am one.\n```\n\nSecond block\n```identity\nI am two.\n```",
    )
    plugin.on_turn_end(turn)
    block = plugin.prompt_block()
    assert block is not None
    assert "I am two." in block

    records = [
        json.loads(line)
        for line in (tmp_path / "chat-1" / "chat_SELF_history.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert all(r["reason"] == "model-auto" for r in records)
    assert "I am one." in records[0]["new_text"]
    assert "I am two." in records[1]["new_text"]
