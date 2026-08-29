"""Tests for the working-memory plugin and MCP server."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.plugins import PluginManager

from diploid_plugins.working_memory.working_memory import WorkingMemoryPlugin
from diploid_plugins.working_memory.working_memory_mcp import WorkingMemoryMcpServer


def _make_config(tmp_path: Path) -> PluginConfig:
    return PluginConfig(
        name="working_memory",
        enabled=True,
        module="diploid_plugins.working_memory",
        prompt_slot="working_memory",
        first_prompt_only=False,
        prompt_order=40,
        state_file="chat_working_memory.json",
        max_prompt_chars=1024,
    )


def test_empty_prompt_block_is_none(tmp_path: Path) -> None:
    plugin = WorkingMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    assert plugin.prompt_block() is None
    assert plugin.event(event="state") == "Working memory is empty."


def test_set_intent_and_append_lists(tmp_path: Path) -> None:
    plugin = WorkingMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    assert "Deploy" in plugin._set_intent("Deploy the service")
    assert "Set up CI" in plugin._append_to_plan("Set up CI")
    assert "smoke" in plugin._append_to_plan("Run smoke tests")
    assert "cache" in plugin._append_open_question("Should we cache the build?")
    assert "regression" in plugin._append_note("No regression tests yet")

    block = plugin.prompt_block()
    assert block is not None
    assert "## Working memory" in block
    assert "- Intent: Deploy the service" in block
    assert "  - Set up CI" in block
    assert "  - Run smoke tests" in block
    assert "  - Should we cache the build?" in block
    assert "  - No regression tests yet" in block

    path = tmp_path / "chat-1" / "chat_working_memory.json"
    assert path.exists()
    state = json.loads(path.read_text())
    assert state["intent"] == "Deploy the service"
    assert state["plan"] == ["Set up CI", "Run smoke tests"]


def test_clear_field_and_all(tmp_path: Path) -> None:
    plugin = WorkingMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    plugin._set_intent("Deploy")
    plugin._append_to_plan("Set up CI")
    plugin._append_note("Note one")

    assert "plan" in plugin._clear("plan")
    assert plugin._state["plan"] == []
    assert plugin._state["intent"] == "Deploy"

    assert "Cleared working memory" in plugin._clear()
    assert plugin._state["intent"] == ""
    assert plugin._state["notes"] == []


def test_prompt_block_respects_max_chars(tmp_path: Path) -> None:
    plugin = WorkingMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    plugin._set_intent("word " * 100)
    block = plugin.prompt_block(max_chars=50)
    assert block is not None
    assert len(block) <= 50
    assert "## Working memory" in block


def test_event_update_via_params(tmp_path: Path) -> None:
    plugin = WorkingMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    reply = plugin.event(event="update", field="intent", value="Plan the deployment")
    assert "Plan the deployment" in reply
    assert plugin._state["intent"] == "Plan the deployment"

    reply = plugin.event(event="update", field="plan", value="Design schema")
    assert "Design schema" in reply
    assert plugin._state["plan"] == ["Design schema"]


def test_event_update_via_raw_args(tmp_path: Path) -> None:
    plugin = WorkingMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    reply = plugin.event(event="update", raw_args='intent "plan the deployment"')
    assert "plan the deployment" in reply
    assert plugin._state["intent"] == "plan the deployment"

    reply = plugin.event(event="update", raw_args="plan first step")
    assert "first step" in reply
    assert plugin._state["plan"] == ["first step"]


def test_event_clear(tmp_path: Path) -> None:
    plugin = WorkingMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    plugin._set_intent("Deploy")
    plugin._append_to_plan("Set up CI")

    assert "intent" in plugin.event(event="clear", field="intent")
    assert plugin._state["intent"] == ""
    assert plugin._state["plan"] == ["Set up CI"]

    assert "Cleared working memory" in plugin.event(event="clear")
    assert plugin._state["plan"] == []


def test_mcp_server_handle_update_and_state(tmp_path: Path) -> None:
    server = WorkingMemoryMcpServer("chat-1", tmp_path, "chat_working_memory.json")
    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "working_memory_update",
                "arguments": {"field": "intent", "value": "Plan the deployment"},
            },
        }
    )
    assert resp is not None
    assert not resp["result"]["isError"]
    assert "Plan the deployment" in resp["result"]["content"][0]["text"]

    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "working_memory_update",
                "arguments": {"field": "plan", "value": "Set up CI"},
            },
        }
    )
    assert resp is not None
    assert not resp["result"]["isError"]

    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "working_memory_clear", "arguments": {"field": "plan"}},
        }
    )
    assert resp is not None
    assert not resp["result"]["isError"]

    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "working_memory_state", "arguments": {}},
        }
    )
    assert resp is not None
    assert not resp["result"]["isError"]
    assert "Plan the deployment" in resp["result"]["content"][0]["text"]
    assert "Set up CI" not in resp["result"]["content"][0]["text"]

    state = json.loads((tmp_path / "chat-1" / "chat_working_memory.json").read_text())
    assert state["intent"] == "Plan the deployment"
    assert state["plan"] == []


def test_mcp_server_run_lists_tools(tmp_path: Path, monkeypatch) -> None:
    server = WorkingMemoryMcpServer("chat-1", tmp_path, "chat_working_memory.json")
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
    assert names == {"working_memory_update", "working_memory_clear", "working_memory_state"}


def test_plugin_manager_fill_prompt_slots_places_working_memory(tmp_path: Path) -> None:
    mgr = PluginManager(
        plugins=[_make_config(tmp_path)],
        sessions_root=tmp_path,
        instance_id="test-instance",
        instance_started_at=0.0,
    )
    mgr.event("chat-1", "working_memory", event="update", field="intent", value="Stay focused")

    slots: dict[str, list[str]] = {"working_memory": []}
    filled = mgr.fill_prompt_slots("chat-1", slots, is_first=False)
    assert "working_memory" in filled
    assert len(filled["working_memory"]) == 1
    assert "Stay focused" in filled["working_memory"][0]
