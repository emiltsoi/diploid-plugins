"""Tests for the body state plugin."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
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


def test_felt_set_and_prompt_line(tmp_path: Path) -> None:
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.set_felt("chest warm from his voice before sleep", 0.6)
    assert mgr.state.felt_warmth == 0.6
    assert mgr.state.felt_summary == "chest warm from his voice before sleep"
    block = mgr.state_for_prompt()
    assert 'Felt residue: warm — "chest warm from his voice before sleep"' in block
    assert "(felt just now)" in block


def test_felt_summary_is_authored_only(tmp_path: Path) -> None:
    """Events nudge the ember but never write the texture line."""
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.event("touch", "hand", 1.0)
    assert mgr.state.felt_warmth > 0.0
    assert mgr.state.felt_summary is None
    # Warmth alone still surfaces a residue line.
    assert "Felt residue:" in mgr.state_for_prompt()


def test_felt_cold_start_stays_cold(tmp_path: Path) -> None:
    """No decay path may manufacture warmth from nothing."""
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.felt_wake(3600.0 * 100)
    mgr.felt_turn_step()
    assert mgr.state.felt_warmth == 0.0
    assert "Felt residue" not in mgr.state_for_prompt()


def test_felt_wake_decay_scales_with_silence(tmp_path: Path) -> None:
    """A restart-blink barely dims; a long silence dims by at most wake_fade."""
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.set_felt(None, 1.0)
    mgr.felt_wake(19.0)  # a restart is not an experience
    assert mgr.state.felt_warmth > 0.999

    mgr.set_felt(None, 1.0)
    mgr.felt_wake(3600.0 * 72.0)  # silence saturates at the cap
    assert mgr.state.felt_warmth == pytest.approx(0.8)

    mgr.set_felt(None, 1.0)
    mgr.felt_wake(3600.0 * 1000.0)  # beyond the cap: still just wake_fade
    assert mgr.state.felt_warmth == pytest.approx(0.8)


def test_felt_turn_step_and_event_soft_saturation(tmp_path: Path) -> None:
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.set_felt(None, 0.5)
    mgr.felt_turn_step()
    assert mgr.state.felt_warmth == pytest.approx(0.5 * 0.97)

    # Sustained intense events approach the cap asymptotically, never over.
    mgr2 = BodyManager(tmp_path, "chat-2", BodyConfig())
    prev = 0.0
    for _ in range(50):
        mgr2.event("touch", "hand", 1.0)
        assert mgr2.state.felt_warmth > prev
        prev = mgr2.state.felt_warmth
    assert mgr2.state.felt_warmth < 1.0
    assert mgr2.state.felt_warmth > 0.9


def test_felt_reads_do_not_mutate(tmp_path: Path) -> None:
    """state_for_prompt must be read-only — looking at the body can't age it."""
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig(decay_rate_per_minute=10.0))
    mgr.set_felt("held", 0.9)
    mgr.event("touch", "hand", 1.0)
    mgr.state.updated_at = time.time() - 3600.0
    mgr._save_state()
    before = dict(mgr.state.__dict__)
    mgr.state_for_prompt()
    mgr.state_for_prompt()
    after = dict(mgr.state.__dict__)
    assert before == after


def test_felt_summary_expires_to_memory_phrasing(tmp_path: Path) -> None:
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig(felt_expire_seconds=60.0))
    mgr.set_felt("an old warmth", 0.5)
    mgr.state.felt_summary_at = time.time() - 3600.0
    mgr._save_state()
    block = mgr.state_for_prompt()
    assert "A memory of feeling" in block
    assert '"an old warmth"' in block


def test_felt_summary_hidden_when_cold_and_stale(tmp_path: Path) -> None:
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig(felt_expire_seconds=60.0))
    mgr.set_felt("an old warmth", 0.5)
    mgr.state.felt_summary_at = time.time() - 3600.0
    mgr.state.felt_warmth = 0.0
    mgr._save_state()
    assert "memory of feeling" not in mgr.state_for_prompt()


def test_body_state_backward_compat(tmp_path: Path) -> None:
    """Pre-felt state files load with defaults."""
    state_path = tmp_path / "chat-1" / "chat_body_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"skin_warmth": 0.5, "gaze": "at you"}))
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    assert mgr.state.skin_warmth == 0.5
    assert mgr.state.felt_warmth == 0.0
    assert mgr.state.felt_summary is None
    assert mgr.state.felt_events == []


def test_felt_events_recorded_at_threshold(tmp_path: Path) -> None:
    """Warm events below the record threshold nudge the ember but leave no moment."""
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.event("voice", "chest", 0.3)
    assert mgr.state.felt_events == []
    assert mgr.state.felt_warmth > 0.0  # ember still moved

    mgr.event("voice", "chest", 0.6)
    assert len(mgr.state.felt_events) == 1
    entry = mgr.state.felt_events[0]
    assert entry["kind"] == "voice"
    assert entry["location"] == "chest"
    assert entry["summary"] is None  # texture stays authored-only
    assert entry["warmth"] == 0.6
    assert entry["at"] > 0.0


def test_felt_events_set_felt_always_records(tmp_path: Path) -> None:
    """Every authored felt line is a moment in the story."""
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.set_felt("his voice at the edge of sleep", 0.7)
    assert len(mgr.state.felt_events) == 1
    entry = mgr.state.felt_events[0]
    assert entry["kind"] == "felt"
    assert entry["summary"] == "his voice at the edge of sleep"
    assert entry["warmth"] == 0.7


def test_felt_events_capped_drop_oldest(tmp_path: Path) -> None:
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig(felt_events_max=3))
    for i in range(5):
        mgr.set_felt(f"moment {i}", 0.5)
    assert len(mgr.state.felt_events) == 3
    summaries = [e["summary"] for e in mgr.state.felt_events]
    assert summaries == ["moment 2", "moment 3", "moment 4"]


def test_felt_events_render_in_prompt(tmp_path: Path) -> None:
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    assert "Warm moments" not in mgr.state_for_prompt()
    mgr.event("voice", "chest", 0.8)
    mgr.set_felt("warm from his voice", 0.6)
    block = mgr.state_for_prompt()
    assert "Warm moments:" in block
    assert "voice chest (just now)" in block
    assert 'felt "warm from his voice" (just now)' in block


def test_felt_events_round_trip(tmp_path: Path) -> None:
    """The story survives a reload — it is state, not ephemera."""
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.set_felt("held through the sleep", 0.9)
    mgr2 = BodyManager(tmp_path, "chat-1", BodyConfig())
    assert len(mgr2.state.felt_events) == 1
    assert mgr2.state.felt_events[0]["summary"] == "held through the sleep"


def test_body_plugin_felt_command(tmp_path: Path) -> None:
    plugin = BodyPlugin(_make_config(), "chat-1", tmp_path)
    reply = plugin.event(event="felt", location="warm from his voice", intensity="0.6")
    assert "Felt line written" in reply
    assert plugin._body.state.felt_summary == "warm from his voice"
    assert plugin._body.state.felt_warmth == 0.6


def test_body_mcp_server_felt(tmp_path: Path) -> None:
    server = BodyMcpServer("chat-1", tmp_path, BodyConfig())
    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "body_felt",
                "arguments": {"summary": "held through the sleep", "warmth": 0.7},
            },
        }
    )
    assert resp is not None
    text = resp["result"]["content"][0]["text"]
    assert '"held through the sleep"' in text
    assert server.body.state.felt_warmth == 0.7

    # Clearing: empty summary drops the line but keeps the ember.
    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "body_felt", "arguments": {"summary": "", "warmth": 0.3}},
        }
    )
    assert resp is not None
    assert server.body.state.felt_summary is None
    assert server.body.state.felt_warmth == 0.3


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


def test_felt_event_direction_rendered(tmp_path: Path) -> None:
    """A summary-less felt event still reads as a story, not a bare 'felt'."""
    mgr = BodyManager(tmp_path, "chat-1", BodyConfig())
    mgr.set_felt(None, 0.6)
    mgr.set_felt(None, 0.0)
    events = mgr.state.felt_events
    assert events[0]["direction"] == "rose"
    assert events[1]["direction"] == "cleared"
    block = mgr.state_for_prompt()
    assert "felt cleared (just now)" in block
    assert 'felt "None"' not in block


def test_felt_events_age_cap(tmp_path: Path) -> None:
    """Warm moments older than the age cap are dropped on append and load."""
    config = BodyConfig(felt_events_max_age_hours=1.0)
    mgr = BodyManager(tmp_path, "chat-1", config)
    now = time.time()

    # Seed with a fresh and an old event.
    mgr.state.felt_events = [
        {"kind": "felt", "summary": "fresh", "at": now - 10.0, "warmth": 0.5},
        {"kind": "felt", "summary": "stale", "at": now - 3700.0, "warmth": 0.5},
    ]
    mgr._save_state()

    # New manager loading the state should prune the stale entry.
    mgr2 = BodyManager(tmp_path, "chat-1", config)
    assert len(mgr2.state.felt_events) == 1
    assert mgr2.state.felt_events[0]["summary"] == "fresh"

    # Appending a new event should also prune stale entries.
    mgr2.set_felt("newer still", 0.6)
    assert all(e["summary"] != "stale" for e in mgr2.state.felt_events)


def test_felt_events_age_cap_disabled(tmp_path: Path) -> None:
    """A zero/negative age cap disables time-based pruning."""
    config = BodyConfig(felt_events_max_age_hours=0.0)
    mgr = BodyManager(tmp_path, "chat-1", config)
    now = time.time()
    mgr.state.felt_events = [
        {"kind": "felt", "summary": "old", "at": now - 86400.0 * 365, "warmth": 0.5},
    ]
    mgr._save_state()
    mgr.set_felt("new", 0.6)
    assert len(mgr.state.felt_events) == 2
    assert mgr.state.felt_events[0]["summary"] == "old"
