"""Tests for partial-turn snapshots in the continuity plugin."""

import json
from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.models import PartialTurn
from diploid_agent.plugins.base import TurnInfo

from diploid_plugins.continuity.continuity import ContinuityPlugin


def test_on_partial_writes_active_turn(tmp_path: Path) -> None:
    chat_dir = tmp_path / "c1"
    chat_dir.mkdir()
    p = ContinuityPlugin(
        PluginConfig(name="continuity", enabled=True, state_file="chat_wake_state.json"),
        "c1",
        tmp_path,
    )
    p.on_partial(
        PartialTurn(
            chat_id="c1",
            session_number=1,
            turn_number=2,
            user_message="hi",
            message_text="msg so far",
            thought_text="thinking",
            updated_at=10.0,
        )
    )
    active = json.loads((chat_dir / "chat_active_turn.json").read_text())
    assert active["message_text"] == "msg so far"
    assert active["thought_text"] == "thinking"


def test_turn_end_removes_active_turn(tmp_path: Path) -> None:
    chat_dir = tmp_path / "c1"
    chat_dir.mkdir()
    p = ContinuityPlugin(
        PluginConfig(name="continuity", enabled=True, state_file="chat_wake_state.json"),
        "c1",
        tmp_path,
    )
    p.on_partial(
        PartialTurn(
            chat_id="c1",
            session_number=1,
            turn_number=2,
            user_message="hi",
            message_text="msg",
            thought_text="thought",
            updated_at=10.0,
        )
    )
    p.on_turn_end(
        TurnInfo(
            chat_id="c1",
            session_id="s1",
            session_number=1,
            turn_number=2,
            updated_at=11.0,
            last_stop_reason="completed",
            user_message="hi",
            reply="done",
        )
    )
    assert not (chat_dir / "chat_active_turn.json").exists()


def test_on_partial_is_throttled(tmp_path: Path) -> None:
    chat_dir = tmp_path / "c1"
    chat_dir.mkdir()
    p = ContinuityPlugin(
        PluginConfig(name="continuity", enabled=True, state_file="chat_wake_state.json"),
        "c1",
        tmp_path,
    )
    for i in range(5):
        p.on_partial(
            PartialTurn(
                chat_id="c1",
                session_number=1,
                turn_number=2,
                user_message="hi",
                message_text=f"msg {i}",
                thought_text="thinking",
                updated_at=1.0 + i * 0.25,
            )
        )
    active = json.loads((chat_dir / "chat_active_turn.json").read_text())
    assert active["message_text"] != "msg 0"
