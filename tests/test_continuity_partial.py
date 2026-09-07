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


class _RecordingRuntime:
    def __init__(self) -> None:
        self.system_notes: list[tuple[str, str]] = []

    def record_system_note(self, chat_id: str, text: str) -> None:
        self.system_notes.append((chat_id, text))


def _wake_context():
    from diploid_agent.models import SessionRecord
    from diploid_agent.plugins.base import WakeContext

    return WakeContext(
        chat_id="c1",
        record=SessionRecord(
            chat_id="c1",
            session_number=1,
            session_id="s1",
            model="m1",
            persona="test",
            cwd="/tmp",
            created_at=0.0,
            updated_at=8.0,
            last_stop_reason="completed",
        ),
        now=20.0,
        instance_id="i2",
        instance_started_at=15.0,
        previous_turn_at=10.0,
        pending_dispatches=[],
    )


def _make_plugin(tmp_path: Path, runtime: object | None = None) -> ContinuityPlugin:
    return ContinuityPlugin(
        PluginConfig(name="continuity", enabled=True, state_file="chat_wake_state.json"),
        "c1",
        tmp_path,
        runtime=runtime,
    )


def test_wake_preserves_interrupted_turn(tmp_path: Path) -> None:
    chat_dir = tmp_path / "c1"
    chat_dir.mkdir()
    (chat_dir / "chat_active_turn.json").write_text(
        json.dumps(
            {
                "session_number": 1,
                "turn_number": 5,
                "user_message": "Approved - go ahead",
                "message_text": "partial reply",
                "thought_text": "mid work",
                "updated_at": 9.0,
            }
        )
    )
    runtime = _RecordingRuntime()
    p = _make_plugin(tmp_path, runtime=runtime)

    p.on_waking(_wake_context())

    assert not (chat_dir / "chat_active_turn.json").exists()
    preserved = json.loads((chat_dir / "chat_interrupted_turn.json").read_text())
    assert preserved["message_text"] == "partial reply"
    assert p._state["interrupted_turn"]["turn_number"] == 5
    assert runtime.system_notes == [
        (
            "c1",
            (
                "Turn 5 was interrupted mid-flight and never recorded; "
                "partial reply preserved in chat_interrupted_turn.json."
            ),
        )
    ]


def test_wake_without_stale_snapshot_sets_no_flag(tmp_path: Path) -> None:
    (tmp_path / "c1").mkdir()
    runtime = _RecordingRuntime()
    p = _make_plugin(tmp_path, runtime=runtime)

    p.on_waking(_wake_context())

    assert "interrupted_turn" not in p._state
    assert not (tmp_path / "c1" / "chat_interrupted_turn.json").exists()
    assert runtime.system_notes == []


def test_prompt_block_warns_about_interrupted_turn(tmp_path: Path) -> None:
    chat_dir = tmp_path / "c1"
    chat_dir.mkdir()
    (chat_dir / "chat_active_turn.json").write_text(
        json.dumps(
            {
                "session_number": 1,
                "turn_number": 5,
                "user_message": "Approved - go ahead",
                "message_text": "partial",
                "updated_at": 9.0,
            }
        )
    )
    p = _make_plugin(tmp_path)
    p.on_waking(_wake_context())

    block = p.prompt_block()
    assert block is not None
    assert "interrupted mid-flight" in block
    assert "chat_interrupted_turn.json" in block

    compact = p.prompt_block(compact=True)
    assert compact is not None
    assert "interrupted" in compact


def test_turn_end_clears_interrupted_flag(tmp_path: Path) -> None:
    chat_dir = tmp_path / "c1"
    chat_dir.mkdir()
    (chat_dir / "chat_active_turn.json").write_text(
        json.dumps({"turn_number": 5, "user_message": "hi", "updated_at": 9.0})
    )
    p = _make_plugin(tmp_path)
    p.on_waking(_wake_context())
    assert "interrupted_turn" in p._state

    p.on_turn_end(
        TurnInfo(
            chat_id="c1",
            session_id="s1",
            session_number=1,
            turn_number=6,
            updated_at=30.0,
            last_stop_reason="completed",
            user_message="next",
            reply="done",
        )
    )
    assert "interrupted_turn" not in p._state


def test_prompt_block_captures_stale_snapshot_without_wake(tmp_path: Path) -> None:
    """A leftover snapshot must be caught even when no wake event fires."""
    chat_dir = tmp_path / "c1"
    chat_dir.mkdir()
    (chat_dir / "chat_active_turn.json").write_text(
        json.dumps(
            {
                "session_number": 1,
                "turn_number": 7,
                "user_message": "Approve, please proceed",
                "message_text": "partial work",
                "updated_at": 9.0,
            }
        )
    )
    runtime = _RecordingRuntime()
    p = _make_plugin(tmp_path, runtime=runtime)
    # Seed last_turn_at so prompt_block does not early-return.
    p._state["last_turn_at"] = 8.0

    block = p.prompt_block()

    assert block is not None
    assert "interrupted mid-flight" in block
    assert (chat_dir / "chat_interrupted_turn.json").exists()
    assert not (chat_dir / "chat_active_turn.json").exists()
    assert len(runtime.system_notes) == 1


def test_prompt_block_does_not_capture_live_turn_snapshot(tmp_path: Path) -> None:
    """During an in-flight turn the snapshot is live, not stale."""
    chat_dir = tmp_path / "c1"
    chat_dir.mkdir()
    runtime = _RecordingRuntime()
    p = _make_plugin(tmp_path, runtime=runtime)
    p._state["last_turn_at"] = 8.0
    p.on_partial(
        PartialTurn(
            chat_id="c1",
            session_number=1,
            turn_number=8,
            user_message="hi",
            message_text="streaming now",
            updated_at=10.0,
        )
    )

    p.prompt_block()

    assert (chat_dir / "chat_active_turn.json").exists()
    assert not (chat_dir / "chat_interrupted_turn.json").exists()
    assert "interrupted_turn" not in p._state
    assert runtime.system_notes == []
