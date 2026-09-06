"""Tests for the self_state continuity plugin."""

from __future__ import annotations

from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.models import SessionRecord
from diploid_agent.plugins.contexts import RecordTurnContext

from diploid_plugins.self_state.self_state import SelfStatePlugin


def _make_config() -> PluginConfig:
    return PluginConfig(
        name="self_state",
        enabled=True,
        state_file="chat_self_state.md",
        prompt_slot="persona_state",
        prompt_order=90,
        max_prompt_chars=512,
    )


def _record(last_stop_reason: str | None = "completed") -> SessionRecord:
    return SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="s1",
        model="m1",
        persona="test",
        cwd="/tmp",
        created_at=0.0,
        updated_at=0.0,
        last_stop_reason=last_stop_reason,
    )


def test_extract_self_state_present(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = "Hello.\n\n<self_state>I am warm.</self_state>"
    stripped, note = p._extract_self_state(reply)
    assert note == "I am warm."
    assert "<self_state>" not in stripped
    assert stripped == "Hello."


def test_extract_self_state_uses_last_block(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = "Start. <self_state>old</self_state> middle <self_state>new</self_state> end"
    stripped, note = p._extract_self_state(reply)
    assert note == "new"
    assert "<self_state>" not in stripped
    assert stripped == "Start.  middle  end"


def test_extract_self_state_missing(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = "Just a normal reply."
    stripped, note = p._extract_self_state(reply)
    assert note is None
    assert stripped == reply


def test_extract_self_state_case_insensitive_and_dotall(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = "Reply.\n<SELF_STATE>\nI\nam\ncurious\n</SELF_STATE>\n"
    stripped, note = p._extract_self_state(reply)
    assert note == "I\nam\ncurious"
    assert "<SELF_STATE>" not in stripped


def test_extract_self_state_ignores_stray_open_mentions(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = (
        "I put state in a `<self_state>` block, and `<self_state>` again.\n"
        "<self_state>I am focused.</self_state>"
    )
    stripped, note = p._extract_self_state(reply)
    assert note == "I am focused."
    assert "block to save" not in note
    assert "I put state in a `<self_state>` block" in stripped
    assert "<self_state>I am focused.</self_state>" not in stripped


def test_extract_self_state_stray_open_after_block(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = "<self_state>I am focused.</self_state> Then I mention `<self_state>` in prose."
    stripped, note = p._extract_self_state(reply)
    assert note == "I am focused."
    assert "in prose" in stripped


def test_extract_self_state_unmatched_close(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = "Text with a stray `</self_state>` mention."
    stripped, note = p._extract_self_state(reply)
    assert note is None
    assert stripped == reply


def test_fallback_note_completed(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    note = p._fallback_note(_record("completed"), "short reply")
    assert "I just replied" in note
    assert "short reply" in note
    assert len(note) <= 200


def test_fallback_note_partial(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    for reason in ("timeout", "stopped", "cancelled"):
        note = p._fallback_note(_record(reason), "partial reply")
        assert "in the middle" in note
        assert len(note) <= 200


def test_prompt_block_returns_saved_note_and_instruction(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I was explaining restarts.")
    block = p.prompt_block()
    assert block is not None
    assert "## State I am resuming from" in block
    assert "I was explaining restarts." in block
    assert "<self_state>" in block
    assert "Example:" in block


def test_prompt_block_empty_returns_none(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    assert p.prompt_block() is None


def test_prompt_block_caps_at_max_chars(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    long_note = "word " * 500
    p._save_state(long_note)
    block = p.prompt_block(max_chars=300)
    assert block is not None
    assert len(block) <= 300
    assert "## State I am resuming from" in block
    assert "At the end of your reply" in block


def test_before_record_turn_strips_and_saves(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = "Visible reply.\n<self_state>I am focused.</self_state>"
    ctx = RecordTurnContext(
        chat_id="chat-1",
        record=_record(),
        turn_number=1,
        reply=reply,
    )
    result = p.before_record_turn(ctx)
    assert result.reply == "Visible reply."
    assert p._load_state() == "I am focused."
    state_path = tmp_path / "chat-1" / "chat_self_state.md"
    assert state_path.exists()
    assert state_path.read_text(encoding="utf-8") == "I am focused."


def test_before_record_turn_saves_fallback_when_no_block(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = "Plain reply."
    ctx = RecordTurnContext(
        chat_id="chat-1",
        record=_record(),
        turn_number=1,
        reply=reply,
    )
    result = p.before_record_turn(ctx)
    assert result.reply == reply
    assert p._load_state()
    assert "I just replied" in p._load_state()


def test_before_record_turn_preserves_existing_state_when_no_block(
    tmp_path: Path,
) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am focused on the continuity work.")
    reply = "Plain reply without a self_state block."
    ctx = RecordTurnContext(
        chat_id="chat-1",
        record=_record(),
        turn_number=2,
        reply=reply,
    )
    result = p.before_record_turn(ctx)
    assert result.reply == reply
    assert p._load_state() == "I am focused on the continuity work."
