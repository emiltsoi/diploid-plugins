"""Tests for the self_state continuity plugin."""

from __future__ import annotations

from pathlib import Path

import pytest
from diploid_agent.config import PluginConfig
from diploid_agent.models import SessionRecord
from diploid_agent.plugins.base import WakeContext
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


def _wake_context(record: SessionRecord | None = None) -> WakeContext:
    return WakeContext(
        chat_id="chat-1",
        record=record or _record(),
        now=0.0,
        instance_id="i1",
        instance_started_at=0.0,
        previous_turn_at=None,
        pending_dispatches=[],
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


def test_prompt_block_empty_no_reminder_returns_none(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._remind = False
    assert p.prompt_block() is None


def test_prompt_block_on_wake_shows_reminder_for_empty_state(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    block = p.prompt_block()
    assert block is not None
    assert "## State I am resuming from" in block
    assert "Update this with a `<self_state>` block" in block


def test_prompt_block_on_wake_includes_note_and_reminder(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I was explaining restarts.")
    block = p.prompt_block()
    assert block is not None
    assert "## State I am resuming from" in block
    assert "I was explaining restarts." in block
    assert "Update this with a `<self_state>` block" in block


def test_prompt_block_follow_up_includes_note_without_reminder(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I was explaining restarts.")
    # First prompt consumes the wake reminder.
    p.prompt_block()
    follow_up = p.prompt_block()
    assert follow_up is not None
    assert "I was explaining restarts." in follow_up
    assert "Update this with a `<self_state>` block" not in follow_up


def test_prompt_block_caps_at_max_chars(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    long_note = "word " * 500
    p._save_state(long_note)
    block = p.prompt_block(max_chars=300)
    assert block is not None
    assert len(block) <= 300
    assert "## State I am resuming from" in block
    assert "Update this with a `<self_state>` block" in block


def test_on_waking_re_enables_reminder(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am focused.")
    # First prompt consumes the wake reminder.
    p.prompt_block()
    follow_up = p.prompt_block()
    assert "Update this with a `<self_state>` block" not in follow_up

    p.on_waking(_wake_context())
    reawakened = p.prompt_block()
    assert reawakened is not None
    assert "I am focused." in reawakened
    assert "Update this with a `<self_state>` block" in reawakened


def test_prompt_block_changed_detects_state_change(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I was explaining restarts.")
    # The wake reminder is pending, so the block is considered changed.
    assert p.prompt_block_changed(0.0) is True
    p._remind = False
    # The state file has a real mtime and it is newer than the reference time.
    assert p.prompt_block_changed(0.0) is True
    # A future reference time means the file has not changed.
    assert p.prompt_block_changed(9_999_999_999.0) is False


def test_prompt_block_changed_no_file_returns_none(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._remind = False
    assert p.prompt_block_changed(0.0) is None


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


def test_before_record_turn_leaves_empty_state_empty_when_no_block(tmp_path: Path) -> None:
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
    assert not p._load_state()
    state_path = tmp_path / "chat-1" / "chat_self_state.md"
    assert not state_path.exists()


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


def _partial(message_text: str):
    from diploid_agent.models import PartialTurn

    return PartialTurn(
        chat_id="chat-1",
        session_number=1,
        turn_number=3,
        user_message="hi",
        message_text=message_text,
        updated_at=1.0,
    )


def test_on_partial_saves_complete_self_state_block(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p.on_partial(_partial("Working on it.\n<self_state>I am mid-task.</self_state>"))
    assert p._load_state() == "I am mid-task."


def test_on_partial_ignores_incomplete_block(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p.on_partial(_partial("Working.\n<self_state>I am still writi"))
    assert p._load_state() == ""
    state_path = tmp_path / "chat-1" / "chat_self_state.md"
    assert not state_path.exists()


def test_on_partial_does_not_rewrite_unchanged_note(tmp_path: Path, monkeypatch) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p.on_partial(_partial("<self_state>I am focused.</self_state>"))
    calls = []
    monkeypatch.setattr(p, "_save_state", lambda text: calls.append(text))
    p.on_partial(_partial("<self_state>I am focused.</self_state> more text"))
    assert calls == []


def test_on_partial_saves_latest_block_when_state_changes(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p.on_partial(_partial("<self_state>I am first.</self_state>"))
    p.on_partial(
        _partial(
            "<self_state>I am first.</self_state> then <self_state>I am second.</self_state>"
        )
    )
    assert p._load_state() == "I am second."


def test_on_waking_resets_streamed_note_cache(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p.on_partial(_partial("<self_state>cached</self_state>"))
    assert p._last_streamed_note == "cached"
    p.on_waking(_wake_context())
    assert p._last_streamed_note is None


@pytest.mark.parametrize(
    "note",
    [
        "I am warm.",
        "I'm warm.",
        "I am.",
        "My chest is warm.",
        "We are working.",
        "We're working.",
        "Our thread holds.",
        "> I am here.",
        "- I am here.",
        "  \t I am here.",
        "i am lowercase.",
    ],
)
def test_is_first_person_accepts_first_person_variants(note: str) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", Path("/nonexistent"))
    assert p._is_first_person(note) is True


@pytest.mark.parametrize(
    "note",
    [
        "The work is done.",
        "This is the state.",
        "Issue A is fixed.",
        "User asked about continuity.",
        "He is here.",
        "It is warm.",
        "They are watching.",
        "first",  # test-data regression guard
        "second",
    ],
)
def test_is_first_person_rejects_non_first_person(note: str) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", Path("/nonexistent"))
    assert p._is_first_person(note) is False


def test_before_record_turn_rejects_third_person_and_keeps_prior_state(
    tmp_path: Path,
) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am focused.")
    reply = "Visible reply.\n<self_state>The work is done.</self_state>"
    ctx = RecordTurnContext(
        chat_id="chat-1",
        record=_record(),
        turn_number=1,
        reply=reply,
    )
    result = p.before_record_turn(ctx)
    assert result.reply == "Visible reply."
    assert p._load_state() == "I am focused."
    assert p._rejected is True
    assert p._remind is True


def test_before_record_turn_accepts_first_person_plural(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    reply = "Done.\n<self_state>We are holding the thread.</self_state>"
    ctx = RecordTurnContext(
        chat_id="chat-1",
        record=_record(),
        turn_number=1,
        reply=reply,
    )
    p.before_record_turn(ctx)
    assert p._load_state() == "We are holding the thread."
    assert p._rejected is False


def test_on_partial_rejects_non_first_person(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p.on_partial(_partial("<self_state>The work is done.</self_state>"))
    assert p._load_state() == ""
    assert p._rejected is True
    assert p._remind is True


def test_prompt_block_warns_after_rejection(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p.on_partial(_partial("<self_state>The work is done.</self_state>"))
    block = p.prompt_block()
    assert block is not None
    assert "not in first person" in block
    assert "I am..." in block


def test_prompt_block_shows_rejected_reminder_under_max_chars(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p.on_partial(_partial("<self_state>The work is done.</self_state>"))
    block = p.prompt_block(max_chars=200)
    assert block is not None
    assert len(block) <= 200
    assert "not in first person" in block
