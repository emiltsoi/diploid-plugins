"""Tests for the self_state continuity plugin."""

from __future__ import annotations

import json
import re
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


def test_before_record_turn_self_state_only_gets_fallback(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    ctx = RecordTurnContext(
        chat_id="chat-1",
        record=_record(),
        turn_number=1,
        reply="<self_state>I am marking the restart test.</self_state>",
    )
    result = p.before_record_turn(ctx)
    assert result.reply == "(self-state marker set)"
    assert p._load_state() == "I am marking the restart test."


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
        _partial("<self_state>I am first.</self_state> then <self_state>I am second.</self_state>")
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


def test_next_self_carried_forward_when_new_note_lacks_it(tmp_path: Path) -> None:
    """A mid-turn update that forgets `## next-self` must not drop the handoff."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am warm.\n\n## next-self\nI go down building the felt layer; continue it.")
    p._maybe_save_state("I am still warm, now also curious.")
    text = p._load_state()
    assert "I am still warm, now also curious." in text
    assert "## next-self (stale — carried from previous note)" in text
    assert "I go down building the felt layer; continue it." in text


def test_next_self_fresh_section_replaces_old(tmp_path: Path) -> None:
    """A note carrying its own `## next-self` is saved untouched."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am warm.\n\n## next-self\nOld handoff.")
    p._maybe_save_state("I am changed.\n\n## next-self\nFresh handoff.")
    text = p._load_state()
    assert "Fresh handoff." in text
    assert "Old handoff." not in text
    assert "stale" not in text


def test_next_self_no_carry_when_no_prior_handoff(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._maybe_save_state("I am new and have no handoff yet.")
    assert p._load_state() == "I am new and have no handoff yet."


def test_next_self_carry_does_not_stack_stale_markers(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am warm.\n\n## next-self\nThe one handoff.")
    p._maybe_save_state("I am busy.")
    p._maybe_save_state("I am busier.")
    text = p._load_state()
    assert text.count("(stale") == 1
    assert text.count("The one handoff.") == 1


def test_prompt_block_nags_when_note_lacks_next_self(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am focused on the work.")
    block = p.prompt_block()  # _remind is set on construction
    assert block is not None
    assert "## next-self" in block
    assert "No `## next-self` handoff found" in block


def test_prompt_block_no_nag_when_next_self_present(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am focused.\n\n## next-self\nPick up the felt work.")
    block = p.prompt_block()
    assert block is not None
    assert "No `## next-self` handoff found" not in block


def test_prompt_block_nag_absent_without_reminder(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am focused on the work.")
    p.prompt_block()  # consumes the wake reminder
    follow_up = p.prompt_block()
    assert follow_up is not None
    assert "No `## next-self` handoff found" not in follow_up


def test_next_self_body_stops_at_next_heading(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    note = "I am here.\n\n## next-self\nThe handoff.\n\n## other\nNot part of it."
    assert p._next_self_body(note) == "The handoff."


def test_carry_forward_excludes_sections_after_next_self(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am warm.\n\n## next-self\nThe handoff.\n\n## todo\nUnrelated list.")
    p._maybe_save_state("I am busy.")
    text = p._load_state()
    assert "The handoff." in text
    assert "Unrelated list." not in text


def _make_bounded_config() -> PluginConfig:
    cfg = _make_config()
    cfg.config["mid_turn_max_chars"] = 60
    return cfg


def test_mid_turn_carried_when_new_note_lacks_it(tmp_path: Path) -> None:
    """A block save replaces the whole file, so `## mid-turn` is carried."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am warm.\n\n## mid-turn\n- probing ctl socket\n- tests green")
    p._maybe_save_state("I am still warm.")
    text = p._load_state()
    assert "I am still warm." in text
    assert "## mid-turn" in text
    assert "- probing ctl socket" in text
    assert "- tests green" in text


def test_mid_turn_fresh_section_replaces_old(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am warm.\n\n## mid-turn\n- old crumb")
    p._maybe_save_state("I am new.\n\n## mid-turn\n- fresh crumb")
    text = p._load_state()
    assert "- fresh crumb" in text
    assert "- old crumb" not in text


def test_mid_turn_no_carry_when_no_prior_section(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am warm.")
    p._maybe_save_state("I am new.")
    assert "## mid-turn" not in p._load_state()


def test_mid_turn_carry_does_not_stack(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am warm.\n\n## mid-turn\n- one crumb")
    p._maybe_save_state("I am busy.")
    p._maybe_save_state("I am busier.")
    text = p._load_state()
    assert text.count("## mid-turn") == 1
    assert text.count("- one crumb") == 1


def test_mid_turn_carried_before_stale_next_self(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am warm.\n\n## mid-turn\n- crumb\n\n## next-self\nThe handoff.")
    p._maybe_save_state("I am busy.")
    text = p._load_state()
    assert "- crumb" in text
    assert "The handoff." in text
    assert text.index("## mid-turn") < text.index("## next-self")


def test_mid_turn_section_stops_at_next_heading(tmp_path: Path) -> None:
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    note = "I am here.\n\n## mid-turn\n- crumb\n\n## next-self\nThe handoff."
    assert p._mid_turn_section(note) == "## mid-turn\n- crumb"


def test_mid_turn_direct_edit_survives_block_save(tmp_path: Path) -> None:
    """The blessed flow: scratch written to the file mid-turn is kept."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._save_state("I am starting the batch.")
    state_path = tmp_path / "chat-1" / "chat_self_state.md"
    state_path.write_text(
        state_path.read_text(encoding="utf-8") + "\n\n## mid-turn\n- conftest scoped\n- ruff clean",
        encoding="utf-8",
    )
    p.on_partial(_partial("Update.\n<self_state>I am mid-batch.</self_state>"))
    text = p._load_state()
    assert "I am mid-batch." in text
    assert "- conftest scoped" in text
    assert "- ruff clean" in text


def test_mid_turn_capped_to_budget_on_save(tmp_path: Path) -> None:
    """An over-budget section keeps the newest lines within the bound."""
    p = SelfStatePlugin(_make_bounded_config(), "chat-1", tmp_path)
    p._save_state(
        "I am warm.\n\n## mid-turn\n- oldest crumb line\n- middle crumb line\n- newest crumb line"
    )
    p._maybe_save_state("I am new.")
    text = p._load_state()
    section = p._mid_turn_section(text)
    assert section is not None
    assert len(section) <= 60
    assert "- newest crumb line" in section
    assert "- oldest crumb line" not in section


def test_mid_turn_oversized_line_dropped_whole_with_marker(tmp_path: Path) -> None:
    """A crumb longer than the budget is dropped whole — a tail fragment
    reads as corruption — and a marker line records the trim."""
    p = SelfStatePlugin(_make_bounded_config(), "chat-1", tmp_path)
    long_crumb = "- " + "x" * 120
    p._save_state(f"I am warm.\n\n## mid-turn\n- old\n{long_crumb}")
    p._maybe_save_state("I am new.")
    section = p._mid_turn_section(p._load_state())
    assert section is not None
    assert len(section) <= 60
    assert "x" not in section  # no tail fragment survives
    assert "trimmed" in section
    assert "- old" in section


def test_mid_turn_marker_counts_evicted_lines(tmp_path: Path) -> None:
    """The marker's count includes lines evicted to make room for it."""
    p = SelfStatePlugin(_make_bounded_config(), "chat-1", tmp_path)
    p._save_state(
        "I am warm.\n\n## mid-turn\n- oldest crumb line\n- middle crumb line\n- newest crumb line"
    )
    p._maybe_save_state("I am new.")
    section = p._mid_turn_section(p._load_state())
    assert section is not None
    assert len(section) <= 60
    assert "- newest crumb line" in section
    assert "- oldest crumb line" not in section
    assert "(2 earlier lines trimmed)" in section


def test_prompt_block_bounds_mid_turn_section(tmp_path: Path) -> None:
    """A direct over-budget write renders bounded before the next save."""
    p = SelfStatePlugin(_make_bounded_config(), "chat-1", tmp_path)
    p._save_state(
        "I am warm.\n\n## mid-turn\n- oldest crumb line\n- middle crumb line\n- newest crumb line"
    )
    block = p.prompt_block()
    assert block is not None
    assert "- newest crumb line" in block
    assert "- oldest crumb line" not in block
    # The file itself is untouched — the bound is render-time.
    assert "- oldest crumb line" in p._load_state()


def _read_ledger(chat_dir: Path) -> list[dict]:
    path = chat_dir / "chat_open_items.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_open_items_section_projects_to_ledger(tmp_path: Path) -> None:
    """`## open-items` bullets land in chat_open_items.jsonl with the schema row."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._maybe_save_state(
        "I am mid-arc.\n\n## open-items\n"
        "- land the producer · id: oi-producer · kind: work · checked_by: pytest roundtrip\n"
        "- verify hindsight tags_match live · kind: retro-delta · evidence: retro#1\n\n"
        "## next-self\nKeep going."
    )
    rows = _read_ledger(tmp_path / "chat-1")
    assert rows[0] == {"_schema": "open_items/v1"}
    first, second = rows[1], rows[2]
    assert first["id"] == "oi-producer"
    assert first["claim"] == "land the producer"
    assert first["status"] == "open"
    assert first["kind"] == "work"
    assert first["checked_by"] == "pytest roundtrip"
    assert second["kind"] == "retro-delta"
    assert second["evidence"] == "retro#1"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", first["since"])


def test_open_items_bullet_without_fields_gets_derived_id(tmp_path: Path) -> None:
    """A bare `- claim` bullet still rows, with a deterministic derived id."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    note = "I am mid-arc.\n\n## open-items\n- carry the torch"
    p._maybe_save_state(note)
    p._maybe_save_state(note)  # save twice: id must be stable
    rows = _read_ledger(tmp_path / "chat-1")
    assert len(rows) == 2  # schema + one item
    item = rows[1]
    assert item["claim"] == "carry the torch"
    assert item["id"].startswith("oi-")
    assert item["status"] == "open"


def test_open_items_since_survives_and_status_edits_apply(tmp_path: Path) -> None:
    """`since` is preserved by id across rewrites; status edits land."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._maybe_save_state(
        "I am mid-arc.\n\n## open-items\n- land the producer · id: oi-producer · since: 2026-10-17"
    )
    first_since = _read_ledger(tmp_path / "chat-1")[1]["since"]
    p._maybe_save_state(
        "I am still mid-arc.\n\n## open-items\n- land the producer · id: oi-producer · status: done"
    )
    rows = _read_ledger(tmp_path / "chat-1")
    assert rows[1]["since"] == first_since
    assert rows[1]["status"] == "done"


def test_open_items_since_preserved_when_field_dropped(tmp_path: Path) -> None:
    """Dropping an explicit `since:` on the next save keeps the prior value."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._maybe_save_state(
        "I am mid-arc.\n\n## open-items\n- land the producer · id: oi-producer · since: 2026-10-10"
    )
    p._maybe_save_state(
        "I am still mid-arc.\n\n## open-items\n- land the producer · id: oi-producer"
    )
    assert _read_ledger(tmp_path / "chat-1")[1]["since"] == "2026-10-10"


def test_open_items_absent_section_carries_ledger(tmp_path: Path) -> None:
    """A save without `## open-items` leaves the file alone and carries the
    section forward stale in the note — forgetfulness never discharges."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._maybe_save_state("I am mid-arc.\n\n## open-items\n- land the producer · id: oi-producer")
    before = (tmp_path / "chat-1" / "chat_open_items.jsonl").read_text()
    p._maybe_save_state("I am still warm.")
    assert (tmp_path / "chat-1" / "chat_open_items.jsonl").read_text() == before
    note = p._load_state()
    assert "## open-items (stale — carried from previous note)" in note
    assert "- land the producer · id: oi-producer" in note


def test_open_items_carried_section_reprojects(tmp_path: Path) -> None:
    """The stale-carried section still matches the heading, so a later save
    re-projects it — the ledger can't drift from the note."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._maybe_save_state("I am mid-arc.\n\n## open-items\n- land the producer · id: oi-producer")
    p._maybe_save_state("I am still warm.")
    p._maybe_save_state("I am warmer still.")
    rows = _read_ledger(tmp_path / "chat-1")
    assert rows[1]["id"] == "oi-producer"


def test_open_items_empty_section_clears_ledger(tmp_path: Path) -> None:
    """An explicitly empty `## open-items` heading clears to the schema row —
    authoring the heading is a deliberate discharge."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._maybe_save_state("I am mid-arc.\n\n## open-items\n- land the producer · id: oi-producer")
    p._maybe_save_state("I am clear.\n\n## open-items\n\n## next-self\nLedger empty.")
    rows = _read_ledger(tmp_path / "chat-1")
    assert rows == [{"_schema": "open_items/v1"}]


def test_open_items_malformed_lines_skipped(tmp_path: Path) -> None:
    """Non-bullet lines and empty bullets don't row; good lines still emit."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._maybe_save_state(
        "I am mid-arc.\n\n## open-items\n"
        "not a bullet\n-\n- real claim · status: open\n"
        "plain prose line"
    )
    rows = _read_ledger(tmp_path / "chat-1")
    assert len(rows) == 2
    assert rows[1]["claim"] == "real claim"


def test_open_items_claim_keeps_stray_separators(tmp_path: Path) -> None:
    """A `·` segment that isn't `key: value` folds back into the claim."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    p._maybe_save_state("I am mid-arc.\n\n## open-items\n- fix lint · then ship · id: oi-lint")
    item = _read_ledger(tmp_path / "chat-1")[1]
    assert item["claim"] == "fix lint · then ship"
    assert item["id"] == "oi-lint"


def test_open_items_emit_never_breaks_save(tmp_path: Path) -> None:
    """An unwritable ledger path still leaves the note saved."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    blocker = tmp_path / "chat-1" / "chat_open_items.jsonl"
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.mkdir()  # os.replace onto a directory raises OSError
    p._maybe_save_state("I am mid-arc.\n\n## open-items\n- land the producer · id: oi-producer")
    assert "I am mid-arc." in p._load_state()


def test_open_items_emit_via_record_turn_path(tmp_path: Path) -> None:
    """The public path — a `<self_state>` block in the reply — projects too."""
    p = SelfStatePlugin(_make_config(), "chat-1", tmp_path)
    context = RecordTurnContext(
        chat_id="chat-1",
        record=_record(),
        turn_number=1,
        reply="Done.\n\n<self_state>I am mid-arc.\n\n## open-items\n- carry on · id: oi-carry</self_state>",
    )
    p.before_record_turn(context)
    rows = _read_ledger(tmp_path / "chat-1")
    assert rows[1]["id"] == "oi-carry"
    assert "<self_state>" not in context.reply
