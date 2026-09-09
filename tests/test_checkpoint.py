"""Tests for the checkpoint/breadcrumb plugin."""

import json
import time
from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.models import PartialTurn
from diploid_agent.plugins.base import TurnInfo

from diploid_plugins.checkpoint.checkpoint import CheckpointPlugin


def _make_plugin(tmp_path: Path) -> CheckpointPlugin:
    return CheckpointPlugin(
        PluginConfig(
            name="checkpoint",
            enabled=True,
            state_file="chat_checkpoint_state.json",
        ),
        "c1",
        tmp_path,
    )


def _partial(
    *,
    turn_number: int = 2,
    message_text: str = "msg",
    thought_text: str = "thinking",
    current_intent: str = "design checkpoint",
    last_side_effect: str = "",
    side_effects: list[dict[str, str]] | None = None,
    updated_at: float | None = None,
) -> PartialTurn:
    return PartialTurn(
        chat_id="c1",
        session_number=1,
        turn_number=turn_number,
        user_message="go",
        message_text=message_text,
        thought_text=thought_text,
        updated_at=updated_at or time.time(),
        current_intent=current_intent,
        last_side_effect=last_side_effect,
        last_side_effect_at=updated_at or time.time(),
        side_effects=side_effects or [],
    )


def test_on_partial_writes_active_checkpoint_and_breadcrumb(tmp_path: Path) -> None:
    (tmp_path / "c1").mkdir()
    p = _make_plugin(tmp_path)
    p.on_partial(_partial())

    active = json.loads((tmp_path / "c1" / "chat_active_checkpoint.json").read_text())
    assert active["turn_number"] == 2
    assert active["current_intent"] == "design checkpoint"

    crumbs = (tmp_path / "c1" / "chat_turn_breadcrumbs.jsonl").read_text().splitlines()
    assert len(crumbs) == 1
    data = json.loads(crumbs[0])
    assert data["type"] == "partial"
    assert data["current_intent"] == "design checkpoint"


def test_side_effect_change_writes_immediately(tmp_path: Path) -> None:
    (tmp_path / "c1").mkdir()
    p = _make_plugin(tmp_path)
    p.on_partial(_partial(message_text="msg 1"))
    p.on_partial(_partial(message_text="msg 1", last_side_effect="exec ls (running)"))

    crumbs = (tmp_path / "c1" / "chat_turn_breadcrumbs.jsonl").read_text().splitlines()
    assert len(crumbs) == 2
    assert json.loads(crumbs[1])["type"] == "tool_update"
    assert json.loads(crumbs[1])["last_side_effect"] == "exec ls (running)"


def test_throttle_suppresses_redundant_partials(tmp_path: Path) -> None:
    (tmp_path / "c1").mkdir()
    p = _make_plugin(tmp_path)
    t = time.time()
    for i in range(3):
        # No significant change between these partials; only the first should write.
        p.on_partial(_partial(message_text="msg", updated_at=t + i * 0.1))

    crumbs = (tmp_path / "c1" / "chat_turn_breadcrumbs.jsonl").read_text().splitlines()
    # Only the first partial is written within the 0.5 s throttle window.
    assert len(crumbs) == 1


def test_after_turn_rotates_and_clears_active(tmp_path: Path) -> None:
    (tmp_path / "c1").mkdir()
    p = _make_plugin(tmp_path)
    p.on_partial(_partial())
    p.after_turn(
        TurnInfo(
            chat_id="c1",
            session_id="s1",
            session_number=1,
            turn_number=2,
            updated_at=time.time(),
            last_stop_reason="completed",
            user_message="go",
            reply="done",
        )
    )

    assert not (tmp_path / "c1" / "chat_active_checkpoint.json").exists()
    assert not (tmp_path / "c1" / "chat_turn_breadcrumbs.jsonl").exists()
    assert (tmp_path / "c1" / "chat_last_turn_breadcrumbs.jsonl").exists()
    last = (tmp_path / "c1" / "chat_last_turn_breadcrumbs.jsonl").read_text().splitlines()
    assert len(last) == 2  # original + completed marker
    assert json.loads(last[-1])["type"] == "completed"


def test_prompt_block_renders_active_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "c1").mkdir()
    p = _make_plugin(tmp_path)
    p.on_partial(
        _partial(
            current_intent="delete stale dirs",
            last_side_effect="exec rmtree (running)",
            side_effects=[{"title": "exec rmtree", "status": "running", "at": 1.0}],
        )
    )

    block = p.prompt_block()
    assert block is not None
    assert "delete stale dirs" in block
    assert "exec rmtree (running)" in block
    assert "turn 2" in block


def test_prompt_block_returns_none_after_completion(tmp_path: Path) -> None:
    (tmp_path / "c1").mkdir()
    p = _make_plugin(tmp_path)
    p.on_partial(_partial())
    p.after_turn(
        TurnInfo(
            chat_id="c1",
            session_id="s1",
            session_number=1,
            turn_number=2,
            updated_at=time.time(),
            last_stop_reason="completed",
            user_message="go",
            reply="done",
        )
    )

    assert p.prompt_block() is None


def test_breadcrumbs_trimming(tmp_path: Path) -> None:
    (tmp_path / "c1").mkdir()
    p = CheckpointPlugin(
        PluginConfig(
            name="checkpoint",
            enabled=True,
            state_file="chat_checkpoint_state.json",
            config={"max_breadcrumbs": 3},
        ),
        "c1",
        tmp_path,
    )
    t = time.time()
    for i in range(5):
        p.on_partial(_partial(message_text=f"msg {i}", updated_at=t + i * 2.0))

    crumbs = (tmp_path / "c1" / "chat_turn_breadcrumbs.jsonl").read_text().splitlines()
    assert len(crumbs) == 3
    assert "msg 4" in json.loads(crumbs[-1])["message_snippet"]
