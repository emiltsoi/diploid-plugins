"""Tests for the bridge/surface handoff plugin."""

from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.models import SessionRecord
from diploid_agent.plugins.base import SessionArchiveContext, WakeContext

from diploid_plugins.bridge.bridge import BridgePlugin
from diploid_plugins.bridge.config import BridgeConfig
from diploid_plugins.bridge.manager import BridgeManager


def _make_plugin(tmp_path: Path, chat_id: str = "chat-1") -> BridgePlugin:
    return BridgePlugin(
        PluginConfig(
            name="bridge",
            enabled=True,
            state_file="chat_bridge_state.json",
            max_prompt_chars=512,
        ),
        chat_id,
        tmp_path,
    )


def _waking_context() -> WakeContext:
    return WakeContext(
        chat_id="chat-1",
        record=None,
        now=1.0,
        instance_id="i1",
        instance_started_at=1.0,
        previous_turn_at=None,
        pending_dispatches=[],
    )


def test_manager_writes_bridge_and_surface(tmp_path: Path) -> None:
    mgr = BridgeManager("chat-1", tmp_path, BridgeConfig())
    chat_dir = tmp_path / "chat-1"
    chat_dir.mkdir()
    (chat_dir / "chat_self_state.md").write_text(
        "I am warm and settled.\n\n## next-self\nPick up the thread."
    )
    (chat_dir / "chat_body_state.json").write_text('{"felt_warmth": 0.6, "felt_summary": "mild"}')
    (chat_dir / "chat_working_memory.json").write_text(
        '{"intent": "finish continuity wave", "open_questions": ["what next?"]}'
    )
    (chat_dir / "chat_TASKS.md").write_text(
        "# Tasks\n\n## Open\n- bridge plugin\n- tests\n\n## Done\n- plan"
    )
    (chat_dir / "chat_wake_state.json").write_text(
        '{"last_user_message": "go", "last_assistant_reply": "done"}'
    )

    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="s1",
        model="m1",
        persona="aurelia",
        cwd="/tmp",
        created_at=0.0,
        updated_at=1.0,
        turn_number=3,
    )
    mgr.write_handoff(record)

    bridge = (chat_dir / "chat_bridge.md").read_text()
    surface = (chat_dir / "chat_surface.md").read_text()

    assert "# BRIDGE" in bridge
    assert "I am warm and settled" in bridge
    assert "## Next-Self Handoff" in bridge
    assert "finish continuity wave" in bridge
    assert "bridge plugin" in bridge

    assert "# SURFACE" in surface
    assert "I am warm and settled" in surface
    assert "user asked: go" in surface
    assert "Body: mild" in surface


def test_prompt_block_returns_surface_and_rejects_stale(tmp_path: Path) -> None:
    mgr = BridgeManager("chat-1", tmp_path, BridgeConfig(surface_stale_hours=1.0))
    chat_dir = tmp_path / "chat-1"
    chat_dir.mkdir()
    (chat_dir / "chat_self_state.md").write_text("I am here.")
    (chat_dir / "chat_body_state.json").write_text('{"felt_warmth": 0.0}')
    (chat_dir / "chat_working_memory.json").write_text('{"intent": ""}')
    (chat_dir / "chat_TASKS.md").write_text("")
    (chat_dir / "chat_wake_state.json").write_text('{}')

    mgr.write_handoff(None)
    block = mgr.prompt_block()
    assert block is not None
    assert "## Bridge surface" in block
    assert "I am here" in block

    # Fake a stale mtime.
    import time

    path = chat_dir / "chat_surface.md"
    import os

    os.utime(path, (time.time() - 7200, time.time() - 7200))

    assert mgr.prompt_block() is None


def test_plugin_before_session_archive_writes_surface(tmp_path: Path) -> None:
    p = _make_plugin(tmp_path)
    chat_dir = tmp_path / "chat-1"
    chat_dir.mkdir()
    (chat_dir / "chat_self_state.md").write_text("I am settled.")
    (chat_dir / "chat_body_state.json").write_text('{"felt_warmth": 0.5}')
    (chat_dir / "chat_working_memory.json").write_text('{"intent": ""}')
    (chat_dir / "chat_TASKS.md").write_text("")
    (chat_dir / "chat_wake_state.json").write_text('{}')

    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="s1",
        model="m1",
        persona="aurelia",
        cwd="/tmp",
        created_at=0.0,
        updated_at=1.0,
        turn_number=2,
    )
    p.before_session_archive(SessionArchiveContext(chat_id="chat-1", old_record=record))

    assert (chat_dir / "chat_surface.md").exists()
    assert (chat_dir / "chat_bridge.md").exists()


def test_prompt_block_after_waking_uses_surface(tmp_path: Path) -> None:
    p = _make_plugin(tmp_path)
    chat_dir = tmp_path / "chat-1"
    chat_dir.mkdir()
    (chat_dir / "chat_self_state.md").write_text("I am resumed.")
    (chat_dir / "chat_body_state.json").write_text('{"felt_warmth": 0.6, "felt_summary": "mild"}')
    (chat_dir / "chat_working_memory.json").write_text('{"intent": "continue"}')
    (chat_dir / "chat_TASKS.md").write_text("")
    (chat_dir / "chat_wake_state.json").write_text('{}')

    p.before_session_archive(
        SessionArchiveContext(
            chat_id="chat-1",
            old_record=SessionRecord(
                chat_id="chat-1",
                session_number=1,
                session_id="s1",
                model="m1",
                persona="aurelia",
                cwd="/tmp",
                created_at=0.0,
                updated_at=1.0,
                turn_number=1,
            ),
        )
    )
    p.on_waking(_waking_context())

    block = p.prompt_block()
    assert block is not None
    assert "I am resumed" in block
    assert "Body: mild" in block
