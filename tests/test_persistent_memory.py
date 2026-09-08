"""Tests for the persistent-memory plugin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    MemoryConfig,
    PersonaConfig,
    PluginConfig,
)
from diploid_agent.models import ChatResult, PartialTurn
from diploid_agent.plugins.base import SleepContext, TurnInfo
from diploid_agent.plugins.contexts import TurnStartContext

from diploid_plugins.persistent_memory.persistent_memory import PersistentMemoryPlugin


class FakeEngine:
    """Engine that returns a fixed reply."""

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply or ""
        self.calls: list[dict[str, Any]] = []

    def prompt(self, request: Any) -> Any:
        from diploid_agent.engine.base import TurnResult

        self.calls.append({"prompt": request.prompt, "model": request.model})
        return TurnResult(reply=self.reply)


class FakeRuntime:
    """Minimal runtime that satisfies the new recall/promote surface."""

    def __init__(self, tmp: Path, recall_reply: str = "") -> None:
        self.sessions_root = tmp
        self.config = Config(
            diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
            persona=PersonaConfig(
                name="test-pilot",
                profile_root=tmp / "persona",
            ),
            harness=HarnessConfig(
                sessions_root=tmp,
                session_store_path=tmp / "sessions.jsonl",
                memory=MemoryConfig(backend="file"),
            ),
        )
        self.recall_queries: list[tuple[str, str, list[str] | None, int | None]] = []
        self.recall_reply = recall_reply
        self.promoted: list[tuple[str, str]] = []
        self.engine = FakeEngine()

    def recall(
        self,
        chat_id: str,
        query: str,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.recall_queries.append((chat_id, query, tags, max_tokens))
        return ChatResult(reply=self.recall_reply)

    def promote(self, chat_id: str, fact: str) -> ChatResult:
        self.promoted.append((chat_id, fact))
        return ChatResult(reply="Promoted.")

    def call_engine_unlocked(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)


def _make_config(tmp_path: Path, max_prompt_chars: int = 1024, **overrides: Any) -> PluginConfig:
    return PluginConfig(
        name="persistent_memory",
        enabled=True,
        module="diploid_plugins.persistent_memory",
        prompt_slot="persistent_memory",
        first_prompt_only=False,
        prompt_order=32,
        state_file="chat_persistent_memory.json",
        max_prompt_chars=max_prompt_chars,
        config={"auto_promote": True, "auto_recall": True, **overrides},
    )


def _turn_start(user_message: str) -> TurnStartContext:
    return TurnStartContext(
        chat_id="chat-1",
        user_message=user_message,
        model=None,
        record=None,
    )


def _write_transcript(
    sessions_root: Path,
    chat_id: str,
    content_pairs: list[tuple[str, str]],
) -> None:
    chat_dir = sessions_root / chat_id.replace("/", "_")
    chat_dir.mkdir(parents=True, exist_ok=True)
    path = chat_dir / "chat_transcript.jsonl"
    with path.open("w") as f:
        for role, content in content_pairs:
            f.write(json.dumps({"role": role, "content": content}) + "\n")


def test_default_prompt_block_is_none(tmp_path: Path) -> None:
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path)
    assert plugin.prompt_block() is None


def test_non_memory_question_does_nothing(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    result = plugin.before_turn(_turn_start("hello"))
    assert result is None
    assert not runtime.recall_queries
    assert plugin.prompt_block() is None


def test_recall_skips_when_short_term_has_answer(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    _write_transcript(
        tmp_path,
        "chat-1",
        [("user", "What about the migration?"), ("assistant", "We say the migration is fine.")],
    )
    result = plugin.before_turn(_turn_start("What did we say about the migration?"))
    assert result is None
    assert not runtime.recall_queries
    assert plugin.prompt_block() is None


def test_recall_injected_when_not_in_short_term(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, recall_reply="- We decided to use the new schema.")
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    result = plugin.before_turn(_turn_start("What did we decide about the schema?"))
    assert result is None
    assert len(runtime.recall_queries) == 1
    assert runtime.recall_queries[0][1] == "What did we decide about the schema?"
    block = plugin.prompt_block()
    assert block is not None
    assert "## Persistent memory" in block
    assert "new schema" in block


def test_recall_strips_backend_prefix(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        tmp_path,
        recall_reply="Memory from previous turns:\n\n- We use Postgres.",
    )
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    plugin.before_turn(_turn_start("What database do we use?"))
    block = plugin.prompt_block()
    assert block is not None
    assert "Memory from previous turns" not in block
    assert "Postgres" in block


def test_prompt_block_respects_max_chars(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, recall_reply="- " + "a" * 2000)
    config = _make_config(tmp_path, max_prompt_chars=100)
    plugin = PersistentMemoryPlugin(config, "chat-1", tmp_path, runtime=runtime)
    plugin.before_turn(_turn_start("What did we do?"))
    block = plugin.prompt_block()
    assert block is not None
    assert len(block) <= 100


def test_auto_promote_from_memory_block(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    turn = TurnInfo(
        chat_id="chat-1",
        session_id="s1",
        session_number=1,
        turn_number=1,
        updated_at=0.0,
        last_stop_reason=None,
        user_message="ok",
        reply="We should track this.\n\n```memory\nWe track everything in one repo.\n```",
        notice=None,
    )
    plugin.after_turn(turn)
    assert len(runtime.promoted) == 1
    assert runtime.promoted[0] == ("chat-1", "We track everything in one repo.")


def _partial(message_text: str) -> PartialTurn:
    return PartialTurn(
        chat_id="chat-1",
        session_number=1,
        turn_number=1,
        user_message="ok",
        message_text=message_text,
        updated_at=1.0,
    )


def test_on_partial_promotes_complete_memory_block(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    plugin.before_turn(_turn_start("hi"))
    plugin.on_partial(_partial("Working.\n```memory\nWe prefer warm restarts.\n```"))
    assert runtime.promoted == [("chat-1", "We prefer warm restarts.")]


def test_on_partial_ignores_incomplete_memory_block(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    plugin.before_turn(_turn_start("hi"))
    plugin.on_partial(_partial("Working.\n```memory\nWe prefer warm"))
    assert not runtime.promoted


def test_after_turn_does_not_double_promote_partial_fact(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    plugin.before_turn(_turn_start("hi"))
    reply = "Done.\n```memory\nWe prefer warm restarts.\n```"
    plugin.on_partial(_partial(reply))
    plugin.after_turn(
        TurnInfo(
            chat_id="chat-1",
            session_id="s1",
            session_number=1,
            turn_number=1,
            updated_at=1.0,
            last_stop_reason=None,
            user_message="ok",
            reply=reply,
            notice=None,
        )
    )
    assert runtime.promoted == [("chat-1", "We prefer warm restarts.")]


def test_auto_promote_disabled(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    plugin = PersistentMemoryPlugin(
        _make_config(tmp_path, auto_promote=False), "chat-1", tmp_path, runtime=runtime
    )
    turn = TurnInfo(
        chat_id="chat-1",
        session_id="s1",
        session_number=1,
        turn_number=1,
        updated_at=0.0,
        last_stop_reason=None,
        user_message="ok",
        reply="```memory\nImportant fact.\n```",
        notice=None,
    )
    plugin.after_turn(turn)
    assert not runtime.promoted


def test_manual_promote_event(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    reply = plugin.event(event="promote", fact="We use blue logos.")
    assert "Promoted" in reply
    assert runtime.promoted == [("chat-1", "We use blue logos.")]


def test_manual_promote_event_raw_args(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    plugin.event(event="promote", raw_args="  We use blue logos.  ")
    assert runtime.promoted == [("chat-1", "We use blue logos.")]


def test_event_state_and_clear(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path, recall_reply="- Fact.")
    plugin = PersistentMemoryPlugin(_make_config(tmp_path), "chat-1", tmp_path, runtime=runtime)
    plugin.before_turn(_turn_start("What did we do?"))
    assert "Persistent memory" in plugin.event(event="state")
    assert "clear" in plugin.event(event="clear")
    assert plugin.prompt_block() is None


def test_sleep_summary_promotes_top_facts(tmp_path: Path) -> None:
    runtime = FakeRuntime(tmp_path)
    runtime.engine.reply = "- We like tea.\n- We prefer morning standups."
    plugin = PersistentMemoryPlugin(
        _make_config(tmp_path, auto_summarize_on_sleep=True),
        "chat-1",
        tmp_path,
        runtime=runtime,
    )
    _write_transcript(
        tmp_path,
        "chat-1",
        [
            ("user", "I like tea."),
            ("assistant", "Noted."),
            ("user", "We prefer morning standups."),
            ("assistant", "Sure."),
        ],
    )
    from diploid_agent.models import SessionRecord

    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="s1",
        model="swe-1-7",
        persona="test-pilot",
        cwd=str(tmp_path),
        created_at=0.0,
        updated_at=0.0,
    )
    plugin.on_sleeping(
        SleepContext(chat_id="chat-1", record=record, reason="idle", now=0.0, instance_id="i1")
    )
    assert runtime.promoted == [
        ("chat-1", "We like tea."),
        ("chat-1", "We prefer morning standups."),
    ]
