"""Tests for the auto_continue plugin and the wake payload user-message hook."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    PluginConfig,
    Secrets,
)
from diploid_agent.models import ChatResult, SessionRecord, WakeEvent
from diploid_agent.plugins.base import TurnInfo
from diploid_agent.plugins.contexts import TurnStartContext
from diploid_agent.runtime.agent_runtime import AgentRuntime

from diploid_plugins.auto_continue.auto_continue import AutoContinuePlugin


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_config(tmp_path: Path) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=_fixture_root(),
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
            session_prune_enabled=False,
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def _timeout_record() -> SessionRecord:
    return SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="s1",
        model="m1",
        persona="test",
        cwd="/tmp",
        created_at=0.0,
        updated_at=0.0,
        last_stop_reason="timeout",
    )


class FakeEngine:
    def prompt(self, *a, **k):
        from diploid_agent.engine import TurnResult

        return TurnResult(reply="ok", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


class FakeWakeQueue:
    def __init__(self) -> None:
        self.events: list[WakeEvent] = []
        self._next_id = 0

    def enqueue(self, event: WakeEvent) -> WakeEvent:
        if not event.id:
            self._next_id += 1
            event.id = f"wake-{self._next_id}"
        self.events.append(event)
        return event

    def pending(
        self,
        chat_id: str | None = None,
        now: float | None = None,
    ) -> list[WakeEvent]:
        events = list(self.events)
        if chat_id is not None:
            events = [e for e in events if e.chat_id == chat_id]
        return events

    def complete(self, event_id: str) -> WakeEvent | None:
        for i, e in enumerate(self.events):
            if e.id == event_id:
                return self.events.pop(i)
        return None


class FakeContextBuilder:
    def __init__(self, triggers: list[str] | None = None) -> None:
        self.triggers = triggers or ["continue", "go on", "proceed", "resume"]

    def is_continuation_message(self, text: str) -> bool:
        normalized = re.sub(r"[^\w\s]", "", text).strip().lower()
        if not normalized:
            return False
        return normalized in {t.strip().lower() for t in self.triggers}


class FakeRuntime:
    def __init__(self) -> None:
        self.wake_queue = FakeWakeQueue()
        self.context_builder = FakeContextBuilder()

    def is_continuation_message(self, text: str) -> bool:
        return self.context_builder.is_continuation_message(text)


def _make_plugin(
    tmp_path: Path,
    runtime: FakeRuntime,
    max_attempts: int = 3,
    stop_reasons: list[str] | None = None,
    delay_seconds: float = 0.0,
) -> AutoContinuePlugin:
    return AutoContinuePlugin(
        PluginConfig(
            name="auto_continue",
            enabled=True,
            module="diploid_plugins.auto_continue",
            prompt_slot="persona_state",
            prompt_order=55,
            state_file="chat_auto_continue.json",
            config={
                "delay_seconds": delay_seconds,
                "max_attempts": max_attempts,
                "stop_reasons": stop_reasons or ["timeout"],
            },
        ),
        "chat-1",
        tmp_path,
        runtime=runtime,
    )


def test_after_turn_schedules_after_timeout(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime)
    before = time.time()

    plugin.after_turn(
        TurnInfo(
            chat_id="chat-1",
            session_id="s1",
            session_number=1,
            turn_number=1,
            updated_at=before,
            last_stop_reason="timeout",
            user_message="hello",
            reply="",
        )
    )

    assert len(runtime.wake_queue.events) == 1
    event = runtime.wake_queue.events[0]
    assert event.chat_id == "chat-1"
    assert event.reason == "auto_continue"
    assert event.payload == {"user_message": "Continue"}
    assert event.silent is False
    assert event.ready is True
    assert event.scheduled_at >= before
    assert plugin._state["attempt"] == 1
    state_path = tmp_path / "chat-1" / "chat_auto_continue.json"
    assert state_path.exists()
    assert '"attempt": 1' in state_path.read_text()


def test_after_turn_does_not_schedule_on_completed_and_resets_attempt(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime)
    plugin._state["attempt"] = 2

    plugin.after_turn(
        TurnInfo(
            chat_id="chat-1",
            session_id="s1",
            session_number=1,
            turn_number=1,
            updated_at=time.time(),
            last_stop_reason="completed",
            user_message="hello",
            reply="ok",
        )
    )

    assert not runtime.wake_queue.events
    assert plugin._state["attempt"] == 0
    state_path = tmp_path / "chat-1" / "chat_auto_continue.json"
    assert state_path.exists()
    assert '"attempt": 0' in state_path.read_text()


def test_after_turn_stops_at_max_attempts(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime, max_attempts=2)

    for _ in range(3):
        plugin.after_turn(
            TurnInfo(
                chat_id="chat-1",
                session_id="s1",
                session_number=1,
                turn_number=1,
                updated_at=time.time(),
                last_stop_reason="timeout",
                user_message="hello",
                reply="",
            )
        )

    assert len(runtime.wake_queue.events) == 2
    assert plugin._state["attempt"] == 0


def test_after_turn_increments_attempt_across_chained_partials(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime, max_attempts=5)

    for i in range(3):
        plugin.after_turn(
            TurnInfo(
                chat_id="chat-1",
                session_id="s1",
                session_number=1,
                turn_number=i + 1,
                updated_at=time.time(),
                last_stop_reason="timeout",
                user_message="hello",
                reply="",
            )
        )

    assert len(runtime.wake_queue.events) == 3
    assert plugin._state["attempt"] == 3
    for event in runtime.wake_queue.events:
        assert event.chat_id == "chat-1"
        assert event.reason == "auto_continue"
        assert event.payload == {"user_message": "Continue"}


def test_before_turn_cancels_pending_and_resets_attempt_on_unrelated_message(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime)
    plugin._schedule(1, 3, 2.0)

    assert len(runtime.wake_queue.events) == 1

    plugin.before_turn(
        TurnStartContext(
            chat_id="chat-1",
            user_message="what is the weather",
            model=None,
            record=_timeout_record(),
            now=time.time(),
        )
    )

    assert not runtime.wake_queue.events
    assert plugin._state["attempt"] == 0


def test_before_turn_cancels_pending_but_keeps_attempt_on_manual_continue(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime)
    plugin._schedule(2, 3, 2.0)

    assert len(runtime.wake_queue.events) == 1

    plugin.before_turn(
        TurnStartContext(
            chat_id="chat-1",
            user_message="Continue",
            model=None,
            record=_timeout_record(),
            now=time.time(),
        )
    )

    assert not runtime.wake_queue.events
    assert plugin._state["attempt"] == 2


def test_after_turn_schedules_after_cancelled(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime, stop_reasons=["timeout", "cancelled"])
    before = time.time()

    plugin.after_turn(
        TurnInfo(
            chat_id="chat-1",
            session_id="s1",
            session_number=1,
            turn_number=1,
            updated_at=before,
            last_stop_reason="cancelled",
            user_message="hello",
            reply="",
        )
    )

    assert len(runtime.wake_queue.events) == 1
    event = runtime.wake_queue.events[0]
    assert event.chat_id == "chat-1"
    assert event.reason == "auto_continue"
    assert event.payload == {"user_message": "Continue"}
    assert event.silent is False
    assert event.ready is True
    assert event.scheduled_at >= before
    assert plugin._state["attempt"] == 1


def test_after_turn_does_not_schedule_after_user_stop(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime, stop_reasons=["timeout", "cancelled"])

    plugin.after_turn(
        TurnInfo(
            chat_id="chat-1",
            session_id="s1",
            session_number=1,
            turn_number=1,
            updated_at=time.time(),
            last_stop_reason="stopped",
            user_message="hello",
            reply="",
        )
    )

    assert not runtime.wake_queue.events


def test_default_stop_reasons_include_cancelled(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    plugin = AutoContinuePlugin(
        PluginConfig(
            name="auto_continue",
            enabled=True,
            module="diploid_plugins.auto_continue",
            prompt_slot="persona_state",
            prompt_order=55,
            state_file="chat_auto_continue.json",
            config={},
        ),
        "chat-1",
        tmp_path,
        runtime=runtime,
    )

    assert "timeout" in plugin._stop_reasons
    assert "cancelled" in plugin._stop_reasons


def test_agent_runtime_wake_uses_payload_user_message(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()

    captured: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(
        runtime.turn_controller,
        "process",
        lambda *a, **k: captured.append((a, k)) or ChatResult(reply="ok"),
    )

    event = runtime.wake_queue.enqueue(
        WakeEvent(
            id="w1",
            chat_id="chat-1",
            reason="test",
            priority=1,
            scheduled_at=0.0,
            created_at=0.0,
            payload={"user_message": "Continue"},
            silent=True,
            ready=True,
        )
    )

    result = runtime.wake("chat-1", event_id=event.id)

    assert result.reply == "ok"
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args[0] == "chat-1"
    assert args[1] == "Continue"
    assert kwargs.get("wake_event") == event
    runtime.shutdown()


def test_after_turn_includes_partial_notice(tmp_path: Path) -> None:
    """A partial turn's notice is kept in the current turn while auto-continue is scheduled."""
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime)
    partial_notice = "The agent reached the time limit..."

    turn = TurnInfo(
        chat_id="chat-1",
        session_id="s1",
        session_number=1,
        turn_number=1,
        updated_at=time.time(),
        last_stop_reason="timeout",
        user_message="hello",
        reply="",
        partial_notice=partial_notice,
    )
    plugin.after_turn(turn)

    assert turn.partial_notice is None
    assert partial_notice in turn.notice
    assert "Working..." in turn.notice
    assert "deferred_notice" not in plugin._state
    assert len(runtime.wake_queue.events) == 1


def test_after_turn_sends_deferred_on_max_attempts(tmp_path: Path) -> None:
    """When auto-continue gives up, the deferred notice is put back into the turn."""
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime, max_attempts=1)
    plugin._state["attempt"] = 1
    partial_notice = "The agent reached the time limit..."

    turn = TurnInfo(
        chat_id="chat-1",
        session_id="s1",
        session_number=1,
        turn_number=1,
        updated_at=time.time(),
        last_stop_reason="timeout",
        user_message="hello",
        reply="",
        partial_notice=partial_notice,
    )
    plugin.after_turn(turn)

    assert turn.notice == partial_notice
    assert turn.partial_notice is None
    assert "deferred_notice" not in plugin._state
    assert len(runtime.wake_queue.events) == 0


def test_after_turn_working_notice_and_exponential_backoff(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime, delay_seconds=1.0, max_attempts=3)
    before = time.time()

    turn = TurnInfo(
        chat_id="chat-1",
        session_id="s1",
        session_number=1,
        turn_number=1,
        updated_at=before,
        last_stop_reason="timeout",
        user_message="hello",
        reply="partial",
    )
    plugin.after_turn(turn)

    assert len(runtime.wake_queue.events) == 1
    event = runtime.wake_queue.events[0]
    assert "Working..." in turn.notice
    assert "attempt 1/3" in turn.notice
    assert event.scheduled_at >= before + 1.0
    assert event.scheduled_at < before + 2.0

    # Second attempt should double the delay.
    plugin._state["attempt"] = 1
    before2 = time.time()
    turn = TurnInfo(
        chat_id="chat-1",
        session_id="s1",
        session_number=1,
        turn_number=2,
        updated_at=before2,
        last_stop_reason="timeout",
        user_message="hello",
        reply="more",
    )
    plugin.after_turn(turn)
    assert len(runtime.wake_queue.events) == 2
    assert "attempt 2/3" in turn.notice
    event2 = runtime.wake_queue.events[1]
    delay2 = event2.scheduled_at - before2
    assert 1.9 <= delay2 < 2.5


def test_per_reason_settings(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    plugin = AutoContinuePlugin(
        PluginConfig(
            name="auto_continue",
            enabled=True,
            module="diploid_plugins.auto_continue",
            prompt_slot="persona_state",
            prompt_order=55,
            state_file="chat_auto_continue.json",
            config={
                "delay_seconds": 2.0,
                "max_attempts": 3,
                "per_reason": {
                    "timeout": {"delay_seconds": 0.5, "max_attempts": 5},
                    "cancelled": {"max_attempts": 0},
                },
            },
        ),
        "chat-1",
        tmp_path,
        runtime=runtime,
    )

    max_attempts, delay, _ = plugin._settings("timeout")
    assert max_attempts == 5
    assert delay == 0.5

    max_attempts, delay, _ = plugin._settings("cancelled")
    assert max_attempts == 0
    assert delay == 2.0


def test_before_turn_manual_continue_drops_deferred_notice(tmp_path: Path) -> None:
    """A manual Continue cancels any pending auto-continue and drops the deferred notice."""
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime)
    plugin._state["deferred_notice"] = "keep going"

    plugin.before_turn(
        TurnStartContext(
            chat_id="chat-1",
            user_message="Continue",
            model=None,
            record=_timeout_record(),
            now=time.time(),
        )
    )

    assert "deferred_notice" not in plugin._state
    assert "send_deferred" not in plugin._state


def test_before_turn_unrelated_message_drops_deferred_notice(tmp_path: Path) -> None:
    """A non-continuation user message cancels auto-continue and drops the
    deferred partial notice so it does not leak into a normal reply."""
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime)
    plugin._state["deferred_notice"] = "keep going"

    plugin.before_turn(
        TurnStartContext(
            chat_id="chat-1",
            user_message="what is the weather",
            model=None,
            record=_timeout_record(),
            now=time.time(),
        )
    )

    assert "send_deferred" not in plugin._state
    assert "deferred_notice" not in plugin._state
    assert plugin._state["attempt"] == 0


def test_after_turn_sends_deferred_on_completed_turn(tmp_path: Path) -> None:
    """A completed turn with a pending deferred notice sends it."""
    runtime = FakeRuntime()
    plugin = _make_plugin(tmp_path, runtime)
    plugin._state["deferred_notice"] = "keep going"
    plugin._state["send_deferred"] = True

    turn = TurnInfo(
        chat_id="chat-1",
        session_id="s1",
        session_number=1,
        turn_number=2,
        updated_at=time.time(),
        last_stop_reason="completed",
        user_message="what is the weather",
        reply="sunny",
    )
    plugin.after_turn(turn)

    assert turn.notice == "keep going"
    assert "deferred_notice" not in plugin._state
    assert "send_deferred" not in plugin._state
