"""Continuity plugin: wake state, instance identity, pending dispatches."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diploid_agent.config import PluginConfig
from diploid_agent.models import PartialTurn
from diploid_agent.plugins.base import SleepContext, StatePlugin, TurnInfo, WakeContext
from diploid_agent.plugins.contexts import TurnErrorContext
from diploid_agent.runtime.plugin_runtime import PluginRuntime

logger = logging.getLogger(__name__)


class ContinuityPlugin(StatePlugin):
    """Tracks time asleep, last turn, and pending work across wake events."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._state: dict[str, Any] = self._load_state()
        self._pending_partial: PartialTurn | None = None
        self._last_partial_write: float = 0.0
        self._throttle_seconds: float = 0.2
        self._self_state_file: str = str(
            config.config.get("self_state_file", "chat_self_state.md")
        )

    def state_path(self) -> Path | None:
        if not self.config.state_file:
            return None
        chat_dir = self._chat_dir()
        return chat_dir / self.config.state_file

    def _chat_dir(self) -> Path:
        return self.sessions_root / self.chat_id.replace("/", "_")

    def _load_state(self) -> dict[str, Any]:
        path = self.state_path()
        if path is None or not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_state(self) -> None:
        path = self.state_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state, indent=2, default=str))

    def _format_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)} seconds"
        if seconds < 3600:
            return f"{int(seconds / 60)} minutes"
        return f"{seconds / 3600:.1f} hours"

    def _format_time(self, when: float) -> str:
        return datetime.fromtimestamp(when, tz=UTC).isoformat()

    def on_waking(self, context: WakeContext) -> None:
        previous_instance = self._state.get("this_instance_id")
        self._state["last_woken_at"] = context.now
        self._state["this_instance_id"] = context.instance_id
        self._state["instance_started_at"] = context.instance_started_at
        rehydration_reason = getattr(context, "rehydration_reason", None)
        if context.wake_event is not None:
            self._state["last_wake_event"] = context.wake_event.reason
        elif rehydration_reason and rehydration_reason != "none":
            self._state["last_wake_event"] = rehydration_reason
        else:
            self._state["last_wake_event"] = "wake"
        if context.previous_turn_at:
            self._state["time_asleep_seconds"] = context.now - context.previous_turn_at
        if context.record:
            self._state["last_session_number"] = context.record.session_number
            self._state["last_turn_number"] = context.record.turn_number
            self._state["last_stop_reason"] = context.record.last_stop_reason
            self._state["last_turn_at"] = context.record.updated_at

        pending = context.pending_dispatches or []
        self._state["pending_dispatches"] = pending
        self._state["had_pending_dispatches"] = bool(pending)

        self._state["instance_changed"] = (
            previous_instance is not None and previous_instance != context.instance_id
        )

        self._capture_interrupted_turn()
        self._save_state()

    def _capture_interrupted_turn(self) -> None:
        """Preserve a leftover active-turn snapshot as an interrupted turn.

        ``chat_active_turn.json`` is removed by ``on_turn_end``; if it still
        exists while no turn is streaming, the previous process died mid-turn
        and ``record_turn`` never ran — its side effects may exist without a
        transcript entry.
        """
        if self._pending_partial is not None:
            # A turn is streaming right now; the snapshot belongs to it.
            return
        active_path = self._active_turn_path()
        if not active_path.exists():
            return
        try:
            data = json.loads(active_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        try:
            interrupted_path = self._chat_dir() / "chat_interrupted_turn.json"
            interrupted_path.write_text(
                json.dumps(data, indent=2, default=str)
            )
            active_path.unlink()
        except OSError:
            return
        self._state["interrupted_turn"] = {
            "turn_number": data.get("turn_number"),
            "session_number": data.get("session_number"),
            "user_message": (data.get("user_message") or "")[:200],
            "updated_at": data.get("updated_at"),
            "current_intent": (data.get("current_intent") or "")[:200],
            "last_side_effect": (data.get("last_side_effect") or "")[:200],
        }
        self._save_state()
        if self._runtime is not None:
            try:
                self._runtime.record_system_note(
                    self.chat_id,
                    f"Turn {data.get('turn_number')} was interrupted mid-flight "
                    "and never recorded; partial reply preserved in "
                    "chat_interrupted_turn.json.",
                )
            except Exception:  # noqa: BLE001
                logger.warning("Could not record interrupted-turn system note")

    def _active_turn_path(self) -> Path:
        return self._chat_dir() / "chat_active_turn.json"

    def _next_self_status(self, interrupted_at: Any) -> str | None:
        """Check whether a `## next-self` handoff survives the interruption.

        The self-state note is authored by the agent; this only detects and
        reports — it never writes the handoff itself.
        """
        path = self._chat_dir() / self._self_state_file
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if not re.search(r"^##\s*next-self\b", text, re.IGNORECASE | re.MULTILINE):
            return (
                "  No `## next-self` handoff survived the interruption — "
                "reconstruct one from this snapshot in your own words."
            )
        try:
            mtime = path.stat().st_mtime
            interrupted_ts = float(interrupted_at) if interrupted_at else None
        except (OSError, TypeError, ValueError):
            return None
        if interrupted_ts is not None and mtime < interrupted_ts:
            return (
                "  Your `## next-self` handoff predates the interrupted turn "
                "— it may be stale."
            )
        return None

    def _write_active_turn(self) -> None:
        if self._pending_partial is None:
            return
        partial = self._pending_partial
        path = self._active_turn_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "session_number": partial.session_number,
                    "turn_number": partial.turn_number,
                    "user_message": partial.user_message,
                    "message_text": partial.message_text,
                    "thought_text": partial.thought_text,
                    "updated_at": partial.updated_at,
                    # getattr keeps this plugin hot-reloadable onto a harness
                    # whose PartialTurn predates the breadcrumb fields.
                    "current_intent": getattr(partial, "current_intent", ""),
                    "last_side_effect": getattr(partial, "last_side_effect", ""),
                    "last_side_effect_at": getattr(partial, "last_side_effect_at", 0.0),
                },
                indent=2,
                default=str,
            )
        )

    def _remove_active_turn(self) -> None:
        path = self._active_turn_path()
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def _flush_active_turn(self) -> None:
        if self._pending_partial is None:
            return
        self._write_active_turn()
        self._last_partial_write = time.time()

    def on_partial(self, partial: PartialTurn) -> None:
        self._pending_partial = partial
        now = partial.updated_at or time.time()
        if now - self._last_partial_write >= self._throttle_seconds:
            self._write_active_turn()
            self._save_state()
            self._last_partial_write = now

    def on_turn_end(self, turn: TurnInfo) -> None:
        self._state.pop("interrupted_turn", None)
        self._state["last_turn_at"] = turn.updated_at
        self._state["last_user_message"] = turn.user_message
        self._state["last_assistant_reply"] = turn.reply
        self._state["last_session_number"] = turn.session_number
        self._state["last_turn_number"] = turn.turn_number
        self._state["last_stop_reason"] = turn.last_stop_reason
        self._save_state()
        self._flush_active_turn()
        self._remove_active_turn()
        self._pending_partial = None

    def on_turn_error(self, context: TurnErrorContext) -> None:
        self._flush_active_turn()

    def prompt_block(self, max_chars: int | None = None, compact: bool = False) -> str | None:
        # A leftover active-turn snapshot at prompt-build time means the last
        # turn died mid-flight — no wake event is required to notice it.
        self._capture_interrupted_turn()

        last_turn_at = self._state.get("last_turn_at")
        if not last_turn_at:
            return None

        instance_id = self._state.get("this_instance_id", "unknown")
        last_session = self._state.get("last_session_number")
        last_turn = self._state.get("last_turn_number")
        last_reason = self._state.get("last_stop_reason") or "unknown"

        if compact:
            parts = ["## Wake state"]
            interrupted = self._state.get("interrupted_turn")
            if interrupted:
                parts.append(
                    f"Turn {interrupted.get('turn_number')} was interrupted "
                    "mid-flight and never recorded; its side effects may exist "
                    "outside the transcript (see chat_interrupted_turn.json)."
                )
            if last_session is not None and last_turn is not None:
                parts.append(
                    f"Last turn: session {last_session}, turn {last_turn}, {last_reason}."
                )
            time_asleep = self._state.get("time_asleep_seconds")
            if time_asleep is not None:
                parts.append(f"Silent for {self._format_duration(time_asleep)}.")
            pending = self._state.get("pending_dispatches") or []
            if pending:
                parts.append(f"Pending work: {len(pending)} dispatch(es).")
            block = " ".join(parts)
            if max_chars is not None and len(block) > max_chars:
                block = block[:max_chars]
            return block

        lines = ["## Wake state"]

        interrupted = self._state.get("interrupted_turn")
        if interrupted:
            lines.append(
                "- Previous turn was interrupted mid-flight and never recorded "
                "— its side effects (commits, edits) may exist outside the "
                "transcript. Partial draft is preserved in "
                "chat_interrupted_turn.json."
            )
            lines.append(
                f"  Interrupted turn {interrupted.get('turn_number')}, "
                f"user asked: {(interrupted.get('user_message') or '')[:80]}"
            )
            intent = interrupted.get("current_intent")
            if intent:
                lines.append(f"  Intent: {intent[:120]}")
            side_effect = interrupted.get("last_side_effect")
            if side_effect:
                lines.append(f"  Last side effect: {side_effect[:120]}")
            handoff = self._next_self_status(interrupted.get("updated_at"))
            if handoff:
                lines.append(handoff)

        instance_started = self._state.get("instance_started_at")
        if instance_started:
            lines.append(
                f"- Instance: {instance_id} (started {self._format_time(instance_started)})"
            )
        else:
            lines.append(f"- Instance: {instance_id}")

        wake_event = self._state.get("last_wake_event")
        if wake_event:
            lines.append(f"- Last wake: {wake_event}")
        if self._state.get("instance_changed"):
            lines.append("- Instance changed since previous wake: yes")

        if last_session is not None and last_turn is not None:
            lines.append(
                f"- Last turn: session {last_session}, turn {last_turn}, "
                f"stop reason {last_reason}, at {self._format_time(last_turn_at)}"
            )

        time_asleep = self._state.get("time_asleep_seconds")
        if time_asleep is not None:
            lines.append(f"- You were silent for {self._format_duration(time_asleep)}.")

        pending = self._state.get("pending_dispatches") or []
        if pending:
            lines.append(f"- Pending background work: {len(pending)} dispatch(es)")
            for d in pending[:5]:
                ctx = d.get("context") or "(no context)"
                lines.append(f"  - {d.get('id')}: {ctx[:60]}")
        else:
            lines.append("- Pending background work: none")

        last_user = self._state.get("last_user_message", "")
        last_reply = self._state.get("last_assistant_reply", "")
        if last_user or last_reply:
            snippet = []
            if last_user:
                snippet.append(f"user asked: {last_user[:80]}")
            if last_reply:
                snippet.append(f"you were saying: {last_reply[:80]}")
            lines.append("- Last exchange: " + "; ".join(snippet))

        active_path = self._active_turn_path()
        if active_path.exists():
            try:
                active = json.loads(active_path.read_text())
                message_text = active.get("message_text", "")[:200]
                thought_text = active.get("thought_text", "")[:200]
                intent = active.get("current_intent", "")[:120]
                side_effect = active.get("last_side_effect", "")[:120]
                if intent:
                    lines.append(f"- Active intent: {intent}")
                if side_effect:
                    lines.append(f"- Active side effect: {side_effect}")
                if message_text:
                    lines.append(f"- Active turn draft: {message_text}")
                if thought_text:
                    lines.append(f"- Active thought: {thought_text}")
            except (json.JSONDecodeError, OSError):
                pass

        block = "\n".join(lines)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block

    def on_sleeping(self, context: SleepContext) -> None:
        self._state["last_process_ended_at"] = context.now
        self._state["last_process_ended_reason"] = context.reason
        self._save_state()
