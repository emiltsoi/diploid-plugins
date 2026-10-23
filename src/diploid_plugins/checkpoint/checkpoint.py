"""Mid-turn checkpoint/breadcrumb plugin."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from diploid_agent.config import PluginConfig
from diploid_agent.models import PartialTurn
from diploid_agent.plugins.base import (
    RecordTurnContext,
    StatePlugin,
    TurnErrorContext,
    TurnInfo,
    WakeContext,
)
from diploid_agent.runtime.plugin_runtime import PluginRuntime

from diploid_plugins.checkpoint.config import CheckpointConfig

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


class CheckpointPlugin(StatePlugin):
    """Write an append-only breadcrumb trail and a live checkpoint snapshot.

    The goal is to make a killed turn leave a durable, first-person trace so the
    next wake can step back into the same wave. This is complementary to the
    continuity plugin's active/interrupted turn snapshot and to the self_state
    and persistent_memory plugins: it keeps an append-only timeline of the turn's
    intent, side effects, and recent text, not just the latest partial.
    """

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._config = CheckpointConfig()
        for key in CheckpointConfig.__dataclass_fields__:
            if key in config.config:
                setattr(self._config, key, config.config[key])

        # Last partial state used to decide whether a new checkpoint is needed.
        self._last_turn_number: int | None = None
        self._last_session_number: int | None = None
        self._last_intent: str = ""
        self._last_side_effect: str = ""
        self._last_side_effect_at: float = 0.0
        self._last_side_effects_count: int = 0
        self._last_message_tail: str = ""
        self._last_thought_tail: str = ""
        self._last_checkpoint_at: float = 0.0

    # ---------------------------------------------------------------- paths

    def _chat_dir(self) -> Path:
        return self.sessions_root / self.chat_id.replace("/", "_")

    def _active_path(self) -> Path:
        return self._chat_dir() / self._config.active_checkpoint_file

    def _breadcrumbs_path(self) -> Path:
        return self._chat_dir() / self._config.breadcrumbs_file

    def _last_breadcrumbs_path(self) -> Path:
        return self._chat_dir() / self._config.last_turn_breadcrumbs_file

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _snip(text: str | None, max_len: int) -> str:
        text = text or ""
        if len(text) <= max_len:
            return text
        return "..." + text[-(max_len - 3) :]

    def _side_effects_recent(
        self, side_effects: list[dict[str, Any]] | None
    ) -> list[dict[str, str]]:
        recent: list[dict[str, str]] = []
        for eff in (side_effects or [])[-3:]:
            if not isinstance(eff, dict):
                continue
            recent.append(
                {
                    "title": str(eff.get("title") or eff.get("toolCallId") or "tool")[:80],
                    "status": str(eff.get("status") or "running")[:40],
                }
            )
        return recent

    def _build_checkpoint(self, partial: PartialTurn, checkpoint_type: str) -> dict[str, Any]:
        side_effects = partial.side_effects or []
        now = _now()
        return {
            "at": now,
            "type": checkpoint_type,
            "session_number": partial.session_number,
            "turn_number": partial.turn_number,
            "user_message": self._snip(partial.user_message, self._config.max_user_message_snippet),
            "current_intent": self._snip(partial.current_intent, 160),
            "last_side_effect": self._snip(partial.last_side_effect, 160),
            "last_side_effect_at": partial.last_side_effect_at or now,
            "side_effects_count": len(side_effects),
            "side_effects_recent": self._side_effects_recent(side_effects),
            "message_snippet": self._snip(partial.message_text, self._config.max_message_snippet),
            "thought_snippet": self._snip(partial.thought_text, self._config.max_thought_snippet),
            "updated_at": partial.updated_at or now,
        }

    def _save_active_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        path = self._active_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(checkpoint, indent=2, default=str))
        except OSError:
            logger.exception("Failed to write active checkpoint")

    def _append_breadcrumb(self, checkpoint: dict[str, Any]) -> None:
        path = self._breadcrumbs_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(checkpoint, default=str) + "\n")
            self._trim_breadcrumbs(path)
        except OSError:
            logger.exception("Failed to append breadcrumb")

    def _trim_breadcrumbs(self, path: Path) -> None:
        """Drop oldest lines if the file exceeds max_breadcrumbs."""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) <= self._config.max_breadcrumbs:
            return
        try:
            path.write_text(
                "\n".join(lines[-self._config.max_breadcrumbs :]) + "\n", encoding="utf-8"
            )
        except OSError:
            logger.exception("Failed to trim breadcrumbs")

    def _load_active_checkpoint(self) -> dict[str, Any] | None:
        path = self._active_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _remove_active_checkpoint(self) -> None:
        path = self._active_path()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove active checkpoint")

    def _rotate_breadcrumbs(self, turn_number: int) -> None:
        """Move the current turn's breadcrumbs to the last-turn file on completion."""
        active = self._breadcrumbs_path()
        last = self._last_breadcrumbs_path()
        try:
            if active.exists():
                # Write a small completed marker at the end.
                with active.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "at": _now(),
                                "type": "completed",
                                "turn_number": turn_number,
                            },
                            default=str,
                        )
                        + "\n"
                    )
                last.write_text(active.read_text(encoding="utf-8"), encoding="utf-8")
                active.unlink()
        except OSError:
            logger.exception("Failed to rotate breadcrumbs")

    # ---------------------------------------------------------------- partial tracking

    def _is_significant_change(self, partial: PartialTurn) -> bool:
        side_effects = partial.side_effects or []
        side_effects_count = len(side_effects)
        message_tail = (partial.message_text or "")[-self._config.max_message_snippet :]
        thought_tail = (partial.thought_text or "")[-self._config.max_thought_snippet :]

        if self._last_turn_number != partial.turn_number:
            return True
        if self._last_session_number != partial.session_number:
            return True
        if self._last_intent != (partial.current_intent or ""):
            return True
        if self._last_side_effect != (partial.last_side_effect or ""):
            return True
        if self._last_side_effects_count != side_effects_count:
            return True
        if self._last_message_tail != message_tail:
            return True
        return self._last_thought_tail != thought_tail

    def _update_last_state(self, partial: PartialTurn) -> None:
        self._last_turn_number = partial.turn_number
        self._last_session_number = partial.session_number
        self._last_intent = partial.current_intent or ""
        self._last_side_effect = partial.last_side_effect or ""
        self._last_side_effect_at = partial.last_side_effect_at or 0.0
        self._last_side_effects_count = len(partial.side_effects or [])
        self._last_message_tail = (partial.message_text or "")[-self._config.max_message_snippet :]
        self._last_thought_tail = (partial.thought_text or "")[-self._config.max_thought_snippet :]

    # ---------------------------------------------------------------- hooks

    def on_partial(self, partial: PartialTurn) -> None:
        now = _now()
        significant = self._is_significant_change(partial)
        throttled = now - self._last_checkpoint_at < self._config.throttle_seconds

        # Always write immediately on a side-effect/intent/text change; otherwise
        # throttle ordinary stream updates to avoid flooding the file.
        if not significant and throttled:
            return

        # Side-effect changes get the type of the new side effect status.
        if self._last_side_effect != (partial.last_side_effect or ""):
            checkpoint_type = "tool_update"
        elif self._last_side_effects_count != len(partial.side_effects or []):
            checkpoint_type = "tool_start"
        else:
            checkpoint_type = "partial"

        checkpoint = self._build_checkpoint(partial, checkpoint_type)
        self._save_active_checkpoint(checkpoint)
        self._append_breadcrumb(checkpoint)
        self._update_last_state(partial)
        self._last_checkpoint_at = now

    def before_record_turn(self, context: RecordTurnContext) -> RecordTurnContext | None:
        # If the active checkpoint exists, write one final pre-record snapshot so the
        # reply and final state are captured before the transcript is committed.
        active = self._load_active_checkpoint()
        if active:
            active["type"] = "completing"
            active["at"] = _now()
            active["reply_snippet"] = self._snip(context.reply, self._config.max_message_snippet)
            self._save_active_checkpoint(active)
            self._append_breadcrumb(active)
        return None

    def after_turn(self, turn: TurnInfo) -> None:
        # The turn has been recorded; archive the breadcrumbs and remove the active
        # checkpoint so the next wake does not show a stale in-progress marker.
        self._rotate_breadcrumbs(turn.turn_number)
        self._remove_active_checkpoint()
        self._last_turn_number = None
        self._last_session_number = None
        self._last_intent = ""
        self._last_side_effect = ""
        self._last_side_effects_count = 0
        self._last_message_tail = ""
        self._last_thought_tail = ""

    def on_turn_error(self, context: TurnErrorContext) -> None:
        active = self._load_active_checkpoint()
        if active:
            active["type"] = "interrupted"
            active["error"] = self._snip(str(context.error), 200)
            active["at"] = _now()
            self._save_active_checkpoint(active)
            self._append_breadcrumb(active)

    def on_waking(self, context: WakeContext) -> None:
        # Nothing to preload; prompt_block checks the active file on demand.
        pass

    def prompt_block(self, max_chars: int | None = None, compact: bool = False) -> str | None:
        checkpoint = self._load_active_checkpoint()
        if not checkpoint:
            return None

        # Ignore stale checkpoints left over from a much older turn.
        now = _now()
        updated_at = checkpoint.get("at") or checkpoint.get("updated_at") or 0
        if now - float(updated_at) > self._config.stale_hours * 3600:
            self._remove_active_checkpoint()
            return None

        turn_number = checkpoint.get("turn_number", 0)
        user_message = checkpoint.get("user_message", "")
        current_intent = checkpoint.get("current_intent", "")
        last_side_effect = checkpoint.get("last_side_effect", "")
        side_effects_count = checkpoint.get("side_effects_count", 0)
        recent = checkpoint.get("side_effects_recent", [])
        checkpoint_type = checkpoint.get("type", "partial")

        lines: list[str] = ["## Turn breadcrumb"]

        if compact:
            lines.append(
                f"I am in the middle of turn {turn_number}. "
                f"Last step: {last_side_effect or 'none'}. "
                f"Intent: {current_intent or 'unknown'}."
            )
        else:
            lines.append(
                f"I am in the middle of turn {turn_number}. If I was interrupted, "
                "continue from the last known step."
            )
            if user_message:
                lines.append(f"You asked: {user_message}")
            if current_intent:
                lines.append(f"My current intent: {current_intent}")
            if last_side_effect:
                lines.append(f"Last side effect: {last_side_effect}")
            if recent:
                lines.append(f"Recent steps ({side_effects_count} total):")
                for eff in recent:
                    lines.append(
                        f"  - {eff.get('title') or 'tool'}: {eff.get('status') or 'running'}"
                    )
            else:
                lines.append(f"Side effects so far: {side_effects_count}")

        if checkpoint_type == "interrupted":
            lines.append("This turn ended with an error.")

        block = "\n".join(lines)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars].rsplit("\n", 1)[0]
        return block
