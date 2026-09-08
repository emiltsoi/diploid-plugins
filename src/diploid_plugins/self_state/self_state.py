"""Self-state continuity plugin."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from diploid_agent.config import PluginConfig
from diploid_agent.models import PartialTurn
from diploid_agent.plugins.base import StatePlugin, WakeContext
from diploid_agent.plugins.contexts import RecordTurnContext
from diploid_agent.runtime.plugin_runtime import PluginRuntime


class SelfStatePlugin(StatePlugin):
    """Save and resume a first-person self-state note across sessions.

    The plugin loads the saved note for the agent and reminds it to keep the note
    up to date, but it never writes the note itself. Only content the agent places
    inside a ``<self_state>`` block is persisted.
    """

    _SELF_STATE_TAG_RE = re.compile(r"</?self_state>", re.IGNORECASE)
    _FIRST_PERSON_RE = re.compile(
        r"(?:"
        r"I(?:[''']m| am|[''']ve| have|[''']d| would|[''']ll| will)?"
        r"|My|Mine|Me|Myself"
        r"|We(?:[''']re| are|[''']ve| have|[''']d| would|[''']ll| will)?"
        r"|Our|Ours|Us|Ourselves"
        r")\b",
        re.IGNORECASE,
    )
    _HEADER = "## State I am resuming from"
    _REMINDER = (
        "Update this with a `<self_state>` block in first person, present tense."
    )
    _REJECTED_REMINDER = (
        "Your last `<self_state>` block was not in first person and was not saved. "
        "Write it as your own direct statement of being: 'I am...' or 'We are...'."
    )

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Any,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._state_path: Path = self._chat_dir() / self.config.state_file
        self._remind: bool = True
        self._rejected: bool = False
        self._last_streamed_note: str | None = None

    def _chat_dir(self) -> Path:
        return self.sessions_root / self.chat_id.replace("/", "_")

    def _load_state(self) -> str:
        try:
            return self._state_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def _save_state(self, text: str) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(text, encoding="utf-8")

    def _state_mtime(self) -> float | None:
        try:
            return self._state_path.stat().st_mtime
        except OSError:
            return None

    def _is_first_person(self, note: str) -> bool:
        """Return True when the note starts as a first-person present statement.

        Leading whitespace and common markdown list/quote markers are ignored.
        """
        lead = note.strip().lstrip(">-*# \t")
        match = self._FIRST_PERSON_RE.match(lead)
        return match is not None and match.start() == 0

    def _extract_self_state(self, reply: str) -> tuple[str, str | None]:
        # Pair each closing tag with the nearest unmatched opening tag so a
        # ``<self_state>`` mention in prose cannot swallow the real block.
        opens: list[tuple[int, int]] = []
        pairs: list[tuple[int, int, int, int]] = []
        for match in self._SELF_STATE_TAG_RE.finditer(reply):
            if match.group(0).startswith("</"):
                if opens:
                    o_start, o_end = opens.pop()
                    pairs.append((o_start, o_end, match.start(), match.end()))
            else:
                opens.append((match.start(), match.end()))
        if not pairs:
            return reply, None
        o_start, o_end, c_start, _ = pairs[-1]
        note = reply[o_end:c_start].strip()
        parts: list[str] = []
        pos = 0
        for s0, _, _, c1 in sorted(pairs):
            if s0 < pos:
                continue
            parts.append(reply[pos:s0])
            pos = c1
        parts.append(reply[pos:])
        return "".join(parts).rstrip(), note

    def on_waking(self, context: WakeContext) -> None:
        self._remind = True
        self._rejected = False
        self._last_streamed_note = None

    def _maybe_save_state(self, note: str) -> None:
        """Save the note if it is first-person; otherwise flag a rejected reminder."""
        if self._is_first_person(note):
            self._save_state(note)
            self._rejected = False
        else:
            self._rejected = True
            self._remind = True

    def on_partial(self, partial: PartialTurn) -> None:
        # Save a <self_state> block as soon as it completes in the stream so the
        # note survives a mid-turn kill; record_turn would be too late.
        _, note = self._extract_self_state(partial.message_text or "")
        if note is not None and note != self._last_streamed_note:
            self._maybe_save_state(note)
            self._last_streamed_note = note

    def prompt_block_changed(self, since: float | None = None) -> bool | None:
        if self._remind:
            return True
        if since is None:
            return True
        mtime = self._state_mtime()
        if mtime is None:
            return None
        return mtime > since

    def before_record_turn(self, context: RecordTurnContext) -> RecordTurnContext:
        stripped, note = self._extract_self_state(context.reply)
        if note is not None:
            self._maybe_save_state(note)
            context.reply = stripped
            if not context.reply.strip():
                context.reply = "(self-state marker set)"
        return context

    def prompt_block(self, max_chars: int | None = None, compact: bool = False) -> str | None:
        note = self._load_state().strip()
        remind = self._remind
        rejected = self._rejected
        self._remind = False
        self._rejected = False

        if not note and not remind:
            return None

        reminder = self._REJECTED_REMINDER if rejected else self._REMINDER
        parts: list[str] = [self._HEADER]
        if note:
            parts.append(note)
        if remind:
            parts.append(reminder)
        block = "\n\n".join(parts)

        if max_chars is not None and len(block) > max_chars:
            base_parts = [self._HEADER]
            if remind:
                base_parts.append(reminder)
            base = "\n\n".join(base_parts)
            note_budget = max_chars - len(base) - (2 if note else 0)
            if note and note_budget > 0:
                block = "\n\n".join(base_parts + [note[:note_budget]])
            else:
                block = block[:max_chars]
        return block
