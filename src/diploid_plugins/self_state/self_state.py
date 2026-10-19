"""Self-state continuity plugin."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diploid_agent.config import PluginConfig
from diploid_agent.models import PartialTurn
from diploid_agent.plugins.base import StatePlugin, WakeContext
from diploid_agent.plugins.contexts import RecordTurnContext
from diploid_agent.runtime.plugin_runtime import PluginRuntime

logger = logging.getLogger(__name__)


class SelfStatePlugin(StatePlugin):
    """Save and resume a first-person self-state note across sessions.

    The plugin loads the saved note for the agent and reminds it to keep the note
    up to date, but it never writes the note itself. Only content the agent places
    inside a ``<self_state>`` block is persisted — plus a bounded ``## mid-turn``
    section the agent may edit directly between block saves, which the plugin
    carries forward (capped at ``mid_turn_max_chars``, default 4096) so a block
    save cannot wipe in-flight scratch notes.
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
    _NEXT_SELF_RE = re.compile(
        r"^##\s*next-self\b[^\n]*\n?(.*?)(?=^##\s|\Z)",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    _MID_TURN_RE = re.compile(
        r"^##\s*mid-turn\b[^\n]*\n?(.*?)(?=^##\s|\Z)",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    _OPEN_ITEMS_RE = re.compile(
        r"^##\s*open-items\b[^\n]*\n?(.*?)(?=^##\s|\Z)",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    _OPEN_ITEMS_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")
    _ITEM_FIELD_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
    _MID_TURN_DEFAULT_MAX_CHARS = 4096
    _STALE_NEXT_SELF_HEADER = "## next-self (stale — carried from previous note)"
    _STALE_OPEN_ITEMS_HEADER = "## open-items (stale — carried from previous note)"
    _OPEN_ITEMS_DEFAULT_FILE = "chat_open_items.jsonl"
    _OPEN_ITEMS_SCHEMA = "open_items/v1"
    _HEADER = "## State I am resuming from"
    _REMINDER = (
        "Update this with a `<self_state>` block in first person, present tense; "
        "end it with `## next-self` for whoever wakes next. Keep `## open-items` "
        "for durable debts — `- <claim> · id: … · status: open · kind: … "
        "· checked_by: …`."
    )
    _NO_HANDOFF_NUDGE = "No `## next-self` handoff found — reconstruct one from the wake state."
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

    def _next_self_body(self, note: str) -> str | None:
        """Return the text after a `## next-self` heading, or None."""
        match = self._NEXT_SELF_RE.search(note)
        if match is None:
            return None
        body = match.group(1).strip()
        return body or None

    def _mid_turn_section(self, note: str) -> str | None:
        """Return the whole `## mid-turn` section (header + body), or None."""
        match = self._MID_TURN_RE.search(note)
        if match is None:
            return None
        return match.group(0).rstrip()

    def _mid_turn_budget(self) -> int:
        try:
            return int(
                self.config.config.get("mid_turn_max_chars", self._MID_TURN_DEFAULT_MAX_CHARS)
            )
        except (TypeError, ValueError):
            return self._MID_TURN_DEFAULT_MAX_CHARS

    def _cap_mid_turn(self, section: str) -> str:
        """Trim a `## mid-turn` section to its budget, keeping the newest lines.

        Lines are dropped whole — a tail-cut crumb reads as file corruption,
        not trimming. When anything is dropped a marker line records the count
        so the trim is visible rather than silent.
        """
        budget = self._mid_turn_budget()
        if len(section) <= budget:
            return section
        header, *rest = section.split("\n")
        kept: list[str] = []
        size = len(header)
        for line in reversed(rest):
            if len(line) + 1 <= budget - size:
                kept.append(line)
                size += len(line) + 1
        kept.reverse()
        dropped = len(rest) - len(kept)
        if dropped:

            def _marker(n: int) -> str:
                return f"- …({n} earlier line{'s' if n > 1 else ''} trimmed)"

            marker = _marker(dropped)
            while kept and len("\n".join([header, marker, *kept])) > budget:
                kept.pop(0)
                dropped += 1
                marker = _marker(dropped)
            kept.insert(0, marker)
        return "\n".join([header, *kept])

    def _bounded_note(self, note: str) -> str:
        """Apply the mid-turn cap to the note's `## mid-turn` section, if any."""
        match = self._MID_TURN_RE.search(note)
        if match is None:
            return note
        capped = self._cap_mid_turn(match.group(0).rstrip())
        if capped == match.group(0):
            return note
        suffix = note[match.end() :]
        return note[: match.start()] + capped + ("\n\n" + suffix if suffix else "")

    def _open_items_section(self, note: str) -> str | None:
        """Return the `## open-items` section body, or None when absent.

        An explicitly empty section returns ``""`` — the author wrote the
        heading, so the ledger clears rather than carrying stale rows.
        """
        match = self._OPEN_ITEMS_RE.search(note)
        if match is None:
            return None
        return match.group(1).strip()

    def _open_items_path(self) -> Path:
        name = self.config.config.get("open_items_file", self._OPEN_ITEMS_DEFAULT_FILE)
        return self._chat_dir() / str(name)

    def _parse_open_item_line(self, line: str) -> dict[str, str] | None:
        """One `## open-items` bullet into a claims row.

        Format: ``- <claim> · key: value · key: value``. The first `·`-segment
        is the claim; later ``key: value`` segments set fields; a `·` segment
        that does not parse folds back into the claim so stray text is never
        silently dropped.
        """
        match = self._OPEN_ITEMS_BULLET_RE.match(line)
        if match is None:
            return None
        segments = match.group(1).split("·")
        claim_parts = [segments[0].strip()]
        fields: dict[str, str] = {}
        for segment in segments[1:]:
            field = self._ITEM_FIELD_RE.match(segment.strip())
            if field:
                fields[field.group(1)] = field.group(2).strip()
            else:
                claim_parts.append(segment.strip())
        claim = " · ".join(part for part in claim_parts if part)
        if not claim:
            return None
        fields["claim"] = claim
        return fields

    def _load_open_items(self) -> dict[str, dict[str, Any]]:
        """Existing ledger rows keyed by id — read for `since` preservation."""
        try:
            lines = self._open_items_path().read_text(encoding="utf-8").splitlines()
        except OSError:
            return {}
        rows: dict[str, dict[str, Any]] = {}
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                rows[row["id"]] = row
        return rows

    def _emit_open_items(self, note: str) -> None:
        """Project the note's `## open-items` section into the claims ledger.

        The section is the author's edit surface; ``chat_open_items.jsonl`` is
        the machine-readable surface the reconciliation detector diffs against
        the world (open-items plan, phase 2). Row shape::

            {"id": ..., "claim": ..., "since": "YYYY-MM-DD", "status": "open",
             "kind"?, "checked_by"?, "evidence"?, ...extra fields passthrough}

        `since` survives rewrites by id; an absent section carries the file —
        a mid-turn update that forgets the heading never discharges the
        ledger. Never raises into the save path: a ledger write must not
        break the note.
        """
        section = self._open_items_section(note)
        if section is None:
            return
        try:
            prior = self._load_open_items()
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            rows: list[dict[str, Any]] = [{"_schema": self._OPEN_ITEMS_SCHEMA}]
            for line in section.splitlines():
                fields = self._parse_open_item_line(line)
                if fields is None:
                    continue
                claim = fields.pop("claim")
                row_id = (
                    fields.pop("id", None)
                    or "oi-" + hashlib.sha1(claim.encode("utf-8")).hexdigest()[:8]
                )
                item: dict[str, Any] = {
                    "id": row_id,
                    "claim": claim,
                    "since": fields.pop("since", "") or prior.get(row_id, {}).get("since") or today,
                    "status": fields.pop("status", "") or "open",
                }
                item.update(fields)
                rows.append(item)
            path = self._open_items_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError:
            logger.warning("Could not write open-items ledger in %s", self._chat_dir())

    def _maybe_save_state(self, note: str) -> None:
        """Save the note if it is first-person; otherwise flag a rejected reminder."""
        if self._is_first_person(note):
            note = self._bounded_note(note)
            if self._mid_turn_section(note) is None:
                # Bounded direct writes (`## mid-turn` scratch the agent edits
                # between `<self_state>` updates) must survive a block save —
                # the block replaces the whole file, so carry the section.
                old_section = self._mid_turn_section(self._load_state())
                if old_section is not None:
                    note = note.rstrip() + "\n\n" + self._cap_mid_turn(old_section)
            if self._next_self_body(note) is None:
                # A mid-turn state update that forgets `## next-self` must not
                # silently drop the last handoff — carry it forward marked stale.
                old_body = self._next_self_body(self._load_state())
                if old_body:
                    note = note.rstrip() + f"\n\n{self._STALE_NEXT_SELF_HEADER}\n{old_body}"
            if self._open_items_section(note) is None:
                # Same courtesy as next-self: a forgotten `## open-items`
                # section must not drop the ledger from the note — carry it
                # forward marked stale, so note and ledger stay consistent.
                old_items = self._open_items_section(self._load_state())
                if old_items:
                    note = note.rstrip() + f"\n\n{self._STALE_OPEN_ITEMS_HEADER}\n{old_items}"
            self._save_state(note)
            self._emit_open_items(note)
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
        note = self._bounded_note(self._load_state().strip())
        remind = self._remind
        rejected = self._rejected
        self._remind = False
        self._rejected = False

        if not note and not remind:
            return None

        reminder = self._REJECTED_REMINDER if rejected else self._REMINDER
        if remind and note and self._next_self_body(note) is None:
            reminder += "\n" + self._NO_HANDOFF_NUDGE
        appends = self._appends_line()
        parts: list[str] = [self._HEADER]
        if note:
            parts.append(note)
        if appends:
            parts.append(appends)
        if remind:
            parts.append(reminder)
        block = "\n\n".join(parts)

        if max_chars is not None and len(block) > max_chars:
            base_parts = [self._HEADER]
            if appends:
                base_parts.append(appends)
            if remind:
                base_parts.append(reminder)
            base = "\n\n".join(base_parts)
            note_budget = max_chars - len(base) - (2 if note else 0)
            if note and note_budget > 0:
                block = "\n\n".join(base_parts + [note[:note_budget]])
            else:
                block = base[:max_chars]
        return block

    def _appends_line(self) -> str | None:
        """Surface the bounded-self-writes append counter — a nudge, not a gate.

        The counter is fed quietly by cron file-triggers on the memory files
        (see the persona's crons.yaml); this line is the slot the negotiated
        design promised to expose it in.
        """
        path = self.sessions_root / "daily_appends.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        day = (data.get("days") or {}).get(time.strftime("%Y-%m-%d")) or {}
        persona = int(day.get("persona") or 0)
        chat = int(day.get(f"chat:{self.chat_id}") or 0)
        if not persona and not chat:
            return None
        budget = int(self.config.config.get("memory_append_budget", 10))
        over = " — over soft budget; compact before more" if max(persona, chat) > budget else ""
        return f"Memory appends today: MEMORY.md {persona}/{budget}, chat {chat}/{budget}{over}."
