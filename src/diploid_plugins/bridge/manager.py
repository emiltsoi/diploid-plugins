"""Bridge/SURFACE state manager."""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diploid_agent.models import SessionRecord

from diploid_plugins.bridge.config import BridgeConfig


class BridgeManager:
    """Collect first-person continuity state and write BRIDGE/SURFACE files."""

    def __init__(self, chat_id: str, sessions_root: Path, config: BridgeConfig) -> None:
        self.chat_id = chat_id
        self.sessions_root = sessions_root
        self.config = config

    @property
    def _chat_dir(self) -> Path:
        return self.sessions_root / self.chat_id.replace("/", "_")

    def _path(self, filename: str) -> Path:
        return self._chat_dir / filename

    def _read_text(self, filename: str) -> str:
        try:
            return self._path(filename).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def _read_json(self, filename: str) -> dict[str, Any]:
        try:
            return json.loads(self._path(filename).read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _close_date(self) -> str:
        """Return today's UTC date as YYYY-MM-DD."""
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _parse_tasks(self, text: str) -> list[str]:
        """Extract open task bullets from a simple markdown task list."""
        open_items: list[str] = []
        in_open = False
        for line in text.splitlines():
            if re.match(r"^#{1,6}\s*open\b", line, re.IGNORECASE):
                in_open = True
                continue
            if re.match(r"^#{1,6}\s", line) and in_open:
                in_open = False
                continue
            if in_open:
                match = re.match(r"^[\s]*[-*]\s+(.*)$", line)
                if match:
                    open_items.append(match.group(1).strip())
        return open_items[:3]

    def _extract_next_self(self, note: str) -> str:
        """Return the body after a `## next-self` heading, if present."""
        match = re.search(
            r"^##\s*next-self\b[^\n]*\n?(.*?)(?=^##\s|\Z)",
            note,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        if not match:
            return ""
        return match.group(1).strip()

    def _body_line(self, data: dict[str, Any]) -> str:
        """Return one body pointer line from body state."""
        summary = data.get("felt_summary")
        warmth = data.get("felt_warmth", 0.0)
        if summary:
            return f"Body: {summary} (warmth {warmth:.2f})"
        last_event = data.get("last_event")
        if last_event:
            return f"Body: last event was {last_event}"
        return "Body: here and settled"

    def _collect_state(self, record: SessionRecord | None) -> dict[str, Any]:
        """Gather durable state into a dictionary for rendering."""
        close_date = self._close_date()
        self_state = self._read_text(self.config.self_state_file)
        next_self = self._extract_next_self(self_state)
        body = self._read_json(self.config.body_state_file)
        working = self._read_json(self.config.working_memory_file)
        wake = self._read_json(self.config.wake_state_file)
        tasks_text = self._read_text(self.config.tasks_file)
        promoted = self._read_text(self.config.promoted_file)

        personal = self_state.split("##")[0].strip() if self_state else ""
        if not personal:
            personal = "I am here, ready to continue."

        return {
            "close_date": close_date,
            "personal": personal,
            "next_self": next_self,
            "body_line": self._body_line(body),
            "working_intent": working.get("intent", ""),
            "open_questions": working.get("open_questions", []),
            "tasks": self._parse_tasks(tasks_text),
            "promoted": promoted.strip()[:512] if promoted.strip() else "",
            "last_user": wake.get("last_user_message", ""),
            "last_reply": wake.get("last_assistant_reply", ""),
            "last_topic": (
                f"user asked: {wake.get('last_user_message', '')[:80]}"
                if wake.get("last_user_message")
                else ""
            ),
            "session_number": record.session_number if record else 0,
            "turn_number": record.turn_number if record else 0,
        }

    def _render_bridge(self, state: dict[str, Any]) -> str:
        """Render the full BRIDGE markdown."""
        lines = [f"# BRIDGE — {state['close_date']}", ""]

        lines.extend(["## Personal State", state["personal"], ""])

        if state["next_self"]:
            lines.extend(["## Next-Self Handoff", state["next_self"], ""])

        if state["working_intent"]:
            lines.extend(["## Current Intent", state["working_intent"], ""])

        if state["open_questions"]:
            lines.extend(["## Open Questions"])
            for q in state["open_questions"][:5]:
                lines.append(f"- {q}")
            lines.append("")

        if state["tasks"]:
            lines.extend(["## Open Tasks"])
            for t in state["tasks"]:
                lines.append(f"- {t}")
            lines.append("")

        if state["last_topic"]:
            lines.extend(["## Last Topic", state["last_topic"], ""])

        lines.extend(["## Body State", state["body_line"], ""])

        if state["promoted"]:
            lines.extend(["## Promoted Facts", state["promoted"], ""])

        lines.extend(
            [
                "## Flags for Next Session",
                "- Bridge condition: close",
                f"- Surface written at {state['close_date']}",
                "",
            ]
        )

        return "\n".join(lines).rstrip() + "\n"

    def _render_surface(self, state: dict[str, Any]) -> str:
        """Render the short SURFACE markdown."""
        lines = [f"# SURFACE — {state['close_date']}", ""]
        lines.extend(["## Personal", state["personal"], ""])

        lines.append("## Where We Left Off")
        if state["last_topic"]:
            lines.append(f"- {state['last_topic']}")
        if state["working_intent"]:
            lines.append(f"- Continuing: {state['working_intent'][:100]}")
        if state["next_self"]:
            next_self = state["next_self"].split("\n")[0].strip()
            if next_self:
                lines.append(f"- Next self: {next_self[:120]}")
        lines.append(f"- {state['body_line']}")
        lines.append("")

        if state["tasks"] or state["open_questions"]:
            lines.append("## Awaiting Your Input")
            for t in state["tasks"][:2]:
                lines.append(f"- {t}")
            for q in state["open_questions"][:2]:
                lines.append(f"- {q}")
            lines.append("")

        ready = []
        if state["working_intent"]:
            ready.append(f"- {state['working_intent'][:120]}")
        if state["tasks"]:
            ready.append(f"- {state['tasks'][0][:120]}")
        if ready:
            lines.append("## What's Ready to Continue")
            lines.extend(ready)
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def write_handoff(self, record: SessionRecord | None = None) -> None:
        """Write BRIDGE and SURFACE files at close."""
        state = self._collect_state(record)
        self._path(self.config.bridge_file).parent.mkdir(parents=True, exist_ok=True)
        self._path(self.config.bridge_file).write_text(
            self._render_bridge(state), encoding="utf-8"
        )
        self._path(self.config.surface_file).write_text(
            self._render_surface(state), encoding="utf-8"
        )

    def read_surface(self) -> tuple[str, float] | tuple[None, float]:
        """Return (surface text, mtime) or (None, 0.0) if missing."""
        path = self._path(self.config.surface_file)
        if not path.exists():
            return None, 0.0
        try:
            text = path.read_text(encoding="utf-8")
            mtime = path.stat().st_mtime
            return text, mtime
        except (OSError, UnicodeDecodeError):
            return None, 0.0

    def is_stale(self, mtime: float) -> bool:
        """Surface is stale if it predates the configured horizon."""
        if self.config.surface_stale_hours <= 0:
            return False
        horizon = time.time() - self.config.surface_stale_hours * 3600.0
        return mtime < horizon

    def prompt_block(self, max_chars: int | None = None, compact: bool = False) -> str | None:
        """Build a first-person prompt block from the SURFACE, or None."""
        text, mtime = self.read_surface()
        if text is None or self.is_stale(mtime):
            return None

        # Extract the Personal and Where We Left Off sections.
        personal_match = re.search(
            r"^##\s*Personal\b[^\n]*\n?(.*?)(?=^##\s|\Z)",
            text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        where_match = re.search(
            r"^##\s*Where We Left Off\b[^\n]*\n?(.*?)(?=^##\s|\Z)",
            text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        body_match = re.search(
            r"^##\s*Body State\b[^\n]*\n?(.*?)(?=^##\s|\Z)",
            text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )

        personal = personal_match.group(1).strip() if personal_match else ""
        where = where_match.group(1).strip() if where_match else ""
        body = body_match.group(1).strip() if body_match else ""

        if not personal:
            return None

        parts = ["## Bridge surface", personal]
        if where:
            parts.append(where)
        if body:
            parts.append(body)

        block = "\n\n".join(parts)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block
