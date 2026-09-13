"""Cron digest prompt-slot plugin.

Renders the last-run status of declarative cron jobs as bounded context —
the "don't interrupt me, but let me know" delivery mode. Reads the
``<chat>/cron/*.last`` files the harness ``CronService`` writes after each
run, plus ``service.last`` for reload warnings (parse errors, dropped jobs).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.plugins.base import StatePlugin
from diploid_agent.runtime.plugin_runtime import PluginRuntime

from diploid_plugins.cron.config import CronDigestConfig

logger = logging.getLogger(__name__)


def _age(now: float, ts: float | None) -> str:
    if not ts:
        return "never"
    minutes = (now - ts) / 60.0
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)}m ago"
    hours = minutes / 60.0
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours / 24.0)}d ago"


class CronPlugin(StatePlugin):
    """Render cron last-run results into the ``cron`` prompt slot."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._config = CronDigestConfig()
        for key in CronDigestConfig.__dataclass_fields__:
            if key in config.config:
                setattr(self._config, key, config.config[key])

    def _results_dir(self) -> Path:
        return self.sessions_root / self.chat_id.replace("/", "_") / self._config.results_dirname

    def _read_last(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def prompt_block(self, max_chars: int | None = None, compact: bool = False) -> str | None:
        out_dir = self._results_dir()
        if not out_dir.is_dir():
            return None
        now = time.time()
        stale_after = self._config.stale_hours * 3600.0

        entries: list[tuple[float, str]] = []
        warnings: list[str] = []
        for last_file in sorted(out_dir.glob("*.last")):
            data = self._read_last(last_file)
            if data is None:
                continue
            if last_file.name == "service.last":
                warnings = [str(w) for w in data.get("warnings", [])]
                continue
            if data.get("delivery") == "silent" and data.get("status") != "disabled":
                continue  # files only — but auto-disable always surfaces
            finished = data.get("finished_at") or 0.0
            if stale_after and finished and now - finished > stale_after:
                continue
            entries.append((finished, data))

        if not entries and not warnings:
            return None

        entries.sort(key=lambda item: item[0], reverse=True)
        entries = entries[: self._config.max_jobs]

        lines = ["## Cron jobs", ""]
        for _finished, data in entries:
            status = data.get("status") or "?"
            summary = (data.get("summary") or "").strip()
            if len(summary) > self._config.max_summary_chars:
                summary = summary[: self._config.max_summary_chars].rstrip() + "…"
            marker = "! " if status in ("failed", "disabled") else ""
            line = (
                f"- {marker}{data.get('job_id', '?')} {status} {_age(now, data.get('finished_at'))}"
            )
            if summary:
                line += f" — {summary}"
            lines.append(line)
        for warning in warnings:
            lines.append(f"- ! service: {warning}")

        block = "\n".join(lines)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block or None


Plugin = CronPlugin
