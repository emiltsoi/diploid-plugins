"""Body-state plugin for diploid-agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from diploid_agent.config import McpServerConfig, PluginConfig
from diploid_agent.memory import MemoryItem
from diploid_agent.plugins.base import SleepContext, StatePlugin, TurnInfo, WakeContext
from diploid_agent.runtime.plugin_runtime import PluginRuntime

from diploid_plugins.body.config import BodyConfig
from diploid_plugins.body.manager import BodyEvent, BodyManager


class BodyPlugin(StatePlugin):
    """Per-chat body state that survives transport restarts."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Any,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._decay_rate = float(config.config.get("decay_rate_per_minute", 0.05))
        self._max_intensity = float(config.config.get("max_intensity", 1.0))
        felt_keys = (
            "felt_turn_retention",
            "felt_wake_fade",
            "felt_silence_cap_hours",
            "felt_surface_threshold",
            "felt_expire_seconds",
            "felt_event_gain",
            "max_felt_summary_chars",
            "felt_events_max",
            "felt_event_record_threshold",
        )
        self._felt_overrides = {
            k: config.config[k] for k in felt_keys if k in config.config
        }
        body_cfg = BodyConfig(
            state_file=config.state_file or "chat_body_state.json",
            max_prompt_chars=config.max_prompt_chars,
            decay_rate_per_minute=self._decay_rate,
            max_intensity=self._max_intensity,
            **self._felt_overrides,
        )
        self._body = BodyManager(sessions_root, chat_id, body_cfg)

    def mcp_server(self) -> McpServerConfig | None:
        return McpServerConfig(
            name="diploid-body",
            command="python",
            args=[
                "-m",
                "diploid_plugins.body.body_mcp",
                "--chat-id",
                "{chat_id}",
                "--sessions-root",
                "{sessions_root}",
                "--state-file",
                self.config.state_file or "chat_body_state.json",
                "--decay-rate",
                str(self._decay_rate),
                "--max-intensity",
                str(self._max_intensity),
                "--felt-config",
                json.dumps(self._felt_overrides),
            ],
        )

    def _body_map_path(self) -> Path | None:
        if self._runtime is None:
            return None
        try:
            return self._runtime.config.persona.profile_root / "references" / "body.md"
        except AttributeError:
            return None

    def _body_map_text(self) -> str | None:
        path = self._body_map_path()
        if path is None or not path.exists():
            return None
        return path.read_text().strip()

    def on_waking(self, context: WakeContext) -> None:
        """Reload from the restored snapshot and let sensations fade."""
        self._body.refresh()
        self._body.decay()
        since = context.previous_turn_at or self._body.state.updated_at
        silence = max(0.0, context.now - since) if since else 0.0
        self._body.felt_wake(silence)

    def on_sleeping(self, context: SleepContext) -> None:
        """Flush body state to disk before the transport can die."""
        self._body._save_state()

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        body_map = self._body_map_text()
        state = self._body.state_for_prompt()
        text = f"{body_map}\n\n{state}" if body_map else state
        if not text:
            return None
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
        return text

    def event(
        self,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> str:
        kind = event
        if not kind:
            return "Usage: /state body <kind> [location] [intensity]"
        location = params.get("location")
        intensity = params.get("intensity", 0.5)
        if raw_args:
            parts = _split_unquoted(raw_args)
            if not location and len(parts) >= 1:
                location = parts[0]
            if len(parts) >= 2:
                try:
                    intensity = float(parts[1])
                except ValueError:
                    intensity = 0.5
        try:
            intensity = float(intensity)
        except (TypeError, ValueError):
            intensity = 0.5
        if kind == "felt":
            summary = location or (raw_args or "")
            self._body.set_felt(summary, intensity)
            return "Felt line written."
        self._body.event(kind, location, intensity)
        return f"Felt: {kind} {location or ''} ({intensity}).".strip()

    def memory_items(self, since: float) -> list[MemoryItem]:
        return [
            MemoryItem(
                content=_body_event_description(event),
                timestamp=_ts_iso(event.timestamp),
                document_id=f"body-{self.chat_id}-{int(event.timestamp * 1000)}",
                session_number=0,
                metadata={
                    "kind": event.kind,
                    "location": event.location,
                    "intensity": event.intensity,
                    "chat_id": self.chat_id,
                },
                tags=["body", "sensation", f"chat:{self.chat_id}"],
            )
            for event in self._body.recent_events(since=since)
        ]

    def on_turn_end(self, turn: TurnInfo) -> None:
        self._body.clear_events_before(turn.updated_at)
        self._body.decay()
        self._body.felt_turn_step()
        self._body._save_state()


def _body_event_description(event: BodyEvent) -> str:
    return (
        f"{event.kind}"
        f"{f' at {event.location}' if event.location else ''}"
        f" (intensity {event.intensity})"
    )


def _ts_iso(when: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(when, tz=UTC).isoformat()


def _split_unquoted(text: str) -> list[str]:
    parts = re.findall(r'"([^"]*)"|(\S+)', text)
    return [q or u for q, u in parts]
