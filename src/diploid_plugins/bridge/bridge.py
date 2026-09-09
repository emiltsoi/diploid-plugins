"""Bridge/SURFACE handoff plugin."""

from __future__ import annotations

from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.models import SessionRecord
from diploid_agent.plugins.base import (
    SessionArchiveContext,
    ShutdownContext,
    SleepContext,
    StatePlugin,
    WakeContext,
)
from diploid_agent.runtime.plugin_runtime import PluginRuntime

from diploid_plugins.bridge.config import BridgeConfig
from diploid_plugins.bridge.manager import BridgeManager


class BridgePlugin(StatePlugin):
    """Write a BRIDGE at close and surface a first-person re-entry at wake."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        cfg = BridgeConfig()
        for key in BridgeConfig.__dataclass_fields__:
            if key in config.config:
                setattr(cfg, key, config.config[key])
        self._manager = BridgeManager(chat_id, sessions_root, cfg)

    def _record_from_context(
        self, context: SessionArchiveContext | SleepContext | ShutdownContext
    ) -> SessionRecord | None:
        record = getattr(context, "old_record", None)
        if isinstance(record, SessionRecord):
            return record
        return None

    def before_session_archive(self, context: SessionArchiveContext) -> None:
        self._manager.write_handoff(self._record_from_context(context))

    def on_sleeping(self, context: SleepContext) -> None:
        self._manager.write_handoff(self._record_from_context(context))

    def on_shutdown(self, context: ShutdownContext) -> None:
        self._manager.write_handoff(self._record_from_context(context))

    def on_waking(self, context: WakeContext) -> None:
        # Pre-load surface freshness; prompt_block does the actual rendering.
        self._manager.read_surface()

    def prompt_block(self, max_chars: int | None = None, compact: bool = False) -> str | None:
        return self._manager.prompt_block(max_chars, compact=compact)
