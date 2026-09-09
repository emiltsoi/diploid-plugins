"""Bridge/SURFACE configuration."""

from dataclasses import dataclass


@dataclass
class BridgeConfig:
    """Default configuration for the bridge/surface handoff."""

    bridge_file: str = "chat_bridge.md"
    surface_file: str = "chat_surface.md"
    self_state_file: str = "chat_self_state.md"
    body_state_file: str = "chat_body_state.json"
    working_memory_file: str = "chat_working_memory.json"
    tasks_file: str = "chat_TASKS.md"
    wake_state_file: str = "chat_wake_state.json"
    promoted_file: str = "chat_PROMOTED.md"
    surface_stale_hours: float = 24.0
    max_surface_chars: int = 512
