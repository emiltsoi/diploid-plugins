"""Checkpoint plugin configuration."""

from dataclasses import dataclass


@dataclass
class CheckpointConfig:
    """Default configuration for the mid-turn checkpoint/breadcrumb plugin."""

    active_checkpoint_file: str = "chat_active_checkpoint.json"
    breadcrumbs_file: str = "chat_turn_breadcrumbs.jsonl"
    last_turn_breadcrumbs_file: str = "chat_last_turn_breadcrumbs.jsonl"
    throttle_seconds: float = 0.5
    stale_hours: float = 24.0
    max_breadcrumbs: int = 256
    max_prompt_chars: int = 512
    max_message_snippet: int = 120
    max_thought_snippet: int = 120
    max_user_message_snippet: int = 120
