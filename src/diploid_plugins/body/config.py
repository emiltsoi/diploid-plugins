"""Configuration dataclass for the body state plugin."""

from dataclasses import dataclass


@dataclass
class BodyConfig:
    enabled: bool = True
    max_prompt_chars: int = 1024
    state_file: str = "chat_body_state.json"
    decay_rate_per_minute: float = 0.05
    max_intensity: float = 1.0
