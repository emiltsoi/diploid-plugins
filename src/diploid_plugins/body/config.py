"""Configuration dataclass for the body state plugin."""

from dataclasses import dataclass


@dataclass
class BodyConfig:
    enabled: bool = True
    max_prompt_chars: int = 1024
    state_file: str = "chat_body_state.json"
    decay_rate_per_minute: float = 0.05
    max_intensity: float = 1.0
    felt_turn_retention: float = 0.97
    felt_wake_fade: float = 0.2
    felt_silence_cap_hours: float = 72.0
    felt_surface_threshold: float = 0.02
    felt_expire_seconds: float = 259200.0
    felt_event_gain: float = 0.15
    max_felt_summary_chars: int = 140
