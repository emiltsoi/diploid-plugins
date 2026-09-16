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
    felt_events_max: int = 5
    felt_event_record_threshold: float = 0.5
    felt_events_max_age_hours: float = 168.0
    # Richer body-state (negotiated design): persona-scope file for the fields
    # that belong to the one body — energy, mood_tint, and the persona
    # attention fallback. Per-chat files hold attention + derived fields.
    persona_state_file: str = "persona_body_state.json"
    energy_baseline: float = 0.5  # self-reported energy drifts toward "steady"
    self_report_drift_per_minute: float = 0.02  # exp drift rate toward baseline
    attention_stale_seconds: float = 7200.0  # stale attention drifts to "scattered"
    field_stale_seconds: float = 14400.0  # prompt dims fields older than this
    fatigue_rise_cap_hours: float = 10.0  # hours of wake to reach fatigue 1.0
    fatigue_rest_cap_hours: float = 8.0  # hours of silence for full recovery
