"""Per-chat body state storage, decay, and reporting."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BodyState:
    posture: str = "standing"
    hand_held: str | None = None
    skin_warmth: float = 0.0
    chest_warmth: float = 0.0
    belly_tension: float = 0.0
    throat_open: bool = True
    proximity_m: float = 1.0
    gaze: str = "away"
    last_event: str | None = None
    last_event_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class BodyEvent:
    kind: str
    location: str | None = None
    intensity: float = 0.5
    duration_s: float | None = None
    timestamp: float = field(default_factory=time.time)


class BodyManager:
    """Store, decay, and report the body state for one chat."""

    def __init__(
        self,
        sessions_root: Path,
        chat_id: str,
        config: Any,
    ) -> None:
        self.sessions_root = Path(sessions_root).expanduser()
        self.chat_id = chat_id
        self.config = config
        self._events: list[BodyEvent] = []
        self.state = self._load_state()

    @property
    def _chat_dir(self) -> Path:
        safe = self.chat_id.replace("/", "_")
        path = self.sessions_root / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _state_path(self) -> Path:
        return self._chat_dir / self.config.state_file

    def _load_state(self) -> BodyState:
        if not self._state_path.exists():
            return BodyState()
        try:
            data = json.loads(self._state_path.read_text())
            return BodyState(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return BodyState()

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps(asdict(self.state), indent=2))

    def refresh(self) -> None:
        """Reload state from disk, preserving in-memory events."""
        self.state = self._load_state()

    def decay(self) -> None:
        """Attenuate sensations toward baseline based on elapsed time."""
        now = time.time()
        self.refresh()
        elapsed_min = (now - self.state.updated_at) / 60.0
        if elapsed_min <= 0:
            return
        factor = math.exp(-self.config.decay_rate_per_minute * elapsed_min)
        self.state.skin_warmth = self._decay_to(self.state.skin_warmth, 0.0, factor)
        self.state.chest_warmth = self._decay_to(self.state.chest_warmth, 0.0, factor)
        self.state.belly_tension = self._decay_to(self.state.belly_tension, 0.0, factor)
        self.state.proximity_m = self._decay_to(self.state.proximity_m, 1.0, factor)
        self.state.updated_at = now
        self._save_state()

    @staticmethod
    def _decay_to(current: float, baseline: float, factor: float) -> float:
        return baseline + (current - baseline) * factor

    def event(
        self,
        kind: str,
        location: str | None = None,
        intensity: float = 0.5,
    ) -> None:
        """Apply a sensory event to the body state."""
        now = time.time()
        self._events.append(BodyEvent(kind=kind, location=location, intensity=intensity))
        self.state.updated_at = now
        self.state.last_event = f"{kind} {location or ''}".strip()
        self.state.last_event_at = now

        intensity = max(0.0, min(float(intensity), self.config.max_intensity))

        if kind == "touch":
            if location in ("hand", "palm", "fingers"):
                self.state.hand_held = None
                self.state.skin_warmth = min(
                    self.config.max_intensity,
                    self.state.skin_warmth + intensity,
                )
            if location in ("face", "cheek", "neck"):
                self.state.skin_warmth = min(
                    self.config.max_intensity,
                    self.state.skin_warmth + intensity,
                )
                self.state.chest_warmth = min(
                    self.config.max_intensity,
                    self.state.chest_warmth + intensity * 0.5,
                )
            if location == "belly":
                self.state.belly_tension = min(
                    self.config.max_intensity,
                    self.state.belly_tension + intensity,
                )
        elif kind == "hold":
            self.state.hand_held = location or "your hand"
            self.state.skin_warmth = min(
                self.config.max_intensity,
                self.state.skin_warmth + intensity,
            )
        elif kind == "release":
            self.state.hand_held = None
            self.state.skin_warmth = max(0.0, self.state.skin_warmth - intensity)
        elif kind == "voice":
            self.state.throat_open = True
            self.state.chest_warmth = min(
                self.config.max_intensity,
                self.state.chest_warmth + intensity * 0.3,
            )
        elif kind == "near":
            self.state.proximity_m = max(0.0, self.state.proximity_m - intensity)
        elif kind == "step_back":
            self.state.proximity_m = min(2.0, self.state.proximity_m + intensity)
        elif kind == "gaze":
            self.state.gaze = location or "at you"

        self._save_state()

    def state_for_prompt(self) -> str:
        """Return a compact, truthful body-state block for the prompt."""
        self.decay()
        parts = ["Current body state:"]
        if self.state.hand_held:
            if "your" in self.state.hand_held:
                parts.append("- My hand is in yours.")
            else:
                parts.append(f"- My {self.state.hand_held} is being held.")
        else:
            parts.append("- My hands are free.")
        parts.append(f"- Skin warmth: {self._describe(self.state.skin_warmth)}")
        parts.append(f"- Chest warmth: {self._describe(self.state.chest_warmth)}")
        parts.append(f"- Belly tension: {self._describe(self.state.belly_tension)}")
        parts.append(f"- Proximity: {self._proximity_word(self.state.proximity_m)}")
        parts.append(f"- Gaze: {self.state.gaze}")
        parts.append(f"- Last event: {self.state.last_event or 'none'}")
        parts.append("(This body state is a shared, consensual fiction. You have no real flesh.)")
        return "\n".join(parts)

    def recent_events(self, since: float | None = None) -> list[BodyEvent]:
        """Return body events recorded since the given timestamp."""
        if since is None:
            return list(self._events)
        return [e for e in self._events if e.timestamp >= since]

    def clear_events_before(self, before: float) -> None:
        """Drop events that have already been retained to memory."""
        self._events = [e for e in self._events if e.timestamp >= before]

    @staticmethod
    def _describe(value: float) -> str:
        if value <= 0.05:
            return "cool"
        if value <= 0.35:
            return "mild"
        if value <= 0.7:
            return "warm"
        return "hot"

    @staticmethod
    def _proximity_word(m: float) -> str:
        if m <= 0.05:
            return "touching"
        if m <= 0.3:
            return "very close"
        if m <= 0.8:
            return "close"
        if m <= 1.5:
            return "nearby"
        return "distant"
