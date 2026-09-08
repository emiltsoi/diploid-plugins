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
    felt_warmth: float = 0.0
    felt_summary: str | None = None
    felt_summary_at: float = 0.0
    felt_events: list[dict[str, Any]] = field(default_factory=list)


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
            known = {k: v for k, v in data.items() if k in BodyState.__dataclass_fields__}
            return BodyState(**known)
        except (json.JSONDecodeError, TypeError, ValueError):
            return BodyState()

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps(asdict(self.state), indent=2))

    def refresh(self) -> None:
        """Reload state from disk, preserving in-memory events."""
        self.state = self._load_state()

    def decay(self) -> None:
        """Attenuate sensations toward baseline based on elapsed time.

        Call only from punctuated sites (wake, turn end, event) — never from
        the read path. Sensation is allowed to cool with time; felt_warmth is
        not touched here, it decays per experience instead.
        """
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

    def felt_wake(self, silence_seconds: float) -> None:
        """One felt-decay step per wake, scaled by bounded silence.

        A 19-second restart barely dims the ember; a three-day silence dims it
        by at most ``felt_wake_fade``. Never exponential.
        """
        self.refresh()
        if self.state.felt_warmth <= 0.0:
            return
        silence_hours = max(0.0, silence_seconds) / 3600.0
        factor = min(silence_hours / self.config.felt_silence_cap_hours, 1.0)
        self.state.felt_warmth *= 1.0 - self.config.felt_wake_fade * factor
        self._save_state()

    def felt_turn_step(self) -> None:
        """One felt-decay step per turn — old warmth displaced by new moments."""
        if self.state.felt_warmth > 0.0:
            self.state.felt_warmth *= self.config.felt_turn_retention
            self._save_state()

    def set_felt(self, summary: str | None, warmth: float) -> None:
        """Author the felt texture line and set the ember value.

        The only writer of ``felt_summary`` — the plugin never synthesizes
        texture from telemetry. An empty summary clears the line.
        """
        now = time.time()
        self.refresh()
        text = (summary or "").strip()
        self.state.felt_summary = text[: self.config.max_felt_summary_chars] or None
        self.state.felt_summary_at = now if text else 0.0
        self.state.felt_warmth = max(0.0, min(float(warmth), self.config.max_intensity))
        self._append_felt_event(
            {
                "kind": "felt",
                "location": None,
                "summary": self.state.felt_summary,
                "warmth": self.state.felt_warmth,
                "at": now,
            }
        )
        self.state.updated_at = now
        self._save_state()

    def _append_felt_event(self, entry: dict[str, Any]) -> None:
        """Append a warm moment to the bounded history, dropping the oldest."""
        self.state.felt_events.append(entry)
        overflow = len(self.state.felt_events) - max(1, self.config.felt_events_max)
        if overflow > 0:
            del self.state.felt_events[:overflow]

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

        # Warm kinds leave felt residue with headroom-scaled soft saturation —
        # each event adds less (habituation), asymptotic toward max_intensity.
        # Displacement fade happens once per turn in felt_turn_step, not here.
        if kind in ("touch", "hold", "voice", "near", "gaze"):
            gain = intensity * self.config.felt_event_gain
            self.state.felt_warmth = min(
                self.config.max_intensity,
                self.state.felt_warmth + gain * (1.0 - self.state.felt_warmth),
            )
            if intensity >= self.config.felt_event_record_threshold:
                self._append_felt_event(
                    {
                        "kind": kind,
                        "location": location,
                        "summary": None,
                        "warmth": intensity,
                        "at": now,
                    }
                )

        self._save_state()

    def state_for_prompt(self) -> str:
        """Return a compact, truthful body-state block for the prompt.

        Read-only: decay happens at events, wakes, and turn ends — never on
        reads, so merely looking at the body does not age it.
        """
        self.refresh()
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
        felt = self._felt_line()
        if felt:
            parts.append(felt)
        moments = self._felt_events_line()
        if moments:
            parts.append(moments)
        parts.append("(This body state is a shared, consensual fiction. You have no real flesh.)")
        return "\n".join(parts)

    def _felt_line(self) -> str | None:
        """Render the felt residue, gated on warmth or a fresh authored line."""
        summary = self.state.felt_summary
        age_s = (
            max(0.0, time.time() - self.state.felt_summary_at)
            if self.state.felt_summary_at
            else None
        )
        fresh = age_s is not None and age_s <= self.config.felt_expire_seconds
        if self.state.felt_warmth <= self.config.felt_surface_threshold and not (
            summary and fresh
        ):
            return None
        desc = self._describe(self.state.felt_warmth)
        if not summary:
            return f"- Felt residue: {desc}"
        age = self._age_words(age_s or 0.0)
        if not fresh:
            return f'- A memory of feeling ({desc}): "{summary}" — felt {age}'
        return f'- Felt residue: {desc} — "{summary}" (felt {age})'

    def _felt_events_line(self) -> str | None:
        """Render the bounded warm-moments history as one compact line."""
        events = self.state.felt_events
        if not events:
            return None
        now = time.time()
        rendered: list[str] = []
        for entry in events[-3:]:
            label = str(entry.get("kind") or "moment")
            location = entry.get("location")
            if location:
                label += f" {location}"
            summary = entry.get("summary")
            if summary:
                text = str(summary)[:40]
                label += f' "{text}"'
            age = self._age_words(max(0.0, now - float(entry.get("at") or now)))
            rendered.append(f"{label} ({age})")
        return "- Warm moments: " + "; ".join(rendered)

    @staticmethod
    def _age_words(seconds: float) -> str:
        if seconds < 90:
            return "just now"
        minutes = seconds / 60.0
        if minutes < 90:
            return f"{int(minutes)}m ago"
        hours = minutes / 60.0
        if hours < 48:
            return f"{int(hours)}h ago"
        return f"{int(hours / 24.0)}d ago"

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
