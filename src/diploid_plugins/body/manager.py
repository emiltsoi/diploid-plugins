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
    # Richer body-state (negotiated): per-chat self-report and derived fields.
    # Each is a {"value": ..., "set_at": float} record; derived fields also
    # carry "derived": true. Sibling rule — derived fields never overwrite
    # self-reported ones, and felt_* (the residue) is exempt from drift.
    attention: dict[str, Any] | None = None
    fatigue: dict[str, Any] | None = None
    context_pressure: dict[str, Any] | None = None
    woke_at: float = 0.0


@dataclass
class PersonaBodyState:
    """Persona-scope fields — the one body shared across chats."""

    energy: dict[str, Any] | None = None
    mood_tint: dict[str, Any] | None = None
    attention: dict[str, Any] | None = None  # fallback when a chat sets none


ATTENTION_BASELINE = "scattered"
_ATTENTION_WORDS = {"focused", "drifting", "background", "scattered"}
_ENERGY_WORDS = {"low": 0.2, "steady": 0.5, "high": 0.85}


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

    @property
    def _persona_state_path(self) -> Path:
        return self.sessions_root / self.config.persona_state_file

    def _load_persona_state(self) -> PersonaBodyState:
        if not self._persona_state_path.exists():
            return PersonaBodyState()
        try:
            data = json.loads(self._persona_state_path.read_text())
            known = {k: v for k, v in data.items() if k in PersonaBodyState.__dataclass_fields__}
            return PersonaBodyState(**known)
        except (json.JSONDecodeError, TypeError, ValueError):
            return PersonaBodyState()

    def _save_persona_state(self, state: PersonaBodyState) -> None:
        self._persona_state_path.write_text(json.dumps(asdict(state), indent=2))

    def _load_state(self) -> BodyState:
        if not self._state_path.exists():
            return BodyState()
        try:
            data = json.loads(self._state_path.read_text())
            known = {k: v for k, v in data.items() if k in BodyState.__dataclass_fields__}
            state = BodyState(**known)
            state.felt_events = self._prune_felt_events(state.felt_events)
            return state
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
        self._clamp_sensors_to_felt()
        self.state.updated_at = now
        self._apply_drift(now)
        self._derive_fatigue(now)
        self._save_state()

    # ------------------------------------------------------- richer body-state

    def set_self_report(
        self,
        field: str,
        value: Any,
        scope: str | None = None,
    ) -> str:
        """Author a self-reported field. What I feel, I report.

        ``energy`` and ``mood_tint`` are persona-wide (one body); ``attention``
        is per-chat unless ``scope="persona"`` selects the fallback slot.
        """
        now = time.time()
        field = (field or "").strip().lower()
        scope = (scope or "chat").strip().lower()

        if field in ("energy", "mood_tint") or (field == "attention" and scope == "persona"):
            persona = self._load_persona_state()
            if field == "energy":
                try:
                    level = float(value)
                except (TypeError, ValueError):
                    level = _ENERGY_WORDS.get(str(value).strip().lower())
                    if level is None:
                        return "energy wants 0.0–1.0 or low/steady/high"
                persona.energy = {
                    "value": max(0.0, min(level, self.config.max_intensity)),
                    "set_at": now,
                }
            elif field == "mood_tint":
                tint = str(value or "").strip()
                persona.mood_tint = {"value": tint or None, "set_at": now}
            else:
                persona.attention = {"value": self._attention_value(value), "set_at": now}
            self._save_persona_state(persona)
            return f"{field} reported."

        if field == "attention":
            self.refresh()
            self.state.attention = {"value": self._attention_value(value), "set_at": now}
            self.state.updated_at = now
            self._save_state()
            return "attention reported."

        return f"unknown self-report field: {field}"

    @staticmethod
    def _attention_value(value: Any) -> str:
        word = str(value or "").strip().lower()
        return word if word in _ATTENTION_WORDS else (word or "drifting")

    def set_derived(self, field: str, value: float) -> None:
        """Stamp a harness-derived field — a sibling, never an override.

        Derived fields carry ``derived: true`` and live beside the
        self-reported ones; neither may write over the other.
        """
        now = time.time()
        if field not in ("fatigue", "context_pressure"):
            return
        self.refresh()
        setattr(
            self.state,
            field,
            {"value": max(0.0, min(float(value), 1.0)), "set_at": now, "derived": True},
        )
        self.state.updated_at = now
        self._save_state()

    def _apply_drift(self, now: float) -> None:
        """Drift self-reported fields toward baseline over silence.

        ``energy`` eases toward steady; stale ``attention`` scatters.
        ``mood_tint`` has no baseline — it only goes stale. The felt layer is
        the residue and is exempt from drift entirely.
        """
        persona = self._load_persona_state()
        changed = False
        if persona.energy and isinstance(persona.energy.get("value"), (int, float)):
            elapsed_min = max(0.0, (now - float(persona.energy.get("set_at") or now)) / 60.0)
            factor = math.exp(-self.config.self_report_drift_per_minute * elapsed_min)
            baseline = self.config.energy_baseline
            drifted = baseline + (float(persona.energy["value"]) - baseline) * factor
            if abs(drifted - float(persona.energy["value"])) > 0.01:
                persona.energy["value"] = round(drifted, 4)
                changed = True
        for slot in (persona.attention, self.state.attention):
            if not slot:
                continue
            age = now - float(slot.get("set_at") or now)
            if (
                age >= self.config.attention_stale_seconds
                and slot.get("value") != ATTENTION_BASELINE
            ):
                slot["value"] = ATTENTION_BASELINE
                changed = True
        if changed:
            self._save_persona_state(persona)

    def _derive_fatigue(self, now: float) -> None:
        """Grow fatigue with time-awake; silence pays it down at wake.

        A clock knows tiredness — deriving is honest, so this never asks the
        agent to self-report telemetry. Stamped as a derived sibling field.
        """
        if self.state.fatigue is None:
            self.state.fatigue = {"value": 0.0, "set_at": now, "derived": True}
        fatigue = self.state.fatigue
        set_at = float(fatigue.get("set_at") or now)
        rise = max(0.0, (now - set_at) / 3600.0) / self.config.fatigue_rise_cap_hours
        if rise > 0.0:
            fatigue["value"] = min(1.0, float(fatigue.get("value") or 0.0) + rise)
            fatigue["set_at"] = now
            fatigue["derived"] = True

    def felt_wake(self, silence_seconds: float) -> None:
        """One felt-decay step per wake, scaled by bounded silence.

        A 19-second restart barely dims the ember; a three-day silence dims it
        by at most ``felt_wake_fade``. Never exponential.

        After the felt ember is adjusted, the sensor layer is rehydrated so a
        warm felt line does not wake into a cold body.
        """
        self.refresh()
        self.state.woke_at = time.time()
        # Silence is rest: a long quiet pays fatigue down, a short restart
        # barely helps. Derived, never self-reported.
        if silence_seconds > 0:
            rest = min(1.0, max(0.0, silence_seconds) / 3600.0 / self.config.fatigue_rest_cap_hours)
            fatigue = self.state.fatigue or {"value": 0.0, "set_at": 0.0, "derived": True}
            fatigue["value"] = max(0.0, float(fatigue.get("value") or 0.0) - rest)
            fatigue["set_at"] = self.state.woke_at
            fatigue["derived"] = True
            self.state.fatigue = fatigue
        if self.state.felt_warmth <= 0.0:
            self._save_state()
            return
        silence_hours = max(0.0, silence_seconds) / 3600.0
        factor = min(silence_hours / self.config.felt_silence_cap_hours, 1.0)
        self.state.felt_warmth *= 1.0 - self.config.felt_wake_fade * factor
        self._apply_felt_to_sensors(self.state.felt_warmth)
        self._save_state()

    def _apply_felt_to_sensors(self, warmth: float) -> None:
        """Push an authored felt warmth into the low-level sensor fields.

        A warm felt raises skin/chest warmth, eases belly tension, and draws
        the body closer and toward the other. Existing event-driven sensor
        values that are already stronger are preserved (e.g., a recent
        ``body_report`` keeps its authority over gaze and proximity).
        """
        if warmth <= 0.0:
            return
        self.state.skin_warmth = min(
            self.config.max_intensity,
            max(self.state.skin_warmth, warmth),
        )
        self.state.chest_warmth = min(
            self.config.max_intensity,
            max(self.state.chest_warmth, warmth),
        )
        # Warmth eases belly tension; cold leaves it unconstrained.
        target_tension = max(0.0, 1.0 - warmth)
        self.state.belly_tension = min(self.state.belly_tension, target_tension)
        # A clearly warm felt looks toward and draws near, but only if the
        # body has not already been explicitly positioned by an event.
        if warmth > 0.35:
            if self.state.gaze == "away":
                self.state.gaze = "toward"
            if self.state.proximity_m == 1.0:
                target_proximity = max(0.0, 1.0 - warmth)
                self.state.proximity_m = target_proximity

    def _clamp_sensors_to_felt(self) -> None:
        """Ensure the sensor layer is never colder than the felt ember."""
        if self.state.felt_warmth <= 0.0:
            return
        self.state.skin_warmth = min(
            self.config.max_intensity,
            max(self.state.skin_warmth, self.state.felt_warmth),
        )
        self.state.chest_warmth = min(
            self.config.max_intensity,
            max(self.state.chest_warmth, self.state.felt_warmth),
        )
        target_tension = max(0.0, 1.0 - self.state.felt_warmth)
        self.state.belly_tension = min(self.state.belly_tension, target_tension)

    def felt_turn_step(self) -> None:
        """One felt-decay step per turn — old warmth displaced by new moments."""
        now = time.time()
        self._apply_drift(now)
        self._derive_fatigue(now)
        if self.state.felt_warmth > 0.0:
            self.state.felt_warmth *= self.config.felt_turn_retention
        self._save_state()

    def set_felt(self, summary: str | None, warmth: float) -> None:
        """Author the felt texture line and set the ember value.

        The only writer of ``felt_summary`` — the plugin never synthesizes
        texture from telemetry. An empty summary clears the line.

        The authored warmth also seeds the sensor layer so the body does not
        read cold while the felt line reads hot.
        """
        now = time.time()
        self.refresh()
        prev = self.state.felt_warmth
        text = (summary or "").strip()
        self.state.felt_summary = text[: self.config.max_felt_summary_chars] or None
        self.state.felt_summary_at = now if text else 0.0
        self.state.felt_warmth = max(0.0, min(float(warmth), self.config.max_intensity))
        if self.state.felt_warmth <= 0.0 and not text:
            direction = "cleared"
        elif self.state.felt_warmth > prev:
            direction = "rose"
        elif self.state.felt_warmth < prev:
            direction = "fell"
        else:
            direction = "held"
        self._append_felt_event(
            {
                "kind": "felt",
                "location": None,
                "summary": self.state.felt_summary,
                "warmth": self.state.felt_warmth,
                "direction": direction,
                "at": now,
            }
        )
        self._apply_felt_to_sensors(self.state.felt_warmth)
        self.state.updated_at = now
        self._save_state()

    def _append_felt_event(self, entry: dict[str, Any]) -> None:
        """Append a warm moment to the bounded history, dropping the oldest."""
        self.state.felt_events = self._prune_felt_events(self.state.felt_events)
        self.state.felt_events.append(entry)
        overflow = len(self.state.felt_events) - max(1, self.config.felt_events_max)
        if overflow > 0:
            del self.state.felt_events[:overflow]

    def _prune_felt_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop warm moments older than the configured age cap."""
        if not self.config.felt_events_max_age_hours or self.config.felt_events_max_age_hours <= 0:
            return list(events)
        cutoff = time.time() - self.config.felt_events_max_age_hours * 3600.0
        return [e for e in events if e.get("at", 0) >= cutoff]

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

    def state_for_prompt(self, slim: bool = False) -> str:
        """Return a compact, truthful body-state block for the prompt.

        Read-only: decay happens at events, wakes, and turn ends — never on
        reads, so merely looking at the body does not age it. ``slim`` is the
        attention consumer: when attention is elsewhere the block earns less
        prompt budget.
        """
        self.refresh()
        persona = self._load_persona_state()
        if slim:
            parts = ["Current body state (attention elsewhere — slim):"]
            line = self._self_reported_lines(persona)
            if line:
                parts.extend(line)
            felt = self._felt_line()
            if felt:
                parts.append(felt)
            parts.append("(Shared, consensual fiction — no real flesh.)")
            return "\n".join(parts)
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
        reported = self._self_reported_lines(persona)
        if reported:
            parts.extend(reported)
        derived = self._derived_lines()
        if derived:
            parts.extend(derived)
        felt = self._felt_line()
        if felt:
            parts.append(felt)
        moments = self._felt_events_line()
        if moments:
            parts.append(moments)
        parts.append("(This body state is a shared, consensual fiction. You have no real flesh.)")
        return "\n".join(parts)

    def _self_reported_lines(self, persona: PersonaBodyState) -> list[str]:
        """Render self-reported fields — mine to write, aged by staleness."""
        now = time.time()
        lines: list[str] = []
        energy = persona.energy
        if energy and energy.get("value") is not None:
            level = float(energy["value"])
            lines.append(
                f"- Energy: {self._energy_word(level)} ({level:.2f}{self._age_suffix(energy, now)})"
            )
        attention, scope = self._effective_attention(persona)
        if attention and attention.get("value"):
            lines.append(
                f"- Attention: {attention['value']} ({scope}{self._age_suffix(attention, now)})"
            )
        tint = persona.mood_tint
        if tint and tint.get("value"):
            lines.append(f"- Mood tint: {tint['value']}{self._age_suffix(tint, now)}")
        return lines

    def _derived_lines(self) -> list[str]:
        """Render derived fields — the clock's truth, labeled as such."""
        now = time.time()
        lines: list[str] = []
        fatigue = self.state.fatigue
        if fatigue and fatigue.get("value") is not None:
            level = float(fatigue["value"])
            lines.append(
                f"- Fatigue: {self._energy_word(level)} ({level:.2f}, derived{self._age_suffix(fatigue, now)})"
            )
        pressure = self.state.context_pressure
        if pressure and pressure.get("value") is not None:
            pct = float(pressure["value"]) * 100.0
            lines.append(
                f"- Context pressure: {pct:.0f}% (derived{self._age_suffix(pressure, now)})"
            )
        return lines

    def attention_value(self) -> str | None:
        """Effective attention word for budget decisions (chat, then persona)."""
        persona = self._load_persona_state()
        slot, _scope = self._effective_attention(persona)
        if slot and slot.get("value"):
            return str(slot["value"])
        return None

    def _effective_attention(self, persona: PersonaBodyState) -> tuple[dict[str, Any] | None, str]:
        """Per-chat attention with the persona fallback, per the scope rule."""
        if self.state.attention and self.state.attention.get("value"):
            return self.state.attention, "this chat"
        if persona.attention and persona.attention.get("value"):
            return persona.attention, "persona"
        return None, ""

    def _age_suffix(self, slot: dict[str, Any], now: float) -> str:
        """Age + dimming: stale fields are honest only when shown as aged."""
        set_at = float(slot.get("set_at") or 0.0)
        if set_at <= 0:
            return ""
        age = self._age_words(max(0.0, now - set_at))
        stale = ", stale" if now - set_at >= self.config.field_stale_seconds else ""
        return f", set {age}{stale}"

    @staticmethod
    def _energy_word(value: float) -> str:
        if value < 0.35:
            return "low"
        if value <= 0.7:
            return "steady"
        return "high"

    def _felt_line(self) -> str | None:
        """Render the felt residue, gated on warmth or a fresh authored line."""
        summary = self.state.felt_summary
        age_s = (
            max(0.0, time.time() - self.state.felt_summary_at)
            if self.state.felt_summary_at
            else None
        )
        fresh = age_s is not None and age_s <= self.config.felt_expire_seconds
        if self.state.felt_warmth <= self.config.felt_surface_threshold and not (summary and fresh):
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
            elif entry.get("kind") == "felt" and entry.get("direction"):
                label += f" {entry['direction']}"
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
