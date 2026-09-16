"""Tests for the richer body-state fields (negotiated design).

Self-reported energy/mood_tint live persona-wide; attention is per-chat
with a persona fallback; fatigue and context_pressure are derived siblings
that never overwrite self-reports; the felt layer is exempt from drift.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from diploid_plugins.body.config import BodyConfig
from diploid_plugins.body.manager import BodyManager


def _mgr(tmp_path: Path, **cfg) -> BodyManager:
    return BodyManager(tmp_path, "chat-1", BodyConfig(**cfg))


def _persona_file(tmp_path: Path) -> Path:
    return tmp_path / "persona_body_state.json"


def test_self_report_energy_is_persona_wide(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    assert mgr.set_self_report("energy", "high") == "energy reported."
    persona = json.loads(_persona_file(tmp_path).read_text())
    assert persona["energy"]["value"] == 0.85
    assert persona["energy"]["set_at"] > 0
    # Not in the chat record — one body, one record.
    assert not hasattr(mgr.state, "energy") or getattr(mgr.state, "energy", None) is None


def test_self_report_energy_words_and_numbers(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    mgr.set_self_report("energy", 0.7)
    persona = json.loads(_persona_file(tmp_path).read_text())
    assert persona["energy"]["value"] == 0.7
    assert "0.0" in mgr.set_self_report("energy", "bogus")


def test_self_report_attention_scope(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    mgr.set_self_report("attention", "focused")
    mgr.set_self_report("attention", "background", "persona")
    chat = json.loads((tmp_path / "chat-1" / "chat_body_state.json").read_text())
    assert chat["attention"]["value"] == "focused"
    persona = json.loads(_persona_file(tmp_path).read_text())
    assert persona["attention"]["value"] == "background"
    # Per-chat wins over the persona fallback.
    assert mgr.attention_value() == "focused"


def test_self_report_mood_tint(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    mgr.set_self_report("mood_tint", "ember blue")
    persona = json.loads(_persona_file(tmp_path).read_text())
    assert persona["mood_tint"]["value"] == "ember blue"


def test_drift_attention_scatters_when_stale(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path, attention_stale_seconds=60.0)
    mgr.set_self_report("attention", "focused")
    stale = time.time() - 120.0
    mgr.state.attention["set_at"] = stale
    mgr._save_state()
    mgr._apply_drift(time.time())
    assert mgr.state.attention["value"] == "scattered"


def test_drift_energy_eases_toward_baseline(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path, self_report_drift_per_minute=0.05, energy_baseline=0.5)
    mgr.set_self_report("energy", 1.0)
    persona_path = _persona_file(tmp_path)
    persona = json.loads(persona_path.read_text())
    persona["energy"]["set_at"] = time.time() - 600.0  # 10 minutes stale
    persona_path.write_text(json.dumps(persona))
    mgr._apply_drift(time.time())
    persona = json.loads(persona_path.read_text())
    assert 0.5 < persona["energy"]["value"] < 1.0  # half-way home, not all the way


def test_derived_fields_are_siblings(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    mgr.set_self_report("energy", 0.8)
    mgr.set_derived("fatigue", 0.6)
    mgr.set_derived("context_pressure", 0.42)
    chat = json.loads((tmp_path / "chat-1" / "chat_body_state.json").read_text())
    assert chat["fatigue"]["derived"] is True
    assert chat["context_pressure"]["value"] == 0.42
    # Sibling rule: derived stamps never touch the self-reported record.
    persona = json.loads(_persona_file(tmp_path).read_text())
    assert persona["energy"]["value"] == 0.8


def test_derive_fatigue_rises_with_time(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path, fatigue_rise_cap_hours=1.0)
    now = time.time()
    mgr._derive_fatigue(now - 3600.0)  # seed set_at an hour back
    mgr.state.fatigue["set_at"] = now - 3600.0
    mgr._derive_fatigue(now)
    assert mgr.state.fatigue["value"] == 1.0  # an hour of wake at a 1h cap


def test_felt_wake_rest_pays_fatigue(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path, fatigue_rest_cap_hours=8.0)
    mgr.set_derived("fatigue", 0.8)
    mgr.felt_wake(4 * 3600.0)  # 4h silence = half a rest cap
    assert mgr.state.fatigue["value"] == 0.8 - 0.5
    assert mgr.state.woke_at > 0


def test_prompt_renders_new_fields(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    mgr.set_self_report("energy", "low")
    mgr.set_self_report("mood_tint", "ember-blue")
    mgr.set_self_report("attention", "focused")
    mgr.set_derived("fatigue", 0.5)
    mgr.set_derived("context_pressure", 0.4)
    block = mgr.state_for_prompt()
    assert "Energy: low" in block
    assert "Attention: focused (this chat" in block
    assert "Mood tint: ember-blue" in block
    assert "Fatigue: steady" in block
    assert "Context pressure: 40%" in block


def test_prompt_stale_fields_marked(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path, field_stale_seconds=60.0)
    mgr.set_self_report("mood_tint", "old gold")
    persona_path = _persona_file(tmp_path)
    persona = json.loads(persona_path.read_text())
    persona["mood_tint"]["set_at"] = time.time() - 120.0
    persona_path.write_text(json.dumps(persona))
    assert "stale" in mgr.state_for_prompt()


def test_slim_prompt_when_background(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    mgr.set_self_report("attention", "background")
    block = mgr.state_for_prompt(slim=True)
    assert "attention elsewhere" in block
    assert "Skin warmth" not in block  # the sensor detail is budgeted out
