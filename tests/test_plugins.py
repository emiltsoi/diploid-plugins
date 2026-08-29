"""Tests for the pluggable state layer."""

from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.plugins import PluginManager

from diploid_plugins.continuity.continuity import ContinuityPlugin
from diploid_plugins.curriculum.curriculum import CurriculumPlugin


def test_curriculum_event_persists_state(tmp_path: Path) -> None:
    config = PluginConfig(
        name="curriculum",
        state_file="chat_curriculum.json",
        prompt_slot="persona_state",
    )
    plugin = CurriculumPlugin(config, "chat-1", tmp_path)
    assert "Klingon" in plugin._set_target_language("Klingon")
    assert plugin._load_state()["target_language"] == "Klingon"
    assert (tmp_path / "chat-1" / "chat_curriculum.json").exists()


def test_continuity_prompt_renders(tmp_path: Path) -> None:
    config = PluginConfig(
        name="continuity",
        state_file="chat_wake_state.json",
        prompt_slot="wake",
    )
    plugin = ContinuityPlugin(config, "chat-1", tmp_path)
    assert plugin.prompt_block() is None

    # Fake a previous turn.
    plugin._state["last_turn_at"] = 1.0
    plugin._state["this_instance_id"] = "harness-123"
    plugin._state["last_session_number"] = 0
    plugin._state["last_turn_number"] = 1
    plugin._state["last_stop_reason"] = "completed"
    block = plugin.prompt_block()
    assert block is not None
    assert "Wake state" in block


def test_plugin_manager_lists_defaults() -> None:
    mgr = PluginManager(
        plugins=[
            PluginConfig(
                name="curriculum",
                module="diploid_plugins.curriculum",
                skill="curriculum",
            ),
        ],
        sessions_root=Path("sessions"),
        instance_id="harness-123",
        instance_started_at=0.0,
        dispatch_store=None,  # type: ignore[arg-type]
    )
    assert mgr.default_skill_names() == ["curriculum"]
