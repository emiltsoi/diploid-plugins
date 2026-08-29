"""Tests for the planner plugin."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    PluginConfig,
    Secrets,
    TimerConfig,
)
from diploid_agent.engine.fake import FakeAgentEngine
from diploid_agent.runtime.agent_runtime import AgentRuntime

from diploid_plugins.planner.planner import PlannerPlugin


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_config(tmp_path: Path, planner_enabled: bool = True, auto_triage: bool = False) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=_fixture_root(),
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            plugins=[
                PluginConfig(
                    name="planner",
                    enabled=planner_enabled,
                    module="diploid_plugins.planner",
                    prompt_slot="persona_state",
                    prompt_order=50,
                    max_prompt_chars=1024,
                    config={
                        "trigger_prefix": ["!plan ", "Plan: "],
                        "auto_triage": auto_triage,
                        "model": None,
                        "max_tasks": 10,
                    },
                ),
            ],
            memory={"backend": "file"},  # type: ignore[arg-type]
            timer=TimerConfig(enabled=False, interval_seconds=0.1),
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def _shell_plan_json() -> str:
    return json.dumps(
        {
            "needs_plan": True,
            "plan_name": "deploy",
            "description": "deploy the app",
            "tasks": [
                {
                    "name": "lint",
                    "type": "shell",
                    "command": "echo ok",
                    "depends_on": [],
                },
                {
                    "name": "report",
                    "type": "noop",
                    "depends_on": ["lint"],
                },
            ],
        }
    )


def _acp_plan_with_model_json() -> str:
    return json.dumps(
        {
            "needs_plan": True,
            "plan_name": "analyze",
            "description": "analyze the request",
            "tasks": [
                {
                    "name": "think",
                    "type": "acp",
                    "prompt": "summarize the request in one word",
                    "acp_model": "glm-5-2",
                    "depends_on": [],
                },
                {
                    "name": "think-alias",
                    "type": "acp",
                    "prompt": "summarize again",
                    "model": "swe-1-7",
                    "depends_on": [],
                },
            ],
        }
    )


def _acp_plan_json() -> str:
    return json.dumps(
        {
            "needs_plan": True,
            "plan_name": "analyze",
            "description": "analyze the request",
            "tasks": [
                {
                    "name": "think",
                    "type": "acp",
                    "prompt": "summarize the request in one word",
                    "depends_on": [],
                },
            ],
        }
    )


@pytest.fixture
def runtime(tmp_path: Path):
    r = AgentRuntime(_make_config(tmp_path))
    r.engine = FakeAgentEngine(default_reply="noted", default_session_id="s1")
    r.task_engine.engine = r.engine
    r.start()
    yield r
    r.shutdown()


def _wait_for_plan_complete(runtime: AgentRuntime, plan_id: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        plan = runtime.plan_manager.get_plan(plan_id)
        if plan is not None and plan.status.value in ("completed", "failed"):
            return
        time.sleep(0.05)
    raise AssertionError("plan did not complete")


def _wait_for_due_wakes(runtime: AgentRuntime, chat_id: str, timeout: float = 2.0) -> None:
    """Wait until at least two due wakes for ``chat_id`` are visible."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wakes = [
            e for e in runtime.wake_queue.pending(chat_id=chat_id, now=time.time() + 1) if e.ready
        ]
        if len(wakes) >= 2:
            return
        time.sleep(0.05)
    raise AssertionError(f"no wakes became due for {chat_id}")


def test_normal_message_passes_through(runtime, tmp_path: Path) -> None:
    result = runtime.process("chat-1", "hello")
    assert "Plan started" not in result.reply
    assert runtime.plan_manager.list_plans(chat_id="chat-1") == []


def test_triggered_shell_plan_creates_and_starts_plan(runtime, tmp_path: Path) -> None:
    runtime.engine.replies = [_shell_plan_json()]

    result = runtime.process("chat-2", "!plan deploy the app")
    assert "Plan started" in result.reply

    plans = runtime.plan_manager.list_plans(chat_id="chat-2")
    assert len(plans) == 1
    plan = plans[0]
    assert plan.name == "deploy"
    assert len(plan.tasks) == 2

    # The first ready task was started automatically.
    _wait_for_plan_complete(runtime, plan.id)
    reloaded = runtime.plan_manager.get_plan(plan.id)
    assert reloaded is not None
    assert reloaded.status == "completed"

    # Task execution enqueues plan_task_update and plan_completed wakes.
    _wait_for_due_wakes(runtime, "chat-2")
    due = runtime.wake_queue.pop_due(now=time.time() + 1)
    chat_due = [e for e in due if e.chat_id == "chat-2"]
    assert len(chat_due) >= 2
    assert chat_due[0].reason == "plan_task_update"
    assert chat_due[-1].reason == "plan_completed"


def test_triggered_acp_plan_runs_and_produces_conclusion(runtime, tmp_path: Path) -> None:
    runtime.engine.replies = [_acp_plan_json(), "insight"]

    result = runtime.process("chat-3", "Plan: analyze the request")
    assert "Plan started" in result.reply

    plans = runtime.plan_manager.list_plans(chat_id="chat-3")
    assert len(plans) == 1
    plan = plans[0]
    assert len(plan.tasks) == 1
    assert plan.tasks[0].type == "acp"

    _wait_for_plan_complete(runtime, plan.id)

    _wait_for_due_wakes(runtime, "chat-3")
    due = runtime.wake_queue.pop_due(now=time.time() + 1)
    chat_due = [e for e in due if e.chat_id == "chat-3"]
    assert len(chat_due) == 2
    assert chat_due[0].reason == "plan_task_update"
    assert chat_due[1].reason == "plan_completed"

    # Run the conclusion wake and check for a final reply.
    conclusion = runtime.wake("chat-3", event_id=chat_due[1].id)
    assert "noted" in conclusion.reply


def test_planner_plugin_trigger_parsing(tmp_path: Path) -> None:
    config = PluginConfig(
        name="planner",
        module="diploid_plugins.planner",
        config={"trigger_prefix": "!plan "},
    )
    plugin = PlannerPlugin(config, "chat-1", tmp_path, runtime=None)
    assert plugin._extract_request("!plan build a thing") == "build a thing"
    assert plugin._extract_request("Plan: build") is None
    assert plugin._extract_request("/plan deploy now") == "deploy now"
    assert plugin._extract_request("hello") is None


def test_planner_plugin_prompt_block_is_none(tmp_path: Path) -> None:
    config = PluginConfig(name="planner", module="diploid_plugins.planner")
    plugin = PlannerPlugin(config, "chat-1", tmp_path, runtime=None)
    assert plugin.prompt_block() is None


@pytest.fixture
def runtime_auto(tmp_path: Path):
    r = AgentRuntime(_make_config(tmp_path, auto_triage=True))
    r.engine = FakeAgentEngine(default_reply="noted", default_session_id="s1")
    r.task_engine.engine = r.engine
    r.start()
    yield r
    r.shutdown()


def test_triggered_acp_plan_with_models(runtime, tmp_path: Path) -> None:
    runtime.engine.replies = [_acp_plan_with_model_json()]

    result = runtime.process("chat-model", "Plan: analyze the request")
    assert "Plan started" in result.reply

    plans = runtime.plan_manager.list_plans(chat_id="chat-model")
    assert len(plans) == 1
    plan = plans[0]
    assert len(plan.tasks) == 2
    assert plan.tasks[0].acp_model == "glm-5-2"
    assert plan.tasks[1].acp_model == "swe-1-7"


def test_auto_triage_creates_plan_without_trigger(runtime_auto, tmp_path: Path) -> None:
    runtime_auto.engine.replies = [_shell_plan_json()]

    result = runtime_auto.process("chat-auto", "please deploy the app")
    assert "Plan started" in result.reply

    plans = runtime_auto.plan_manager.list_plans(chat_id="chat-auto")
    assert len(plans) == 1
    plan = plans[0]
    assert plan.name == "deploy"
    assert len(plan.tasks) == 2
