"""Tests for the cron digest prompt-slot plugin."""

import json
import time
from pathlib import Path

from diploid_agent.config import PluginConfig

from diploid_plugins.cron.cron import CronPlugin


def _plugin(tmp_path: Path, **cfg) -> CronPlugin:
    config = PluginConfig(name="cron", prompt_slot="cron", config=cfg)
    return CronPlugin(config, "chat-1", tmp_path)


def _write_last(chat_dir: Path, job: str, status: str, summary: str, ago: float = 0.0) -> None:
    cron_dir = chat_dir / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    (cron_dir / f"{job}.last").write_text(
        json.dumps(
            {
                "job_id": job,
                "status": status,
                "finished_at": time.time() - ago,
                "summary": summary,
            }
        )
    )


def test_no_results_dir_returns_none(tmp_path: Path) -> None:
    assert _plugin(tmp_path).prompt_block() is None


def test_renders_sorted_bounded_lines(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chat-1"
    _write_last(chat_dir, "tidy", "ok", "cleaned 3 notes", ago=3600)
    _write_last(chat_dir, "nightly", "failed", "boom", ago=60)
    block = _plugin(tmp_path).prompt_block()
    assert block is not None
    assert "## Cron jobs" in block
    assert "tidy ok" in block
    assert "! nightly failed" in block  # failures marked
    assert block.index("nightly") < block.index("tidy")  # newest first


def test_service_warnings_render(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chat-1"
    cron_dir = chat_dir / "cron"
    cron_dir.mkdir(parents=True)
    (cron_dir / "service.last").write_text(
        json.dumps({"warnings": ["crons.yaml: parse error"], "updated_at": time.time()})
    )
    block = _plugin(tmp_path).prompt_block()
    assert "! service: crons.yaml: parse error" in block


def test_max_chars_cap(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chat-1"
    for i in range(20):
        _write_last(chat_dir, f"job-{i:02}", "ok", "x" * 90)
    block = _plugin(tmp_path).prompt_block(max_chars=400)
    assert block is not None and len(block) <= 400
    # default max_jobs bounds the listing
    block = _plugin(tmp_path, max_jobs=3).prompt_block()
    assert block.count("- job-") == 3
