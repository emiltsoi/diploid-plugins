"""Cron digest plugin configuration."""

from dataclasses import dataclass


@dataclass
class CronDigestConfig:
    """Default configuration for the cron digest prompt-slot plugin."""

    results_dirname: str = "cron"  # sessions/<chat>/cron/
    max_jobs: int = 8
    max_summary_chars: int = 100
    stale_hours: float = 0.0  # 0 = show all jobs regardless of age
