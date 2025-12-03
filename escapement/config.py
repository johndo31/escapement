"""
Configuration for Escapement.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EscapementConfig(BaseModel):
    """Configuration options for Escapement."""

    # Storage settings
    storage_dir: Path = Field(
        default_factory=lambda: Path.cwd() / ".escapement",
        description="Directory to store trace files",
    )
    database_name: str = "traces.db"

    # Recording settings
    enabled: bool = True
    auto_record: bool = True  # Automatically record all LLM calls
    capture_local_state: bool = False  # Capture local variables (expensive)
    max_trace_steps: int = 10000  # Prevent runaway agents

    # Replay settings
    strict_replay: bool = True  # Fail if cache miss during replay
    allow_fork: bool = True  # Allow forking from replay to live

    # Loop detection
    loop_detection_enabled: bool = True
    max_identical_calls: int = 10  # Kill after N identical LLM calls
    loop_detection_window: int = 20  # Check last N calls for loops

    # Privacy settings
    scrub_pii: bool = False  # Enable PII scrubbing (requires additional setup)
    redact_patterns: list[str] = Field(default_factory=list)

    # Provider settings
    providers: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "openai": {"enabled": True},
            "anthropic": {"enabled": True},
        }
    )

    @classmethod
    def from_env(cls) -> EscapementConfig:
        """Load configuration from environment variables."""
        config = cls()

        if storage_dir := os.environ.get("ESCAPEMENT_STORAGE_DIR"):
            config.storage_dir = Path(storage_dir)

        if enabled := os.environ.get("ESCAPEMENT_ENABLED"):
            config.enabled = enabled.lower() in ("true", "1", "yes")

        if loop_max := os.environ.get("ESCAPEMENT_MAX_IDENTICAL_CALLS"):
            config.max_identical_calls = int(loop_max)

        return config

    def get_db_path(self) -> Path:
        """Get the full path to the database file."""
        return self.storage_dir / self.database_name

    def ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: EscapementConfig | None = None


def get_config() -> EscapementConfig:
    """Get the global configuration."""
    global _config
    if _config is None:
        _config = EscapementConfig.from_env()
    return _config


def set_config(config: EscapementConfig) -> None:
    """Set the global configuration."""
    global _config
    _config = config
