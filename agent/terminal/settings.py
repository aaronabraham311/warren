# mypy: disable-error-code="explicit-any"
"""Validated, atomic persistence for non-secret terminal preferences."""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TerminalSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    persona: Literal["default", "dirt"] = "default"
    max_cost_usd: float = Field(default=1.25, gt=0.0, le=10.0)
    color: Literal["auto", "always", "never"] = "auto"
    animation: bool = True
    show_cost: bool = True


def settings_path(state_dir: Path = Path(".warren")) -> Path:
    return state_dir / "settings.json"


def history_path(state_dir: Path = Path(".warren")) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "history"


def load_settings(
    state_dir: Path = Path(".warren"),
    *,
    warn: Callable[[str], None] | None = None,
) -> TerminalSettings:
    """Load preferences, falling back safely when the file is corrupt or invalid."""
    path = settings_path(state_dir)
    if not path.exists():
        return TerminalSettings()
    try:
        return TerminalSettings.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        reason = "file could not be read"
    except ValidationError:
        reason = "settings failed schema validation"
    except ValueError:
        reason = "file is not valid JSON"
    message = f"Ignoring invalid terminal settings: {reason}."
    if warn is None:
        warnings.warn(message, UserWarning, stacklevel=2)
    else:
        warn(message)
    return TerminalSettings()


def save_settings(settings: TerminalSettings, state_dir: Path = Path(".warren")) -> Path:
    """Atomically replace settings.json after flushing file and directory metadata."""
    state_dir.mkdir(parents=True, exist_ok=True)
    destination = settings_path(state_dir)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_dir,
            prefix=".settings-",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temporary = Path(fh.name)
            fh.write(settings.model_dump_json(indent=2) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(state_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
