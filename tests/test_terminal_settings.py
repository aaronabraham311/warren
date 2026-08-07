import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agent.terminal.settings import (
    TerminalSettings,
    history_path,
    load_settings,
    save_settings,
    settings_path,
)


def test_defaults_and_paths_are_repository_local(tmp_path: Path) -> None:
    assert TerminalSettings() == TerminalSettings(
        persona="default", max_cost_usd=1.25, color="auto", animation=True, show_cost=True
    )
    assert settings_path(tmp_path) == tmp_path / "settings.json"
    assert history_path(tmp_path) == tmp_path / "history"
    assert tmp_path.is_dir()


def test_settings_round_trip_and_file_contains_only_schema_fields(tmp_path: Path) -> None:
    settings = TerminalSettings(
        persona="dirt", max_cost_usd=3.5, color="never", animation=False, show_cost=False
    )
    path = save_settings(settings, tmp_path)
    assert load_settings(tmp_path) == settings
    assert set(json.loads(path.read_text())) == {
        "persona",
        "max_cost_usd",
        "color",
        "animation",
        "show_cost",
    }
    assert not list(tmp_path.glob(".settings-*.tmp"))


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        '{"persona":"unknown"}',
        '{"max_cost_usd":0}',
        '{"api_key":"secret"}',
    ],
)
def test_invalid_or_secret_bearing_settings_fall_back_with_warning(
    tmp_path: Path, payload: str
) -> None:
    tmp_path.mkdir(exist_ok=True)
    settings_path(tmp_path).write_text(payload)
    messages: list[str] = []
    assert load_settings(tmp_path, warn=messages.append) == TerminalSettings()
    assert len(messages) == 1
    assert "Ignoring invalid terminal settings" in messages[0]
    assert "secret" not in messages[0]


def test_missing_settings_are_silent_defaults(tmp_path: Path) -> None:
    messages: list[str] = []
    assert load_settings(tmp_path, warn=messages.append) == TerminalSettings()
    assert messages == []


def test_atomic_replace_failure_preserves_previous_settings_and_cleans_temp(
    tmp_path: Path,
) -> None:
    original = TerminalSettings(persona="default")
    save_settings(original, tmp_path)
    with patch("agent.terminal.settings.os.replace", side_effect=OSError("disk failure")):
        with pytest.raises(OSError, match="disk failure"):
            save_settings(TerminalSettings(persona="dirt"), tmp_path)
    assert load_settings(tmp_path) == original
    assert not list(tmp_path.glob(".settings-*.tmp"))


def test_settings_validation_rejects_out_of_bounds_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TerminalSettings(max_cost_usd=10.01)
    with pytest.raises(ValidationError):
        TerminalSettings.model_validate({"api_key": "secret"})


def test_default_paths_honor_warren_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "terminal-state"
    monkeypatch.setenv("WARREN_STATE_DIR", str(configured))

    assert settings_path() == configured / "settings.json"
    assert history_path() == configured / "history"
