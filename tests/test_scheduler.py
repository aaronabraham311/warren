"""Tests for nightly scheduler scripts and supporting configuration."""

import argparse
import plistlib
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
PLIST_TEMPLATE = SCRIPTS_DIR / "com.warren.agent.plist.template"
INSTALL_CRON = SCRIPTS_DIR / "install_cron.sh"
INSTALL_SCHEDULER = SCRIPTS_DIR / "install_scheduler.sh"


def test_plist_template_exists() -> None:
    assert PLIST_TEMPLATE.exists()


def test_install_scheduler_exists() -> None:
    assert (SCRIPTS_DIR / "install_scheduler.sh").exists()


def test_uninstall_scheduler_exists() -> None:
    assert (SCRIPTS_DIR / "uninstall_scheduler.sh").exists()


def test_install_cron_exists() -> None:
    assert (SCRIPTS_DIR / "install_cron.sh").exists()


def test_plist_template_has_no_api_keys() -> None:
    """Template must carry no key material at all — agent/run.py reads .env directly."""
    content = PLIST_TEMPLATE.read_text()
    assert "EnvironmentVariables" not in content
    assert "API_KEY" not in content
    # No pattern that looks like a real key (sk-ant- prefix or long hex/base64 strings)
    assert "sk-ant-" not in content


def test_plist_template_schedule_is_sunday_at_7am() -> None:
    plist = plistlib.loads(PLIST_TEMPLATE.read_bytes())
    schedule = plist["StartCalendarInterval"]
    assert schedule == {"Weekday": 0, "Hour": 7, "Minute": 0}


def _program_arguments() -> list[str]:
    """The argv launchd will exec, straight out of the plist template."""
    plist = plistlib.loads(PLIST_TEMPLATE.read_bytes())
    args = plist["ProgramArguments"]
    assert isinstance(args, list)
    return [str(arg) for arg in args]


def test_plist_template_runs_gem_hunt() -> None:
    """The scheduled macOS run is gem-hunt mode, not the US GARP default."""
    assert "--gem-hunt" in _program_arguments()


def test_plist_program_arguments_order() -> None:
    """--gem-hunt is an agent.run argument, not an interpreter one."""
    args = _program_arguments()
    assert args[1:] == ["-m", "agent.run", "--gem-hunt"]
    assert args[0].endswith("/python")


def test_cron_entry_runs_gem_hunt() -> None:
    assert "-m agent.run --gem-hunt" in INSTALL_CRON.read_text()


def test_cron_entry_schedule_is_sunday_at_7am_et() -> None:
    content = INSTALL_CRON.read_text()
    assert 'CRON_TIMEZONE="CRON_TZ=America/New_York"' in content
    assert 'CRON_ENTRY="0 7 * * 0 ' in content


def test_cron_entry_keeps_flock_guard_and_cd() -> None:
    """Adding the flag must not disturb the overlap guard or module resolution."""
    content = INSTALL_CRON.read_text()
    assert "$FLOCK_BIN -n $PROJECT_DIR/logs/.nightly.lock" in content
    assert "cd $PROJECT_DIR &&" in content


def test_install_cron_replaces_stale_entry() -> None:
    """A pre-existing entry (e.g. installed before --gem-hunt) is rewritten, not skipped."""
    content = INSTALL_CRON.read_text()
    assert 'grep -vF "$CRON_MATCH"' in content


def test_install_scheduler_unloads_before_load() -> None:
    """launchd keeps the loaded argv until unloaded — a bare `load` would no-op."""
    content = INSTALL_SCHEDULER.read_text()
    unload = content.index("launchctl unload")
    load = content.index("launchctl load")
    assert unload < load


def test_scripts_are_executable() -> None:
    for name in ("install_scheduler.sh", "uninstall_scheduler.sh", "install_cron.sh"):
        path = SCRIPTS_DIR / name
        assert path.stat().st_mode & 0o111, f"{name} is not executable"


def _make_parser() -> argparse.ArgumentParser:
    """The real agent/run.py parser — not a copy, so it cannot drift from production."""
    from agent.run import build_parser

    return build_parser()


def test_run_cli_accepts_ticker() -> None:
    args = _make_parser().parse_args(["AAPL"])
    assert args.ticker == "AAPL"


def test_run_cli_default_ticker_is_none_for_nightly() -> None:
    """No ticker means nightly mode (screen the universe), not an implicit AAPL."""
    args = _make_parser().parse_args([])
    assert args.ticker is None


def test_run_cli_gem_hunt_flag_defaults_off() -> None:
    args = _make_parser().parse_args([])
    assert args.gem_hunt is False


def test_run_cli_accepts_gem_hunt() -> None:
    args = _make_parser().parse_args(["--gem-hunt"])
    assert args.gem_hunt is True


def test_run_cli_gem_hunt_composes_with_skip_ticker_validation() -> None:
    args = _make_parser().parse_args(["--gem-hunt", "--skip-ticker-validation"])
    assert args.gem_hunt is True
    assert args.skip_ticker_validation is True


def test_scheduled_argv_parses_to_gem_hunt_run() -> None:
    """End-to-end on the argv launchd ships: nightly mode, gem-hunt on."""
    module_args = _program_arguments()[3:]  # drop <python> -m agent.run
    args = _make_parser().parse_args(module_args)
    assert args.gem_hunt is True
    assert args.ticker is None


def test_resolve_persona_gem_hunt_forces_dirt() -> None:
    from agent.persona import DirtPersona
    from agent.run import resolve_persona

    assert isinstance(resolve_persona("default", gem_hunt=True), DirtPersona)


def test_resolve_persona_default_when_no_gem_hunt() -> None:
    from agent.persona import DefaultPersona
    from agent.run import resolve_persona

    assert isinstance(resolve_persona("default", gem_hunt=False), DefaultPersona)


def test_resolve_persona_explicit_dirt_still_works() -> None:
    from agent.persona import DirtPersona
    from agent.run import resolve_persona

    assert isinstance(resolve_persona("dirt", gem_hunt=False), DirtPersona)
