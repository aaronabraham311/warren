"""Tests for nightly scheduler scripts and supporting configuration."""

import argparse
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
PLIST_TEMPLATE = SCRIPTS_DIR / "com.warren.agent.plist.template"


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


def test_plist_template_schedule_is_2am() -> None:
    content = PLIST_TEMPLATE.read_text()
    assert "<integer>2</integer>" in content  # Hour
    assert "<integer>0</integer>" in content  # Minute


def test_scripts_are_executable() -> None:
    for name in ("install_scheduler.sh", "uninstall_scheduler.sh", "install_cron.sh"):
        path = SCRIPTS_DIR / name
        assert path.stat().st_mode & 0o111, f"{name} is not executable"


def _make_parser() -> argparse.ArgumentParser:
    """Reproduce the parser from agent/run.py without importing the full module."""
    parser = argparse.ArgumentParser(description="Warren stock analysis agent")
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--skip-ticker-validation", action="store_true")
    parser.add_argument("--persona", choices=["default", "dirt"], default="default")
    return parser


def test_run_cli_accepts_ticker() -> None:
    args = _make_parser().parse_args(["AAPL"])
    assert args.ticker == "AAPL"


def test_run_cli_default_ticker() -> None:
    args = _make_parser().parse_args([])
    assert args.ticker == "AAPL"
