from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import pytest

from agent.models import AnalysisOutput
from agent.tools.base import ToolResultOk
from data_sources.yfinance_client import PriceData
from eval.analysis.dump_theses import dump_thesis
from eval.tool_fixtures import record_tool_result
from tests.conftest import make_end_turn, make_tool_use

_ANALYSIS_JSON = """{
  "ticker": "AAPL",
  "analysis_type": "holding",
  "recommendation": "hold",
  "confidence": 0.72,
  "thesis": "Apple has a durable moat and strong free cash flow but trades near fair value.",
  "lynch_signals": {"pros": ["dominant brand", "consistent earnings"], "cons": []},
  "buffett_signals": {"pros": ["high ROE", "strong FCF", "consumer moat"], "cons": []},
  "key_risks": ["valuation stretched", "China exposure"],
  "data_quality_notes": []
}"""

MockClaude = Callable[[list[anthropic.types.Message]], MagicMock]


@pytest.fixture()
def fixtures_root(tmp_path: Path) -> Path:
    """A fixture tree covering AAPL's get_quote and nothing else."""
    root = tmp_path / "fixtures"
    price = PriceData(
        ticker="AAPL",
        current_price=190.5,
        previous_close=188.0,
        day_change_pct=1.33,
        volume=50_000_000,
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        data_age_hours=1,
    )
    record_tool_result("AAPL", "get_quote", {"ticker": "AAPL"}, ToolResultOk(data=price), root)
    return root


@pytest.fixture()
def log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "logs"
    monkeypatch.setattr("eval.analysis.dump_theses._LOG_DIR", d)
    return d


def test_dump_thesis_replays_a_ticker_with_fixtures(
    fixtures_root: Path,
    log_dir: Path,
    mock_claude: MockClaude,
) -> None:
    client = mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(_ANALYSIS_JSON),
        ]
    )

    result = dump_thesis("AAPL", client, fixtures_root=fixtures_root)

    assert isinstance(result, AnalysisOutput)
    assert result.ticker == "AAPL"
    assert result.recommendation == "hold"
    assert "moat" in result.thesis


def test_dump_thesis_returns_none_without_fixtures(
    tmp_path: Path,
    mock_claude: MockClaude,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = mock_claude([])  # any API call would pop from an empty queue → IndexError

    result = dump_thesis("NKE", client, fixtures_root=tmp_path / "empty")

    assert result is None
    client.messages.create.assert_not_called()
    assert "no recorded tool fixtures" in capsys.readouterr().out
