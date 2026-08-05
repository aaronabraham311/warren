import json
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from agent.models import AnalysisOutput, LynchBuffettSignals
from agent.persona import DefaultPersona, DirtPersona
from agent.tools.base import ToolResultOk
from data_sources.yfinance_client import PriceData
from eval.golden_set import (
    EvalExample,
    EvalExpectations,
    NumericalGrounding,
    RecommendationExpectation,
)
from eval.grader import EvalGrade
from eval.runner import resolve_persona, run_eval
from eval.tool_fixtures import record_tool_result
from storage.models import EvalRun, Run
from tests.conftest import make_end_turn, make_tool_use

_ANALYSIS_JSON = """{
  "ticker": "AAPL",
  "analysis_type": "holding",
  "recommendation": "hold",
  "confidence": 0.72,
  "thesis": "Apple trades at 28.4x earnings with 22% ROE and a 6.1% FCF yield; the moat is intact.",
  "lynch_signals": {"pros": ["consistent earnings"], "cons": []},
  "buffett_signals": {"pros": ["high ROE", "strong FCF"], "cons": []},
  "key_risks": ["valuation stretched", "China exposure"],
  "data_quality_notes": []
}"""


def _example(ticker: str, persona: str = "default") -> EvalExample:
    return EvalExample(
        ticker=ticker,
        notes="test",
        last_curated=date(2026, 1, 1),
        persona=persona,
        expectations=EvalExpectations(
            recommendation=RecommendationExpectation(allowed=["hold", "buy"]),
            numerical_grounding=NumericalGrounding(min_specific_numbers=3),
        ),
    )


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
    monkeypatch.setattr("eval.runner._LOG_DIR", d)
    return d


MockClaude = Callable[[list[anthropic.types.Message]], MagicMock]


def test_run_eval_grades_a_ticker_with_fixtures(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    mock_claude: MockClaude,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(_ANALYSIS_JSON),
        ]
    )

    grades = run_eval(
        examples=[_example("AAPL")],
        client=client,
        eval_run_id="eval-fixed",
        fixtures_root=fixtures_root,
    )

    assert len(grades) == 1
    assert grades[0].ticker == "AAPL"
    assert grades[0].passed, grades[0].overall_notes
    out = capsys.readouterr().out
    assert "✅ AAPL:" in out
    assert "Result: 1/1 examples passed" in out


def test_ticker_without_fixtures_fails_without_an_llm_call(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    mock_claude: MockClaude,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole point of has_tool_fixtures: never burn a Sonnet run on a guaranteed failure."""
    client = mock_claude([])  # any API call pops from an empty queue → IndexError

    grades = run_eval(
        examples=[_example("NKE")],
        client=client,
        eval_run_id="eval-fixed",
        fixtures_root=fixtures_root,
    )

    assert not grades[0].passed
    assert [c.check_name for c in grades[0].checks] == ["fixture_missing"]
    client.messages.create.assert_not_called()
    assert "❌ NKE:" in capsys.readouterr().out


def test_agent_exception_becomes_a_run_completed_failure(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_claude: MockClaude,
) -> None:
    client = mock_claude([])

    def _boom(**_kw: object) -> None:
        raise RuntimeError("model exploded")

    monkeypatch.setattr("eval.runner.analyze_ticker", _boom)

    grades = run_eval(
        examples=[_example("AAPL")],
        client=client,
        eval_run_id="eval-fixed",
        fixtures_root=fixtures_root,
    )

    assert not grades[0].passed
    assert grades[0].checks[0].check_name == "run_completed"
    assert "model exploded" in grades[0].checks[0].actual


def test_temperature_zero_reaches_the_api(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    mock_claude: MockClaude,
) -> None:
    """Leg 2 of determinism. The closest we can get offline to the 12/13-stability criterion."""
    client = mock_claude([make_end_turn(_ANALYSIS_JSON)])

    run_eval(
        examples=[_example("AAPL")],
        client=client,
        eval_run_id="eval-fixed",
        fixtures_root=fixtures_root,
    )

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0.0


def test_output_flag_writes_one_evalgrade_per_ticker(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    tmp_path: Path,
    mock_claude: MockClaude,
) -> None:
    client = mock_claude([make_end_turn(_ANALYSIS_JSON)])
    out_path = tmp_path / "runs" / "eval-2026-05-10.json"

    run_eval(
        output_path=out_path,
        examples=[_example("AAPL"), _example("NKE")],
        client=client,
        eval_run_id="eval-fixed",
        fixtures_root=fixtures_root,
    )

    assert out_path.exists(), "parent dirs are created"
    payload = json.loads(out_path.read_text())
    assert [EvalGrade.model_validate(g).ticker for g in payload] == ["AAPL", "NKE"]


def test_eval_runs_table_gets_one_row_per_ticker(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    mock_claude: MockClaude,
) -> None:
    client = mock_claude([make_end_turn(_ANALYSIS_JSON)])

    run_eval(
        examples=[_example("AAPL"), _example("NKE")],
        client=client,
        eval_run_id="eval-fixed",
        fixtures_root=fixtures_root,
    )

    with Session(db_engine) as session:
        rows = session.scalars(select(EvalRun).order_by(EvalRun.example_ticker)).all()
        assert [r.example_ticker for r in rows] == ["AAPL", "NKE"]
        assert [r.passed for r in rows] == [True, False]
        # check_results round-trips as JSON so a later run can diff it.
        assert json.loads(rows[0].check_results or "")[0]["severity"] in ("must", "should")
        # eval_runs.run_id is an FK — the Run row must exist.
        assert session.get(Run, "eval-fixed") is not None


def test_fixed_run_id_overwrites_rather_than_duplicates(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    mock_claude: MockClaude,
) -> None:
    """Leg 3: re-running under a pinned --eval-run-id replaces its rows in place."""
    client = mock_claude([make_end_turn(_ANALYSIS_JSON), make_end_turn(_ANALYSIS_JSON)])
    for _ in range(2):
        run_eval(
            examples=[_example("AAPL")],
            client=client,
            eval_run_id="eval-fixed",
            fixtures_root=fixtures_root,
        )

    with Session(db_engine) as session:
        rows = session.scalars(select(EvalRun)).all()
        assert len(rows) == 1


def test_identical_mocked_runs_produce_identical_grades(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    mock_claude: MockClaude,
) -> None:
    client = mock_claude(
        [
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(_ANALYSIS_JSON),
            make_tool_use("get_quote", {"ticker": "AAPL"}),
            make_end_turn(_ANALYSIS_JSON),
        ]
    )
    grades = [
        run_eval(
            examples=[_example("AAPL")],
            client=client,
            eval_run_id="eval-fixed",
            fixtures_root=fixtures_root,
        )
        for _ in range(2)
    ]

    assert grades[0][0].model_dump() == grades[1][0].model_dump()


def test_resolve_persona_maps_the_persona_field() -> None:
    assert isinstance(resolve_persona(_example("AAPL")), DefaultPersona)
    assert isinstance(resolve_persona(_example("AAPL", persona="dirt")), DirtPersona)


def _capture_persona(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace analyze_ticker with a spy that records the persona it was handed."""
    captured: dict[str, object] = {}

    def _spy(**kwargs: object) -> AnalysisOutput:
        captured["persona"] = kwargs["persona"]
        return AnalysisOutput(
            ticker="AAPL",
            analysis_type="discovery",
            recommendation="hold",
            confidence=0.5,
            thesis="Trades at 6.1x EV/EBIT with 0.8x NCAV; steady 12% ROE.",
            lynch_signals=LynchBuffettSignals(pros=[], cons=[]),
            buffett_signals=LynchBuffettSignals(pros=[], cons=[]),
            key_risks=["value trap risk"],
        )

    monkeypatch.setattr("eval.runner.analyze_ticker", _spy)
    return captured


def test_dirt_example_replays_under_dirt_persona(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    mock_claude: MockClaude,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `persona: dirt` example must reach analyze_ticker under DirtPersona."""
    captured = _capture_persona(monkeypatch)
    run_eval(
        examples=[_example("AAPL", persona="dirt")],
        client=mock_claude([]),
        eval_run_id="eval-fixed",
        fixtures_root=fixtures_root,
    )
    assert isinstance(captured["persona"], DirtPersona)


def test_default_example_still_replays_under_default_persona(
    db_engine: Engine,
    fixtures_root: Path,
    log_dir: Path,
    mock_claude: MockClaude,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_persona(monkeypatch)
    run_eval(
        examples=[_example("AAPL")],
        client=mock_claude([]),
        eval_run_id="eval-fixed",
        fixtures_root=fixtures_root,
    )
    assert isinstance(captured["persona"], DefaultPersona)
