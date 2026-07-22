"""Tests for the Streamlit Eval page — data layer + headless AppTest integration.

Data-layer tests exercise `eval_run_summaries` (per-run pass-rate aggregation + the
prompt-version join), `load_eval_grades` (parsing the `check_results` JSON string), and
`diff_eval_runs` (identical runs → no changes, and correct fix/regression classification
and net counts). The AppTest cases run `dashboard/pages/eval.py` headlessly against a temp
file DB to cover the acceptance criteria that depend on rendering: the empty-DB info, the
<2-runs diff guard, the net-change banner, and the red/green highlighted flips.
"""

import json
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

import storage.engine as eng
from dashboard.data import (
    EvalCheckResult,
    diff_eval_runs,
    eval_run_summaries,
    load_eval_grades,
)
from storage.models import Base, EvalRun, PromptVersion, Run

_EVAL_PAGE = str(Path(__file__).resolve().parents[1] / "dashboard" / "pages" / "eval.py")
_BASE = datetime(2026, 6, 29, 7, 0, 0, tzinfo=timezone.utc)


def _check(name: str, passed: bool, *, severity: str = "must") -> dict[str, object]:
    return {
        "check_name": name,
        "passed": passed,
        "expected": f"{name} expected",
        "actual": f"{name} actual",
        "severity": severity,
    }


def _make_run(
    run_id: str, *, prompt_version_id: int | None = None, started: datetime = _BASE
) -> Run:
    return Run(id=run_id, prompt_version_id=prompt_version_id, started_at=started, status="success")


def _make_eval_run(
    run_id: str, ticker: str, *, passed: bool, checks: list[dict[str, object]]
) -> EvalRun:
    return EvalRun(
        run_id=run_id,
        example_ticker=ticker,
        passed=passed,
        check_results=json.dumps(checks),
        diff_notes=f"{sum(1 for c in checks if c['passed'])}/{len(checks)} checks passed",
    )


# --------------------------------------------------------------------------- #
# Data layer — eval_run_summaries
# --------------------------------------------------------------------------- #


def test_summaries_pass_rate_and_version(db_session: Session) -> None:
    db_session.add(PromptVersion(id=1, version_tag="v1-baseline"))
    db_session.add(_make_run("run-a", prompt_version_id=1))
    db_session.add_all(
        [
            _make_eval_run("run-a", "AAPL", passed=True, checks=[_check("c1", True)]),
            _make_eval_run("run-a", "MSFT", passed=True, checks=[_check("c1", True)]),
            _make_eval_run("run-a", "GOOG", passed=False, checks=[_check("c1", False)]),
        ]
    )
    db_session.commit()

    summaries = eval_run_summaries(db_session)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.run_id == "run-a"
    assert s.total == 3
    assert s.passed == 2
    assert s.pass_rate == pytest.approx(2 / 3)
    assert s.version_tag == "v1-baseline"


def test_summaries_newest_first_and_null_version(db_session: Session) -> None:
    db_session.add_all(
        [
            _make_run("old", started=_BASE - timedelta(days=2)),
            _make_run("new", started=_BASE),
        ]
    )
    db_session.add_all(
        [
            _make_eval_run("old", "AAPL", passed=True, checks=[_check("c1", True)]),
            _make_eval_run("new", "AAPL", passed=False, checks=[_check("c1", False)]),
        ]
    )
    db_session.commit()

    summaries = eval_run_summaries(db_session)
    assert [s.run_id for s in summaries] == ["new", "old"]
    assert summaries[0].version_tag is None  # no linked prompt version


def test_summaries_empty_run_has_zero_pass_rate() -> None:
    # pass_rate must not divide by zero when a run somehow has no graded examples.
    from dashboard.data import EvalRunSummary

    assert EvalRunSummary("r", None, None, total=0, passed=0).pass_rate == 0.0


# --------------------------------------------------------------------------- #
# Data layer — load_eval_grades
# --------------------------------------------------------------------------- #


def test_load_grades_parses_check_results(db_session: Session) -> None:
    db_session.add(_make_run("run-a"))
    db_session.add(
        _make_eval_run(
            "run-a", "AAPL", passed=False, checks=[_check("c1", True), _check("c2", False)]
        )
    )
    db_session.commit()

    grades = load_eval_grades(db_session, "run-a")
    assert set(grades) == {"AAPL"}
    aapl = grades["AAPL"]
    assert aapl["c1"].passed is True
    assert aapl["c2"].passed is False
    assert aapl["c2"].expected == "c2 expected"
    assert aapl["c2"].actual == "c2 actual"
    assert aapl["c2"].severity == "must"


def test_load_grades_empty_payload(db_session: Session) -> None:
    db_session.add(_make_run("run-a"))
    db_session.add(EvalRun(run_id="run-a", example_ticker="AAPL", passed=True, check_results=None))
    db_session.commit()

    grades = load_eval_grades(db_session, "run-a")
    assert grades == {"AAPL": {}}


# --------------------------------------------------------------------------- #
# Data layer — diff_eval_runs
# --------------------------------------------------------------------------- #


def _grade(passed: bool) -> EvalCheckResult:
    return EvalCheckResult("c", passed, "exp", "act", "must")


def test_diff_identical_runs_has_no_changes() -> None:
    grades = {"AAPL": {"c1": _grade(True), "c2": _grade(False)}}
    diff = diff_eval_runs("a", "b", grades, grades)
    assert diff.ticker_diffs == []
    assert diff.fixes == 0
    assert diff.regressions == 0


def test_diff_classifies_fix_and_regression() -> None:
    grades_a = {
        "AAPL": {"c1": _grade(False), "c2": _grade(True)},  # c1 will fix, c2 will regress
        "MSFT": {"c1": _grade(True)},  # unchanged → dropped
    }
    grades_b = {
        "AAPL": {"c1": _grade(True), "c2": _grade(False)},
        "MSFT": {"c1": _grade(True)},
    }
    diff = diff_eval_runs("a", "b", grades_a, grades_b)

    assert diff.fixes == 1
    assert diff.regressions == 1
    assert [td.ticker for td in diff.ticker_diffs] == ["AAPL"]  # MSFT unchanged, omitted
    kinds = {c.check_name: c.kind for c in diff.ticker_diffs[0].changes}
    assert kinds == {"c1": "fix", "c2": "regression"}


def test_diff_missing_check_is_other_not_counted() -> None:
    # A check present in only one run is a change, but neither a fix nor a regression.
    grades_a = {"AAPL": {"c1": _grade(True)}}
    grades_b = {"AAPL": {"c1": _grade(True), "c2": _grade(True)}}
    diff = diff_eval_runs("a", "b", grades_a, grades_b)

    assert diff.fixes == 0
    assert diff.regressions == 0
    assert len(diff.ticker_diffs) == 1
    (change,) = diff.ticker_diffs[0].changes
    assert change.check_name == "c2"
    assert change.old is None and change.new is True
    assert change.kind == "other"


# --------------------------------------------------------------------------- #
# AppTest integration
# --------------------------------------------------------------------------- #


@pytest.fixture()
def eval_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[SimpleNamespace, None, None]:
    """Temp file-backed warren.db wired into storage.engine for AppTest."""
    db_path = tmp_path / "warren.db"
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setenv("WARREN_DB", str(db_path))
    monkeypatch.setenv("WARREN_LOGS_DIR", str(logs_dir))
    monkeypatch.setattr(eng, "engine", None)
    engine = eng.get_engine()
    Base.metadata.create_all(engine)
    yield SimpleNamespace(engine=engine)
    eng.engine = None


def _seed_two_runs(engine: Engine) -> None:
    """Baseline `run-a` and current `run-b`: AAPL fixes a check, MSFT regresses one."""
    with Session(engine) as session:
        session.add(PromptVersion(id=1, version_tag="v1"))
        session.add(PromptVersion(id=2, version_tag="v2"))
        session.add(_make_run("run-a", prompt_version_id=1, started=_BASE - timedelta(days=1)))
        session.add(_make_run("run-b", prompt_version_id=2, started=_BASE))
        session.flush()  # FK: runs before eval_runs
        session.add_all(
            [
                _make_eval_run("run-a", "AAPL", passed=False, checks=[_check("numerical", False)]),
                _make_eval_run("run-b", "AAPL", passed=True, checks=[_check("numerical", True)]),
                _make_eval_run(
                    "run-a", "MSFT", passed=True, checks=[_check("recommendation", True)]
                ),
                _make_eval_run(
                    "run-b", "MSFT", passed=False, checks=[_check("recommendation", False)]
                ),
            ]
        )
        session.commit()


def test_apptest_empty_db_shows_info(eval_env: SimpleNamespace) -> None:
    at = AppTest.from_file(_EVAL_PAGE).run()
    assert not at.exception
    assert any("No eval runs yet" in info.value for info in at.info)


def test_apptest_single_run_shows_diff_guard(eval_env: SimpleNamespace) -> None:
    with Session(eval_env.engine) as session:
        session.add(_make_run("run-a"))
        session.flush()
        session.add(_make_eval_run("run-a", "AAPL", passed=True, checks=[_check("c1", True)]))
        session.commit()

    at = AppTest.from_file(_EVAL_PAGE).run()
    assert not at.exception
    # Pass-rate section renders, but the diff section stops with its own info.
    assert any("at least 2 eval runs" in info.value for info in at.info)


def test_apptest_diff_shows_fix_and_regression(eval_env: SimpleNamespace) -> None:
    _seed_two_runs(eval_env.engine)
    at = AppTest.from_file(_EVAL_PAGE).run()
    assert not at.exception

    # Default selection: baseline=index 1 (older run-a), current=index 0 (newer run-b).
    # AAPL fixed one check (green success), MSFT regressed one (red error).
    success_text = " ".join(s.value for s in at.success)
    error_text = " ".join(e.value for e in at.error)
    assert "FIXED" in success_text and "numerical" in success_text
    assert "REGRESSION" in error_text and "recommendation" in error_text
    # Net-change banner: one fix, one regression → the error-styled net banner.
    assert any("+1 fixed" in e.value and "-1 regressions" in e.value for e in at.error)

    # Both changed tickers get an expander; unchanged ones would be omitted.
    labels = [e.label for e in at.expander]
    assert any("AAPL" in label for label in labels)
    assert any("MSFT" in label for label in labels)


def test_apptest_identical_runs_show_no_changes(eval_env: SimpleNamespace) -> None:
    with Session(eval_env.engine) as session:
        session.add(_make_run("run-a", started=_BASE - timedelta(days=1)))
        session.add(_make_run("run-b", started=_BASE))
        session.flush()
        for rid in ("run-a", "run-b"):
            session.add(_make_eval_run(rid, "AAPL", passed=True, checks=[_check("c1", True)]))
        session.commit()

    at = AppTest.from_file(_EVAL_PAGE).run()
    assert not at.exception
    # No fixes, no regressions → the neutral "No differences" info and no ticker expanders.
    assert any("No differences" in info.value for info in at.info)
    assert not at.expander
