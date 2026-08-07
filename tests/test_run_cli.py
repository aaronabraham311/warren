from argparse import Namespace
from unittest.mock import patch

import pytest

from agent.run import _print_result, _request_from_args, main
from agent.service import RunMode, RunResult, ScreeningSummary


def _result(*, screening: ScreeningSummary | None = None) -> RunResult:
    return RunResult(
        run_id="run-1",
        status="success",
        ticker_results=(),
        total_cost_usd=0.0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_tool_calls=0,
        duration_seconds=1.0,
        error_msg=None,
        screening=screening,
    )


def test_batch_args_map_to_existing_modes_and_gem_forces_dirt() -> None:
    ticker = _request_from_args(
        Namespace(ticker="aapl", gem_hunt=False, persona="default", skip_ticker_validation=True)
    )
    assert ticker.mode is RunMode.TICKERS
    assert ticker.tickers == ["AAPL"]
    assert ticker.skip_ticker_validation is True

    nightly = _request_from_args(
        Namespace(ticker=None, gem_hunt=False, persona="dirt", skip_ticker_validation=False)
    )
    assert nightly.mode is RunMode.DISCOVERY
    assert nightly.persona == "dirt"

    gem = _request_from_args(
        Namespace(ticker=None, gem_hunt=True, persona="default", skip_ticker_validation=False)
    )
    assert gem.mode is RunMode.GEM_HUNT


def test_batch_output_preserves_screening_summary(capsys: pytest.CaptureFixture[str]) -> None:
    _print_result(
        _result(
            screening=ScreeningSummary(
                confirmed_count=4,
                needs_deeper_fetch_count=2,
                source_error_count=1,
                selected_candidates=("AAPL", "MSFT"),
                gem_hunt=True,
            )
        )
    )
    output = capsys.readouterr().out
    assert "Gem screen: 4 confirmed, 2 need deeper fetch, 1 source errors" in output
    assert "Screening surfaced 4 candidates; analysing top 2: ['AAPL', 'MSFT']" in output


def test_main_calls_shared_service_without_shelling_out() -> None:
    args = Namespace(
        ticker="AAPL",
        gem_hunt=False,
        persona="default",
        skip_ticker_validation=False,
    )
    with (
        patch("agent.run.build_parser") as parser,
        patch("agent.run.execute_run", return_value=_result()) as execute,
    ):
        parser.return_value.parse_args.return_value = args
        main()

    request = execute.call_args.args[0]
    assert request.mode is RunMode.TICKERS
    assert request.tickers == ["AAPL"]
