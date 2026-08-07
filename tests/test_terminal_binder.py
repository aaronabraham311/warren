from __future__ import annotations

from io import StringIO

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from agent.models import AnalysisOutput, LynchBuffettSignals
from agent.service import RunResult, TickerRunResult
from agent.terminal.binder import ResultBinder, build_binder_document


def _result(
    *,
    thesis: str = "**Demand** is durable despite [valuation](https://example.test).",
) -> RunResult:
    analysis = AnalysisOutput(
        ticker="AMD",
        analysis_type="holding",
        recommendation="hold",
        confidence=0.68,
        thesis=thesis,
        lynch_signals=LynchBuffettSignals(pros=["*growth*"], cons=["price"]),
        buffett_signals=LynchBuffettSignals(pros=["moat"], cons=["- cycle"]),
        key_risks=["**Helios** timing", "api_key=secret"],
        data_quality_notes=["`Estimate` only"],
    )
    return RunResult(
        run_id="run-binder",
        status="success",
        ticker_results=(TickerRunResult("AMD", "holding", analysis=analysis),),
        total_cost_usd=0.125,
        total_input_tokens=120,
        total_output_tokens=30,
        total_tool_calls=4,
        duration_seconds=2.5,
        error_msg=None,
    )


def test_document_has_fixed_pages_clean_content_and_existing_facts_only() -> None:
    document = build_binder_document(
        _result(),
        evidence_tools=("Market quote", "Regulatory filing"),
    )

    assert (document.ticker, document.recommendation, document.confidence) == (
        "AMD",
        "HOLD",
        "68%",
    )
    assert [page.label for page in document.pages] == [
        "Summary",
        "Thesis",
        "Signals",
        "Risks",
        "Evidence",
    ]
    rendered = "\n".join(block.body for page in document.pages for block in page.blocks)
    assert "Demand is durable despite valuation (https://example.test)." in rendered
    assert "Helios timing" in rendered
    assert "Market quote" in rendered
    assert "• cycle" in rendered
    assert "• • cycle" not in rendered
    assert "[redacted]" in rendered
    assert "api_key=secret" not in rendered
    assert "**" not in rendered
    assert "`" not in rendered


def test_document_rejects_results_without_exactly_one_analysis() -> None:
    result = _result()
    empty = RunResult(
        run_id=result.run_id,
        status="failed",
        ticker_results=(),
        total_cost_usd=0.0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_tool_calls=0,
        duration_seconds=0.0,
        error_msg="failed",
    )
    with pytest.raises(ValueError, match="exactly one"):
        build_binder_document(empty)


@pytest.mark.parametrize("keys", ["q", "\x1b", "\x03", "\x04", "2j3k4l5h1?q"])
def test_binder_navigation_and_close_keys_restore_application(keys: str) -> None:
    output_text = StringIO()
    output = Vt100_Output.from_pty(output_text, term="xterm-256color", enable_bell=False)
    with create_pipe_input() as input_pipe:
        input_pipe.send_text(keys)
        ResultBinder(input=input_pipe, output=output).run(build_binder_document(_result()))

    transcript = output_text.getvalue()
    assert "AMD" in transcript
    assert "Summary" in transcript
    assert "\x1b[?1049h" in transcript
    assert "\x1b[?1049l" in transcript
