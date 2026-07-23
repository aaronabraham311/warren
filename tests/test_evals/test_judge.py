"""Unit tests for the Sonnet-5-backed thesis judge — offline, via a stub client.

Covers the determinism guarantees that matter: no sampling params (Sonnet 5 rejects them),
thinking disabled, a forced record_verdict tool, and verdict caching so re-runs are free.
"""

import sqlite3
from unittest.mock import MagicMock

import anthropic
import pytest

from agent.models import SONNET_5
from data_sources.cache import CacheStore
from eval.judge import JudgeVerdict, SonnetThesisJudge


def _verdict_response(passes: bool, reasoning: str) -> MagicMock:
    block = anthropic.types.ToolUseBlock(
        id="tu_1",
        name="record_verdict",
        input={"passes": passes, "reasoning": reasoning},
        type="tool_use",
    )
    response = MagicMock()
    response.content = [block]
    return response


def _stub_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = response
    return client


def test_judge_pins_sonnet_5_and_omits_sampling_params() -> None:
    client = _stub_client(_verdict_response(True, "engages the topic"))
    judge = SonnetThesisJudge(client)

    verdict = judge.judge(thesis="…", concept=["moat"], ticker="AAPL")

    assert verdict == JudgeVerdict(passes=True, reasoning="engages the topic")
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == SONNET_5
    # Sonnet 5 rejects sampling params with a 400 — the judge must pass none.
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_verdict"}


def test_judge_caches_verdicts_across_calls() -> None:
    client = _stub_client(_verdict_response(False, "no real engagement"))
    cache = CacheStore(sqlite3.connect(":memory:"))
    judge = SonnetThesisJudge(client, cache)

    first = judge.judge(thesis="same thesis", concept=["take rate"], ticker="PYPL")
    second = judge.judge(thesis="same thesis", concept=["take rate"], ticker="PYPL")

    assert first == second == JudgeVerdict(passes=False, reasoning="no real engagement")
    # Second call served from cache — the model is hit exactly once.
    assert client.messages.create.call_count == 1


def test_judge_raises_when_no_verdict_tool_call() -> None:
    response = MagicMock()
    response.content = [anthropic.types.TextBlock(text="oops", type="text")]
    judge = SonnetThesisJudge(_stub_client(response))
    with pytest.raises(ValueError, match="record_verdict"):
        judge.judge(thesis="…", concept=["moat"], ticker="AAPL")
