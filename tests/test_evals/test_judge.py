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
from eval.judge import (
    HumanVerdictJudge,
    JudgeUnavailableError,
    JudgeVerdict,
    SemanticRequest,
    SonnetThesisJudge,
    canonical_request_key,
)


def _verdict_response(
    passes: bool,
    reasoning: str,
    check_ids: list[str] | None = None,
) -> MagicMock:
    ids = check_ids or ["semantic_concept"]
    block = anthropic.types.ToolUseBlock(
        id="tu_1",
        name="record_verdicts",
        input={
            "verdicts": [
                {"check_id": check_id, "passes": passes, "reasoning": reasoning} for check_id in ids
            ]
        },
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

    assert verdict.passes
    assert verdict.reasoning == "engages the topic"
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == SONNET_5
    # Sonnet 5 rejects sampling params with a 400 — the judge must pass none.
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_verdicts"}


def test_judge_caches_verdicts_across_calls() -> None:
    client = _stub_client(_verdict_response(False, "no real engagement"))
    cache = CacheStore(sqlite3.connect(":memory:"))
    judge = SonnetThesisJudge(client, cache)

    first = judge.judge(thesis="same thesis", concept=["take rate"], ticker="PYPL")
    second = judge.judge(thesis="same thesis", concept=["take rate"], ticker="PYPL")

    assert first == second
    assert not first.passes
    assert first.reasoning == "no real engagement"
    # Second call served from cache — the model is hit exactly once.
    assert client.messages.create.call_count == 1


def test_judge_raises_when_no_verdict_tool_call() -> None:
    response = MagicMock()
    response.content = [anthropic.types.TextBlock(text="oops", type="text")]
    judge = SonnetThesisJudge(_stub_client(response))
    with pytest.raises(JudgeUnavailableError, match="record_verdicts"):
        judge.judge(thesis="…", concept=["moat"], ticker="AAPL")


def test_multiple_checks_are_sent_in_one_live_request() -> None:
    requests = [
        SemanticRequest(check_id="risk", text="risk text", concept=["leverage"], ticker="AAPL"),
        SemanticRequest(check_id="thesis", text="thesis text", concept=["moat"], ticker="AAPL"),
    ]
    client = _stub_client(_verdict_response(True, "supported", ["risk", "thesis"]))
    verdicts = SonnetThesisJudge(client).judge_many(requests)
    assert set(verdicts) == {"risk", "thesis"}
    assert client.messages.create.call_count == 1


def test_human_verdicts_use_canonical_collision_safe_keys() -> None:
    left = SemanticRequest(
        check_id="risk", text="a", concept=["x|y", "z"], ticker="AAPL", rubric="r"
    )
    right = SemanticRequest(
        check_id="risk", text="a", concept=["x", "y|z"], ticker="AAPL", rubric="r"
    )
    left_key = canonical_request_key(left, judge_id="human")
    right_key = canonical_request_key(right, judge_id="human")
    assert left_key != right_key

    judge = HumanVerdictJudge({left_key: JudgeVerdict(passes=True, reasoning="reviewed")})
    assert judge.judge_many([left])["risk"].passes
    with pytest.raises(JudgeUnavailableError, match="no verdict"):
        judge.judge_many([right])
