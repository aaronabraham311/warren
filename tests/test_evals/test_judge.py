"""Offline unit tests for blinded semantic judges.

Covers the determinism guarantees that matter: no sampling params (Sonnet 5 rejects them),
thinking disabled, a forced record_verdict tool, and verdict caching so re-runs are free.
"""

import json
import sqlite3
from unittest.mock import MagicMock

import anthropic
import pytest

from agent.models import LUNA_5_6, SONNET_5
from data_sources.cache import CacheStore
from eval.judge import (
    HumanVerdictJudge,
    JudgeUnavailableError,
    JudgeVerdict,
    OpenAIThesisJudge,
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


def _openai_response(records: list[dict[str, object]]) -> MagicMock:
    response = MagicMock()
    response.output_parsed = {"verdicts": records}
    return response


def _stub_openai_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.responses.parse.return_value = response
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


def test_sonnet_judge_accepts_json_encoded_verdict_list_with_trailing_text() -> None:
    records = [
        {"check_id": "risk", "passes": True, "reasoning": "engages leverage"},
        {"check_id": "thesis", "passes": False, "reasoning": "does not support moat"},
    ]
    response = MagicMock()
    response.content = [
        anthropic.types.ToolUseBlock(
            id="toolu_1",
            name="record_verdicts",
            input={"verdicts": json.dumps(records) + "}\n"},
            type="tool_use",
        )
    ]
    requests = [
        SemanticRequest(check_id="risk", text="risk", concept=["x"], ticker="AAPL"),
        SemanticRequest(check_id="thesis", text="thesis", concept=["y"], ticker="AAPL"),
    ]

    verdicts = SonnetThesisJudge(_stub_client(response)).judge_many(requests)

    assert verdicts["risk"].passes
    assert not verdicts["thesis"].passes


def test_openai_judge_batches_strict_blinded_requests() -> None:
    requests = [
        SemanticRequest(
            check_id="risk", text="leveraged balance sheet", concept=["leverage"], ticker="AAPL"
        ),
        SemanticRequest(
            check_id="thesis", text="durable switching costs", concept=["moat"], ticker="AAPL"
        ),
    ]
    client = _stub_openai_client(
        _openai_response(
            [
                {"check_id": "risk", "passes": True, "reasoning": "engages leverage"},
                {"check_id": "thesis", "passes": True, "reasoning": "supports moat"},
            ]
        )
    )

    verdicts = OpenAIThesisJudge(client).judge_many(requests)

    assert set(verdicts) == {"risk", "thesis"}
    assert all(verdict.passes for verdict in verdicts.values())
    assert client.responses.parse.call_count == 1
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs["model"] == LUNA_5_6
    assert kwargs["reasoning"] == {"effort": "medium"}
    assert kwargs["text_format"].__name__ == "_VerdictBatch"
    assert kwargs["store"] is False
    wire_items = json.loads(kwargs["input"].split("\n", 1)[1])
    assert all(
        set(item) == {"check_id", "company", "rubric", "acceptable_framings", "candidate_text"}
        for item in wire_items
    )
    assert wire_items[0]["candidate_text"] == "leveraged balance sheet"
    assert wire_items[1]["candidate_text"] == "durable switching costs"


def test_openai_judge_reuses_canonical_cache() -> None:
    client = _stub_openai_client(
        _openai_response(
            [{"check_id": "semantic_concept", "passes": False, "reasoning": "unsupported"}]
        )
    )
    cache = CacheStore(sqlite3.connect(":memory:"))
    judge = OpenAIThesisJudge(client, cache, reasoning_effort="medium")

    first = judge.judge(thesis="same thesis", concept=["moat"], ticker="AAPL")
    second = judge.judge(thesis="same thesis", concept=["moat"], ticker="AAPL")

    assert first == second
    assert not first.passes
    assert client.responses.parse.call_count == 1
    assert judge.judge_id.startswith(f"openai:{LUNA_5_6}:medium:")


@pytest.mark.parametrize(
    "records",
    [
        [{"check_id": "risk", "passes": True, "reasoning": "only one"}],
        [
            {"check_id": "risk", "passes": True, "reasoning": "first"},
            {"check_id": "risk", "passes": False, "reasoning": "duplicate"},
        ],
    ],
    ids=["omitted", "duplicate"],
)
def test_openai_judge_rejects_incomplete_or_duplicate_ids(
    records: list[dict[str, object]],
) -> None:
    requests = [
        SemanticRequest(check_id="risk", text="risk", concept=["x"], ticker="AAPL"),
        SemanticRequest(check_id="thesis", text="thesis", concept=["y"], ticker="AAPL"),
    ]
    judge = OpenAIThesisJudge(_stub_openai_client(_openai_response(records)))

    with pytest.raises(JudgeUnavailableError, match="verdict ids mismatch"):
        judge.judge_many(requests)


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
