"""Blinded, batched semantic judging for eval concepts.

The grader supplies only the candidate text and rubric: judges never see the producing
provider, the preferred recommendation, or another judge's vote.  Live Sonnet requests
are batched per ticker and cached per item.  ``HumanVerdictJudge`` provides the same
contract from a deterministic local verdict mapping for offline adjudication.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Protocol

import anthropic
from pydantic import BaseModel, ConfigDict, Field

from agent.models import SONNET_5
from data_sources.cache import CacheStore, make_key

_VERDICT_TTL_HOURS = 90 * 24
_JUDGE_MODEL = SONNET_5
_JUDGE_PROMPT_VERSION = "v3-batched-blind"

_SYSTEM = (
    "You are a blinded equity-analysis evaluator. Grade each independent item only against "
    "its supplied rubric. You are not told which model produced the analysis or what a golden "
    "recommendation label says. Judge substantive reasoning, not vocabulary: a bare keyword "
    "does not pass, while an evidence-backed conclusion phrased differently does. For an any-of "
    "rubric, substantively engaging one framing is sufficient. Return one verdict for every id."
)

_VERDICTS_TOOL: anthropic.types.ToolParam = {
    "name": "record_verdicts",
    "description": "Record one semantic grading verdict for every supplied opaque check id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "check_id": {"type": "string"},
                        "passes": {"type": "boolean"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["check_id", "passes", "reasoning"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    },
}


class SemanticRequest(BaseModel):
    """One blinded semantic check. ``check_id`` is stable and contains no gold answer."""

    model_config = ConfigDict(frozen=True)

    check_id: str = Field(min_length=1)
    text: str
    concept: list[str] = Field(min_length=1)
    ticker: str
    rubric: str = "Substantively engages at least one supplied framing."


class JudgeVote(BaseModel):
    judge_id: str
    passes: bool
    reasoning: str


class JudgeVerdict(BaseModel):
    passes: bool
    reasoning: str
    agreement: bool = True
    votes: list[JudgeVote] = []


class _VerdictRecord(BaseModel):
    check_id: str
    passes: bool
    reasoning: str


class _VerdictBatch(BaseModel):
    verdicts: list[_VerdictRecord]


class JudgeUnavailableError(RuntimeError):
    """A configured judge could not produce a complete set of verdicts."""


class ThesisJudge(Protocol):
    """Compatibility name for a field-neutral semantic judge."""

    def judge(self, *, thesis: str, concept: list[str], ticker: str) -> JudgeVerdict: ...

    def judge_many(self, requests: Sequence[SemanticRequest]) -> dict[str, JudgeVerdict]: ...


def canonical_request_key(request: SemanticRequest, *, judge_id: str) -> str:
    """Collision-safe content key for cached or human verdicts."""
    payload = {
        "prompt_version": _JUDGE_PROMPT_VERSION,
        "judge_id": judge_id,
        "check_id": request.check_id,
        "ticker": request.ticker,
        "text": request.text,
        "concept": request.concept,
        "rubric": request.rubric,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _single_request(thesis: str, concept: list[str], ticker: str) -> SemanticRequest:
    return SemanticRequest(
        check_id="semantic_concept",
        text=thesis,
        concept=concept,
        ticker=ticker,
    )


class HumanVerdictJudge:
    """Deterministic local judge backed by canonical-request-keyed human verdicts."""

    def __init__(
        self,
        verdicts: Mapping[str, JudgeVerdict | bool],
        judge_id: str = "human",
    ) -> None:
        self._verdicts = verdicts
        self.judge_id = judge_id

    def judge(self, *, thesis: str, concept: list[str], ticker: str) -> JudgeVerdict:
        request = _single_request(thesis, concept, ticker)
        return self.judge_many([request])[request.check_id]

    def judge_many(self, requests: Sequence[SemanticRequest]) -> dict[str, JudgeVerdict]:
        result: dict[str, JudgeVerdict] = {}
        for request in requests:
            key = canonical_request_key(request, judge_id=self.judge_id)
            raw = self._verdicts.get(key)
            if raw is None:
                raise JudgeUnavailableError(
                    f"{self.judge_id} has no verdict for {request.check_id} ({key})"
                )
            verdict = (
                raw
                if isinstance(raw, JudgeVerdict)
                else JudgeVerdict(passes=raw, reasoning="human verdict")
            )
            result[request.check_id] = verdict.model_copy(
                update={
                    "votes": [
                        JudgeVote(
                            judge_id=self.judge_id,
                            passes=verdict.passes,
                            reasoning=verdict.reasoning,
                        )
                    ]
                }
            )
        return result


class JudgePanel:
    """Require agreement across independently configured blinded judges."""

    def __init__(self, judges: Sequence[tuple[str, ThesisJudge]]) -> None:
        if not judges:
            raise ValueError("a judge panel needs at least one judge")
        self._judges = tuple(judges)

    def judge(self, *, thesis: str, concept: list[str], ticker: str) -> JudgeVerdict:
        request = _single_request(thesis, concept, ticker)
        return self.judge_many([request])[request.check_id]

    def judge_many(self, requests: Sequence[SemanticRequest]) -> dict[str, JudgeVerdict]:
        votes_by_check: dict[str, list[JudgeVote]] = {request.check_id: [] for request in requests}
        for judge_id, judge in self._judges:
            try:
                verdicts = judge.judge_many(requests)
            except Exception as exc:  # one missing judge is an explicit eval outcome
                raise JudgeUnavailableError(f"judge {judge_id!r} failed: {exc}") from exc
            for request in requests:
                verdict = verdicts.get(request.check_id)
                if verdict is None:
                    raise JudgeUnavailableError(f"judge {judge_id!r} omitted {request.check_id!r}")
                votes_by_check[request.check_id].append(
                    JudgeVote(
                        judge_id=judge_id,
                        passes=verdict.passes,
                        reasoning=verdict.reasoning,
                    )
                )

        result: dict[str, JudgeVerdict] = {}
        for check_id, votes in votes_by_check.items():
            agreement = len({vote.passes for vote in votes}) == 1
            passes = agreement and votes[0].passes
            reasoning = "; ".join(f"{vote.judge_id}: {vote.reasoning}" for vote in votes)
            result[check_id] = JudgeVerdict(
                passes=passes,
                reasoning=reasoning,
                agreement=agreement,
                votes=votes,
            )
        return result


class SonnetThesisJudge:
    """Pinned Sonnet semantic judge; uncached checks are sent in one batched request."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        cache: CacheStore | None = None,
        model: str = _JUDGE_MODEL,
    ) -> None:
        self._client = client
        self._cache = cache
        self._model = model

    @property
    def judge_id(self) -> str:
        return f"anthropic:{self._model}"

    def judge(self, *, thesis: str, concept: list[str], ticker: str) -> JudgeVerdict:
        request = _single_request(thesis, concept, ticker)
        return self.judge_many([request])[request.check_id]

    def judge_many(self, requests: Sequence[SemanticRequest]) -> dict[str, JudgeVerdict]:
        result: dict[str, JudgeVerdict] = {}
        missing: list[SemanticRequest] = []
        keys: dict[str, str] = {}
        for request in requests:
            digest = canonical_request_key(request, judge_id=self.judge_id)
            key = make_key("eval_judge", digest)
            keys[request.check_id] = key
            cached = self._cache.get(key) if self._cache is not None else None
            if cached is None:
                missing.append(request)
            else:
                result[request.check_id] = JudgeVerdict.model_validate_json(cached)

        if missing:
            fresh = self._call_many(missing)
            result.update(fresh)
            if self._cache is not None:
                for check_id, verdict in fresh.items():
                    self._cache.set(keys[check_id], verdict.model_dump_json(), _VERDICT_TTL_HOURS)

        omitted = {request.check_id for request in requests} - set(result)
        if omitted:
            raise JudgeUnavailableError(f"judge omitted verdicts: {sorted(omitted)}")
        return result

    def _call_many(self, requests: Sequence[SemanticRequest]) -> dict[str, JudgeVerdict]:
        items = [
            {
                "check_id": request.check_id,
                "company": request.ticker,
                "rubric": request.rubric,
                "acceptable_framings": request.concept,
                "candidate_text": request.text,
            }
            for request in requests
        ]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max(512, 256 * len(requests)),
            thinking={"type": "disabled"},
            system=_SYSTEM,
            tools=[_VERDICTS_TOOL],
            tool_choice={"type": "tool", "name": "record_verdicts"},
            messages=[
                {
                    "role": "user",
                    "content": "Grade every item in this JSON array:\n"
                    + json.dumps(items, ensure_ascii=False),
                }
            ],
        )
        for block in response.content:
            if isinstance(block, anthropic.types.ToolUseBlock) and block.name == "record_verdicts":
                batch = _VerdictBatch.model_validate(block.input)
                expected = {request.check_id for request in requests}
                actual = {record.check_id for record in batch.verdicts}
                if actual != expected or len(actual) != len(batch.verdicts):
                    raise JudgeUnavailableError(
                        "judge verdict ids mismatch: "
                        f"expected {sorted(expected)}, got {sorted(actual)}"
                    )
                return {
                    record.check_id: JudgeVerdict(
                        passes=record.passes,
                        reasoning=record.reasoning,
                        votes=[
                            JudgeVote(
                                judge_id=self.judge_id,
                                passes=record.passes,
                                reasoning=record.reasoning,
                            )
                        ],
                    )
                    for record in batch.verdicts
                }
        raise JudgeUnavailableError("judge returned no record_verdicts tool call")
