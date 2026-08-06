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
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from agent.models import LUNA_5_6, SONNET_5
from agent.providers.base import ReasoningEffort
from data_sources.cache import CacheStore, make_key

_VERDICT_TTL_HOURS = 90 * 24
_JUDGE_MODEL = SONNET_5
JUDGE_PROMPT_VERSION = "v3-batched-blind"

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
    votes: list[JudgeVote] = Field(default_factory=list)


class _VerdictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    passes: bool
    reasoning: str


class _VerdictBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
        "prompt_version": JUDGE_PROMPT_VERSION,
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

    @property
    def judge_id(self) -> str:
        return "panel:" + ",".join(judge_id for judge_id, _judge in self._judges)

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
                batch = _VerdictBatch.model_validate(_normalize_sonnet_tool_input(block.input))
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


def _normalize_sonnet_tool_input(raw_input: object) -> object:
    """Accept a JSON-encoded verdict list while keeping its records strictly validated."""
    if not isinstance(raw_input, Mapping):
        return raw_input
    raw_verdicts = raw_input.get("verdicts")
    if not isinstance(raw_verdicts, str):
        return raw_input
    start = raw_verdicts.find("[")
    if start < 0:
        return raw_input
    try:
        verdicts, _end = json.JSONDecoder().raw_decode(raw_verdicts[start:])
    except json.JSONDecodeError:
        return raw_input
    return {**raw_input, "verdicts": verdicts}


class OpenAIThesisJudge:
    """Pinned, blinded OpenAI judge using one strict structured-output batch."""

    def __init__(
        self,
        client: OpenAI,
        cache: CacheStore | None = None,
        model: str = LUNA_5_6,
        reasoning_effort: ReasoningEffort = "medium",
    ) -> None:
        self._client = client
        self._cache = cache
        self._model = model
        self._reasoning_effort = reasoning_effort

    @property
    def judge_id(self) -> str:
        return f"openai:{self._model}:{self._reasoning_effort}:{JUDGE_PROMPT_VERSION}"

    def judge(self, *, thesis: str, concept: list[str], ticker: str) -> JudgeVerdict:
        request = _single_request(thesis, concept, ticker)
        return self.judge_many([request])[request.check_id]

    def judge_many(self, requests: Sequence[SemanticRequest]) -> dict[str, JudgeVerdict]:
        request_ids = [request.check_id for request in requests]
        if len(set(request_ids)) != len(request_ids):
            raise JudgeUnavailableError("judge requests contain duplicate check ids")

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
                try:
                    result[request.check_id] = JudgeVerdict.model_validate_json(cached)
                except Exception as exc:
                    raise JudgeUnavailableError(
                        f"cached OpenAI verdict is malformed for {request.check_id!r}"
                    ) from exc

        if missing:
            fresh = self._call_many(missing)
            result.update(fresh)
            if self._cache is not None:
                for check_id, verdict in fresh.items():
                    self._cache.set(keys[check_id], verdict.model_dump_json(), _VERDICT_TTL_HOURS)

        omitted = set(request_ids) - set(result)
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
        input_text = "Grade every item in this JSON array:\n" + json.dumps(
            items, ensure_ascii=False
        )
        max_output_tokens = max(512, 256 * len(requests))

        try:
            parse = getattr(self._client.responses, "parse", None)
            if callable(parse):
                response = parse(
                    model=self._model,
                    instructions=_SYSTEM,
                    input=input_text,
                    max_output_tokens=max_output_tokens,
                    reasoning={"effort": self._reasoning_effort},
                    store=False,
                    text_format=_VerdictBatch,
                )
                raw_batch = response.output_parsed
                batch = (
                    raw_batch
                    if isinstance(raw_batch, _VerdictBatch)
                    else _VerdictBatch.model_validate(raw_batch)
                )
            else:
                response = self._client.responses.create(
                    model=self._model,
                    instructions=_SYSTEM,
                    input=input_text,
                    max_output_tokens=max_output_tokens,
                    reasoning={"effort": self._reasoning_effort},
                    store=False,
                    stream=False,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "eval_judge_verdicts",
                            "schema": _VerdictBatch.model_json_schema(),
                            "strict": True,
                        }
                    },
                )
                output_text = getattr(response, "output_text", None)
                if not isinstance(output_text, str) or not output_text:
                    raise JudgeUnavailableError("OpenAI judge returned no structured output")
                batch = _VerdictBatch.model_validate_json(output_text)
        except JudgeUnavailableError:
            raise
        except Exception as exc:
            raise JudgeUnavailableError(f"OpenAI judge unavailable: {exc}") from exc

        expected = {request.check_id for request in requests}
        actual = {record.check_id for record in batch.verdicts}
        if actual != expected or len(actual) != len(batch.verdicts):
            raise JudgeUnavailableError(
                f"judge verdict ids mismatch: expected {sorted(expected)}, got {sorted(actual)}"
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
