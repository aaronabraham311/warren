"""LLM-as-judge for the ``thesis_must_mention`` grading family.

The substring checks in ``eval.grader`` ask *does the thesis contain this exact word*.
That fails analytically-sound theses on vocabulary alone (a thesis discussing Chevron's
"integrated" cost advantage in different words fails ``breakeven_or_cost_position``). This
module grades the *idea* instead: a small Claude call decides whether the thesis
substantively reasons about the concept, however it's phrased.

Three determinism guarantees, mirroring the eval harness's invariants:

1. **Pinned model.** ``claude-sonnet-5`` (``_JUDGE_MODEL``), never the routed agent model.
2. **No sampling params + thinking off.** Sonnet 5 rejects ``temperature``/``top_p``/``top_k``
   with a 400 — the eval's ``temperature=0`` is agent-only — so the judge passes none, and
   sets ``thinking={"type": "disabled"}`` for a cheap, stable verdict.
3. **Cached verdicts.** Keyed on ``sha256(model + thesis + concept)`` in the shared
   ``CacheStore``, so re-running the same eval re-reads verdicts instead of paying for them.

The judge is a ``Protocol`` so tests inject a deterministic fake and never touch the
network — ``grade_analysis`` takes ``judge=None`` and keeps the substring behavior.
"""

from typing import Protocol

import anthropic
from pydantic import BaseModel

from agent.models import SONNET_5
from data_sources.cache import CacheStore, make_key

# yfinance-style basics: verdicts are a pure function of (thesis, concept, model), so a
# long TTL is a re-run cache, not a freshness window. 90 days matches the fixture policy.
_VERDICT_TTL_HOURS = 90 * 24

_JUDGE_MODEL = SONNET_5

# Bumped whenever the judge's grading semantics change, so verdicts cached under an older
# prompt are not reused. v2: align the judge with the golden set's any-of semantics.
_JUDGE_PROMPT_VERSION = "v2"

_SYSTEM = (
    "You grade an equity-analysis thesis against an expected topic that is defined by a set of "
    "acceptable framings (an any-of list). The expectation is SATISFIED if the thesis "
    "substantively engages AT LEAST ONE of those framings for the given company — it need not "
    "engage all of them, nor the broader narrative that connects them. Judge the idea, not the "
    "vocabulary: reasoning about any one framing in different words passes; a bare keyword drop "
    "with no real engagement does not. Do not demand more than one framing, or a specific angle "
    "the list does not require. Be strict about substance but faithful to the any-of semantics, "
    "and record your verdict with the record_verdict tool."
)

_VERDICT_TOOL: anthropic.types.ToolParam = {
    "name": "record_verdict",
    "description": "Record whether the thesis substantively engages the expected topic.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passes": {
                "type": "boolean",
                "description": "True if the thesis substantively reasons about the topic.",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence justifying the verdict.",
            },
        },
        "required": ["passes", "reasoning"],
    },
}


class JudgeVerdict(BaseModel):
    passes: bool
    reasoning: str


def _prompt(thesis: str, concept: list[str], ticker: str) -> str:
    alternatives = "\n".join(f"  - {framing}" for framing in concept)
    return (
        f"Company: {ticker}\n"
        f"Expected topic — satisfied by substantively engaging ANY ONE of these framings:\n"
        f"{alternatives}\n\n"
        f"Thesis:\n{thesis}\n\n"
        f"Does the thesis substantively reason about at least one of the framings above "
        f"for {ticker}? Engaging a single framing is enough; do not require the others."
    )


class ThesisJudge(Protocol):
    """Grades whether a thesis engages an expected topic. Implementations must be pure
    given ``(thesis, concept, ticker)`` so the eval stays reproducible."""

    def judge(self, *, thesis: str, concept: list[str], ticker: str) -> JudgeVerdict: ...


class SonnetThesisJudge:
    """A ``ThesisJudge`` backed by a pinned Sonnet 5 call, with a verdict cache."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        cache: CacheStore | None = None,
        model: str = _JUDGE_MODEL,
    ) -> None:
        self._client = client
        self._cache = cache
        self._model = model

    def judge(self, *, thesis: str, concept: list[str], ticker: str) -> JudgeVerdict:
        key = make_key("eval_judge", _JUDGE_PROMPT_VERSION, self._model, thesis, "|".join(concept))
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return JudgeVerdict.model_validate_json(cached)

        verdict = self._call(thesis, concept, ticker)

        if self._cache is not None:
            self._cache.set(key, verdict.model_dump_json(), _VERDICT_TTL_HOURS)
        return verdict

    def _call(self, thesis: str, concept: list[str], ticker: str) -> JudgeVerdict:
        # No temperature/top_p/top_k — Sonnet 5 rejects sampling params (HTTP 400).
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            thinking={"type": "disabled"},
            system=_SYSTEM,
            tools=[_VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "record_verdict"},
            messages=[{"role": "user", "content": _prompt(thesis, concept, ticker)}],
        )
        for block in response.content:
            if isinstance(block, anthropic.types.ToolUseBlock) and block.name == "record_verdict":
                return JudgeVerdict.model_validate(dict(block.input))
        raise ValueError(f"judge returned no record_verdict tool call for {ticker}")
