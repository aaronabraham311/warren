"""Golden eval set — hand-curated expectations for the agent's analysis output.

Each file under ``eval/examples/{ticker}.yaml`` describes the *envelope* of acceptable
output for one ticker: which recommendations are allowed, which topics the thesis must
(and must not) engage, how many Buffett/Lynch signals must be surfaced, which risks must
appear, and how many specific numbers must be cited. Nothing here hard-codes a single
expected answer — the eval runner asserts membership in the envelope, so a prompt change
that shifts a recommendation from ``hold`` to ``buy`` on an ambiguous ticker is visible
without an LLM-as-judge.

The expectation fields mirror ``agent.persona._ANALYSIS_OUTPUT_SCHEMA`` exactly:
``recommendation``, ``thesis``, ``buffett_signals.pros/cons``, ``lynch_signals.pros/cons``,
and ``key_risks``.

Every model sets ``extra="forbid"``: a misspelled expectation key must fail loudly, since
a silently-ignored key is an assertion that never runs.

    from eval.golden_set import load_all_examples
    for example in load_all_examples():
        ...
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_sources.symbols import TICKER_PATTERN

EXAMPLES_DIR = Path(__file__).parent / "examples"

Recommendation = Literal["buy", "sell", "hold"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SignalCount(_Strict):
    """Minimum number of entries required in a signals list, e.g. ``{min_count: 2}``."""

    min_count: int = Field(ge=0)


class SignalsExpectation(_Strict):
    """Per-philosophy expectations over ``{buffett,lynch}_signals.{pros,cons}``."""

    pros: SignalCount | None = None
    cons: SignalCount | None = None


class ThesisMention(_Strict):
    """A keyword group; the thesis satisfies it by mentioning any one member."""

    any_of: list[str] = Field(min_length=1)


class RecommendationExpectation(_Strict):
    allowed: list[Recommendation] = Field(min_length=1)
    preferred: Recommendation | None = None

    @model_validator(mode="after")
    def _preferred_is_allowed(self) -> RecommendationExpectation:
        if self.preferred is not None and self.preferred not in self.allowed:
            raise ValueError(
                f"preferred={self.preferred!r} is not in allowed={self.allowed!r}",
            )
        return self


class KeyRisksExpectation(_Strict):
    must_include_one_of: list[str] = []


class NumericalGrounding(_Strict):
    min_specific_numbers: int = Field(default=3, ge=0)
    no_hallucinated_format: bool = True


class EvalExpectations(_Strict):
    recommendation: RecommendationExpectation
    thesis_must_mention: list[ThesisMention] = []
    thesis_must_not_mention: list[str] = []
    buffett_signals: SignalsExpectation = SignalsExpectation()
    lynch_signals: SignalsExpectation = SignalsExpectation()
    key_risks: KeyRisksExpectation = KeyRisksExpectation()
    numerical_grounding: NumericalGrounding = NumericalGrounding()


class EvalExample(_Strict):
    """One curated ticker expectation file."""

    ticker: str = Field(pattern=TICKER_PATTERN)
    notes: str = Field(min_length=1)
    last_curated: date
    expectations: EvalExpectations


def _stem_for(ticker: str) -> str:
    """``BRK.B`` → ``brk_b`` — the filename stem a ticker must live under."""
    return ticker.replace(".", "_").lower()


def load_eval_example(path: Path) -> EvalExample:
    """Parse and validate one golden YAML file.

    Raises ``ValueError`` if the file does not validate, or if its filename disagrees with
    the ticker it declares (a copy-paste guard).
    """
    parsed = yaml.safe_load(path.read_text())
    try:
        example = EvalExample.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001 — re-raised with the offending path attached
        raise ValueError(f"{path}: {exc}") from exc

    if path.stem != _stem_for(example.ticker):
        raise ValueError(
            f"{path}: filename stem {path.stem!r} does not match "
            f"ticker {example.ticker!r} (expected {_stem_for(example.ticker)!r})",
        )
    return example


def load_all_examples(directory: Path = EXAMPLES_DIR) -> list[EvalExample]:
    """Load every ``*.yaml`` under ``directory``, sorted by filename."""
    return [load_eval_example(p) for p in sorted(directory.glob("*.yaml"))]
