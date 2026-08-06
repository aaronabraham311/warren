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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data_sources.symbols import TICKER_PATTERN

EXAMPLES_DIR = Path(__file__).parent / "examples"

Recommendation = Literal["buy", "sell", "hold"]
Persona = Literal["default", "dirt"]


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


class RiskConcept(_Strict):
    """One structured risk concept, with wording variants for semantic grading."""

    concept: str = Field(min_length=1)
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
    must_include_one_of: list[RiskConcept] = []

    @field_validator("must_include_one_of", mode="before")
    @classmethod
    def _upgrade_legacy_strings(cls, value: object) -> object:
        """Keep older golden files valid while making the in-memory contract structured."""
        if not isinstance(value, list):
            return value
        return [
            {"concept": item, "any_of": [item]} if isinstance(item, str) else item for item in value
        ]


class NumericalGrounding(_Strict):
    min_specific_numbers: int = Field(default=3, ge=0)
    no_hallucinated_format: bool = True


class DeepValueExpectation(_Strict):
    """Deep-value (DIRT) check toggles — each fires the matching grader check when true.

    Present only on ``persona: dirt`` examples; every check it configures is a ``must`` (a
    DIRT thesis that never surfaces EV/EBIT, NCAV, a substantive value-trap assessment, or
    the universe note is a regression in the deep-value path, not a stylistic drift).
    """

    require_ev_ebit: bool = False
    require_ncav: bool = False
    require_value_trap_assessment: bool = False
    require_universe_note: bool = False
    # Opt-in because only examples with a grounded regional forensic fixture can assert it.
    # The check traces ownership/RPT/buyback/catalyst claims to compact EvidenceRef IDs.
    require_forensic_citations: bool = False
    require_decision_contract: bool = False
    require_decision_recomputation: bool = False
    allowed_decision_outcomes: list[Literal["buy", "watchlist", "pass"]] = []


class ClosabilityExpectation(_Strict):
    """Envelope for the compact, cited G14 decision fields.

    Unknown coverage is intentionally a first-class allowed outcome. Bounds let a golden
    example constrain confidence without pinning one exact score or recommendation.
    """

    allowed_status: list[Literal["supported", "constrained", "unknown"]] = Field(min_length=1)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_score: float | None = Field(default=None, ge=0.0, le=1.0)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    min_reasons: int = Field(default=1, ge=0)
    require_unknown_semantics: bool = False
    require_observable_or_contractual_catalyst: bool = False

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> ClosabilityExpectation:
        if (
            self.min_score is not None
            and self.max_score is not None
            and self.min_score > self.max_score
        ):
            raise ValueError("closability min_score cannot exceed max_score")
        if (
            self.min_confidence is not None
            and self.max_confidence is not None
            and self.min_confidence > self.max_confidence
        ):
            raise ValueError("closability min_confidence cannot exceed max_confidence")
        return self


class EvalExpectations(_Strict):
    recommendation: RecommendationExpectation
    thesis_must_mention: list[ThesisMention] = []
    thesis_must_not_mention: list[str] = []
    buffett_signals: SignalsExpectation = SignalsExpectation()
    lynch_signals: SignalsExpectation = SignalsExpectation()
    key_risks: KeyRisksExpectation = KeyRisksExpectation()
    numerical_grounding: NumericalGrounding = NumericalGrounding()
    deep_value: DeepValueExpectation | None = None
    closability: ClosabilityExpectation | None = None


class EvalExample(_Strict):
    """One curated ticker expectation file."""

    ticker: str = Field(pattern=TICKER_PATTERN)
    notes: str = Field(min_length=1)
    last_curated: date
    persona: Persona = "default"
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
