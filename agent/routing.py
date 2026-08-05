from collections.abc import Callable, Sequence
from typing import Literal, Protocol, runtime_checkable

import anthropic

from agent.models import HAIKU_4_5, OPUS_4_7, SONNET_4_6, DirtSignals, LynchBuffettSignals

# The set of models the routing layer may select. Kept in sync with agent.models
# by tests/test_routing.py (a drift guard asserts get_args(ModelID) matches the
# constants below). select() returns the agent.models constants, not these strings.
ModelID = Literal[
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

Phase = Literal["screen", "deep", "synthesize"]

# A synthesis call is marked by the orchestrator appending a user message whose text
# contains this token. Phase detection reads conversation state (the messages), not
# the iteration number, so routing adapts to what the agent has actually done.
SYNTHESIS_PHASE_MARKER = "[phase:synthesize]"


@runtime_checkable
class RoutingPolicy(Protocol):
    """Structural interface the eval harness can swap without touching analyze_ticker()."""

    def select(
        self,
        iteration: int,
        messages: list[anthropic.types.MessageParam],
        ticker: str | None,
    ) -> ModelID: ...


class _Analysis(Protocol):
    """Structural shape of agent.loop.AnalysisOutput the Opus trigger reads.

    Declared locally (rather than importing AnalysisOutput) so routing.py has no
    dependency on agent.loop — agent.loop imports the routing types, so importing
    back would create a cycle. Members are read-only properties so that covariant
    fields (e.g. AnalysisOutput.recommendation: Literal[...]) satisfy the protocol.
    """

    @property
    def confidence(self) -> float: ...
    @property
    def recommendation(self) -> str: ...
    @property
    def lynch_signals(self) -> LynchBuffettSignals: ...
    @property
    def buffett_signals(self) -> LynchBuffettSignals: ...
    # Populated only in deep-value (DIRT) mode; None for the default persona, where
    # the Lynch/Buffett pros/cons carry the signal instead. Declared read-only for
    # the same covariance reason as the fields above. Imported directly from
    # agent.models (same module as LynchBuffettSignals), so no import cycle arises.
    @property
    def dirt_signals(self) -> DirtSignals | None: ...


class OpusTrigger(Protocol):
    def should_fire(self, analyses: Sequence[_Analysis]) -> bool: ...


class DefaultOpusTrigger:
    """Decides when the most expensive model (Opus) is warranted — Tech Spec §4.2.

    Fires when at least one of three independent conditions holds. Each catches a
    distinct failure mode: (1) uncertain-but-acting, (2) the two frameworks telling
    opposite stories, (3) a high-stakes sell. The goal is 0–2 fires per nightly run.
    """

    def should_fire(self, analyses: Sequence[_Analysis]) -> bool:
        # Condition 1: ≥2 holdings are low-confidence (<0.5) yet recommend an action.
        low_conf_action = [a for a in analyses if a.confidence < 0.5 and a.recommendation != "hold"]
        cond_low_conf = len(low_conf_action) >= 2

        # Condition 2: Lynch and Buffett total signal counts disagree by ≥3 for any holding.
        def _total(s: LynchBuffettSignals) -> int:
            return len(s.pros) + len(s.cons)

        cond_signal_split = any(
            abs(_total(a.lynch_signals) - _total(a.buffett_signals)) >= 3 for a in analyses
        )

        # Condition 3: any sell recommendation in the current run.
        cond_sell = any(a.recommendation == "sell" for a in analyses)

        # Condition 4 (deep-value / DIRT mode): a contested or fragile deep-value setup.
        # In DIRT mode the Lynch/Buffett pros/cons are empty (dirt_signals is populated
        # instead), so condition 2 never escalates a contested cheap call. This condition
        # fires when either:
        #   (a) source verification found aggregator-vs-filing discrepancies — the numbers
        #       the whole thesis rests on are in doubt, so a contested cheap call warrants
        #       the strongest model; or
        #   (b) a deep NCAV discount (≥30%) is paired with a non-net-cash balance sheet
        #       (net_cash_positive is explicitly False) — "cheap but fragile", the classic
        #       value-trap shape where the discount may be the market pricing distress.
        # dirt_signals is None for the default persona, so this is a strict no-op there and
        # default-persona routing is byte-for-byte unchanged.
        def _dirt_contested(s: DirtSignals) -> bool:
            deep_discount = s.ncav_discount_pct is not None and s.ncav_discount_pct >= 30.0
            fragile = s.net_cash_positive is False
            return s.aggregator_discrepancies_found or (deep_discount and fragile)

        cond_dirt = any(
            a.dirt_signals is not None and _dirt_contested(a.dirt_signals) for a in analyses
        )

        # All conditions are computed before the disjunction so each is evaluated
        # independently — no single condition can short-circuit the others away.
        return cond_low_conf or cond_signal_split or cond_sell or cond_dirt


def _no_analyses() -> list[_Analysis]:
    return []


class PhaseBasedRouting:
    """Default v1 routing (PRD §6.3):
    screen     → Haiku   (cheap, many tickers)
    deep       → Sonnet  (reasoning-heavy, tool use)
    synthesize → Opus    (only when the OpusTrigger criteria are met)

    Opus needs the run's accumulated analyses to evaluate its trigger. Since
    select() is called per-LLM-call and does not receive them, the orchestrator
    supplies them via `get_analyses` (default: none, so Opus never fires in the
    current single-ticker loop). Swapping in live analyses needs no loop change.
    """

    def __init__(
        self,
        opus_trigger: OpusTrigger | None = None,
        get_analyses: Callable[[], list[_Analysis]] = _no_analyses,
    ) -> None:
        self._opus_trigger = opus_trigger or DefaultOpusTrigger()
        self._get_analyses = get_analyses

    def select(
        self,
        iteration: int,
        messages: list[anthropic.types.MessageParam],
        ticker: str | None,
    ) -> ModelID:
        phase = self._detect_phase(messages, ticker)
        if phase == "screen":
            return HAIKU_4_5
        if phase == "synthesize" and self._opus_trigger.should_fire(self._get_analyses()):
            return OPUS_4_7
        return SONNET_4_6

    def _detect_phase(
        self, messages: list[anthropic.types.MessageParam], ticker: str | None
    ) -> Phase:
        # No specific ticker → a screening pass over the universe.
        if ticker is None:
            return "screen"
        # An explicit synthesis marker in the conversation → final synthesis.
        if _has_synthesis_marker(messages):
            return "synthesize"
        # A concrete ticker under active analysis.
        return "deep"


def _has_synthesis_marker(messages: list[anthropic.types.MessageParam]) -> bool:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and SYNTHESIS_PHASE_MARKER in content:
            return True
    return False


class HardcodedSonnetRouting:
    """Always routes to Sonnet 4.6 — a stable eval baseline for routing comparisons."""

    def select(
        self,
        iteration: int,
        messages: list[anthropic.types.MessageParam],
        ticker: str | None,
    ) -> ModelID:
        return SONNET_4_6
