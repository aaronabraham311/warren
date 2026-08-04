from typing import get_args, get_protocol_members

import anthropic

from agent.models import (
    HAIKU_4_5,
    OPUS_4_7,
    SONNET_4_6,
    AnalysisOutput,
    DirtSignals,
    LynchBuffettSignals,
)
from agent.routing import (
    SYNTHESIS_PHASE_MARKER,
    DefaultOpusTrigger,
    HardcodedSonnetRouting,
    ModelID,
    PhaseBasedRouting,
    RoutingPolicy,
)


def _analysis(
    *,
    recommendation: str = "hold",
    confidence: float = 0.8,
    lynch_signals: list[str] | None = None,
    buffett_signals: list[str] | None = None,
    dirt_signals: DirtSignals | None = None,
) -> AnalysisOutput:
    return AnalysisOutput(
        ticker="AAPL",
        analysis_type="holding",
        recommendation=recommendation,
        confidence=confidence,
        thesis="t" * 10,
        lynch_signals=LynchBuffettSignals(pros=lynch_signals or [], cons=[]),
        buffett_signals=LynchBuffettSignals(pros=buffett_signals or [], cons=[]),
        key_risks=["risk"],
        dirt_signals=dirt_signals,
    )


def _synthesis_messages() -> list[anthropic.types.MessageParam]:
    return [{"role": "user", "content": f"{SYNTHESIS_PHASE_MARKER} produce final synthesis"}]


# ── Acceptance criterion 1: screen phase → Haiku ────────────────────────────────
def test_select_returns_haiku_for_screen_phase() -> None:
    # ticker=None is the screening pass over the universe.
    assert PhaseBasedRouting().select(1, [], ticker=None) == HAIKU_4_5


# ── Acceptance criterion 2: a single sell fires the Opus trigger ─────────────────
def test_opus_trigger_fires_on_single_sell() -> None:
    assert DefaultOpusTrigger().should_fire([_analysis(recommendation="sell")]) is True


# ── Acceptance criterion 3: confident hold/buy with balanced signals → no fire ──
def test_opus_trigger_does_not_fire_for_confident_balanced_run() -> None:
    analyses = [
        _analysis(
            recommendation="hold",
            confidence=0.7,
            lynch_signals=["a"],
            buffett_signals=["b"],
        ),
        _analysis(
            recommendation="buy",
            confidence=0.9,
            lynch_signals=["a", "b"],
            buffett_signals=["c", "d"],
        ),
    ]
    assert DefaultOpusTrigger().should_fire(analyses) is False


# ── Acceptance criterion 4: HardcodedSonnetRouting drops in with no loop changes ─
def test_both_policies_satisfy_routing_policy_protocol() -> None:
    # analyze_ticker() only depends on the RoutingPolicy structural interface, so
    # any policy satisfying it swaps in with a single constructor-argument change.
    def takes_policy(policy: RoutingPolicy) -> str:
        return policy.select(1, [], "AAPL")

    assert takes_policy(PhaseBasedRouting()) == SONNET_4_6
    assert takes_policy(HardcodedSonnetRouting()) == SONNET_4_6
    assert isinstance(PhaseBasedRouting(), RoutingPolicy)
    assert isinstance(HardcodedSonnetRouting(), RoutingPolicy)


# ── Acceptance criterion 5: each Opus condition fires independently ──────────────
def test_opus_condition_low_confidence_action_alone() -> None:
    # Two low-confidence, non-hold holdings; signals balanced, no sell.
    analyses = [
        _analysis(recommendation="buy", confidence=0.3),
        _analysis(recommendation="buy", confidence=0.4),
    ]
    assert DefaultOpusTrigger().should_fire(analyses) is True


def test_opus_condition_signal_split_alone() -> None:
    # Confident hold (so conditions 1 and 3 are false) but Lynch/Buffett disagree by ≥3.
    analyses = [
        _analysis(
            recommendation="hold",
            confidence=0.9,
            lynch_signals=["a", "b", "c"],
            buffett_signals=[],
        )
    ]
    assert DefaultOpusTrigger().should_fire(analyses) is True


def test_opus_condition_sell_alone() -> None:
    # A single confident sell; conditions 1 and 2 are false.
    analyses = [_analysis(recommendation="sell", confidence=0.9)]
    assert DefaultOpusTrigger().should_fire(analyses) is True


# ── DIRT / deep-value condition 4 ───────────────────────────────────────────────
def test_opus_condition_dirt_aggregator_discrepancy_alone() -> None:
    # Benign on conditions 1–3 (confident hold, balanced/empty signals, no sell), but
    # source verification flagged aggregator-vs-filing discrepancies → contested cheap call.
    analyses = [
        _analysis(
            recommendation="hold",
            confidence=0.9,
            dirt_signals=DirtSignals(aggregator_discrepancies_found=True),
        )
    ]
    assert DefaultOpusTrigger().should_fire(analyses) is True


def test_opus_condition_dirt_cheap_but_fragile_alone() -> None:
    # Deep NCAV discount paired with a non-net-cash balance sheet = value-trap shape.
    analyses = [
        _analysis(
            recommendation="hold",
            confidence=0.9,
            dirt_signals=DirtSignals(ncav_discount_pct=45.0, net_cash_positive=False),
        )
    ]
    assert DefaultOpusTrigger().should_fire(analyses) is True


def test_opus_condition_dirt_benign_does_not_fire() -> None:
    # A clean deep-value setup: cheap and net-cash-positive, no discrepancies → no fire.
    analyses = [
        _analysis(
            recommendation="hold",
            confidence=0.9,
            dirt_signals=DirtSignals(
                ncav_discount_pct=45.0,
                net_cash_positive=True,
                aggregator_discrepancies_found=False,
            ),
        )
    ]
    assert DefaultOpusTrigger().should_fire(analyses) is False


def test_opus_condition_dirt_none_is_noop() -> None:
    # Default persona: dirt_signals is None. A confident, balanced hold must not fire —
    # proving condition 4 leaves default-persona routing unchanged.
    analyses = [_analysis(recommendation="hold", confidence=0.9)]
    assert DefaultOpusTrigger().should_fire(analyses) is False


# ── Acceptance criterion 6: RoutingPolicy is a Protocol, not an ABC ─────────────
def test_routing_policy_is_a_protocol() -> None:
    # get_protocol_members raises TypeError if RoutingPolicy is not a Protocol.
    assert "select" in get_protocol_members(RoutingPolicy)


# ── Synthesize phase wiring ─────────────────────────────────────────────────────
def test_synthesize_phase_routes_to_opus_when_trigger_fires() -> None:
    routing = PhaseBasedRouting(get_analyses=lambda: [_analysis(recommendation="sell")])
    assert routing.select(1, _synthesis_messages(), "AAPL") == OPUS_4_7


def test_synthesize_phase_falls_back_to_sonnet_without_trigger() -> None:
    # Default get_analyses returns [], so the trigger never fires.
    assert PhaseBasedRouting().select(1, _synthesis_messages(), "AAPL") == SONNET_4_6


def test_deep_phase_routes_to_sonnet() -> None:
    # A concrete ticker with no synthesis marker is the deep-analysis phase.
    assert PhaseBasedRouting().select(1, [], "AAPL") == SONNET_4_6


# ── Drift guard: ModelID stays in sync with agent.models ────────────────────────
def test_model_id_matches_models_constants() -> None:
    assert set(get_args(ModelID)) == {HAIKU_4_5, SONNET_4_6, OPUS_4_7}
