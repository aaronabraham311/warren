"""Seed a realistic demo run into warren.db + a JSONL trace, for dashboard demos.

This is dev tooling, not part of a real run: it lets you view the Streamlit Today
page with a full spread of holdings/discovery cards, varied recommendations, a
⚠️ data-quality card, and complete reasoning traces (tool args + outputs + LLM
turns, in sequence) without waiting for an overnight agent run.

Usage::

    python -m dashboard.seed_demo                 # -> $WARREN_DB (default warren.db)
    python -m dashboard.seed_demo --db demo.db --logs-dir demo_logs

It is idempotent (merges a fixed run id), so re-running just refreshes the demo run.
Then launch the dashboard::

    streamlit run dashboard/app.py
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session

from agent.models import DirtDecisionContract, DirtDownsideFloorAssumption, DirtScenarioAssumption
from agent.tools.dirt_scenarios import ModelDirtScenariosInput, model_dirt_scenarios
from storage.models import Analysis, Base, PromptVersion, Run

RUN_ID = "demo-run"
_START = datetime(2026, 6, 29, 6, 30, 0, tzinfo=timezone.utc)

# Two prompt versions so the History page shows real version tags (and the join is
# exercised end-to-end). The latest run uses v2; older runs use v1; one history run
# below is left unlinked so the page's "unknown version" fallback is demoable too.
_V1_ID, _V2_ID = 1, 2
_PROMPT_VERSIONS = [
    PromptVersion(id=_V1_ID, version_tag="v1-baseline", notes="Initial persona-free prompt."),
    PromptVersion(
        id=_V2_ID, version_tag="v2-persona", notes="Lynch/Buffett persona system prompt."
    ),
]


@dataclass(frozen=True)
class _DemoTicker:
    """One ticker's analysis row + the numbers its demo trace will echo back."""

    ticker: str
    analysis_type: str
    recommendation: str
    confidence: float
    thesis: str
    lynch_signals: list[str]
    buffett_signals: list[str]
    key_risks: list[str]
    data_quality_notes: list[str]
    price: float
    pe: float
    peg: float | None
    dirt_signals: dict[str, object] | None = None
    dirt_decision: dict[str, object] | None = None


def _demo_decision(
    outcome: str,
    terminal_prices: tuple[float, float, float],
    *,
    outcome_reason: str,
) -> dict[str, object]:
    valuation_date = _START.date()
    terminal_date = date(2028, 6, 29)
    scenarios = []
    for case, probability, price in zip(
        ("bear", "base", "bull"), (0.25, 0.50, 0.25), terminal_prices, strict=True
    ):
        scenarios.append(
            {
                "case": case,
                "probability": probability,
                "assumption": f"{case.title()} operating and valuation case",
                "rationale": "Demo assumptions tied to the latest annual filing",
                "terminal_price": price,
                "terminal_date": terminal_date,
                "cash_flows": [
                    {
                        "date": date(2027, 6, 29),
                        "amount": 5.0,
                        "kind": "dividend",
                        "source_ref": "demo:annual-report:p42",
                    },
                    {
                        "date": terminal_date,
                        "amount": price,
                        "kind": "terminal_sale",
                        "source_ref": "demo:scenario",
                    },
                ],
            }
        )
    entry_conditions = (
        [
            {
                "description": "Enter when price reaches the computed hurdle price",
                "metric": "share_price",
                "operator": "lte",
                "threshold": 82.0,
                "currency": "USD",
                "source_ref": "demo:quote",
            }
        ]
        if outcome == "watchlist"
        else []
    )
    inp = ModelDirtScenariosInput.model_validate(
        {
            "valuation_date": valuation_date,
            "currency": "USD",
            "current_price": 100.0,
            "horizon_years": 2,
            "scenarios": scenarios,
            "downside_floor": {
                "basis": "tangible_book",
                "gross": 90.0,
                "haircut": 0.20,
                "source_ref": "demo:annual-report:p18",
                "as_of": date(2025, 12, 31),
                "adjustments": ["Exclude goodwill"],
                "confidence": "medium",
            },
            "catalysts": [
                {
                    "description": "Board-authorized capital return",
                    "category": "capital_return",
                    "evidence_strength": "contractual",
                    "expected_by": date(2027, 6, 30),
                    "source_ref": "demo:announcement:1",
                    "failure_condition": "Authorization is withdrawn",
                }
            ],
            "failure_thesis": "Asset coverage erodes or the capital return is withdrawn",
            "outcome": outcome,
            "outcome_reason": outcome_reason,
            "entry_conditions": entry_conditions,
            "blocking_unknowns": (
                ["Controller blocks minority realization"] if outcome == "pass" else []
            ),
            "monitoring_metrics": (
                [
                    {
                        "metric": "tangible_book_per_share",
                        "failure_threshold": 70.0,
                        "cadence": "quarterly",
                        "source_ref": "demo:annual-report:p18",
                        "rationale": "Protects the downside floor",
                    },
                    {
                        "metric": "capital_return_completion_pct",
                        "warning_threshold": 25.0,
                        "cadence": "event",
                        "source_ref": "demo:announcement:1",
                        "rationale": "Tracks the discount-closing mechanism",
                    },
                ]
                if outcome == "buy"
                else []
            ),
        }
    )
    return model_dirt_scenarios(inp).model_dump(mode="json")


def _decision_tool_input(decision_payload: dict[str, object]) -> dict[str, object]:
    """Recover the exact pure-tool input represented by a seeded decision contract."""
    decision = DirtDecisionContract.model_validate(decision_payload)
    return ModelDirtScenariosInput(
        valuation_date=decision.valuation_date,
        currency=decision.currency,
        current_price=decision.current_price,
        horizon_years=decision.horizon_years,
        scenarios=[
            DirtScenarioAssumption.model_validate(scenario.model_dump())
            for scenario in decision.scenarios
        ],
        downside_floor=DirtDownsideFloorAssumption.model_validate(
            decision.downside_floor.model_dump()
        ),
        catalysts=decision.catalysts,
        failure_thesis=decision.failure_thesis,
        outcome=decision.outcome,
        outcome_reason=decision.outcome_reason,
        entry_conditions=decision.entry_conditions,
        blocking_unknowns=decision.blocking_unknowns,
        monitoring_metrics=decision.monitoring_metrics,
    ).model_dump(mode="json")


_BUY_DECISION = _demo_decision(
    "buy", (85.0, 160.0, 235.0), outcome_reason="Return clears the hurdle with a cited catalyst"
)
_WATCH_DECISION = _demo_decision(
    "watchlist", (75.0, 135.0, 195.0), outcome_reason="Expected return is near 17%, below hurdle"
)
_PASS_DECISION = _demo_decision(
    "pass", (85.0, 160.0, 235.0), outcome_reason="Structural control trap blocks realization"
)


_DEMO: list[_DemoTicker] = [
    _DemoTicker(
        "AAPL",
        "holding",
        "buy",
        0.82,
        "Apple keeps compounding services revenue while buybacks shrink the share count. "
        "Free cash flow remains exceptional and the install base is a durable moat.",
        ["Dominant consumer brand", "Consistent double-digit EPS growth"],
        ["High ROIC", "Owner-friendly capital return", "Wide consumer moat"],
        ["Valuation near historical highs", "China demand softness"],
        [],
        207.42,
        29.1,
        2.3,
    ),
    _DemoTicker(
        "INTC",
        "holding",
        "sell",
        0.74,
        "Intel keeps losing share to TSMC/AMD and the foundry turnaround is burning cash "
        "with no clear inflection. Better capital homes exist.",
        ["Earnings trend deteriorating"],
        ["Eroding competitive position", "Negative free cash flow"],
        ["Foundry execution risk", "Dividend already cut"],
        ["Forward estimates revised down twice in the last quarter"],
        21.18,
        0.0,
        None,
    ),
    _DemoTicker(
        "KO",
        "holding",
        "hold",
        0.71,
        "Coca-Cola is a steady dividend grower trading at fair value. No catalyst to add today.",
        ["Stalwart, predictable earnings"],
        ["55+ years of dividend growth", "Global distribution moat"],
        ["Sugar/health headwinds", "FX drag on overseas revenue"],
        [],
        63.05,
        24.6,
        3.1,
    ),
    _DemoTicker(
        "NVDA",
        "discovery",
        "buy",
        0.91,
        "NVIDIA owns the AI accelerator stack end-to-end (hardware + CUDA). Data-center "
        "demand continues to outrun supply and gross margins are extraordinary.",
        ["Fast grower with expanding TAM"],
        ["Software-like margins", "CUDA ecosystem lock-in"],
        ["Cyclicality of hyperscaler capex", "Rich valuation"],
        [],
        131.26,
        47.8,
        1.1,
    ),
    _DemoTicker(
        "TSLA",
        "discovery",
        "sell",
        0.68,
        "Auto margins are compressing under price cuts while the robotaxi optionality is "
        "years out and richly priced in. Risk/reward skews negative here.",
        ["Margins contracting"],
        ["Valuation detached from current cash flows"],
        ["Demand elasticity", "Execution on FSD timeline"],
        ["PEG unavailable — fundamentals tool returned partial data"],
        248.50,
        78.4,
        None,
    ),
    _DemoTicker(
        "GOOG",
        "discovery",
        "hold",
        0.58,
        "Alphabet is cheap on core search FCF but AI-search disruption is a real overhang. "
        "Wait for a clearer read before initiating.",
        ["Reasonable PEG"],
        ["Search advertising moat"],
        ["AI disruption to search", "Regulatory pressure"],
        [],
        178.30,
        22.4,
        1.4,
    ),
    _DemoTicker(
        "DIR.MI",
        "discovery",
        "buy",
        0.84,
        "DIR clears the 20% probability-weighted hurdle with a cited capital return.",
        ["Asset play"],
        ["Downside floor"],
        ["Catalyst execution"],
        [],
        100.0,
        5.2,
        None,
        {"ev_ebit": 5.2, "closability_status": "supported"},
        _BUY_DECISION,
    ),
    _DemoTicker(
        "KPL.WA",
        "discovery",
        "hold",
        0.72,
        "KPL is a watchlist candidate near 17% expected IRR; wait for the hurdle entry price.",
        ["Asset play"],
        ["Net cash"],
        ["Entry-price discipline"],
        [],
        100.0,
        6.1,
        None,
        {"ev_ebit": 6.1, "closability_status": "unknown"},
        _WATCH_DECISION,
    ),
    _DemoTicker(
        "FRO.MI",
        "discovery",
        "hold",
        0.88,
        "FRO is a pass despite optical upside because a structural control trap "
        "blocks realization.",
        ["Asset play"],
        [],
        ["Structural control trap"],
        [],
        100.0,
        4.4,
        None,
        {"ev_ebit": 4.4, "closability_status": "constrained"},
        _PASS_DECISION,
    ),
]


_DEMO_BY_TICKER = {d.ticker: d for d in _DEMO}


def _analysis(d: _DemoTicker) -> Analysis:
    outcome = None if d.dirt_decision is None else d.dirt_decision.get("outcome")
    weighted_irr = (
        None if d.dirt_decision is None else d.dirt_decision.get("probability_weighted_irr")
    )
    return Analysis(
        run_id=RUN_ID,
        ticker=d.ticker,
        analysis_type=d.analysis_type,
        recommendation=d.recommendation,
        confidence=d.confidence,
        thesis=d.thesis,
        lynch_signals=d.lynch_signals,
        buffett_signals=d.buffett_signals,
        key_risks=d.key_risks,
        data_quality_notes=d.data_quality_notes,
        tool_calls_made=2 + (1 if d.dirt_decision is not None else 0),
        tokens_used=21_000,
        dirt_signals=d.dirt_signals,
        dirt_decision=d.dirt_decision,
        decision_outcome=outcome if isinstance(outcome, str) else None,
        probability_weighted_irr=(
            float(weighted_irr) if isinstance(weighted_irr, (int, float)) else None
        ),
        created_at=_START,
    )


@dataclass(frozen=True)
class _HistoryRun:
    """A past run for the History-page archive: a date, a prompt version, and its rows.

    Each row reuses a demo ticker's thesis/signals but overrides recommendation and
    confidence so the same ticker shows different calls over time. ``prompt_version_id``
    of ``None`` leaves the run unlinked, exercising the page's "unknown version" fallback.
    """

    run_id: str
    days_ago: int
    prompt_version_id: int | None
    rows: list[tuple[str, str, float]]  # (ticker, recommendation, confidence)


# A spread of past runs over the prior two weeks: repeated tickers with changing calls,
# every recommendation type, a range of confidences, and one unversioned run.
_HISTORY: list[_HistoryRun] = [
    _HistoryRun(
        "demo-hist-1",
        days_ago=2,
        prompt_version_id=_V2_ID,
        rows=[
            ("AAPL", "buy", 0.78),
            ("NVDA", "buy", 0.88),
            ("INTC", "hold", 0.55),
            ("KO", "hold", 0.69),
        ],
    ),
    _HistoryRun(
        "demo-hist-2",
        days_ago=5,
        prompt_version_id=_V1_ID,
        rows=[
            ("AAPL", "hold", 0.61),
            ("TSLA", "sell", 0.72),
            ("GOOG", "buy", 0.64),
            ("NVDA", "buy", 0.90),
        ],
    ),
    _HistoryRun(
        "demo-hist-3",
        days_ago=9,
        prompt_version_id=_V1_ID,
        rows=[("AAPL", "buy", 0.70), ("INTC", "sell", 0.80), ("KO", "hold", 0.66)],
    ),
    _HistoryRun(
        "demo-hist-4",
        days_ago=14,
        prompt_version_id=None,
        rows=[("AAPL", "sell", 0.52), ("GOOG", "hold", 0.58)],
    ),
]


def _history_analysis(run_id: str, when: datetime, ticker: str, rec: str, conf: float) -> Analysis:
    d = _DEMO_BY_TICKER[ticker]
    return Analysis(
        run_id=run_id,
        ticker=ticker,
        analysis_type=d.analysis_type,
        recommendation=rec,
        confidence=conf,
        thesis=d.thesis,
        lynch_signals=d.lynch_signals,
        buffett_signals=d.buffett_signals,
        key_risks=d.key_risks,
        data_quality_notes=[],
        tool_calls_made=2,
        tokens_used=19_500,
        created_at=when,
    )


def _trace_events() -> list[dict[str, object]]:
    """A sequential per-ticker trace: plan → fetch quote → fetch fundamentals → synthesise."""
    events: list[dict[str, object]] = [
        {"ts": _START.isoformat(), "run_id": RUN_ID, "event": "run_started"}
    ]
    for i, d in enumerate(_DEMO):
        events.append({"run_id": RUN_ID, "event": "ticker_started", "ticker": d.ticker})
        events.append(
            {
                "run_id": RUN_ID,
                "event": "llm_call",
                "ticker": d.ticker,
                "model": "claude-opus-4-8",
                "input_tokens": 9200 + i * 130,
                "output_tokens": 540 + i * 20,
                "cost_usd": 0.041 + i * 0.003,
            }
        )
        events.append(
            {
                "run_id": RUN_ID,
                "event": "tool_call",
                "ticker": d.ticker,
                "tool": "get_quote",
                "input": {"ticker": d.ticker},
                "status": "ok",
                "cached": i % 2 == 0,
                "latency_ms": 118 + i * 7,
                "output": json.dumps(
                    {
                        "ticker": d.ticker,
                        "price": d.price,
                        "previous_close": round(d.price * 0.991, 2),
                        "day_change_pct": 0.91,
                        "volume": 5_400_000 + i * 210_000,
                    }
                ),
            }
        )
        events.append(
            {
                "run_id": RUN_ID,
                "event": "tool_call",
                "ticker": d.ticker,
                "tool": "get_fundamentals",
                "input": {"ticker": d.ticker, "period": "annual"},
                "status": "ok",
                "cached": False,
                "latency_ms": 233 + i * 11,
                "output": json.dumps(
                    {
                        "ticker": d.ticker,
                        "pe": d.pe,
                        "peg": d.peg,
                        "roe": round(0.18 + i * 0.02, 3),
                        "revenue_growth": round(0.06 + i * 0.015, 3),
                    }
                ),
            }
        )
        if d.dirt_decision is not None:
            events.append(
                {
                    "run_id": RUN_ID,
                    "event": "tool_call",
                    "ticker": d.ticker,
                    "tool": "model_dirt_scenarios",
                    "input": _decision_tool_input(d.dirt_decision),
                    "status": "ok",
                    "cached": True,
                    "latency_ms": 1,
                    "output": json.dumps(d.dirt_decision),
                }
            )
        events.append(
            {
                "run_id": RUN_ID,
                "event": "llm_call",
                "ticker": d.ticker,
                "model": "claude-opus-4-8",
                "input_tokens": 11_800 + i * 140,
                "output_tokens": 820 + i * 25,
                "cost_usd": 0.058 + i * 0.004,
            }
        )
        events.append(
            {
                "run_id": RUN_ID,
                "event": "ticker_completed",
                "ticker": d.ticker,
                "decision_outcome": (
                    None if d.dirt_decision is None else d.dirt_decision.get("outcome")
                ),
                "probability_weighted_irr": (
                    None
                    if d.dirt_decision is None
                    else d.dirt_decision.get("probability_weighted_irr")
                ),
            }
        )
    return events


def seed_demo(db_path: str, logs_dir: str) -> None:
    """Write the demo run + analyses to ``db_path`` and the JSONL trace to ``logs_dir``."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    run = Run(
        id=RUN_ID,
        prompt_version_id=_V2_ID,
        started_at=_START,
        completed_at=_START + timedelta(minutes=8, seconds=42),
        status="success",
        total_input_tokens=148_200,
        total_output_tokens=12_840,
        total_cost_usd=0.6431,
        num_tool_calls=sum(2 + (1 if d.dirt_decision is not None else 0) for d in _DEMO),
    )
    all_run_ids = [RUN_ID, *(h.run_id for h in _HISTORY)]
    with Session(engine) as session:
        # Analyses have an autoincrement PK, so merge() re-inserts rather than updating;
        # clear this seed's rows first so re-running stays idempotent (no dupes / unique-
        # constraint errors on (run_id, ticker)).
        session.execute(delete(Analysis).where(Analysis.run_id.in_(all_run_ids)))
        for pv in _PROMPT_VERSIONS:
            session.merge(pv)
        session.merge(run)
        for h in _HISTORY:
            started = _START - timedelta(days=h.days_ago)
            session.merge(
                Run(
                    id=h.run_id,
                    prompt_version_id=h.prompt_version_id,
                    started_at=started,
                    completed_at=started + timedelta(minutes=7),
                    status="success",
                    total_cost_usd=0.51,
                    num_tool_calls=len(h.rows) * 2,
                )
            )
        session.flush()
        for d in _DEMO:
            session.merge(_analysis(d))
        for h in _HISTORY:
            when = _START - timedelta(days=h.days_ago)
            for ticker, rec, conf in h.rows:
                session.merge(_history_analysis(h.run_id, when, ticker, rec, conf))
        session.commit()

    log_path = Path(logs_dir) / f"{RUN_ID}.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for event in _trace_events():
            fh.write(json.dumps(event) + "\n")

    hist_analyses = sum(len(h.rows) for h in _HISTORY)
    print(
        f"Seeded run {RUN_ID!r} ({len(_DEMO)} analyses) + {len(_HISTORY)} history runs "
        f"({hist_analyses} analyses) into {db_path} + trace at {log_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a demo run for the Streamlit dashboard.")
    parser.add_argument("--db", default=os.environ.get("WARREN_DB", "warren.db"))
    parser.add_argument("--logs-dir", default=os.environ.get("WARREN_LOGS_DIR", "logs/runs"))
    args = parser.parse_args()
    seed_demo(args.db, args.logs_dir)


if __name__ == "__main__":
    main()
