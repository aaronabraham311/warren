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
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from storage.models import Analysis, Base, Run

RUN_ID = "demo-run"
_START = datetime(2026, 6, 29, 6, 30, 0, tzinfo=timezone.utc)


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
]


def _analysis(d: _DemoTicker) -> Analysis:
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
        tool_calls_made=2,
        tokens_used=21_000,
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
        events.append({"run_id": RUN_ID, "event": "ticker_completed", "ticker": d.ticker})
    return events


def seed_demo(db_path: str, logs_dir: str) -> None:
    """Write the demo run + analyses to ``db_path`` and the JSONL trace to ``logs_dir``."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    run = Run(
        id=RUN_ID,
        started_at=_START,
        completed_at=_START + timedelta(minutes=8, seconds=42),
        status="success",
        total_input_tokens=148_200,
        total_output_tokens=12_840,
        total_cost_usd=0.6431,
        num_tool_calls=len(_DEMO) * 2,
    )
    with Session(engine) as session:
        session.merge(run)
        session.flush()
        for d in _DEMO:
            session.merge(_analysis(d))
        session.commit()

    log_path = Path(logs_dir) / f"{RUN_ID}.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for event in _trace_events():
            fh.write(json.dumps(event) + "\n")

    print(f"Seeded run {RUN_ID!r} ({len(_DEMO)} analyses) into {db_path} + trace at {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a demo run for the Streamlit dashboard.")
    parser.add_argument("--db", default=os.environ.get("WARREN_DB", "warren.db"))
    parser.add_argument("--logs-dir", default=os.environ.get("WARREN_LOGS_DIR", "logs/runs"))
    args = parser.parse_args()
    seed_demo(args.db, args.logs_dir)


if __name__ == "__main__":
    main()
