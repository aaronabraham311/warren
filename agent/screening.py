"""Haiku-powered screening pass — cheap PASS/FAIL filter over the full universe.

Runs five quantitative thresholds via claude-haiku-4-5-20251001 to surface 3–5
discovery candidates per night. Two execution paths:

  use_batch_api=True  — Anthropic Batch API (50% cost discount, async, ~5 min)
  use_batch_api=False — sequential synchronous calls (immediate, for local dev)

Cooldown suppression is the caller's responsibility: filter the universe with
``agent.cooldown.filter_universe_for_cooldown`` *before* passing it here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
from anthropic.types import TextBlock

from storage.logger import RunLogger

SCREENING_MODEL = "claude-haiku-4-5-20251001"
_BATCH_POLL_INTERVAL = 30.0

DEFAULT_SCREEN_CRITERIA: dict[str, float] = {
    "pe_max": 30.0,
    "peg_max": 1.5,
    "roe_min": 0.12,
    "de_max": 1.0,
    "rev_growth_min": 0.05,
}


@dataclass
class ScreeningResult:
    candidates: list[str]
    pass_rate: float
    batch_id: str | None  # None for the sequential path


def screening_prompt(ticker: str, criteria: dict[str, float]) -> str:
    return (
        f"You are doing a quick quantitative screen on {ticker}. "
        "Based on available data, does this stock pass the following criteria?\n\n"
        "Criteria:\n"
        f"- P/E ratio ≤ {criteria.get('pe_max', 30)}\n"
        f"- PEG ratio ≤ {criteria.get('peg_max', 1.5)}\n"
        f"- ROE ≥ {criteria.get('roe_min', 0.12)} (12%)\n"
        f"- Debt/equity ≤ {criteria.get('de_max', 1.0)}\n"
        f"- Revenue growth (3Y CAGR) ≥ {criteria.get('rev_growth_min', 0.05)} (5%)\n\n"
        "Respond with exactly one word: PASS or FAIL."
    )


def run_screening_pass(
    universe: list[str],
    system_prompt: str,
    criteria: dict[str, float] | None = None,
    use_batch_api: bool = True,
    logger: RunLogger | None = None,
    _sleep: Callable[[float], None] | None = None,
) -> ScreeningResult:
    """Screen the universe and return tickers that passed.

    Args:
        universe: Tickers to screen (cooldown-filtered by the caller).
        system_prompt: The persona system prompt — passed verbatim to Haiku.
        criteria: Threshold overrides; defaults to DEFAULT_SCREEN_CRITERIA.
        use_batch_api: True → Batch API (async); False → sequential (immediate).
        logger: Optional RunLogger; emits phase_started / phase_completed events.
        _sleep: Test seam for the batch-polling sleep (defaults to time.sleep).
    """
    effective_criteria = criteria if criteria is not None else DEFAULT_SCREEN_CRITERIA
    sleep_fn = _sleep if _sleep is not None else time.sleep

    if logger is not None:
        logger.log(
            "phase_started",
            phase="screening",
            universe_size=len(universe),
            model=SCREENING_MODEL,
            use_batch_api=use_batch_api,
        )

    if use_batch_api:
        result = _run_batch_screening(universe, system_prompt, effective_criteria, sleep_fn)
    else:
        result = _run_sequential_screening(universe, system_prompt, effective_criteria)

    if logger is not None:
        logger.log(
            "phase_completed",
            phase="screening",
            candidates_surfaced=result.candidates,
            pass_rate=result.pass_rate,
            batch_id=result.batch_id,
        )

    return result


def _run_batch_screening(
    universe: list[str],
    system_prompt: str,
    criteria: dict[str, float],
    sleep_fn: Callable[[float], None],
) -> ScreeningResult:
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": f"screen-{ticker}",
                "params": {
                    "model": SCREENING_MODEL,
                    "max_tokens": 10,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": screening_prompt(ticker, criteria)}],
                },
            }
            for ticker in universe
        ]
    )

    while True:
        status = client.messages.batches.retrieve(batch.id)
        if status.processing_status == "ended":
            break
        sleep_fn(_BATCH_POLL_INTERVAL)

    candidates: list[str] = []
    for item in client.messages.batches.results(batch.id):
        if item.result.type != "succeeded":
            continue
        content = item.result.message.content
        if content and isinstance(content[0], TextBlock) and "PASS" in content[0].text.upper():
            candidates.append(item.custom_id.removeprefix("screen-"))

    pass_rate = len(candidates) / len(universe) if universe else 0.0
    return ScreeningResult(candidates=candidates, pass_rate=pass_rate, batch_id=batch.id)


def _run_sequential_screening(
    universe: list[str],
    system_prompt: str,
    criteria: dict[str, float],
) -> ScreeningResult:
    client = anthropic.Anthropic()
    candidates: list[str] = []

    for ticker in universe:
        response = client.messages.create(
            model=SCREENING_MODEL,
            max_tokens=10,
            system=system_prompt,
            messages=[{"role": "user", "content": screening_prompt(ticker, criteria)}],
        )
        if response.content and isinstance(response.content[0], TextBlock):
            if "PASS" in response.content[0].text.upper():
                candidates.append(ticker)

    pass_rate = len(candidates) / len(universe) if universe else 0.0
    return ScreeningResult(candidates=candidates, pass_rate=pass_rate, batch_id=None)
