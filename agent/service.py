"""Shared Warren run orchestration for batch and interactive clients."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

import anthropic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.budget import Budget, RunContext
from agent.cancellation import CancellationToken, NeverCancelToken, RunCancelledError
from agent.cooldown import filter_universe_for_cooldown
from agent.events import EventSink, NullEventSink
from agent.locking import RunLock
from agent.loop import CostAbortedError, analyze_ticker
from agent.models import DEFAULT_MODEL_ID, AnalysisOutput
from agent.persona import DefaultPersona, DirtPersona
from agent.portfolio import (
    Holding,
    WatchlistEntry,
    load_portfolio,
    load_watchlist,
    sync_holdings_to_db,
    sync_watchlist_to_db,
)
from agent.routing import PhaseBasedRouting
from agent.screening import GEM_HUNT_SCREEN_CRITERIA, run_screening_pass, screen_ticker_value
from agent.tools._clients import yfinance_client
from agent.universe import get_current_universe, get_gem_hunt_universe
from data_sources.symbols import TICKER_PATTERN, canonical_symbol
from data_sources.yfinance_client import PriceData
from storage.engine import (
    ensure_prompt_version,
    get_session,
    migrate,
    upsert_analysis,
    write_run_start,
)
from storage.logger import RunLogger
from storage.models import AnalysisData, RunStatus
from storage.recovery import reconcile_orphans

_PORTFOLIO_FILE = Path("data/portfolio.csv")
_WATCHLIST_FILE = Path("data/watchlist.csv")
MAX_SCREEN_CANDIDATES = 3
_GEM_HUNT_SCREEN_WORKERS = 8

AnalysisType = Literal["holding", "discovery"]
PersonaName = Literal["default", "dirt"]


class Analyzer(Protocol):
    def __call__(
        self,
        *,
        ticker: str,
        persona: DefaultPersona | DirtPersona,
        routing_policy: PhaseBasedRouting,
        run_context: RunContext,
        client: anthropic.Anthropic,
        portfolio_context: str,
    ) -> AnalysisOutput: ...


class RunMode(StrEnum):
    """The deterministic workflow selected before any model call."""

    TICKERS = "tickers"
    PORTFOLIO = "portfolio"
    DISCOVERY = "discovery"
    GEM_HUNT = "gem_hunt"


class RunRequest(BaseModel):
    """Validated input to the shared run service."""

    model_config = ConfigDict(frozen=True)

    mode: RunMode
    tickers: list[str] = Field(default_factory=list, max_length=4)
    persona: PersonaName = "default"
    max_cost_usd: float = Field(default=1.25, gt=0.0, le=10.0)
    max_candidates: int = Field(default=MAX_SCREEN_CANDIDATES, ge=1, le=10)
    skip_ticker_validation: bool = False

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, tickers: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw in tickers:
            ticker = canonical_symbol(raw)
            if re.fullmatch(TICKER_PATTERN, ticker) is None:
                raise ValueError(f"invalid ticker: {raw!r}")
            if ticker in normalized:
                raise ValueError(f"duplicate ticker: {ticker}")
            normalized.append(ticker)
        return normalized

    @model_validator(mode="after")
    def validate_mode_tickers(self) -> RunRequest:
        if self.mode is RunMode.TICKERS and not self.tickers:
            raise ValueError("tickers mode requires at least one ticker")
        if self.mode is not RunMode.TICKERS and self.tickers:
            raise ValueError(f"{self.mode.value} mode does not accept explicit tickers")
        return self


@dataclass(frozen=True)
class TickerRunResult:
    ticker: str
    analysis_type: AnalysisType
    analysis: AnalysisOutput | None = None
    error: str | None = None


@dataclass(frozen=True)
class ScreeningSummary:
    confirmed_count: int
    needs_deeper_fetch_count: int
    source_error_count: int
    selected_candidates: tuple[str, ...]
    gem_hunt: bool


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    ticker_results: tuple[TickerRunResult, ...]
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_tool_calls: int
    duration_seconds: float
    error_msg: str | None
    screening: ScreeningSummary | None = None

    @property
    def analyses(self) -> tuple[AnalysisOutput, ...]:
        return tuple(item.analysis for item in self.ticker_results if item.analysis is not None)


def logs_dir() -> Path:
    """Configured JSONL/sidecar root shared by batch, terminal, and dashboard clients."""

    return Path(os.environ.get("WARREN_LOGS_DIR", "logs/runs"))


def state_dir() -> Path:
    """Repository-local terminal/runtime state root."""

    return Path(os.environ.get("WARREN_STATE_DIR", ".warren"))


def build_portfolio_context(portfolio_file: Path = _PORTFOLIO_FILE) -> str:
    if not portfolio_file.exists():
        return ""
    holdings = load_portfolio(portfolio_file, validate_tickers=False)
    if not holdings:
        return ""
    lines = ["User's current portfolio holdings:"]
    for holding in holdings:
        lines.append(
            f"  {holding.ticker}: {holding.shares:.2f} shares, "
            f"cost basis ${holding.cost_basis:.2f} (purchased {holding.purchase_date})"
        )
    return "\n".join(lines)


def sync_input_data(
    skip_ticker_validation: bool,
    *,
    portfolio_file: Path = _PORTFOLIO_FILE,
    watchlist_file: Path = _WATCHLIST_FILE,
) -> None:
    """Validate and snapshot portfolio/watchlist data before a run begins."""

    holdings = load_portfolio(portfolio_file, validate_tickers=not skip_ticker_validation)
    entries = load_watchlist(watchlist_file) if watchlist_file.exists() else []
    current_prices: dict[str, float] = {}
    for holding in holdings:
        price = yfinance_client().get_price(holding.ticker)
        if isinstance(price, PriceData) and price.current_price is not None:
            current_prices[holding.ticker] = price.current_price
    with get_session() as session:
        sync_holdings_to_db(holdings, session, current_prices)
        sync_watchlist_to_db(entries, session)


def resolve_persona(persona_name: str, gem_hunt: bool) -> DefaultPersona | DirtPersona:
    """Resolve the requested persona; gem hunt always forces DIRT."""

    if gem_hunt or persona_name == "dirt":
        return DirtPersona()
    return DefaultPersona()


def _analysis_data(result: AnalysisOutput) -> AnalysisData:
    dirt_signals = (
        result.dirt_signals.model_dump(mode="json") if result.dirt_signals is not None else None
    )
    dirt_decision = (
        result.dirt_decision.model_dump(mode="json") if result.dirt_decision is not None else None
    )
    return AnalysisData(
        analysis_type=result.analysis_type,
        recommendation=result.recommendation,
        confidence=result.confidence,
        thesis=result.thesis,
        lynch_signals=result.lynch_signals.model_dump(),
        buffett_signals=result.buffett_signals.model_dump(),
        key_risks=result.key_risks,
        data_quality_notes=result.data_quality_notes,
        tool_calls_made=result.tool_calls_made,
        tokens_used=result.tokens_used,
        termination_reason=result.termination_reason,
        dirt_signals=dirt_signals,
        dirt_decision=dirt_decision,
        decision_outcome=None if result.dirt_decision is None else result.dirt_decision.outcome,
        probability_weighted_irr=(
            None if result.dirt_decision is None else result.dirt_decision.probability_weighted_irr
        ),
    )


def _analyze_and_persist(
    ticker: str,
    analysis_type: AnalysisType,
    run_id: str,
    budget: Budget,
    logger: RunLogger,
    persona: DefaultPersona | DirtPersona,
    routing_policy: PhaseBasedRouting,
    client: anthropic.Anthropic,
    portfolio_context: str,
    *,
    cancellation: CancellationToken | None = None,
    event_sink: EventSink | None = None,
    analyzer: Analyzer | None = None,
) -> AnalysisOutput:
    """Analyze one ticker, append lifecycle events, and persist its structured result."""

    token_count_before = budget.total_input_tokens + budget.total_output_tokens
    tool_count_before = budget.total_tool_calls
    token = cancellation if cancellation is not None else NeverCancelToken()
    token.raise_if_cancelled()
    run_context = RunContext(
        run_id=run_id,
        budget=budget,
        logger=logger,
        cancellation=token,
        event_sink=event_sink if event_sink is not None else NullEventSink(),
    )
    logger.log("ticker_started", ticker=ticker, phase="deep", model=DEFAULT_MODEL_ID)
    run_analyzer = analyzer if analyzer is not None else analyze_ticker
    result = run_analyzer(
        ticker=ticker,
        persona=persona,
        routing_policy=routing_policy,
        run_context=run_context,
        client=client,
        portfolio_context=portfolio_context,
    )
    result.analysis_type = analysis_type
    result.tool_calls_made = budget.total_tool_calls - tool_count_before
    result.tokens_used = budget.total_input_tokens + budget.total_output_tokens - token_count_before
    decision = result.dirt_decision
    logger.log(
        "ticker_completed",
        ticker=ticker,
        recommendation=result.recommendation,
        confidence=result.confidence,
        iterations=run_context.iterations,
        tokens=result.tokens_used,
        cost_usd=budget.total_cost_usd,
        termination=result.termination_reason,
        decision_outcome=None if decision is None else decision.outcome,
        probability_weighted_irr=None if decision is None else decision.probability_weighted_irr,
        hurdle_irr=None if decision is None else decision.hurdle_irr,
        required_entry_price=None if decision is None else decision.required_entry_price,
    )
    upsert_analysis(run_id, ticker, _analysis_data(result))
    return result


def _select_targets(
    request: RunRequest,
    *,
    logger: RunLogger,
    cancellation: CancellationToken,
    holdings: list[Holding],
    watchlist: list[WatchlistEntry],
) -> tuple[list[tuple[str, AnalysisType]], ScreeningSummary | None]:
    if request.mode is RunMode.TICKERS:
        return [(ticker, "holding") for ticker in request.tickers], None
    if request.mode is RunMode.PORTFOLIO:
        return [(holding.ticker, "holding") for holding in holdings], None

    cancellation.raise_if_cancelled()
    watchlist_tickers = [entry.ticker for entry in watchlist]
    with get_session() as session:
        if request.mode is RunMode.GEM_HUNT:
            universe = get_gem_hunt_universe(session, watchlist_tickers)
        else:
            universe = get_current_universe(session, watchlist_tickers)
        cooldown = filter_universe_for_cooldown(universe, session, recent_news={})
    logger.log(
        "discovery_cooldown_applied",
        suppressed_count=len(cooldown.suppressed),
        suppressed_tickers=cooldown.suppressed,
    )
    cancellation.raise_if_cancelled()
    if request.mode is RunMode.GEM_HUNT:
        screening = run_screening_pass(
            cooldown.active,
            criteria=GEM_HUNT_SCREEN_CRITERIA,
            logger=logger,
            max_workers=_GEM_HUNT_SCREEN_WORKERS,
            screen_fn=screen_ticker_value,
            rank=True,
            cancellation=cancellation,
        )
        candidates = screening.surfaced[: request.max_candidates]
    else:
        screening = run_screening_pass(
            cooldown.active,
            logger=logger,
            cancellation=cancellation,
        )
        candidates = screening.candidates[: request.max_candidates]
    for rank, ticker in enumerate(candidates, start=1):
        logger.log("candidate_selected", ticker=ticker, rank=rank)
    targets: list[tuple[str, AnalysisType]] = [
        (holding.ticker, "holding") for holding in holdings
    ] + [(candidate, "discovery") for candidate in candidates]
    return targets, ScreeningSummary(
        confirmed_count=len(screening.candidates),
        needs_deeper_fetch_count=len(screening.needs_deeper_fetch),
        source_error_count=len(screening.source_errors),
        selected_candidates=tuple(candidates),
        gem_hunt=request.mode is RunMode.GEM_HUNT,
    )


def _run_targets(
    targets: list[tuple[str, AnalysisType]],
    *,
    run_id: str,
    budget: Budget,
    logger: RunLogger,
    persona: DefaultPersona | DirtPersona,
    routing_policy: PhaseBasedRouting,
    client: anthropic.Anthropic,
    portfolio_context: str,
    cancellation: CancellationToken,
    event_sink: EventSink,
) -> tuple[RunStatus, str | None, list[TickerRunResult]]:
    results: list[TickerRunResult] = []
    status: RunStatus = "success"
    error_msg: str | None = None
    for ticker, analysis_type in targets:
        cancellation.raise_if_cancelled()
        try:
            analysis = _analyze_and_persist(
                ticker,
                analysis_type,
                run_id,
                budget,
                logger,
                persona,
                routing_policy,
                client,
                portfolio_context,
                cancellation=cancellation,
                event_sink=event_sink,
            )
            results.append(TickerRunResult(ticker, analysis_type, analysis=analysis))
        except RunCancelledError:
            raise
        except CostAbortedError as exc:
            status = "cost_aborted"
            error_msg = str(exc)
            results.append(TickerRunResult(ticker, analysis_type, error=error_msg))
            break
        except Exception as exc:
            ticker_error = str(exc)
            logger.log("ticker_failed", ticker=ticker, error=ticker_error)
            results.append(TickerRunResult(ticker, analysis_type, error=ticker_error))
    return status, error_msg, results


def execute_run(
    request: RunRequest,
    *,
    event_sink: EventSink | None = None,
    cancellation: CancellationToken | None = None,
    log_dir: Path | None = None,
    runtime_state_dir: Path | None = None,
    client: anthropic.Anthropic | None = None,
) -> RunResult:
    """Execute one Warren workflow and always finalize its durable trace safely."""

    sink = event_sink if event_sink is not None else NullEventSink()
    token = cancellation if cancellation is not None else NeverCancelToken()
    selected_log_dir = log_dir if log_dir is not None else logs_dir()
    selected_state_dir = runtime_state_dir if runtime_state_dir is not None else state_dir()
    run_id = str(uuid4())
    started_at = datetime.now(timezone.utc)

    with RunLock(selected_state_dir).acquire(run_id=run_id, mode=request.mode.value):
        migrate()
        reconcile_orphans(selected_log_dir)
        sync_input_data(request.skip_ticker_validation)
        persona = resolve_persona(request.persona, request.mode is RunMode.GEM_HUNT)
        routing_policy = PhaseBasedRouting()
        prompt_version_id = ensure_prompt_version(
            version_tag="v1",
            persona_system_prompt=persona.system_prompt,
            routing_policy_name=type(routing_policy).__name__,
        )
        portfolio_context = build_portfolio_context()
        holdings = load_portfolio(_PORTFOLIO_FILE, validate_tickers=False)
        watchlist = load_watchlist(_WATCHLIST_FILE) if _WATCHLIST_FILE.exists() else []
        logger = RunLogger(run_id, selected_log_dir, event_sink=sink)
        budget = Budget(max_cost_usd=request.max_cost_usd)
        run_status: RunStatus = "success"
        error_msg: str | None = None
        ticker_results: list[TickerRunResult] = []
        screening_summary: ScreeningSummary | None = None
        try:
            write_run_start(run_id, started_at, prompt_version_id=prompt_version_id)
            logger.log(
                "run_started",
                mode=request.mode.value,
                tickers=request.tickers,
            )
            targets, screening_summary = _select_targets(
                request,
                logger=logger,
                cancellation=token,
                holdings=holdings,
                watchlist=watchlist,
            )
            run_status, error_msg, ticker_results = _run_targets(
                targets,
                run_id=run_id,
                budget=budget,
                logger=logger,
                persona=persona,
                routing_policy=routing_policy,
                client=client if client is not None else anthropic.Anthropic(),
                portfolio_context=portfolio_context,
                cancellation=token,
                event_sink=sink,
            )
        except RunCancelledError as exc:
            run_status = "cancelled"
            error_msg = str(exc)
            logger.log("run_cancelled", error_msg=error_msg)
        except KeyboardInterrupt:
            run_status = "cancelled"
            error_msg = "Run interrupted by user"
            logger.log("run_cancelled", error_msg=error_msg)
        except Exception as exc:
            run_status = "failed"
            error_msg = str(exc)
            logger.log("run_failed", error_msg=error_msg)
        finally:
            duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
            try:
                logger.log(
                    "run_completed",
                    status=run_status,
                    total_cost_usd=budget.total_cost_usd,
                    duration_seconds=duration_seconds,
                    error_msg=error_msg,
                )
                with get_session() as session:
                    logger.flush_to_db(session)
            finally:
                logger.close()

    return RunResult(
        run_id=run_id,
        status=run_status,
        ticker_results=tuple(ticker_results),
        total_cost_usd=budget.total_cost_usd,
        total_input_tokens=budget.total_input_tokens,
        total_output_tokens=budget.total_output_tokens,
        total_tool_calls=budget.total_tool_calls,
        duration_seconds=duration_seconds,
        error_msg=error_msg,
        screening=screening_summary,
    )
