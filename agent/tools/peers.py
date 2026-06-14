import csv
import statistics
from pathlib import Path

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import yfinance_client
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk
from data_sources.yfinance_client import FundamentalsData, ValuationData

_UNIVERSE_FILES = (Path("data/portfolio.csv"), Path("data/watchlist.csv"))

_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["MSFT", "GOOG", "META", "NVDA"],
    "Healthcare": ["JNJ", "UNH", "PFE", "MRK"],
    "Financial Services": ["JPM", "BAC", "GS", "WFC"],
    "Consumer Cyclical": ["AMZN", "HD", "MCD", "NKE"],
    "Communication Services": ["GOOG", "META", "DIS", "NFLX"],
    "Industrials": ["HON", "GE", "CAT", "MMM"],
    "Consumer Defensive": ["KO", "PG", "WMT", "COST"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Basic Materials": ["LIN", "APD", "FCX", "NEM"],
    "Real Estate": ["AMT", "PLD", "EQIX", "SPG"],
    "Utilities": ["NEE", "DUK", "SO", "AEP"],
}

# True = higher is better (percentile 100 = highest value)
# False = lower is better (percentile 100 = lowest value)
_METRIC_DIRECTION: dict[str, bool] = {
    "pe_ratio": False,
    "ev_to_ebit": False,
    "pb_ratio": False,
    "roe_pct": True,
    "gross_margin_pct": True,
    "fcf_yield": True,
}


class GetPeerComparisonInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Subject ticker, e.g. AAPL")
    peers: list[str] | None = Field(
        default=None,
        description=(
            "Explicit peer tickers (e.g. ['MSFT', 'GOOG']). "
            "When omitted the tool auto-resolves peers from the sector map "
            "or the portfolio+watchlist universe."
        ),
    )


class PeerMetrics(BaseModel):
    ticker: str
    pe_ratio: float | None
    ev_to_ebit: float | None
    pb_ratio: float | None
    roe_pct: float | None
    gross_margin_pct: float | None
    fcf_yield: float | None


class MetricSummary(BaseModel):
    peer_median: float | None
    ticker_rank: int | None
    ticker_percentile: float | None


class PeerComparison(BaseModel):
    ticker: str
    peers: list[str]
    all_metrics: list[PeerMetrics]
    summary: dict[str, MetricSummary]


def _universe() -> list[str]:
    tickers: list[str] = []
    for path in _UNIVERSE_FILES:
        if not path.exists():
            continue
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                ticker = (row.get("ticker") or "").strip().upper()
                if ticker and ticker not in tickers:
                    tickers.append(ticker)
    return tickers


def _resolve_peers(
    ticker: str,
    explicit_peers: list[str] | None,
    subject_fundamentals: FundamentalsData | None,
) -> list[str]:
    if explicit_peers is not None:
        return [p.upper() for p in explicit_peers if p.upper() != ticker]
    if subject_fundamentals is not None and subject_fundamentals.sector:
        sector_peers = [
            p for p in _SECTOR_PEERS.get(subject_fundamentals.sector, []) if p != ticker
        ]
        if len(sector_peers) >= 2:
            return sector_peers
    return [t for t in _universe() if t != ticker]


def _build_peer_metrics(
    ticker: str,
    fundamentals: FundamentalsData | None,
    valuation: ValuationData | None,
) -> PeerMetrics:
    return PeerMetrics(
        ticker=ticker,
        pe_ratio=fundamentals.pe_ratio if fundamentals else None,
        ev_to_ebit=valuation.ev_to_ebit if valuation else None,
        pb_ratio=fundamentals.pb_ratio if fundamentals else None,
        roe_pct=fundamentals.roe_pct if fundamentals else None,
        gross_margin_pct=fundamentals.gross_margin_pct if fundamentals else None,
        fcf_yield=valuation.fcf_yield if valuation else None,
    )


def _compute_summaries(subject: str, all_metrics: list[PeerMetrics]) -> dict[str, MetricSummary]:
    summaries: dict[str, MetricSummary] = {}
    for metric, higher_is_better in _METRIC_DIRECTION.items():
        values: list[tuple[float, str]] = []
        for pm in all_metrics:
            v = getattr(pm, metric)
            if isinstance(v, (int, float)):
                values.append((float(v), pm.ticker))

        if not values:
            summaries[metric] = MetricSummary(
                peer_median=None, ticker_rank=None, ticker_percentile=None
            )
            continue

        peer_values = [v for v, t in values if t != subject]
        peer_median: float | None = (
            round(statistics.median(peer_values), 4) if peer_values else None
        )

        ticker_val = next((v for v, t in values if t == subject), None)
        if ticker_val is None:
            summaries[metric] = MetricSummary(
                peer_median=peer_median, ticker_rank=None, ticker_percentile=None
            )
            continue

        sorted_vals = sorted(values, key=lambda x: x[0], reverse=higher_is_better)
        rank = next(i + 1 for i, (_, t) in enumerate(sorted_vals) if t == subject)
        n = len(sorted_vals)
        percentile = round((n - rank) / max(n - 1, 1) * 100, 1)

        summaries[metric] = MetricSummary(
            peer_median=peer_median,
            ticker_rank=rank,
            ticker_percentile=percentile,
        )
    return summaries


class GetPeerComparisonTool(Tool):
    name = "get_peer_comparison"
    description = (
        "Compare a ticker's key multiples and margins against sector peers to judge "
        "relative value and quality. Metrics: P/E, EV/EBIT, P/B, ROE, gross margin, "
        "FCF yield. Returns per-metric peer medians and the ticker's rank/percentile "
        "within the peer set. Peers auto-resolved from sector when not specified."
    )
    input_schema = GetPeerComparisonInput
    output_schema = PeerComparison

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, GetPeerComparisonInput)
        ticker = tool_input.ticker
        try:
            yf = yfinance_client()

            subject_fund_result = yf.get_fundamentals(ticker)
            subject_fund: FundamentalsData | None = (
                subject_fund_result if isinstance(subject_fund_result, FundamentalsData) else None
            )
            subject_val_result = yf.get_valuation_multiples(ticker)
            subject_val: ValuationData | None = (
                subject_val_result if isinstance(subject_val_result, ValuationData) else None
            )

            if subject_fund is None and subject_val is None:
                return ToolResultError(
                    error_code="not_found",
                    message=f"No data available for subject ticker {ticker}",
                    retryable=False,
                )

            peers = _resolve_peers(ticker, tool_input.peers, subject_fund)

            all_metrics: list[PeerMetrics] = [
                _build_peer_metrics(ticker, subject_fund, subject_val)
            ]
            valid_peers: list[str] = []
            for peer in peers:
                fund_result = yf.get_fundamentals(peer)
                val_result = yf.get_valuation_multiples(peer)
                fund: FundamentalsData | None = (
                    fund_result if isinstance(fund_result, FundamentalsData) else None
                )
                val: ValuationData | None = (
                    val_result if isinstance(val_result, ValuationData) else None
                )
                if fund is None and val is None:
                    continue
                valid_peers.append(peer)
                all_metrics.append(_build_peer_metrics(peer, fund, val))

            if len(valid_peers) < 2:
                return ToolResultError(
                    error_code="not_found",
                    message=(
                        f"Could not retrieve data for at least 2 peers of {ticker}; "
                        f"got {len(valid_peers)}. Try passing explicit peers."
                    ),
                    retryable=False,
                )

        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"get_peer_comparison failed for {ticker}: {exc}",
                retryable=False,
            )

        return ToolResultOk(
            data=PeerComparison(
                ticker=ticker,
                peers=valid_peers,
                all_metrics=all_metrics,
                summary=_compute_summaries(ticker, all_metrics),
            )
        )
