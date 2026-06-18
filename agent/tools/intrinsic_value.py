"""estimate_intrinsic_value — owner-earnings DCF + margin of safety (W3).

A conservative two-stage discounted-cash-flow valuation built on the multi-year
``get_financials`` foundation plus the current price:

* **Owner-earnings DCF** — project the most recent free-cash-flow (Buffett's
  owner-earnings proxy, ``CFO - capex``) at a conservative growth rate for a
  projection window, discount each year, add a Gordon-growth terminal value, and
  divide the equity value by shares outstanding → intrinsic value per share.
* **Margin of safety** = ``(intrinsic - price) / intrinsic`` — positive means the
  market price sits below estimated intrinsic value.
* **Reverse DCF** — the near-term growth rate the *current* price implies, found by
  bisection, so the agent can judge whether the market's baked-in growth is plausible.

All DCF math lives in pure module-level helpers so it is deterministically testable
with fixed inputs (see ``tests/test_tools.py``). The tool itself only fetches data and
assembles the output; like ``peers.py`` it owns its output models rather than the client.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from agent.budget import RunContext
from agent.tools._clients import yfinance_client
from agent.tools.base import (
    Tool,
    ToolResult,
    ToolResultError,
    ToolResultOk,
    error_from_data_source,
)
from data_sources.errors import DataSourceError
from data_sources.yfinance_client import FinancialsHistory, PriceData, YFinanceClient


class DCFAssumptions(BaseModel):
    growth_rate: float = Field(description="Near-term annual owner-earnings growth")
    discount_rate: float = Field(description="Required return / WACC proxy")
    terminal_growth_rate: float = Field(description="Perpetuity growth after the projection window")
    projection_years: int = Field(description="Length of the explicit projection window, in years")


class IntrinsicValue(BaseModel):
    ticker: str
    as_of: date
    current_price: float | None
    owner_earnings_base: int | None
    owner_earnings_source: Literal["free_cash_flow", "cfo_minus_capex"] | None
    shares_outstanding: int | None
    intrinsic_equity_value: int | None
    intrinsic_value_per_share: float | None
    margin_of_safety: float | None
    reverse_dcf_implied_growth: float | None
    assumptions: DCFAssumptions
    data_age_hours: int
    source: Literal["yfinance"] = "yfinance"


# Conservative defaults — all overridable via the tool input. A 10% discount rate with
# 8% near-term growth tapering to a 2.5% perpetuity (roughly long-run GDP) keeps the
# estimate cautious rather than optimistic, in the spirit of a margin of safety.
_DEFAULTS = DCFAssumptions(
    growth_rate=0.08,
    discount_rate=0.10,
    terminal_growth_rate=0.025,
    projection_years=10,
)


def _intrinsic_equity_value(base: float, a: DCFAssumptions) -> float | None:
    """Two-stage DCF equity value: discounted projection window + discounted terminal value.

    Returns ``None`` when the terminal value is not finite (discount ≤ terminal growth).
    """
    if a.discount_rate <= a.terminal_growth_rate:
        return None
    pv_projection = 0.0
    for t in range(1, a.projection_years + 1):
        cash_flow = base * (1.0 + a.growth_rate) ** t
        pv_projection += cash_flow / (1.0 + a.discount_rate) ** t
    final_cash_flow = base * (1.0 + a.growth_rate) ** a.projection_years
    terminal_value = (
        final_cash_flow
        * (1.0 + a.terminal_growth_rate)
        / (a.discount_rate - a.terminal_growth_rate)
    )
    pv_terminal = terminal_value / (1.0 + a.discount_rate) ** a.projection_years
    return pv_projection + pv_terminal


def _reverse_dcf_growth(target_equity_value: float, base: float, a: DCFAssumptions) -> float | None:
    """The near-term growth rate that makes the DCF equity value equal the market value.

    Bisection over near-term growth ∈ [-50%, +100%] (the explicit-window growth may exceed
    the discount rate; only the terminal value needs discount > terminal growth). The equity
    value is monotonically increasing in growth, so the bracket holds a unique root. Returns
    ``None`` when the base is non-positive or the target lies outside the bracket.
    """
    if base <= 0:
        return None

    lo, hi = -0.50, 1.0

    def value_at(g: float) -> float | None:
        return _intrinsic_equity_value(base, a.model_copy(update={"growth_rate": g}))

    lo_val, hi_val = value_at(lo), value_at(hi)
    if lo_val is None or hi_val is None:
        return None
    if not (lo_val <= target_equity_value <= hi_val):
        return None

    for _ in range(100):
        mid = (lo + hi) / 2.0
        mid_val = value_at(mid)
        if mid_val is None:
            return None
        if mid_val < target_equity_value:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 4)


def _owner_earnings_base(
    hist: FinancialsHistory,
) -> tuple[int | None, Literal["free_cash_flow", "cfo_minus_capex"] | None]:
    """Most recent owner-earnings figure: reported free cash flow, else CFO − capex.

    yfinance reports capex as a negative number, so ``cfo + capex`` is CFO − |capex|.
    """
    fcf = YFinanceClient._series(hist.cash_flow, "free_cash_flow")
    if fcf:
        return int(fcf[0]), "free_cash_flow"
    cfo = YFinanceClient._series(hist.cash_flow, "cfo")
    capex = YFinanceClient._series(hist.cash_flow, "capex")
    if cfo and capex:
        return int(cfo[0] + capex[0]), "cfo_minus_capex"
    return None, None


def _shares_outstanding(hist: FinancialsHistory) -> int | None:
    shares = YFinanceClient._series(hist.balance_sheet, "shares_outstanding")
    if shares and shares[0] > 0:
        return int(shares[0])
    return None


class EstimateIntrinsicValueInput(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$", description="Stock ticker, e.g. AAPL")
    growth_rate: float | None = Field(
        default=None, description="Near-term annual owner-earnings growth (default 0.08)"
    )
    discount_rate: float | None = Field(
        default=None, description="Required return / discount rate (default 0.10)"
    )
    terminal_growth_rate: float | None = Field(
        default=None, description="Perpetuity growth after the projection window (default 0.025)"
    )
    projection_years: int | None = Field(
        default=None, description="Explicit projection window in years (default 10)"
    )


def _assumptions(inp: EstimateIntrinsicValueInput) -> DCFAssumptions:
    return DCFAssumptions(
        growth_rate=inp.growth_rate if inp.growth_rate is not None else _DEFAULTS.growth_rate,
        discount_rate=(
            inp.discount_rate if inp.discount_rate is not None else _DEFAULTS.discount_rate
        ),
        terminal_growth_rate=(
            inp.terminal_growth_rate
            if inp.terminal_growth_rate is not None
            else _DEFAULTS.terminal_growth_rate
        ),
        projection_years=(
            inp.projection_years if inp.projection_years is not None else _DEFAULTS.projection_years
        ),
    )


class EstimateIntrinsicValueTool(Tool):
    name = "estimate_intrinsic_value"
    description = (
        "Estimate a stock's intrinsic value with a conservative owner-earnings DCF and report "
        "the margin of safety vs the current price. Projects free cash flow (owner-earnings "
        "proxy) at a configurable growth rate, discounts it, adds a Gordon-growth terminal "
        "value, and divides by shares to get intrinsic value per share. Also returns the "
        "reverse-DCF growth rate the current price implies, so you can judge whether the "
        "market's baked-in growth is plausible. Growth, discount, terminal-growth, and "
        "projection-years are overridable; conservative defaults are echoed in the output."
    )
    input_schema = EstimateIntrinsicValueInput
    output_schema = IntrinsicValue

    def run(self, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        assert isinstance(tool_input, EstimateIntrinsicValueInput)
        ticker = tool_input.ticker
        assumptions = _assumptions(tool_input)
        try:
            yf = yfinance_client()
            hist_result = yf.get_financials(ticker)
            if isinstance(hist_result, DataSourceError):
                return error_from_data_source(hist_result)

            price_result = yf.get_price(ticker)
            current_price = (
                price_result.current_price if isinstance(price_result, PriceData) else None
            )

            base, base_source = _owner_earnings_base(hist_result)
            shares = _shares_outstanding(hist_result)

            equity_value: float | None = None
            intrinsic_per_share: float | None = None
            margin_of_safety: float | None = None
            implied_growth: float | None = None

            if base is not None and base > 0:
                equity_value = _intrinsic_equity_value(float(base), assumptions)
                if equity_value is not None and shares is not None and shares > 0:
                    intrinsic_per_share = round(equity_value / shares, 2)
                    if intrinsic_per_share != 0 and current_price is not None:
                        margin_of_safety = round(
                            (intrinsic_per_share - current_price) / intrinsic_per_share, 4
                        )
                    if current_price is not None:
                        target = current_price * shares
                        implied_growth = _reverse_dcf_growth(target, float(base), assumptions)

            return ToolResultOk(
                data=IntrinsicValue(
                    ticker=ticker.upper(),
                    as_of=date.today(),
                    current_price=current_price,
                    owner_earnings_base=base,
                    owner_earnings_source=base_source,
                    shares_outstanding=shares,
                    intrinsic_equity_value=int(equity_value) if equity_value is not None else None,
                    intrinsic_value_per_share=intrinsic_per_share,
                    margin_of_safety=margin_of_safety,
                    reverse_dcf_implied_growth=implied_growth,
                    assumptions=assumptions,
                    data_age_hours=hist_result.data_age_hours,
                )
            )
        except Exception as exc:
            return ToolResultError(
                error_code="unknown",
                message=f"estimate_intrinsic_value failed for {ticker}: {exc}",
                retryable=False,
            )
