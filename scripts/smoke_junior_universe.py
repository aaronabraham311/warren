"""Live Yahoo coverage and market-cap smoke check for junior-market fallbacks."""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
import statistics
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_sources.errors import DataSourceError  # noqa: E402
from data_sources.yfinance_client import YFinanceClient  # noqa: E402


def _tickers(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return [row["ticker"] for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=90)
    args = parser.parse_args()

    tickers: list[str] = []
    for venue in ("milan", "madrid", "warsaw"):
        tickers += _tickers(Path("data") / f"{venue}.csv")
    tickers = random.Random(12).sample(tickers, min(args.sample_size, len(tickers)))

    client = YFinanceClient(sqlite3.connect(":memory:"))
    resolved = 0
    market_caps: list[int] = []
    for ticker in tickers:
        result = client.get_valuation_multiples(ticker)
        if isinstance(result, DataSourceError):
            continue
        resolved += 1
        if result.market_cap_usd is not None:
            market_caps.append(result.market_cap_usd)

    coverage = resolved / len(tickers) if tickers else 0.0
    market_cap_coverage = len(market_caps) / len(tickers) if tickers else 0.0
    median_cap = int(statistics.median(market_caps)) if market_caps else 0
    print(
        f"sample={len(tickers)} resolved={resolved} coverage={coverage:.1%} "
        f"market_caps={len(market_caps)} market_cap_coverage={market_cap_coverage:.1%} "
        f"median_market_cap_usd={median_cap}"
    )
    if coverage < 0.90:
        raise SystemExit("Yahoo resolution coverage is below 90%")
    if market_cap_coverage < 0.80:
        raise SystemExit("Yahoo market-cap coverage is below 80%")
    # The ticket's €10m–€80m target maps approximately to this deliberately
    # slightly wider USD smoke band. G13 owns exact runtime FX thresholds.
    if not 10_000_000 <= median_cap <= 90_000_000:
        raise SystemExit("median market capitalization is outside the junior target band")


if __name__ == "__main__":
    main()
