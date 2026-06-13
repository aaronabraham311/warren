"""Fixture loading and recording utilities for yfinance integration tests.

Fixture layout:
    eval/fixtures/{TICKER}/{client}/{method}/{input_hash}.json

The input_hash is the first 8 hex chars of sha256(json.dumps(input_dict, sort_keys=True)).
Error fixtures use descriptive names like ``error_not_found.json`` instead.

Recording (requires live network + valid API keys):
    python -m eval.fixtures --record AAPL MSFT GOOG
"""

import json
from hashlib import sha256
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def _input_hash(ticker: str) -> str:
    return sha256(json.dumps({"ticker": ticker}, sort_keys=True).encode()).hexdigest()[:8]


def load_fixture(
    ticker: str, client: str, method: str, name: str | None = None
) -> dict[str, object]:
    """Return the raw fixture dict for the given ticker/client/method.

    If *name* is omitted the hash of ``{"ticker": ticker}`` is used as the filename stem.
    """
    stem = name if name is not None else _input_hash(ticker)
    path = FIXTURES_DIR / ticker / client / method / f"{stem}.json"
    with open(path) as fh:
        data: dict[str, object] = json.load(fh)
    return data


def record_fixtures(ticker: str, output_dir: Path) -> None:
    """Fetch live data from yfinance and write fixture files under *output_dir*.

    Overwrites any existing fixture files for *ticker*.
    Requires network access; no API key needed for yfinance.
    """
    import sqlite3

    import yfinance as yf

    key = _input_hash(ticker)

    def _write(subdir: str, data: dict[str, object]) -> None:
        path = output_dir / ticker / "yfinance" / subdir / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
        print(f"recorded {path}")

    t = yf.Ticker(ticker)

    # get_price: fast_info attributes
    fi = t.fast_info
    _write(
        "get_price",
        {
            "last_price": getattr(fi, "last_price", None),
            "previous_close": getattr(fi, "previous_close", None),
            "three_month_average_volume": getattr(fi, "three_month_average_volume", None),
        },
    )

    # get_fundamentals: .info fields we consume
    info: dict[str, object] = t.info
    relevant = [
        "trailingPE", "priceToBook", "returnOnEquity", "debtToEquity",
        "freeCashflow", "operatingMargins", "profitMargins",
        "lastFiscalYearEnd", "regularMarketPrice", "currentPrice", "pegRatio",
    ]
    _write("get_fundamentals", {k: info.get(k) for k in relevant})

    # get_growth_metrics: peg from .info, financials rows
    fin_data: dict[str, object] = {
        "pegRatio": info.get("pegRatio"),
        "lastFiscalYearEnd": info.get("lastFiscalYearEnd"),
        "regularMarketPrice": info.get("regularMarketPrice"),
    }
    try:
        fin = t.financials
        for metric in ("Total Revenue", "Net Income"):
            if metric in fin.index:
                series = fin.loc[metric].dropna()
                fin_data[metric] = list(series.values)
    except Exception:
        pass
    _write("get_growth_metrics", fin_data)
    del sqlite3  # imported for type-check clarity only; suppress F401
