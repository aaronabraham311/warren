"""Fixture loading and recording utilities for the data-fetcher tests.

Fixture layout:
    eval/fixtures/{TICKER}/{client}/{method}/{input_hash}.json

The input_hash is the first 8 hex chars of sha256(json.dumps(input_dict, sort_keys=True)).
Every method of a given ticker shares the ticker's hash, since the only input is the
ticker. Error fixtures use descriptive names like ``error_not_found.json`` instead.

Each fixture stores the *raw upstream payload* a client consumes (not the parsed model
output), so tests mock at the network boundary and exercise the real parsing path:

    yfinance  get_price / get_fundamentals / get_growth_metrics → the ``.info`` /
              ``fast_info`` / ``financials`` fields we read
    edgar     get_filing_section → {company_tickers, submissions, filing_html}
    finnhub   get_news → {"items": [...]} ; get_basic_financials → the basics dict

This same layout powers the Week-6 eval harness, which loads from these paths.

Recording (requires live network + valid API keys):
    python -m eval.fixtures --record AAPL MSFT GOOG

yfinance and EDGAR need only network; Finnhub additionally needs ``FINNHUB_API_KEY``
and is skipped (with a warning) when it is unset, so recording never hard-fails.
"""

import json
import os
import sqlite3
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


def _write_fixture(output_dir: Path, ticker: str, client: str, method: str, data: object) -> None:
    key = _input_hash(ticker)
    path = output_dir / ticker / client / method / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"recorded {path}")


def record_fixtures(ticker: str, output_dir: Path) -> None:
    """Fetch live data from all three clients and write fixture files under *output_dir*.

    Overwrites any existing fixture files for *ticker*. Requires network access.
    yfinance and EDGAR need no key; Finnhub is skipped with a warning when
    ``FINNHUB_API_KEY`` is unset, so recording never hard-fails.
    """
    record_yfinance(ticker, output_dir)
    record_edgar(ticker, output_dir)
    record_finnhub(ticker, output_dir)


def record_yfinance(ticker: str, output_dir: Path) -> None:
    """Record the yfinance fixtures (no API key required)."""
    import yfinance as yf

    def _write(subdir: str, data: dict[str, object]) -> None:
        _write_fixture(output_dir, ticker, "yfinance", subdir, data)

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
        "trailingPE",
        "priceToBook",
        "returnOnEquity",
        "debtToEquity",
        "freeCashflow",
        "grossMargins",
        "operatingMargins",
        "profitMargins",
        "lastFiscalYearEnd",
        "regularMarketPrice",
        "currentPrice",
        "pegRatio",
        "sector",
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

    # get_quality_metrics: income statement, balance sheet, cash flow rows
    quality_data: dict[str, object] = {
        "lastFiscalYearEnd": info.get("lastFiscalYearEnd"),
        "regularMarketPrice": info.get("regularMarketPrice"),
        "income_statement": {},
        "balance_sheet": {},
        "cashflow": {},
    }
    try:
        fin = t.financials
        inc: dict[str, object] = {}
        for metric in (
            "Gross Profit",
            "Total Revenue",
            "Operating Income",
            "Net Income",
            "Pretax Income",
            "Tax Provision",
        ):
            if metric in fin.index:
                inc[metric] = [float(v) for v in fin.loc[metric].dropna().values]
        quality_data["income_statement"] = inc
    except Exception:
        pass
    try:
        bs = t.balance_sheet
        bsd: dict[str, object] = {}
        for metric in (
            "Total Assets",
            "Current Liabilities",
            "Stockholders Equity",
            "Total Debt",
            "Cash And Cash Equivalents",
        ):
            if metric in bs.index:
                bsd[metric] = [float(v) for v in bs.loc[metric].dropna().values]
        quality_data["balance_sheet"] = bsd
    except Exception:
        pass
    try:
        cf = t.cashflow
        cfd: dict[str, object] = {}
        for metric in ("Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"):
            if metric in cf.index:
                cfd[metric] = [float(v) for v in cf.loc[metric].dropna().values]
        quality_data["cashflow"] = cfd
    except Exception:
        pass
    _write("get_quality_metrics", quality_data)


def record_edgar(ticker: str, output_dir: Path) -> None:
    """Record the EDGAR get_filing_section fixture (no API key required).

    Captures the three raw HTTP bodies the client fetches — the ticker→CIK map,
    the submissions history, and the most recent 10-K's HTML — in one file.
    """
    from data_sources.edgar_client import EDGARClient

    client = EDGARClient(sqlite3_memory())
    company_tickers = client._get("https://www.sec.gov/files/company_tickers.json").text
    cik = client._resolve_cik(ticker)
    submissions = client._get(f"{client.BASE_URL}/submissions/CIK{cik}.json").text
    filing = client._select_filing(cik, "10-K", None)
    filing_html = client._get(filing.url).text
    _write_fixture(
        output_dir,
        ticker,
        "edgar",
        "get_filing_section",
        {
            "company_tickers": json.loads(company_tickers),
            "submissions": json.loads(submissions),
            "filing_html": filing_html,
        },
    )


def record_finnhub(ticker: str, output_dir: Path) -> None:
    """Record the Finnhub fixtures. Requires ``FINNHUB_API_KEY``; skipped if unset."""
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        print("skipping finnhub: FINNHUB_API_KEY not set")
        return

    from data_sources.finnhub_client import FinnhubClient

    client = FinnhubClient(sqlite3_memory(), api_key=api_key)
    news = client.client.company_news(ticker, _from="2023-01-01", to="2023-12-31")
    _write_fixture(output_dir, ticker, "finnhub", "get_news", {"items": news})
    basics = client.client.company_basic_financials(ticker, "all")
    _write_fixture(output_dir, ticker, "finnhub", "get_basic_financials", basics)


def sqlite3_memory() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")
