"""S&P 500 constituent list — keyless Wikipedia scrape.

Scrapes the ``constituents`` table on the Wikipedia "List of S&P 500 companies"
page (free, no API key). Returns the raw ticker symbols (Wikipedia's dotted class
symbols like ``BRK.B`` are normalised to yfinance's ``BRK-B`` form).

Mirrors the ``GDELTClient`` shape: a thin ``requests.Session`` wrapper that returns
``DataSourceError`` (never raises) on transport or parse failure, with an injectable
session so tests stay offline. There is no SQLite cache here — the weekly snapshot in
``storage.models.UniverseSnapshot`` (managed by ``agent.universe``) is the cache.
"""

import requests
from bs4 import BeautifulSoup

from data_sources.errors import DataSourceError

_DEFAULT_TIMEOUT = 15
_USER_AGENT = "Warren/1.0 (research; github.com/aaronabraham311/warren)"


class SP500Client:
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def __init__(
        self,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = _USER_AGENT

    def get_sp500_constituents(self) -> list[str] | DataSourceError:
        try:
            resp = self._session.get(self.URL, timeout=self._timeout)
            resp.raise_for_status()
            html = resp.text
        except requests.RequestException as exc:
            return DataSourceError(error_code="network", message=str(exc))

        try:
            return self._parse(html)
        except (ValueError, AttributeError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))

    @staticmethod
    def _parse(html: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            raise ValueError("constituents table not found in Wikipedia response")

        tickers: list[str] = []
        for row in table.find_all("tr")[1:]:  # row 0 is the header
            cell = row.find("td")
            if cell is None:
                continue
            ticker = cell.get_text(strip=True).replace(".", "-")
            if ticker:
                tickers.append(ticker)

        if not tickers:
            raise ValueError("no tickers parsed from constituents table")
        return tickers
