"""Global-exchange constituent lists — keyless Wikipedia scrapes.

Gem-hunt mode screens three non-US venues, each with its canonical Yahoo suffix:

  - Euronext Growth Milan  → ``.MI``  (scrapes the FTSE MIB index wikitable)
  - Bolsa de Madrid        → ``.MC``  (scrapes the IBEX 35 index wikitable)
  - GPW Warsaw             → ``.WA``  (scrapes the WIG20 index wikitable)

There is no free constituent API for these exchanges, so each client does a
best-effort scrape of the relevant public Wikipedia index page (no key), extracts
the ticker column, and appends the Yahoo suffix. This is a *thin* scrape: on ANY
transport or parse failure it returns ``DataSourceError`` (never raises), and the
caller (``agent.universe``) then falls back to the committed ``data/{exchange}.csv``
— the CSV is the reliable, authoritative slice for the universe; the live fetch only
opportunistically widens it.

Mirrors ``SP500Client``: a thin ``requests.Session`` wrapper with an injectable
session so tests stay offline, and no SQLite cache here — the weekly
``storage.models.UniverseSnapshot`` (managed by ``agent.universe``) is the cache.
"""

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from data_sources.errors import DataSourceError

_DEFAULT_TIMEOUT = 15
_USER_AGENT = "Warren/1.0 (research; github.com/aaronabraham311/warren)"

# Wikipedia header cells that name the ticker column, lowercased.
_TICKER_HEADERS = ("ticker", "symbol")


@dataclass(frozen=True)
class ExchangeSpec:
    """Static config for one exchange's keyless constituent scrape."""

    key: str  # "milan" | "madrid" | "warsaw"
    suffix: str  # Yahoo suffix, e.g. ".MI"
    url: str  # public Wikipedia index page
    display_name: str


MILAN = ExchangeSpec(
    key="milan",
    suffix=".MI",
    url="https://en.wikipedia.org/wiki/FTSE_MIB",
    display_name="Euronext Growth Milan",
)
MADRID = ExchangeSpec(
    key="madrid",
    suffix=".MC",
    url="https://en.wikipedia.org/wiki/IBEX_35",
    display_name="Bolsa de Madrid",
)
WARSAW = ExchangeSpec(
    key="warsaw",
    suffix=".WA",
    url="https://en.wikipedia.org/wiki/WIG20",
    display_name="GPW Warsaw",
)

EXCHANGE_SPECS: dict[str, ExchangeSpec] = {s.key: s for s in (MILAN, MADRID, WARSAW)}


class ExchangeClient:
    """Best-effort keyless scraper for one exchange's constituent tickers."""

    def __init__(
        self,
        spec: ExchangeSpec,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self._spec = spec
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = _USER_AGENT

    def get_constituents(self) -> list[str] | DataSourceError:
        try:
            resp = self._session.get(self._spec.url, timeout=self._timeout)
            resp.raise_for_status()
            html = resp.text
        except requests.RequestException as exc:
            return DataSourceError(error_code="network", message=str(exc))

        try:
            return self._parse(html, self._spec.suffix)
        except (ValueError, AttributeError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))

    @staticmethod
    def _parse(html: str, suffix: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"class": "wikitable"})
        if not isinstance(table, Tag):
            raise ValueError("no wikitable found in Wikipedia response")

        rows = table.find_all("tr")
        if not rows:
            raise ValueError("wikitable has no rows")

        header_cells = [c.get_text(strip=True).lower() for c in rows[0].find_all(["th", "td"])]
        col = next(
            (i for i, h in enumerate(header_cells) if any(k in h for k in _TICKER_HEADERS)),
            0,
        )

        tickers: list[str] = []
        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= col:
                continue
            raw = cells[col].get_text(strip=True).upper()
            if not raw:
                continue
            symbol = raw if raw.endswith(suffix.upper()) else f"{raw}{suffix}"
            tickers.append(symbol)

        if not tickers:
            raise ValueError("no tickers parsed from wikitable")
        return tickers
