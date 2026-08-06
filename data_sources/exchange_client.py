"""Junior-market constituent orchestration over heterogeneous identity sources."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from data_sources.bme_client import BMEGrowthSource
from data_sources.errors import DataSourceError
from data_sources.euronext_client import EuronextProductDirectorySource
from data_sources.security_identity import ConstituentSource, SecurityIdentity
from data_sources.tradingview_client import TradingViewScannerSource

_USER_AGENT = "Warren/1.0 (research; github.com/aaronabraham311/warren)"


@dataclass(frozen=True)
class ExchangeSpec:
    """Static configuration for a junior venue and its identity source."""

    key: str
    suffix: str
    display_name: str
    currency: str
    source: ConstituentSource


MILAN = ExchangeSpec(
    key="milan",
    suffix=".MI",
    display_name="Euronext Growth Milan",
    currency="EUR",
    source=EuronextProductDirectorySource(mics="EXGM", suffix=".MI", venue="euronext_growth_milan"),
)
MADRID = ExchangeSpec(
    key="madrid",
    suffix=".MC",
    display_name="BME Growth",
    currency="EUR",
    source=BMEGrowthSource(mtf_segment="BMEGrowth", suffix=".MC", venue="bme_growth"),
)
WARSAW = ExchangeSpec(
    key="warsaw",
    suffix=".WA",
    display_name="NewConnect",
    currency="PLN",
    source=TradingViewScannerSource(
        country="poland", exchange="NEWCONNECT", suffix=".WA", venue="newconnect"
    ),
)

EXCHANGE_SPECS: dict[str, ExchangeSpec] = {s.key: s for s in (MILAN, MADRID, WARSAW)}


class ExchangeClient:
    """Fetch typed identities for one exchange without raising expected failures."""

    def __init__(
        self,
        spec: ExchangeSpec,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self._spec = spec
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = _USER_AGENT

    def get_constituents(self) -> list[SecurityIdentity] | DataSourceError:
        """Return source-grounded identities; the universe projects canonical tickers."""
        return self._spec.source.fetch(self._session)
