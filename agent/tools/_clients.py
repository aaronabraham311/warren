"""Lazy, process-wide data-source client singletons for the tools layer.

Tools are instantiated arg-free (``GetQuoteTool()``) and ``run(tool_input, ctx)``
takes no client, but the cached clients each need a ``sqlite3.Connection``. These
getters bind one shared connection (on the cache DB at ``$WARREN_DB``, default
``warren.db``) to lazily-built clients. ``finnhub_client()`` returns ``None`` when
``FINNHUB_API_KEY`` is unset so callers can degrade gracefully.

``reset_clients()`` exists for tests to clear the singletons (and close the
connection) between cases — see the autouse fixture in ``tests/conftest.py``.
"""

import os
import sqlite3
from decimal import Decimal

import anthropic

from data_sources.edgar_client import EDGARClient
from data_sources.filing_translation import (
    AnthropicPageTranslator,
    AnthropicSdkTranslationTransport,
    PageTranslator,
)
from data_sources.finnhub_client import FinnhubClient
from data_sources.gdelt_client import GDELTClient
from data_sources.ofac_client import OFACClient
from data_sources.stored_filings import StoredFilingClient
from data_sources.yfinance_client import YFinanceClient
from storage.artifacts import ArtifactStore

_conn: sqlite3.Connection | None = None
_yf: YFinanceClient | None = None
_edgar: EDGARClient | None = None
_finnhub: FinnhubClient | None = None
_finnhub_resolved = False
_gdelt: GDELTClient | None = None
_ofac: OFACClient | None = None
_stored_filings: StoredFilingClient | None = None


def _connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = os.environ.get("WARREN_DB", "warren.db")
        _conn = sqlite3.connect(path, check_same_thread=False)
    return _conn


def yfinance_client() -> YFinanceClient:
    global _yf
    if _yf is None:
        _yf = YFinanceClient(_connection())
    return _yf


def edgar_client() -> EDGARClient:
    global _edgar
    if _edgar is None:
        _edgar = EDGARClient(_connection())
    return _edgar


def finnhub_client() -> FinnhubClient | None:
    global _finnhub, _finnhub_resolved
    if not _finnhub_resolved:
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        _finnhub = FinnhubClient(_connection(), api_key=api_key) if api_key else None
        _finnhub_resolved = True
    return _finnhub


def gdelt_client() -> GDELTClient:
    global _gdelt
    if _gdelt is None:
        _gdelt = GDELTClient(_connection())
    return _gdelt


def ofac_client() -> OFACClient:
    global _ofac
    if _ofac is None:
        _ofac = OFACClient(_connection())
    return _ofac


def stored_filing_client() -> StoredFilingClient:
    global _stored_filings
    if _stored_filings is None:
        _stored_filings = StoredFilingClient(
            _connection(), ArtifactStore(), translator=_translation_provider()
        )
    return _stored_filings


def _translation_provider() -> PageTranslator | None:
    model = os.environ.get("WARREN_TRANSLATION_MODEL", "").strip()
    input_rate = os.environ.get("WARREN_TRANSLATION_INPUT_USD_PER_MILLION_TOKENS", "").strip()
    output_rate = os.environ.get("WARREN_TRANSLATION_OUTPUT_USD_PER_MILLION_TOKENS", "").strip()
    if not model or not input_rate or not output_rate:
        return None
    return AnthropicPageTranslator(
        AnthropicSdkTranslationTransport(anthropic.Anthropic()),
        model=model,
        input_usd_per_million_tokens=Decimal(input_rate),
        output_usd_per_million_tokens=Decimal(output_rate),
    )


def reset_clients() -> None:
    global _conn, _yf, _edgar, _finnhub, _finnhub_resolved, _gdelt, _ofac, _stored_filings
    if _conn is not None:
        _conn.close()
    _conn = None
    _yf = None
    _edgar = None
    _finnhub = None
    _finnhub_resolved = False
    _gdelt = None
    _ofac = None
    _stored_filings = None
