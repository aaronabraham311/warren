"""Lazy, process-wide data-source client singletons for the tools layer.

Tools are instantiated arg-free (``GetQuoteTool()``) and ``run(tool_input, ctx)``
takes no client, but the cached clients each need a ``sqlite3.Connection``. These
getters bind one shared connection (on the cache DB at ``$WARREN_DB``, default
``warren.db``) to lazily-built clients. ``finnhub_client()`` and
``opensanctions_client()`` return ``None`` when the respective API key env var is
unset so callers can degrade gracefully.

``reset_clients()`` exists for tests to clear the singletons (and close the
connection) between cases — see the autouse fixture in ``tests/conftest.py``.
"""

import os
import sqlite3

from data_sources.edgar_client import EDGARClient
from data_sources.finnhub_client import FinnhubClient
from data_sources.gdelt_client import GDELTClient
from data_sources.opensanctions_client import OpenSanctionsClient
from data_sources.yfinance_client import YFinanceClient

_conn: sqlite3.Connection | None = None
_yf: YFinanceClient | None = None
_edgar: EDGARClient | None = None
_finnhub: FinnhubClient | None = None
_finnhub_resolved = False
_gdelt: GDELTClient | None = None
_opensanctions: OpenSanctionsClient | None = None
_opensanctions_resolved = False


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


def opensanctions_client() -> OpenSanctionsClient | None:
    global _opensanctions, _opensanctions_resolved
    if not _opensanctions_resolved:
        api_key = os.environ.get("OPENSANCTIONS_API_KEY", "")
        _opensanctions = OpenSanctionsClient(_connection(), api_key=api_key) if api_key else None
        _opensanctions_resolved = True
    return _opensanctions


def reset_clients() -> None:
    global _conn, _yf, _edgar, _finnhub, _finnhub_resolved
    global _gdelt, _opensanctions, _opensanctions_resolved
    if _conn is not None:
        _conn.close()
    _conn = None
    _yf = None
    _edgar = None
    _finnhub = None
    _finnhub_resolved = False
    _gdelt = None
    _opensanctions = None
    _opensanctions_resolved = False
