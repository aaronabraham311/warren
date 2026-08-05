"""TradingView scanner adapter for the NewConnect junior market."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from data_sources.errors import DataSourceError
from data_sources.security_identity import SecurityIdentity


@dataclass(frozen=True)
class TradingViewScannerSource:
    """Fetch NewConnect symbols from TradingView's keyless scanner endpoint."""

    country: str
    exchange: str
    suffix: str
    venue: str
    timeout: int = 15

    @property
    def url(self) -> str:
        return f"https://scanner.tradingview.com/{self.country}/scan"

    def fetch(self, session: requests.Session) -> list[SecurityIdentity] | DataSourceError:
        body: dict[str, list[dict[str, str]] | list[str] | list[int]] = {
            "filter": [
                {"left": "type", "operation": "equal", "right": "stock"},
                {"left": "exchange", "operation": "equal", "right": self.exchange},
            ],
            "columns": ["name", "description", "isin", "market_cap_basic"],
            "range": [0, 500],
        }
        try:
            response = session.post(self.url, json=body, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            return DataSourceError(error_code="network", message=str(exc))

        try:
            payload = response.json()
            rows = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(rows, list):
                raise ValueError("TradingView response has no data list")
            resolved_at = datetime.now(timezone.utc)
            identities: list[SecurityIdentity] = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("TradingView response contains a malformed row")
                values = row.get("d")
                source_symbol = row.get("s")
                if not isinstance(values, list) or len(values) < 3:
                    raise ValueError("TradingView row is missing its name column")
                symbol_raw, legal_name, isin = values[:3]
                if not all(isinstance(value, str) for value in (symbol_raw, legal_name, isin)):
                    raise ValueError("TradingView identity fields must be strings")
                symbol = symbol_raw.upper()
                if not symbol:
                    raise ValueError("TradingView exchange symbol cannot be empty")
                if len(isin) != 12:
                    raise ValueError("TradingView row has an invalid ISIN")
                canonical = (
                    symbol if symbol.endswith(self.suffix.upper()) else f"{symbol}{self.suffix}"
                )
                source_ids: dict[str, str] = {}
                if isinstance(source_symbol, str) and source_symbol:
                    source_ids["tradingview_symbol"] = source_symbol
                identities.append(
                    SecurityIdentity(
                        canonical_ticker=canonical,
                        venue=self.venue,
                        mic=None,
                        exchange_symbol=symbol,
                        isin=isin.upper(),
                        legal_name=legal_name,
                        identity_source_url=self.url,
                        resolved_at=resolved_at,
                        aliases=(symbol,),
                        source_ids=source_ids,
                    )
                )
            if not identities:
                raise ValueError("TradingView response yielded no equity identities")
            return identities
        except (TypeError, ValueError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))
