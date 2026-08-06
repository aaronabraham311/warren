"""Euronext Product Directory adapter for Euronext Growth Milan (EXGM)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from data_sources.errors import DataSourceError
from data_sources.security_identity import SecurityIdentity


@dataclass(frozen=True)
class EuronextProductDirectorySource:
    """Fetch identities from Euronext's legacy DataTables product endpoint."""

    mics: str
    suffix: str
    venue: str
    exclude_name_prefix: str = "W "
    timeout: int = 15
    page_size: int = 300

    URL = "https://live.euronext.com/en/pd_es/data/stocks"

    def fetch(self, session: requests.Session) -> list[SecurityIdentity] | DataSourceError:
        try:
            rows: list[object] = []
            offset = 0
            while True:
                try:
                    response = session.post(
                        self.URL,
                        params={
                            "mics": self.mics,
                            "display_datapoints": "dp_stocks",
                            "display_filters": "df_stocks",
                            "display_type": "all",
                        },
                        data={
                            "iDisplayStart": str(offset),
                            "iDisplayLength": str(self.page_size),
                            "args[initialLetter]": "",
                        },
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                except requests.RequestException as exc:
                    return DataSourceError(error_code="network", message=str(exc))
                payload = response.json()
                page = payload.get("aaData") if isinstance(payload, dict) else None
                if not isinstance(page, list):
                    raise ValueError("Euronext response has no aaData list")
                rows.extend(page)
                if len(page) < self.page_size:
                    break
                offset += self.page_size

            resolved_at = datetime.now(timezone.utc)
            identities: list[SecurityIdentity] = []
            for row in rows:
                if not isinstance(row, list) or len(row) < 5:
                    raise ValueError("Euronext response contains a malformed row")
                name_html, isin, symbol = row[1], row[2], row[3]
                if not all(isinstance(value, str) for value in (name_html, isin, symbol)):
                    raise ValueError("Euronext identity fields must be strings")
                legal_name = BeautifulSoup(name_html, "lxml").get_text(strip=True)
                if not legal_name or not isin or not symbol:
                    raise ValueError("Euronext identity fields cannot be empty")
                if legal_name.startswith(self.exclude_name_prefix):
                    continue
                canonical = symbol.upper()
                if not canonical.endswith(self.suffix.upper()):
                    canonical = f"{canonical}{self.suffix}"
                identities.append(
                    SecurityIdentity(
                        canonical_ticker=canonical,
                        venue=self.venue,
                        mic=self.mics,
                        exchange_symbol=symbol.upper(),
                        isin=isin.upper(),
                        legal_name=legal_name,
                        identity_source_url=self.URL,
                        resolved_at=resolved_at,
                        aliases=(symbol.upper(),),
                        source_ids={"isin": isin.upper(), "mic": self.mics},
                    )
                )
            if not identities:
                raise ValueError("Euronext response yielded no equity identities")
            return identities
        except (TypeError, ValueError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))
