"""BME Growth adapter with the required ISIN-to-ticker enrichment call."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from data_sources.errors import DataSourceError
from data_sources.security_identity import SecurityIdentity


@dataclass(frozen=True)
class BMEGrowthSource:
    """Fetch BME Growth companies, then resolve each ISIN to its exchange ticker."""

    mtf_segment: str
    suffix: str
    venue: str
    timeout: int = 15

    LIST_URL = "https://apiweb.bolsasymercados.es/Market/v1/EQ/ListedCompanies"
    DETAILS_URL = "https://apiweb.bolsasymercados.es/Market/v1/EQ/ShareDetailsInfo"

    def _get_json(
        self,
        session: requests.Session,
        url: str,
        *,
        params: dict[str, str | int],
    ) -> object | DataSourceError:
        """Keep transport failures distinct from malformed response bodies."""
        try:
            response = session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            return DataSourceError(error_code="network", message=str(exc))
        try:
            payload: object = response.json()
            return payload
        except (TypeError, ValueError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))

    def fetch(self, session: requests.Session) -> list[SecurityIdentity] | DataSourceError:
        common: dict[str, str | int] = {
            "tradingSystem": "MTF",
            "mtfSegment": self.mtf_segment,
        }
        payload = self._get_json(
            session,
            self.LIST_URL,
            params={
                **common,
                "ISIN": "",
                "sectorKey": "",
                "subsectorKey": "",
                "page": 0,
                "pageSize": 0,
            },
        )
        if isinstance(payload, DataSourceError):
            return payload
        try:
            companies = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(companies, list):
                raise ValueError("BME response has no company data list")

            resolved_at = datetime.now(timezone.utc)
            identities: list[SecurityIdentity] = []
            for company in companies:
                if not isinstance(company, dict):
                    raise ValueError("BME response contains a malformed company")
                isin = company.get("isin")
                legal_name = company.get("name")
                company_key = company.get("companyKey")
                if not isinstance(isin, str) or not isin:
                    raise ValueError("BME company is missing an ISIN")
                details = self._get_json(
                    session,
                    self.DETAILS_URL,
                    params={**common, "ISIN": isin},
                )
                if isinstance(details, DataSourceError):
                    return details
                ticker = details.get("ticker") if isinstance(details, dict) else None
                if not isinstance(ticker, str) or not ticker:
                    raise ValueError(f"BME did not resolve ticker for {isin}")
                if legal_name is not None and not isinstance(legal_name, str):
                    raise ValueError("BME legal name must be a string")
                canonical = ticker.upper()
                if not canonical.endswith(self.suffix.upper()):
                    canonical = f"{canonical}{self.suffix}"
                source_ids = {"isin": isin.upper()}
                if isinstance(company_key, str) and company_key:
                    source_ids["company_key"] = company_key
                issuer_code = details.get("issuerCode") if isinstance(details, dict) else None
                if isinstance(issuer_code, str) and issuer_code:
                    source_ids["issuer_code"] = issuer_code
                identities.append(
                    SecurityIdentity(
                        canonical_ticker=canonical,
                        venue=self.venue,
                        mic=None,
                        exchange_symbol=ticker.upper(),
                        isin=isin.upper(),
                        legal_name=legal_name,
                        identity_source_url=self.DETAILS_URL,
                        resolved_at=resolved_at,
                        aliases=(ticker.upper(),),
                        source_ids=source_ids,
                    )
                )
            if not identities:
                raise ValueError("BME response yielded no equity identities")
            return identities
        except (TypeError, ValueError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))
