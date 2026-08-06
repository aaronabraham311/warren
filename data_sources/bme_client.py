"""BME Growth adapter with the required ISIN-to-ticker enrichment call."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import ceil

import requests

from data_sources.errors import DataSourceError, ErrorStage
from data_sources.filing_models import (
    DocumentKind,
    DocumentRef,
    FilingsArchive,
    IssuerIdentity,
    SourceSystem,
    stable_filing_id,
)
from data_sources.regional_http import RegionalHttpClient
from data_sources.security_identity import SecurityIdentity
from data_sources.security_master import SecurityMaster


@dataclass(frozen=True)
class BMEGrowthSource:
    """Fetch BME Growth companies, then resolve each ISIN to its exchange ticker."""

    mtf_segment: str
    suffix: str
    venue: str
    timeout: int = 15

    LIST_URL = "https://apiweb.bolsasymercados.es/Market/v1/EQ/ListedCompanies"
    DETAILS_URL = "https://apiweb.bolsasymercados.es/Market/v1/EQ/ShareDetailsInfo"
    MIN_DETAIL_COMPLETENESS = 0.8

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
                if not isinstance(legal_name, str) or not legal_name.strip():
                    raise ValueError("BME company is missing a legal name")
                details = self._get_json(
                    session,
                    self.DETAILS_URL,
                    params={**common, "ISIN": isin},
                )
                if isinstance(details, DataSourceError):
                    continue
                ticker = details.get("ticker") if isinstance(details, dict) else None
                if not isinstance(ticker, str) or not ticker:
                    continue
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
                        legal_name=legal_name.strip(),
                        identity_source_url=self.DETAILS_URL,
                        resolved_at=resolved_at,
                        aliases=(ticker.upper(),),
                        source_ids=source_ids,
                    )
                )
            required = max(1, ceil(len(companies) * self.MIN_DETAIL_COMPLETENESS))
            if len(identities) < required:
                raise ValueError(
                    "BME ticker resolution was incomplete: "
                    f"resolved {len(identities)} of {len(companies)} companies"
                )
            return identities
        except (TypeError, ValueError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))


@dataclass(frozen=True)
class BMEGrowthFilingsSource:
    """Discover BME Growth issuer publications through its public archive.

    Network policy, caching, and retries belong to ``RegionalHttpClient``. This
    adapter only constructs allowlisted BME requests and normalizes responses.
    Identity matching is deliberately delegated to the persisted security master;
    the public archive is never searched using a guessed ticker or ISIN.
    """

    http: RegionalHttpClient
    security_master: SecurityMaster
    page_size: int = 50

    DETAILS_URL = "https://apiweb.bolsasymercados.es/Market/v1/EQ/ShareDetailsInfo"
    DOCUMENTS_URL = "https://apiweb.bolsasymercados.es/Market/v1/EQ/MtfEquity/Documents"
    FINANCIAL_URL = "https://apiweb.bolsasymercados.es/Market/v1/EQ/MtfEquity/FinancialInformation"
    DOCUMENT_BASE_URL = "https://apiweb.bolsasymercados.es/Market"
    ARCHIVE_START = date(1990, 1, 1)
    MAX_PAGES = 100

    def list_filings(
        self,
        identity: IssuerIdentity,
        *,
        kinds: Sequence[DocumentKind] | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 50,
    ) -> FilingsArchive | DataSourceError:
        if limit < 1:
            return self._error("parse", "filing limit must be at least one", stage="discovery")
        if from_date is not None and to_date is not None and from_date > to_date:
            return self._error("parse", "from_date cannot be after to_date", stage="discovery")

        resolved = self.security_master.resolve(
            ticker=identity.canonical_ticker,
            venue=identity.venue,
            isin=identity.isin,
        )
        if isinstance(resolved, DataSourceError):
            return resolved
        if identity.venue != "bme_growth":
            return self._error(
                "parse",
                f"BME Growth adapter cannot serve venue {identity.venue!r}",
                stage="identity",
            )

        issuer = IssuerIdentity(
            canonical_ticker=resolved.canonical_ticker,
            venue=resolved.venue,
            isin=resolved.isin,
            legal_name=resolved.legal_name,
        )
        company_key = self._company_key(
            isin=resolved.isin,
            exchange_symbol=resolved.exchange_symbol,
        )
        if isinstance(company_key, DataSourceError):
            return company_key
        query_start = from_date or self.ARCHIVE_START
        query_end = to_date or date.today()
        common_documents = {
            "companyKey": company_key,
            "documentTypes": (
                "InsideInformation,OtherRelevantInformation,Prospectuses,Notices,RelevantFacts"
            ),
            "language": "es",
            "from": query_start.strftime("%Y%m%d"),
            "to": query_end.strftime("%Y%m%d"),
        }
        documents = self._fetch_archive(self.DOCUMENTS_URL, common_documents, issuer)
        if isinstance(documents, DataSourceError):
            return documents
        financial = self._fetch_archive(
            self.FINANCIAL_URL,
            {
                "companyKey": company_key,
                "language": "es",
                "mtfSegment": "BMEGrowth",
                "order": "Period DESC",
                "fromYear": str(query_start.year),
                "toYear": str(query_end.year),
            },
            issuer,
        )
        if isinstance(financial, DataSourceError):
            return financial

        discovered: dict[str, DocumentRef] = {}
        for filing in (*documents[0], *financial[0]):
            existing = discovered.get(filing.filing_id)
            if existing is not None and (
                existing.upstream_id != filing.upstream_id
                or existing.direct_document_url != filing.direct_document_url
            ):
                return self._error(
                    "parse",
                    f"BME filing {filing.upstream_id!r} changed identity across archives",
                    stage="discovery",
                )
            if existing is None or (
                existing.document_kind is not DocumentKind.ANNUAL
                and filing.document_kind is DocumentKind.ANNUAL
            ):
                discovered[filing.filing_id] = filing

        wanted = set(kinds) if kinds is not None else None
        all_filings = [
            filing
            for filing in discovered.values()
            if (wanted is None or filing.document_kind in wanted)
            and (from_date is None or filing.publication_date >= from_date)
            and (to_date is None or filing.publication_date <= to_date)
        ]
        all_filings.sort(key=lambda item: item.publication_date, reverse=True)
        truncated = len(all_filings) > limit
        filings = all_filings[:limit]
        warnings = [
            "BME does not publish an archive completeness boundary; coverage bounds "
            "reflect returned filings only."
        ]
        if truncated:
            warnings.append(
                "BME results were truncated at the requested limit; additional filings exist."
            )
        dates = [item.publication_date for item in filings]
        fetched_times = [item.fetched_at for item in (*documents[0], *financial[0])]
        return FilingsArchive(
            issuer=issuer,
            filings=filings,
            coverage_start=min(dates) if dates else None,
            coverage_end=max(dates) if dates else None,
            pages_exhausted=documents[1] and financial[1] and not truncated,
            fetched_at=max(fetched_times) if fetched_times else datetime.now(timezone.utc),
            warnings=warnings,
        )

    def _company_key(self, *, isin: str, exchange_symbol: str) -> str | DataSourceError:
        response = self.http.get_text(
            self.DETAILS_URL,
            params={
                "tradingSystem": "MTF",
                "mtfSegment": "BMEGrowth",
                "ISIN": isin,
            },
        )
        if isinstance(response, DataSourceError):
            return self._with_source(response, stage="identity")
        try:
            payload = json.loads(response.text)
            if not isinstance(payload, dict):
                raise ValueError("BME share-details response must be an object")
            if str(payload.get("isin", "")).upper() != isin:
                raise ValueError("BME share-details response returned a different ISIN")
            if str(payload.get("ticker", "")).upper() != exchange_symbol.upper():
                raise ValueError("BME share-details response returned a different ticker")
            if payload.get("mtfSegment") != "BMEGrowth":
                raise ValueError("BME share-details response is not a BME Growth listing")
            value = payload.get("issuerCode")
            if not isinstance(value, str) or not value:
                raise ValueError("BME share-details response omitted issuerCode")
            return value
        except (json.JSONDecodeError, ValueError) as exc:
            return self._error("parse", str(exc), stage="identity")

    def _fetch_archive(
        self,
        url: str,
        base_params: Mapping[str, str],
        issuer: IssuerIdentity,
    ) -> tuple[list[DocumentRef], bool] | DataSourceError:
        page = 0
        filings: list[DocumentRef] = []
        while True:
            if page >= self.MAX_PAGES:
                return self._error(
                    "parse", "BME archive exceeded the hard page limit", stage="discovery"
                )
            response = self.http.get_text(
                url,
                params={
                    **base_params,
                    "page": str(page),
                    "pagesize": str(self.page_size),
                },
            )
            if isinstance(response, DataSourceError):
                return self._with_source(response, stage="discovery")
            try:
                records, has_more = self._parse_page(
                    json.loads(response.text),
                    expected_page=page,
                    expected_company_key=base_params["companyKey"],
                )
                filings.extend(
                    self._document_ref(
                        record,
                        issuer,
                        response.fetched_at,
                        etag=response.etag,
                        last_modified=response.last_modified,
                    )
                    for record in records
                )
                if not has_more:
                    return filings, True
                if not records:
                    raise ValueError("BME archive claims more results after an empty page")
                page += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return self._error("parse", str(exc), stage="discovery")

    @staticmethod
    def _parse_page(
        payload: object, *, expected_page: int, expected_company_key: str
    ) -> tuple[list[Mapping[str, object]], bool]:
        if not isinstance(payload, dict):
            raise ValueError("BME archive response must be a JSON object")
        raw_records = payload.get("data")
        if not isinstance(raw_records, list):
            raise ValueError("BME archive response has no data list")
        records: list[Mapping[str, object]] = []
        for item in raw_records:
            if not isinstance(item, dict):
                raise ValueError("BME archive contains a malformed filing")
            records.append(item)
        params = payload.get("params")
        if not isinstance(params, dict):
            raise ValueError("BME archive response has no params object")
        if params.get("page") != expected_page:
            raise ValueError("BME archive returned an unexpected page")
        if params.get("companyKey") != expected_company_key:
            raise ValueError("BME archive response changed issuer companyKey")
        page_size = params.get("pageSize")
        total_results = payload.get("totalResults")
        has_more = payload.get("hasMoreResults")
        if not isinstance(page_size, int) or page_size < 1:
            raise ValueError("BME archive returned an invalid pageSize")
        if not isinstance(total_results, int) or total_results < 0:
            raise ValueError("BME archive returned an invalid totalResults")
        if not isinstance(has_more, bool):
            raise ValueError("BME archive returned an invalid hasMoreResults")
        total_pages = (total_results + page_size - 1) // page_size
        if total_pages > BMEGrowthFilingsSource.MAX_PAGES:
            raise ValueError("BME archive exceeds the hard page limit")
        if total_results == 0:
            if expected_page != 0 or records or has_more:
                raise ValueError("BME empty archive pagination metadata is inconsistent")
            return records, False
        if expected_page >= total_pages:
            raise ValueError("BME archive returned a page beyond totalResults")
        expected_more = expected_page + 1 < total_pages
        if has_more is not expected_more:
            raise ValueError("BME hasMoreResults contradicts totalResults/pageSize")
        expected_records = min(page_size, total_results - expected_page * page_size)
        if len(records) != expected_records:
            raise ValueError("BME archive page row count contradicts pagination metadata")
        for record in records:
            if record.get("companyKey") != expected_company_key:
                raise ValueError("BME filing record changed issuer companyKey")
            if record.get("segment") != "BMEGrowth":
                raise ValueError("BME filing record is not from BMEGrowth")
        return records, has_more

    def _document_ref(
        self,
        record: Mapping[str, object],
        issuer: IssuerIdentity,
        fetched_at: datetime,
        *,
        etag: str | None,
        last_modified: str | None,
    ) -> DocumentRef:
        upstream_id = BMEGrowthFilingsSource._required_string(record, "id")
        title = BMEGrowthFilingsSource._required_string(record, "title")
        publication_date = BMEGrowthFilingsSource._compact_date(record.get("date"))
        if publication_date is None:
            raise ValueError(f"BME filing {upstream_id!r} has an invalid date")
        raw_kind = BMEGrowthFilingsSource._required_string(record, "documentType")
        kind = self._kind(record, raw_kind)
        direct_url = self._document_url(self._required_string(record, "url"))
        url_error = self.http.validate_url(direct_url)
        if url_error is not None:
            raise ValueError(url_error.message)
        annexes = record.get("annexes")
        if not isinstance(annexes, list):
            raise ValueError(f"BME filing {upstream_id!r} has an invalid annexes list")
        attachment_names: list[str] = []
        for annex in annexes:
            if not isinstance(annex, dict):
                raise ValueError(f"BME filing {upstream_id!r} has a malformed annex")
            annex_id = self._required_string(annex, "id")
            annex_title = self._required_string(annex, "title")
            annex_url = self._document_url(self._required_string(annex, "url"))
            url_error = self.http.validate_url(annex_url)
            if url_error is not None:
                raise ValueError(url_error.message)
            attachment_names.append(f"{annex_title} [BME {annex_id}]")
        amendment_ids: set[str] = set()
        for field in ("replaces", "rectifies"):
            relation = record.get(field)
            if relation is not None:
                if not isinstance(relation, dict):
                    raise ValueError(f"BME filing {upstream_id!r} has malformed {field}")
                amendment_ids.add(self._required_string(relation, "id"))
        if len(amendment_ids) > 1:
            raise ValueError(f"BME filing {upstream_id!r} has conflicting amendment targets")
        amendment_of = next(iter(amendment_ids), None)
        identity = issuer.isin
        if identity is None:  # Kept defensive despite the regional DocumentRef contract.
            raise ValueError("BME filing identity has no ISIN")
        filing_id = stable_filing_id(SourceSystem.BME, issuer.venue, identity, upstream_id)
        supersedes = (
            stable_filing_id(SourceSystem.BME, issuer.venue, identity, amendment_of)
            if amendment_of
            else None
        )
        return DocumentRef(
            filing_id=filing_id,
            issuer=issuer,
            source_system=SourceSystem.BME,
            upstream_id=upstream_id,
            document_kind=kind,
            title=title,
            publication_date=publication_date,
            original_language="es",
            landing_page_url=direct_url,
            direct_document_url=direct_url,
            mime_type="application/pdf",
            attachment_names=attachment_names,
            amended=amendment_of is not None,
            supersedes_filing_id=supersedes,
            fetched_at=fetched_at,
            etag=etag,
            last_modified=last_modified,
        )

    @staticmethod
    def _kind(record: Mapping[str, object], document_type: str) -> DocumentKind:
        if document_type == "InsideInformation":
            return DocumentKind.INSIDE_INFORMATION
        if document_type == "Prospectuses":
            return DocumentKind.ADMISSION
        period = str(record.get("financialInfoPeriod") or "").upper()
        if period == "AN":
            return DocumentKind.ANNUAL
        if period in {"S1", "S2", "SE"}:
            return DocumentKind.HALF_YEAR
        if period.startswith(("Q", "T")):
            return DocumentKind.QUARTERLY
        raw_types = record.get("types")
        if not isinstance(raw_types, list):
            raise ValueError("BME filing has an invalid types list")
        labels: list[str] = []
        keys: set[str] = set()
        for value in raw_types:
            if not isinstance(value, dict):
                raise ValueError("BME filing has a malformed type")
            keys.add(str(value.get("key") or ""))
            labels.append(str(value.get("value") or "").casefold())
        combined = " ".join(labels)
        if "001" in keys or "anual" in combined or "annual" in combined:
            return DocumentKind.ANNUAL
        if "semestr" in combined or "half-year" in combined:
            return DocumentKind.HALF_YEAR
        if "trimestr" in combined or "quarter" in combined:
            return DocumentKind.QUARTERLY
        if "junta" in combined or "shareholder" in combined:
            return DocumentKind.SHAREHOLDER_MEETING
        if document_type in {"OtherRelevantInformation", "RelevantFacts", "Notices"}:
            return DocumentKind.OTHER_RELEVANT
        return DocumentKind.OTHER

    def _document_url(self, value: str) -> str:
        if value.startswith("/MTFDocuments/"):
            return f"{self.DOCUMENT_BASE_URL}{value}"
        raise ValueError(
            f"BME document URL must use the official /Market/MTFDocuments/ prefix: {value!r}"
        )

    @staticmethod
    def _required_string(record: Mapping[str, object], field: str) -> str:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"BME filing field {field!r} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _compact_date(value: object) -> date | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if len(normalized) == 8 and normalized.isdigit():
            return datetime.strptime(normalized, "%Y%m%d").date()
        try:
            return date.fromisoformat(normalized)
        except ValueError:
            return None

    @staticmethod
    def _error(error_code: str, message: str, *, stage: ErrorStage) -> DataSourceError:
        return DataSourceError(
            error_code=error_code,
            message=message,
            stage=stage,
            source=SourceSystem.BME.value,
        )

    @staticmethod
    def _with_source(error: DataSourceError, *, stage: ErrorStage) -> DataSourceError:
        return DataSourceError(
            error_code=error.error_code,
            message=error.message,
            stage=error.stage or stage,
            source=error.source or SourceSystem.BME.value,
        )
