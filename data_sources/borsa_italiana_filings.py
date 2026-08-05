"""Official Borsa Italiana filing discovery for Euronext Growth Milan."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Protocol
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from pydantic import ValidationError

from data_sources.errors import DataSourceError, ErrorStage
from data_sources.filing_models import (
    DocumentKind,
    DocumentRef,
    FilingsArchive,
    IssuerIdentity,
    SourceSystem,
    stable_filing_id,
)
from data_sources.regional_http import HttpDocument
from data_sources.security_master import ResolvedSecurityIdentity, SecurityMaster

DEFAULT_ARCHIVE_URL = (
    "https://www.borsaitaliana.it/borsa/azioni/elenco-completo-documenti-societari.html"
)
_VENUE = "euronext_growth_milan"


class TextGetter(Protocol):
    """The narrow shared HTTP interface used by regional archive adapters."""

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError: ...

    def validate_url(self, url: str) -> DataSourceError | None: ...


class BorsaItalianaFilingsSource:
    """Parse Borsa Italiana's issuer-specific corporate-documents HTML table.

    Borsa publishes one row per PDF and exposes neither a formal coverage interval
    nor structured attachment/supersession relationships. Consequently each PDF is
    a separate ``DocumentRef`` and coverage bounds are explicitly labelled inferred.
    """

    MAX_PAGES = 100

    def __init__(
        self,
        *,
        http: TextGetter,
        security_master: SecurityMaster,
        archive_url: str = DEFAULT_ARCHIVE_URL,
    ) -> None:
        self._http = http
        self._security_master = security_master
        self._archive_url = archive_url

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
            return self._error("parse", "filing limit must be positive")
        if from_date is not None and to_date is not None and from_date > to_date:
            return self._error("parse", "from_date cannot be after to_date")
        resolved = self._resolve_identity(identity)
        if isinstance(resolved, DataSourceError):
            return resolved
        issuer = IssuerIdentity(
            canonical_ticker=resolved.canonical_ticker,
            venue=resolved.venue,
            isin=resolved.isin,
            legal_name=resolved.legal_name,
        )
        selected_kinds = frozenset(kinds) if kinds is not None else None
        filings: list[DocumentRef] = []
        observed_dates: list[date] = []
        warnings = [
            "Borsa Italiana publishes no explicit archive coverage bounds; coverage dates "
            "are inferred from rows observed during this fetch.",
            "The archive exposes one PDF per row and no structured attachment or "
            "supersession relationships.",
        ]
        fetched_at: datetime | None = None
        page_number = 1
        exhausted = False

        while True:
            if page_number > self.MAX_PAGES:
                return self._error("parse", "Borsa archive exceeded the hard page limit")
            response = self._http.get_text(
                self._archive_url,
                params={"isin": resolved.isin, "page": str(page_number)},
            )
            if isinstance(response, DataSourceError):
                return response
            fetched_at = max(fetched_at, response.fetched_at) if fetched_at else response.fetched_at
            parsed = self._parse_page(response, issuer)
            if isinstance(parsed, DataSourceError):
                return parsed
            page_filings, next_page, page_warnings = parsed
            warnings.extend(page_warnings)
            if not page_filings and next_page is not None:
                return self._error("parse", "Borsa archive page has pagination but no filing rows")

            oldest_on_page: date | None = None
            matching_on_page: list[DocumentRef] = []
            for filing in page_filings:
                observed_dates.append(filing.publication_date)
                oldest_on_page = (
                    filing.publication_date
                    if oldest_on_page is None
                    else min(oldest_on_page, filing.publication_date)
                )
                if to_date is not None and filing.publication_date > to_date:
                    continue
                if from_date is not None and filing.publication_date < from_date:
                    continue
                if selected_kinds is not None and filing.document_kind not in selected_kinds:
                    continue
                matching_on_page.append(filing)

            remaining = limit - len(filings)
            filings.extend(matching_on_page[:remaining])
            if len(filings) == limit:
                has_more_matching = len(matching_on_page) > remaining or next_page is not None
                exhausted = not has_more_matching
                if has_more_matching:
                    warnings.append(f"Results truncated at requested limit {limit}.")
                break
            if next_page is None:
                exhausted = True
                break
            if next_page <= page_number:
                return self._error("parse", "Borsa archive pagination did not advance")
            if next_page > self.MAX_PAGES:
                return self._error("parse", "Borsa archive exceeded the hard page limit")
            if from_date is not None and oldest_on_page is not None and oldest_on_page < from_date:
                warnings.append("Pagination stopped after passing the requested start date.")
                break
            page_number = next_page

        if fetched_at is None:
            raise AssertionError("at least one archive page is always fetched")
        filings.sort(key=lambda item: item.publication_date, reverse=True)
        filings = list({item.filing_id: item for item in filings}.values())
        coverage_start = min(observed_dates) if observed_dates else None
        coverage_end = max(observed_dates) if observed_dates else None
        if from_date is not None and coverage_start is not None and from_date < coverage_start:
            warnings.append(
                f"No filing before {coverage_start.isoformat()} was observed; the inferred "
                "bound does not prove older documents are unavailable."
            )
        if to_date is not None and coverage_end is not None and to_date > coverage_end:
            warnings.append(
                f"No filing after {coverage_end.isoformat()} was observed; the inferred bound "
                "does not prove later documents are unavailable."
            )
        return FilingsArchive(
            issuer=issuer,
            filings=filings,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            pages_exhausted=exhausted,
            fetched_at=fetched_at,
            warnings=list(dict.fromkeys(warnings)),
        )

    def _resolve_identity(
        self, identity: IssuerIdentity
    ) -> ResolvedSecurityIdentity | DataSourceError:
        if identity.venue != _VENUE:
            return self._error(
                "parse",
                f"Borsa Italiana adapter requires venue {_VENUE!r}",
                stage="identity",
            )
        if identity.isin is None:
            return self._error("parse", "EGM filing discovery requires an ISIN", stage="identity")
        resolved = self._security_master.resolve(
            ticker=identity.canonical_ticker,
            venue=_VENUE,
            isin=identity.isin,
        )
        if isinstance(resolved, DataSourceError):
            return resolved
        if identity.legal_name is not None and identity.legal_name != resolved.legal_name:
            return self._error(
                "parse",
                "supplied issuer legal name contradicts the persisted security identity",
                stage="identity",
            )
        return resolved

    def _parse_page(
        self, response: HttpDocument, issuer: IssuerIdentity
    ) -> tuple[list[DocumentRef], int | None, list[str]] | DataSourceError:
        try:
            soup = BeautifulSoup(response.text, "lxml")
            table = next(
                (
                    candidate
                    for candidate in soup.select("table")
                    if "Documenti societari disponibili" in candidate.get_text(" ", strip=True)
                    or candidate.select_one('a[href*="/documenti/documenti.htm?filename="]')
                    is not None
                ),
                None,
            )
            if table is None:
                raise ValueError("corporate-documents table not found")
            filings: list[DocumentRef] = []
            warnings: list[str] = []
            rows = table.select("tbody tr")
            for row in rows:
                parsed = self._parse_row(row, response, issuer)
                if isinstance(parsed, DataSourceError):
                    warnings.append(parsed.message)
                else:
                    filings.append(parsed)
            if rows and not filings:
                raise ValueError("corporate-documents table contained rows but none were valid")
            next_page = self._next_page(soup, response.url, issuer.isin or "")
            return filings, next_page, warnings
        except (TypeError, ValueError, ValidationError) as exc:
            return self._error("parse", f"could not parse Borsa archive HTML: {exc}")

    def _parse_row(
        self, row: Tag, response: HttpDocument, issuer: IssuerIdentity
    ) -> DocumentRef | DataSourceError:
        try:
            cells = row.find_all("td", recursive=False)
            if len(cells) != 2:
                raise ValueError("filing row must contain document and date cells")
            link = cells[0].find("a", href=True)
            if not isinstance(link, Tag):
                raise ValueError("filing row has no PDF link")
            href = str(link["href"])
            direct_url = urljoin(response.url, href)
            landing_url = response.url
            for url in (landing_url, direct_url):
                url_error = self._http.validate_url(url)
                if url_error is not None:
                    raise ValueError(url_error.message)
            upstream_id, filename = self._document_identity(href)
            publication_date = datetime.strptime(
                cells[1].get_text(" ", strip=True), "%d/%m/%Y"
            ).date()
            link.extract()
            raw_title = " ".join(cells[0].get_text(" ", strip=True).split())
            title = re.sub(r"^\d{4}\s+", "", raw_title)
            title = re.sub(r"\s*-\s*\(\s*\)\s*$", "", title).strip()
            if not title:
                raise ValueError("filing title is empty")
            return DocumentRef(
                filing_id=stable_filing_id(
                    SourceSystem.BORSA_ITALIANA,
                    issuer.venue,
                    issuer.isin or "",
                    upstream_id,
                ),
                issuer=issuer,
                source_system=SourceSystem.BORSA_ITALIANA,
                upstream_id=upstream_id,
                document_kind=self._document_kind(title),
                title=title,
                publication_date=publication_date,
                original_language="it",
                landing_page_url=landing_url,
                direct_document_url=direct_url,
                mime_type="application/pdf",
                attachment_names=[filename],
                fetched_at=response.fetched_at,
                etag=response.etag,
                last_modified=response.last_modified,
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            return self._error("parse", f"skipping malformed Borsa filing row: {exc}")

    def _next_page(self, soup: BeautifulSoup, base_url: str, expected_isin: str) -> int | None:
        next_link = soup.select_one('[data-bb-view="pagination"] a[title="Successiva"]')
        current_values = parse_qs(urlparse(base_url).query).get("page", ["1"])
        current_page = int(current_values[0]) if current_values[0].isdigit() else 1
        if not isinstance(next_link, Tag) or not next_link.has_attr("href"):
            candidates: list[tuple[int, Tag]] = []
            for link in soup.select(".m-pagination a[href]"):
                href_values = parse_qs(urlparse(str(link["href"])).query).get("page")
                if href_values and href_values[0].isdigit():
                    page = int(href_values[0])
                    if page > current_page:
                        candidates.append((page, link))
            if not candidates:
                return None
            _, next_link = min(candidates, key=lambda item: item[0])
        next_url = urljoin(base_url, str(next_link["href"]))
        url_error = self._http.validate_url(next_url)
        if url_error is not None:
            raise ValueError(url_error.message)
        query = parse_qs(urlparse(next_url).query)
        if query.get("isin", [expected_isin])[0].upper() != expected_isin.upper():
            raise ValueError("next-page link changes issuer ISIN")
        values = query.get("page")
        if not values or not values[0].isdigit():
            raise ValueError("next-page link has no numeric page")
        return int(values[0])

    @staticmethod
    def _document_identity(href: str) -> tuple[str, str]:
        values = parse_qs(urlparse(href).query).get("filename")
        if not values:
            raise ValueError("PDF link has no filename parameter")
        filename = PurePosixPath(values[0]).name
        match = re.fullmatch(r"([0-9]+)\.pdf", filename, flags=re.IGNORECASE)
        if match is None:
            raise ValueError("PDF filename has no stable numeric document ID")
        return match.group(1), filename

    @staticmethod
    def _document_kind(title: str) -> DocumentKind:
        value = title.casefold()
        if "revisione" in value or "revisore" in value:
            return DocumentKind.AUDITOR
        if "semestral" in value or "bilancio intermedio" in value:
            return DocumentKind.HALF_YEAR
        if "trimestral" in value or "resoconto intermedio" in value:
            return DocumentKind.QUARTERLY
        if "bilancio" in value or "relazione finanziaria annuale" in value:
            return DocumentKind.ANNUAL
        if "ammission" in value:
            return DocumentKind.ADMISSION
        if "assemblea" in value:
            return DocumentKind.SHAREHOLDER_MEETING
        if "statuto" in value or "governance" in value:
            return DocumentKind.GOVERNANCE
        return DocumentKind.OTHER

    @staticmethod
    def _error(code: str, message: str, *, stage: ErrorStage = "discovery") -> DataSourceError:
        return DataSourceError(
            error_code=code,
            message=message,
            stage=stage,
            source=SourceSystem.BORSA_ITALIANA,
        )
