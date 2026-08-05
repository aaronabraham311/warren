"""Fail-closed Biznes PAP boundary for NewConnect filing discovery."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from data_sources.errors import DataSourceError
from data_sources.filing_models import DocumentKind, FilingsArchive, IssuerIdentity
from data_sources.regional_http import RegionalHttpClient
from data_sources.security_master import SecurityMaster

_VENUE = "newconnect"
_PAP_BASE = "https://biznes.pap.pl"
_KNOWN_COMPANY_IDS: dict[str, str] = {"PLCRFRG00016": "1490"}  # CFG.WA, official PAP ID
_DATE_RE = re.compile(r"\b(20\d{2})[-./](\d{2})[-./](\d{2})\b")
_MAX_YEARS = 10


@dataclass(frozen=True)
class PapPeriodicRow:
    """Neutral fields observed in PAP's generic periodic-report table."""

    company_id: str
    report_number: str
    publication_date: date
    detail_url: str


class NewConnectFilingsSource:
    """Probe PAP safely but refuse to invent EBI/ESPI filing provenance.

    PAP identifies issuers by a numeric provider ID. That identity mapping and
    the generic periodic-table shape are verified, but the current plain HTTP
    transport is blocked by Incapsula and no verified channel/detail contract is
    available. Consequently this source never emits successful DocumentRefs.
    """

    def __init__(
        self,
        http: RegionalHttpClient,
        security_master: SecurityMaster,
        *,
        company_ids: Mapping[str, str] | None = None,
        default_history_years: int = 2,
    ) -> None:
        if default_history_years < 1:
            raise ValueError("default_history_years must be positive")
        self._http = http
        self._security_master = security_master
        self._company_ids = dict(_KNOWN_COMPANY_IDS if company_ids is None else company_ids)
        self._default_history_years = default_history_years

    def list_filings(
        self,
        identity: IssuerIdentity,
        *,
        kinds: Sequence[DocumentKind] | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 50,
    ) -> FilingsArchive | DataSourceError:
        del kinds  # Filtering is unavailable until verified source provenance exists.
        if limit < 1:
            return self._error("parse", "filing limit must be positive")
        if from_date and to_date and from_date > to_date:
            return self._error("parse", "from_date cannot be after to_date")
        resolved = self._security_master.resolve(
            ticker=identity.canonical_ticker,
            venue=identity.venue,
            isin=identity.isin,
        )
        if isinstance(resolved, DataSourceError):
            return resolved
        if resolved.venue != _VENUE:
            return DataSourceError(
                error_code="not_found",
                message=f"identity is listed on {resolved.venue}, not {_VENUE}",
                stage="identity",
                source="security_master",
            )
        company_id = self._company_ids.get(resolved.isin)
        if company_id is None:
            return DataSourceError(
                error_code="not_found",
                message=f"no verified PAP company ID for ISIN {resolved.isin}",
                stage="identity",
                source="security_master",
            )

        upper = to_date or date.today()
        lower = from_date or date(upper.year - self._default_history_years + 1, 1, 1)
        if upper.year - lower.year + 1 > _MAX_YEARS:
            return self._error(
                "parse", f"PAP history request exceeds the {_MAX_YEARS}-year safety limit"
            )

        # This generic periodic endpoint is evidence for the raw table shape only.
        # It cannot establish whether a row came from EBI or ESPI/PAP.
        url = f"{_PAP_BASE}/articles/periodic/{upper.year}/{upper.month}/{upper.day}"
        response = self._http.get_text(
            url,
            params={
                "company": company_id,
                "selectCompany": company_id,
                "selectDay": "true",
                "limit": "25",
                "page": "0",
            },
        )
        if isinstance(response, DataSourceError):
            return DataSourceError(
                error_code=response.error_code,
                message=response.message,
                stage=response.stage or "discovery",
                source="biznes_pap",
            )
        if self._is_waf_challenge(response.text):
            return self._error("network", "Biznes PAP returned an Incapsula access challenge")
        return self._error(
            "stale_data",
            "NewConnect filing discovery is unsupported: PAP content was fetched, but the "
            "EBI/ESPI channel and detail-page provenance contract is unverified; a "
            "browser-capable verified transport is required",
        )

    @staticmethod
    def _is_waf_challenge(text: str) -> bool:
        lowered = text.casefold()
        return "_incapsula_resource" in lowered or "incapsula incident" in lowered

    @staticmethod
    def _error(code: str, message: str) -> DataSourceError:
        return DataSourceError(
            error_code=code,
            message=message,
            stage="discovery",
            source="biznes_pap",
        )


def parse_pap_periodic_rows(
    payload: str,
    *,
    company_id: str,
    base_url: str = _PAP_BASE,
) -> list[PapPeriodicRow]:
    """Parse neutral fields from the observed generic PAP periodic table.

    The result intentionally carries no SourceSystem and cannot be promoted to a
    DocumentRef without a separately verified EBI/ESPI channel contract.
    """
    html = _unwrap_ajax(payload)
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("div.tableComponent table tbody")
    if not isinstance(table, Tag):
        raise ValueError("official PAP page has no periodic report table")

    records: list[PapPeriodicRow] = []
    publication_date: date | None = None
    report_rows = 0
    for row in table.find_all("tr", recursive=False):
        day_cell = row.select_one("td.day")
        if isinstance(day_cell, Tag):
            publication_date = _date_text(day_cell.get_text(" ", strip=True))
            continue
        report_rows += 1
        cells = row.find_all("td", recursive=False)
        if len(cells) != 4:
            continue
        company_link = cells[2].find("a", href=True)
        if not isinstance(company_link, Tag):
            continue
        company_values = parse_qs(urlparse(str(company_link["href"])).query)
        if company_values.get("company") != [company_id] or company_values.get("selectCompany") != [
            company_id
        ]:
            continue
        detail = cells[3].find("a", href=re.compile(r"^/wiadomosci/"))
        report_number = cells[1].get_text(" ", strip=True)
        if not isinstance(detail, Tag) or not report_number or publication_date is None:
            continue
        detail_url = urljoin(base_url, str(detail["href"]))
        parsed_detail = urlparse(detail_url)
        if (
            parsed_detail.scheme != "https"
            or parsed_detail.hostname != "biznes.pap.pl"
            or not parsed_detail.path.startswith("/wiadomosci/")
        ):
            continue
        records.append(
            PapPeriodicRow(
                company_id=company_id,
                report_number=report_number,
                publication_date=publication_date,
                detail_url=detail_url,
            )
        )
    if report_rows and not records:
        raise ValueError("official PAP periodic table contained no valid rows for the issuer")
    return records


def _unwrap_ajax(payload: str) -> str:
    stripped = payload.lstrip()
    if not stripped.startswith("["):
        return payload
    commands = json.loads(payload)
    if not isinstance(commands, list):
        raise ValueError("PAP Ajax response must be a command list")
    fragments = [
        str(command["data"])
        for command in commands
        if isinstance(command, dict)
        and command.get("command") in {"insert", "replaceWith"}
        and isinstance(command.get("data"), str)
    ]
    if not fragments:
        raise ValueError("PAP Ajax response contains no inserted HTML")
    return "\n".join(fragments)


def _date_text(value: str) -> date:
    match = _DATE_RE.search(value)
    if match is None:
        raise ValueError("PAP day header has no publication date")
    return date(int(match[1]), int(match[2]), int(match[3]))
