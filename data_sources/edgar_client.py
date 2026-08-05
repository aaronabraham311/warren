import json
import re
import sqlite3
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from data_sources.cache import CacheStore, make_key
from data_sources.errors import DataSourceError, ErrorStage
from data_sources.filing_models import (
    ExtractionMethod,
    SourceSystem,
    stable_filing_id,
)
from data_sources.filing_models import FilingSection as FilingSection

FilingType = Literal["10-K", "10-Q", "8-K", "DEF 14A"]

_EFTS_BASE = "https://efts.sec.gov"
SectionName = Literal[
    "business",
    "risk_factors",
    "mdna",
    "financial_statements",
    "executive_summary",
    "compensation",
    "related_party",
]

MAX_CHARS = 200_000  # approx 50K tokens at 4 chars/token

# Cache TTLs, in hours. Filings change at most quarterly; CIK map is stable.
TTL_HOURS: dict[str, float] = {
    "10-K": 2160.0,
    "10-Q": 2160.0,
    "8-K": 24.0,
    "DEF 14A": 2160.0,
    "SC 13G": 720.0,  # 30 days; 13G/D holders change infrequently
}  # 90d / 90d / 24h / 90d / 30d
CIK_TTL_HOURS = 168.0  # 7 days

# Section → (start Item, end-boundary Items). The first matching end Item after the
# start anchor bounds the slice. executive_summary has no Item anchor (synthesized).
SECTION_BOUNDARIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "business": ("Item 1", ("Item 1A", "Item 2")),
    "risk_factors": ("Item 1A", ("Item 1B", "Item 2")),
    "mdna": ("Item 7", ("Item 7A", "Item 8")),
    "financial_statements": ("Item 8", ("Item 9",)),
    # Part III items — often "incorporated by reference" in 10-K; full detail in DEF 14A.
    "compensation": ("Item 11", ("Item 12",)),
    "related_party": ("Item 13", ("Item 14", "Item 15")),
}
EXEC_SUMMARY_LINES = 400

# DEF 14A proxy statements use prose headings rather than Item numbers.
_DEF14A_SECTION_ANCHORS: dict[str, re.Pattern[str]] = {
    "compensation": re.compile(r"executive\s+compensation", re.IGNORECASE),
    "related_party": re.compile(
        r"certain\s+relationships|related\s+(?:party|person)\s+transactions",
        re.IGNORECASE,
    ),
}
# Candidate next-section headings used to bound a DEF 14A slice.
_DEF14A_END_RE = re.compile(
    r"certain\s+relationships|audit\s+committee|director\s+independence"
    r"|stockholder\s+(?:proposals|information)|security\s+ownership",
    re.IGNORECASE,
)

# Extracts dollar-magnitude amounts ("$10 million", "$2.3B") and bare percentages ("43.8%").
_FIGURE_RE = re.compile(
    r"\$\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|trillion|B|M|T)\b|[\d]+(?:\.\d+)?\s*%",
    re.IGNORECASE,
)


class SC13Holder(BaseModel):
    name: str
    form_type: str  # "SC 13G", "SC 13G/A", "SC 13D", "SC 13D/A"
    filing_date: date


# ── Internal sentinels (mapped to DataSourceError before returning) ───────────


class _NotFoundError(Exception):
    pass


class _NetworkError(Exception):
    pass


class _RateLimitError(Exception):
    pass


class _ParseError(Exception):
    pass


@dataclass
class _SelectedFiling:
    accession: str
    primary_document: str
    filing_date: date
    fiscal_year: int
    url: str


# ── Helpers (avoid Any leaking from json.loads) ───────────────────────────────


def _as_str_list(v: object) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    return []


def _year_of(iso_date: str) -> int | None:
    try:
        return date.fromisoformat(iso_date).year
    except ValueError:
        return None


def _item_regex(item_label: str) -> re.Pattern[str]:
    # "Item 7" → matches "Item 7" / "ITEM 7." but NOT "Item 7A"; "Item 1A" → only "Item 1A".
    num = item_label.split()[-1].lower()
    return re.compile(rf"item\s+{re.escape(num)}(?![0-9a-z])", re.IGNORECASE)


# ── EDGARClient ───────────────────────────────────────────────────────────────


class EDGARClient:
    BASE_URL = "https://data.sec.gov"
    # EDGAR requires a descriptive User-Agent — requests without it may be blocked.
    HEADERS = {"User-Agent": "Warren/1.0 (personal research tool; contact@example.com)"}

    def __init__(
        self,
        db_conn: sqlite3.Connection,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cache = CacheStore(db_conn)
        self._sleep = _sleep
        # Parsed ticker→CIK map, memoized so the ~1MB bulk file is parsed at most
        # once per process (the CacheStore only holds the raw JSON text).
        self._cik_map: dict[str, str] | None = None
        # Default headers on the session guarantee the User-Agent is present on
        # every request — a request without it cannot be constructed.
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)

    # ── Public API ────────────────────────────────────────────────────────

    def get_filing_section(
        self,
        ticker: str,
        filing_type: FilingType,
        section: SectionName,
        fiscal_year: int | None = None,
    ) -> FilingSection | DataSourceError:
        key = make_key("edgar", ticker.upper(), filing_type, section, str(fiscal_year))
        cached = self._cache.get(key)
        if cached is not None:
            return FilingSection.model_validate_json(cached)

        stage: ErrorStage = "identity"
        try:
            cik = self._resolve_cik(ticker)
            stage = "discovery"
            filing = self._select_filing(cik, filing_type, fiscal_year)
            stage = "download"
            html = self._get(filing.url).text
            stage = "extract"
            raw_text = self._extract_section(html, section, filing_type)
        except _NotFoundError as exc:
            return DataSourceError(
                error_code="not_found", message=str(exc), stage=stage, source=SourceSystem.EDGAR
            )
        except _NetworkError as exc:
            return DataSourceError(
                error_code="network", message=str(exc), stage=stage, source=SourceSystem.EDGAR
            )
        except _RateLimitError as exc:
            return DataSourceError(
                error_code="rate_limit",
                message=str(exc),
                stage=stage,
                source=SourceSystem.EDGAR,
            )
        # _ParseError plus the stdlib exceptions raised while interpreting EDGAR
        # responses (a non-JSON 200 body, a malformed/missing field, a bad date).
        # The client's contract is to return DataSourceError, never raise to callers.
        except (_ParseError, json.JSONDecodeError, ValueError, KeyError, AttributeError) as exc:
            return DataSourceError(
                error_code="parse", message=str(exc), stage=stage, source=SourceSystem.EDGAR
            )

        text = raw_text[:MAX_CHARS]
        result = FilingSection(
            ticker=ticker.upper(),
            filing_type=filing_type,
            section=section,
            fiscal_year=filing.fiscal_year,
            filing_date=filing.filing_date,
            text=text,
            word_count=len(text.split()),
            truncated=len(raw_text) > MAX_CHARS,
            source_url=filing.url,
            filing_id=stable_filing_id(SourceSystem.EDGAR, "SEC", ticker, filing.accession),
            venue="SEC",
            source_system=SourceSystem.EDGAR,
            source_language="en",
            extraction_method=ExtractionMethod.HTML,
            key_figures_extracted=_FIGURE_RE.findall(raw_text)[:20],
        )
        self._cache.set(key, result.model_dump_json(), TTL_HOURS[filing_type])
        return result

    def get_sc13_holders(self, ticker: str) -> list[SC13Holder] | DataSourceError:
        """Return 5%+ beneficial owners from recent SC 13G/D filings via EDGAR EFTS."""
        key = make_key("edgar_sc13", ticker.upper())
        cached = self._cache.get(key)
        if cached is not None:
            raw = json.loads(cached)
            if isinstance(raw, list):
                return [SC13Holder.model_validate(h) for h in raw]

        stage: ErrorStage = "identity"
        try:
            cik = self._resolve_cik(ticker)
            stage = "discovery"
            holders = self._fetch_sc13_holders(cik)
        except _NotFoundError as exc:
            return DataSourceError(
                error_code="not_found", message=str(exc), stage=stage, source=SourceSystem.EDGAR
            )
        except _NetworkError as exc:
            return DataSourceError(
                error_code="network", message=str(exc), stage=stage, source=SourceSystem.EDGAR
            )
        except _RateLimitError as exc:
            return DataSourceError(
                error_code="rate_limit",
                message=str(exc),
                stage=stage,
                source=SourceSystem.EDGAR,
            )
        except (_ParseError, json.JSONDecodeError, ValueError, KeyError, AttributeError) as exc:
            return DataSourceError(
                error_code="parse", message=str(exc), stage=stage, source=SourceSystem.EDGAR
            )

        self._cache.set(
            key,
            json.dumps([h.model_dump() for h in holders], default=str),
            TTL_HOURS["SC 13G"],
        )
        return holders

    def _fetch_sc13_holders(self, cik: str) -> list[SC13Holder]:
        """Search EDGAR EFTS for SC 13G/D filings naming this company."""
        submissions_text = self._get(f"{self.BASE_URL}/submissions/CIK{cik}.json").text
        submissions = json.loads(submissions_text)
        if not isinstance(submissions, dict):
            raise _ParseError(f"unexpected submissions shape for CIK{cik}")
        company_name = submissions.get("name")
        if not isinstance(company_name, str) or not company_name:
            raise _ParseError(f"no company name in submissions for CIK{cik}")

        cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        q = urllib.parse.quote(f'"{company_name}"')
        url = (
            f"{_EFTS_BASE}/LATEST/search-index"
            f"?q={q}"
            f"&forms=SC+13G%2CSC+13G%2FA%2CSC+13D%2CSC+13D%2FA"
            f"&dateRange=custom&startdt={cutoff}"
        )
        resp_text = self._get(url).text
        data = json.loads(resp_text)

        if not isinstance(data, dict):
            raise _ParseError("unexpected EFTS response shape")
        hits_obj = data.get("hits", {})
        if not isinstance(hits_obj, dict):
            return []
        hit_list = hits_obj.get("hits", [])
        if not isinstance(hit_list, list):
            return []

        seen: dict[str, SC13Holder] = {}
        for hit in hit_list:
            if not isinstance(hit, dict):
                continue
            source = hit.get("_source", {})
            if not isinstance(source, dict):
                continue
            entity_name = source.get("entity_name")
            form_type = source.get("form_type")
            file_date_str = source.get("file_date")
            if (
                not isinstance(entity_name, str)
                or not entity_name
                or not isinstance(form_type, str)
                or not isinstance(file_date_str, str)
            ):
                continue
            try:
                file_date = date.fromisoformat(file_date_str)
            except ValueError:
                continue
            holder = SC13Holder(name=entity_name, form_type=form_type, filing_date=file_date)
            existing = seen.get(entity_name)
            if existing is None or file_date > existing.filing_date:
                seen[entity_name] = holder

        return list(seen.values())

    # ── HTTP (single choke point; always carries the User-Agent) ───────────

    def _get(self, url: str) -> requests.Response:
        try:
            resp = self._session.get(url, timeout=30)
        except requests.RequestException as exc:
            self._sleep(0.1)
            raise _NetworkError(f"request failed for {url}: {exc}") from exc
        # Stay at ≤10 req/sec per EDGAR fair-use policy. No backoff: errors surface now.
        self._sleep(0.1)
        if resp.status_code == 429:
            raise _RateLimitError(f"HTTP 429 for {url}")
        if resp.status_code != 200:
            raise _NetworkError(f"HTTP {resp.status_code} for {url}")
        return resp

    # ── Step 1: ticker → CIK via the authoritative bulk map ────────────────

    def _resolve_cik(self, ticker: str) -> str:
        # SEC's map spells share classes with a dash: BRK.B → BRK-B.
        want = ticker.upper().replace(".", "-")
        cik = self._cik_map_lookup(want)
        if cik is None:
            raise _NotFoundError(f"ticker {want} not found in EDGAR")
        return cik

    def _cik_map_lookup(self, want: str) -> str | None:
        if self._cik_map is None:
            key = make_key("edgar_cik_map")
            raw = self._cache.get(key)
            if raw is None:
                raw = self._get("https://www.sec.gov/files/company_tickers.json").text
                self._cache.set(key, raw, CIK_TTL_HOURS)
            self._cik_map = self._parse_cik_map(raw)
        return self._cik_map.get(want)

    @staticmethod
    def _parse_cik_map(raw: str) -> dict[str, str]:
        data = json.loads(raw)
        out: dict[str, str] = {}
        if isinstance(data, dict):
            for entry in data.values():
                if not isinstance(entry, dict):
                    continue
                ticker = entry.get("ticker")
                cik = entry.get("cik_str")
                if isinstance(ticker, str) and isinstance(cik, int):
                    out[ticker.upper()] = f"{cik:010d}"
        return out

    # ── Steps 2-3: submissions history → pick the filing ───────────────────

    def _select_filing(
        self, cik: str, filing_type: str, fiscal_year: int | None
    ) -> _SelectedFiling:
        data = json.loads(self._get(f"{self.BASE_URL}/submissions/CIK{cik}.json").text)
        recent = data.get("filings", {}).get("recent", {}) if isinstance(data, dict) else {}
        if not isinstance(recent, dict):
            raise _ParseError(f"unexpected submissions shape for CIK{cik}")

        forms = _as_str_list(recent.get("form"))
        filing_dates = _as_str_list(recent.get("filingDate"))
        accessions = _as_str_list(recent.get("accessionNumber"))
        documents = _as_str_list(recent.get("primaryDocument"))
        report_dates = _as_str_list(recent.get("reportDate"))

        # Arrays are parallel and newest-first; the first match is the most recent.
        for i, form in enumerate(forms):
            if form != filing_type:
                continue
            report_iso = report_dates[i] if i < len(report_dates) else ""
            filing_iso = filing_dates[i] if i < len(filing_dates) else ""
            fy = _year_of(report_iso) or _year_of(filing_iso)
            if fy is None:
                continue
            if fiscal_year is not None and fy != fiscal_year:
                continue
            accession = accessions[i]
            primary = documents[i] if i < len(documents) else ""
            if not accession or not primary:
                continue
            url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession.replace('-', '')}/{primary}"
            )
            return _SelectedFiling(
                accession=accession,
                primary_document=primary,
                filing_date=date.fromisoformat(filing_iso),
                fiscal_year=fy,
                url=url,
            )

        which = f" for fiscal year {fiscal_year}" if fiscal_year is not None else ""
        raise _NotFoundError(f"no {filing_type} filing found{which}")

    # ── Steps 4-5: parse HTML, extract the requested section ───────────────

    def _extract_section(self, html: str, section: str, filing_type: str = "") -> str:
        try:
            text = BeautifulSoup(html, "lxml").get_text("\n")
        except Exception as exc:  # noqa: BLE001 — bs4 may raise varied parser errors
            raise _ParseError(f"failed to parse filing HTML: {exc}") from exc

        if section == "executive_summary":
            lines = [ln for ln in text.splitlines() if ln.strip()]
            return "\n".join(lines[:EXEC_SUMMARY_LINES])

        # DEF 14A proxy statements use prose headings rather than Item numbers.
        if filing_type == "DEF 14A" and section in _DEF14A_SECTION_ANCHORS:
            anchor = _DEF14A_SECTION_ANCHORS[section]
            m_start = anchor.search(text)
            if not m_start:
                return text.strip()
            start_pos = m_start.start()
            end_pos = len(text)
            for m_end in _DEF14A_END_RE.finditer(text, m_start.end()):
                if m_end.start() > start_pos:
                    end_pos = m_end.start()
                    break
            return text[start_pos:end_pos].strip()

        start_label, end_labels = SECTION_BOUNDARIES[section]
        starts = list(_item_regex(start_label).finditer(text))
        if not starts:
            # Header not located — return the whole document rather than nothing.
            return text.strip()

        # Last occurrence skips the table-of-contents reference near the top.
        start = starts[-1]
        end_pos = len(text)
        for end_label in end_labels:
            m = _item_regex(end_label).search(text, start.end())
            if m is not None:
                end_pos = min(end_pos, m.start())
        return text[start.start() : end_pos].strip()
