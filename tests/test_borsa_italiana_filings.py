"""Offline replay of Borsa Italiana's real issuer-documents HTML shape."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlparse

from sqlalchemy.orm import Session

from data_sources.borsa_italiana_filings import (
    DEFAULT_ARCHIVE_URL,
    BorsaItalianaFilingsSource,
)
from data_sources.errors import DataSourceError
from data_sources.filing_models import (
    DocumentKind,
    FilingsArchive,
    IssuerIdentity,
    SourceSystem,
    stable_filing_id,
)
from data_sources.regional_http import HttpDocument
from data_sources.security_master import SecurityMaster
from storage.models import SecurityIdentityRecord

FIXTURES = Path(__file__).parents[1] / "eval" / "fixtures" / "EXGM" / "borsa_italiana"
FETCHED_AT = datetime(2026, 8, 5, 14, 30, tzinfo=timezone.utc)


class _FixtureHttp:
    def __init__(self, names: list[str]) -> None:
        self._names = iter(names)
        self.calls: list[tuple[str, Mapping[str, str] | None]] = []

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError:
        self.calls.append((url, params))
        name = next(self._names)
        request_url = f"{url}?{urlencode(params or {})}"
        return HttpDocument(
            url=request_url,
            text=(FIXTURES / name).read_text(),
            fetched_at=FETCHED_AT,
            etag='"official-html-v1"',
            last_modified="Wed, 05 Aug 2026 14:30:00 GMT",
            status_code=200,
        )

    def validate_url(self, url: str) -> DataSourceError | None:
        parsed = urlparse(url)
        if parsed.scheme == "https" and parsed.hostname == "www.borsaitaliana.it":
            return None
        return DataSourceError(
            error_code="parse",
            message=f"refusing non-official or non-HTTPS archive URL: {url}",
            stage="discovery",
            source=SourceSystem.BORSA_ITALIANA,
        )


class _ErrorHttp(_FixtureHttp):
    def __init__(self) -> None:
        super().__init__([])

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError:
        return DataSourceError(
            error_code="rate_limit",
            message="slow down",
            stage="discovery",
            source=SourceSystem.BORSA_ITALIANA,
        )


def _identity(db_session: Session) -> IssuerIdentity:
    db_session.add(
        SecurityIdentityRecord(
            venue="euronext_growth_milan",
            isin="IT0001463063",
            canonical_ticker="D.MI",
            mic="EXGM",
            exchange_symbol="D",
            legal_name="DIRECTA SIM",
            identity_source_url="https://live.euronext.com/en/pd_es/data/stocks",
            resolved_at=FETCHED_AT,
            is_active=True,
            superseded_by_isin=None,
        )
    )
    db_session.commit()
    return IssuerIdentity(
        canonical_ticker="D.MI",
        venue="euronext_growth_milan",
        isin="IT0001463063",
        legal_name="DIRECTA SIM",
    )


def _source(db_session: Session, http: _FixtureHttp) -> BorsaItalianaFilingsSource:
    return BorsaItalianaFilingsSource(
        http=http,
        security_master=SecurityMaster(db_session),
    )


def test_parses_recorded_official_html_and_follows_next_page(db_session: Session) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp(["documents_page_1.html", "documents_page_2.html"])

    result = _source(db_session, http).list_filings(identity)

    assert isinstance(result, FilingsArchive)
    assert result.pages_exhausted is True
    assert [item.upstream_id for item in result.filings] == ["1047904", "1046443", "1039001"]
    annual = result.filings[0]
    assert annual.title == "Fascicolo bilancio consolidato 2025"
    assert annual.document_kind is DocumentKind.ANNUAL
    assert annual.direct_document_url == (
        "https://www.borsaitaliana.it/documenti/documenti.htm?"
        "filename=/media/borsa/db/pdf/new/1047904.pdf"
    )
    assert annual.attachment_names == ["1047904.pdf"]
    assert annual.filing_id == stable_filing_id(
        SourceSystem.BORSA_ITALIANA,
        identity.venue,
        identity.isin or "",
        "1047904",
    )
    assert http.calls == [
        (DEFAULT_ARCHIVE_URL, {"isin": "IT0001463063", "page": "1"}),
        (DEFAULT_ARCHIVE_URL, {"isin": "IT0001463063", "page": "2"}),
    ]
    assert result.coverage_start == date(2024, 6, 19)
    assert result.coverage_end == date(2026, 4, 13)
    assert any("inferred" in warning for warning in result.warnings)
    assert any("no structured attachment" in warning for warning in result.warnings)


def test_filters_real_html_kinds_and_dates_across_pages(db_session: Session) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp(["documents_page_1.html", "documents_page_2.html"])

    result = _source(db_session, http).list_filings(
        identity,
        kinds=[DocumentKind.HALF_YEAR],
        from_date=date(2025, 1, 1),
        to_date=date(2025, 12, 31),
    )

    assert isinstance(result, FilingsArchive)
    assert [item.upstream_id for item in result.filings] == ["1046443"]
    assert len(http.calls) == 2


def test_limit_stops_html_pagination_and_marks_archive_non_exhaustive(
    db_session: Session,
) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp(["documents_page_1.html"])

    result = _source(db_session, http).list_filings(identity, limit=1)

    assert isinstance(result, FilingsArchive)
    assert result.pages_exhausted is False
    assert any("truncated" in warning.lower() for warning in result.warnings)


def test_strict_security_master_identity_precedes_http(db_session: Session) -> None:
    _identity(db_session)
    http = _FixtureHttp(["documents_page_1.html"])

    result = _source(db_session, http).list_filings(
        IssuerIdentity(
            canonical_ticker="FAKE.MI",
            venue="euronext_growth_milan",
            isin="IT0001463063",
        )
    )

    assert isinstance(result, DataSourceError)
    assert result.stage == "identity"
    assert http.calls == []


def test_shared_transport_error_passes_through(db_session: Session) -> None:
    identity = _identity(db_session)

    result = _source(db_session, _ErrorHttp()).list_filings(identity)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "rate_limit"


def test_missing_official_table_is_parse_error(db_session: Session) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp(["documents_page_1.html"])
    original = http.get_text

    def without_table(
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError:
        document = original(url, params=params, cache_ttl_hours=cache_ttl_hours)
        assert isinstance(document, HttpDocument)
        return HttpDocument(**{**document.__dict__, "text": "<html><body></body></html>"})

    http.get_text = without_table  # type: ignore[method-assign]

    result = _source(db_session, http).list_filings(identity)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
    assert "table not found" in result.message


def test_tampered_pdf_host_is_rejected(db_session: Session) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp(["documents_page_2.html"])
    original = http.get_text

    def tampered(
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError:
        document = original(url, params=params, cache_ttl_hours=cache_ttl_hours)
        assert isinstance(document, HttpDocument)
        return HttpDocument(
            **{
                **document.__dict__,
                "text": document.text.replace(
                    "/documenti/documenti.htm?filename=", "https://evil.example/?filename="
                ),
            }
        )

    http.get_text = tampered  # type: ignore[method-assign]

    result = _source(db_session, http).list_filings(identity)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
    assert "none were valid" in result.message


def test_correction_without_target_is_not_falsely_linked(db_session: Session) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp(["documents_page_2.html"])
    original = http.get_text

    def correction(
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError:
        document = original(url, params=params, cache_ttl_hours=cache_ttl_hours)
        assert isinstance(document, HttpDocument)
        return HttpDocument(
            **{**document.__dict__, "text": document.text.replace("Statuto", "Rettifica statuto")}
        )

    http.get_text = correction  # type: ignore[method-assign]

    result = _source(db_session, http).list_filings(identity)

    assert isinstance(result, FilingsArchive)
    assert len(result.filings) == 1
    assert result.filings[0].title == "Rettifica statuto sociale"
    assert result.filings[0].amended is False
    assert result.filings[0].supersedes_filing_id is None
    assert any("no structured attachment or supersession" in warning for warning in result.warnings)


def test_real_title_classification_prioritizes_auditor_over_annual() -> None:
    assert (
        BorsaItalianaFilingsSource._document_kind(
            "Relazione Società di Revisione KPMG sul bilancio consolidato 2025"
        )
        is DocumentKind.AUDITOR
    )
    assert (
        BorsaItalianaFilingsSource._document_kind("RELAZIONE FINANZIARIA ANNUALE")
        is DocumentKind.ANNUAL
    )


def test_exact_limit_on_final_page_is_complete_without_truncation_warning(
    db_session: Session,
) -> None:
    identity = _identity(db_session)
    result = _source(db_session, _FixtureHttp(["documents_page_2.html"])).list_filings(
        identity, limit=1
    )

    assert isinstance(result, FilingsArchive)
    assert result.pages_exhausted is True
    assert not any("truncated" in warning.lower() for warning in result.warnings)


def test_rejects_pagination_beyond_hard_ceiling(db_session: Session) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp(["documents_page_1.html"])
    original = http.get_text

    def excessive_page(
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError:
        document = original(url, params=params, cache_ttl_hours=cache_ttl_hours)
        assert isinstance(document, HttpDocument)
        return HttpDocument(
            **{
                **document.__dict__,
                "text": document.text.replace("page=2", "page=101"),
            }
        )

    http.get_text = excessive_page  # type: ignore[method-assign]
    result = _source(db_session, http).list_filings(identity)

    assert isinstance(result, DataSourceError)
    assert "hard page limit" in result.message
