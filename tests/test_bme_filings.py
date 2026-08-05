import json
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import cast

from sqlalchemy.orm import Session

from data_sources.bme_client import BMEGrowthFilingsSource
from data_sources.errors import DataSourceError
from data_sources.filing_models import (
    DocumentKind,
    FilingsArchive,
    FilingsSource,
    IssuerIdentity,
    SourceSystem,
    stable_filing_id,
)
from data_sources.regional_http import HttpDocument, RegionalHttpClient
from data_sources.security_master import SecurityMaster
from storage.models import SecurityIdentityRecord

FIXTURES = Path(__file__).parents[1] / "eval" / "fixtures" / "BMEGROWTH" / "bme" / "filings"
FETCHED_AT = datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc)
ISIN = "ES0105509006"


class _FixtureHttp:
    def __init__(self, responses: dict[str, list[str | DataSourceError]]) -> None:
        self.responses = {url: list(items) for url, items in responses.items()}
        self.calls: list[tuple[str, Mapping[str, str] | None]] = []

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError:
        del cache_ttl_hours
        self.calls.append((url, params))
        result = self.responses[url].pop(0)
        if isinstance(result, DataSourceError):
            return result
        return HttpDocument(
            url=url,
            text=result,
            fetched_at=FETCHED_AT,
            etag='"fixture"',
            last_modified="Wed, 05 Aug 2026 14:00:00 GMT",
            status_code=200,
        )

    @staticmethod
    def validate_url(url: str) -> DataSourceError | None:
        if url.startswith("https://apiweb.bolsasymercados.es/Market/"):
            return None
        return DataSourceError(
            error_code="parse",
            message=f"unofficial BME URL: {url}",
            stage="discovery",
            source="bme",
        )


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def _responses() -> dict[str, list[str | DataSourceError]]:
    return {
        BMEGrowthFilingsSource.DETAILS_URL: [_fixture("share_details.json")],
        BMEGrowthFilingsSource.DOCUMENTS_URL: [
            _fixture("documents_page_0.json"),
            _fixture("documents_page_1.json"),
        ],
        BMEGrowthFilingsSource.FINANCIAL_URL: [_fixture("financial_page_0.json")],
    }


def _identity() -> IssuerIdentity:
    return IssuerIdentity(
        canonical_ticker="480S.MC",
        venue="bme_growth",
        isin=ISIN,
        legal_name="untrusted caller name",
    )


def _master(db_session: Session) -> SecurityMaster:
    db_session.add(
        SecurityIdentityRecord(
            venue="bme_growth",
            isin=ISIN,
            canonical_ticker="480S.MC",
            mic="XGRO",
            exchange_symbol="480S",
            legal_name="SOLUCIONES CUATROOCHENTA, S.A.",
            identity_source_url=(
                "https://www.bolsasymercados.es/es/bme-growth/cotizaciones-e-indices/"
                "empresas-cotizadas/ficha.cuatroochenta-es0105509006.html"
            ),
            resolved_at=FETCHED_AT,
            is_active=True,
            superseded_by_isin=None,
        )
    )
    db_session.flush()
    return SecurityMaster(db_session)


def _source(
    db_session: Session,
    responses: dict[str, list[str | DataSourceError]] | None = None,
) -> tuple[BMEGrowthFilingsSource, _FixtureHttp]:
    http = _FixtureHttp(responses or _responses())
    source = BMEGrowthFilingsSource(
        http=cast(RegionalHttpClient, http), security_master=_master(db_session), page_size=2
    )
    contract: FilingsSource = source
    assert contract is source
    return source, http


def test_bme_uses_share_details_then_merges_and_deduplicates_official_archives(
    db_session: Session,
) -> None:
    source, http = _source(db_session)

    result = source.list_filings(_identity())

    assert isinstance(result, FilingsArchive)
    assert result.issuer.legal_name == "SOLUCIONES CUATROOCHENTA, S.A."
    assert result.pages_exhausted is True
    assert result.coverage_start == date(2025, 12, 19)
    assert result.coverage_end == date(2026, 4, 30)
    assert [item.upstream_id for item in result.filings] == ["118463", "115936", "115930"]
    assert result.filings[0].document_kind is DocumentKind.ANNUAL
    assert result.filings[0].direct_document_url == (
        "https://apiweb.bolsasymercados.es/Market/MTFDocuments/OtraInfRelevante/"
        "2026/04/05509_OtraInfRelev_20260429_1.pdf"
    )
    correction = result.filings[1]
    original = result.filings[2]
    assert correction.amended is True
    assert correction.supersedes_filing_id == original.filing_id
    assert correction.filing_id == stable_filing_id(SourceSystem.BME, "bme_growth", ISIN, "115936")
    assert correction.etag == '"fixture"'
    assert [call[0] for call in http.calls] == [
        source.DETAILS_URL,
        source.DOCUMENTS_URL,
        source.DOCUMENTS_URL,
        source.FINANCIAL_URL,
    ]
    assert http.calls[0][1] == {
        "tradingSystem": "MTF",
        "mtfSegment": "BMEGrowth",
        "ISIN": ISIN,
    }
    assert http.calls[1][1] == {
        "companyKey": "05509",
        "documentTypes": (
            "InsideInformation,OtherRelevantInformation,Prospectuses,Notices,RelevantFacts"
        ),
        "language": "es",
        "from": "19900101",
        "to": date.today().strftime("%Y%m%d"),
        "page": "0",
        "pagesize": "2",
    }
    assert http.calls[2][1] == {**http.calls[1][1], "page": "1"}
    assert http.calls[3][1] == {
        "companyKey": "05509",
        "language": "es",
        "mtfSegment": "BMEGrowth",
        "order": "Period DESC",
        "fromYear": "1990",
        "toYear": str(date.today().year),
        "page": "0",
        "pagesize": "2",
    }


def test_bme_annex_and_rectification_keep_provenance(db_session: Session) -> None:
    responses = _responses()
    documents_payload = responses[BMEGrowthFilingsSource.DOCUMENTS_URL][0]
    assert isinstance(documents_payload, str)
    payload = json.loads(documents_payload)
    record = payload["data"][0]
    record["annexes"] = [
        {
            "id": "118464",
            "title": "Informe de auditoría",
            "url": "/MTFDocuments/OtraInfRelevante/2026/04/05509_Auditoria_20260429.pdf",
        }
    ]
    record["rectifies"] = {"id": "118400"}
    responses[BMEGrowthFilingsSource.DOCUMENTS_URL][0] = json.dumps(payload)
    source, _ = _source(db_session, responses)

    result = source.list_filings(_identity())

    assert isinstance(result, FilingsArchive)
    filing = result.filings[0]
    assert filing.attachment_names == ["Informe de auditoría [BME 118464]"]
    assert filing.amended is True
    assert filing.supersedes_filing_id == stable_filing_id(
        SourceSystem.BME, "bme_growth", ISIN, "118400"
    )


def test_bme_filters_then_reports_honest_derived_coverage(db_session: Session) -> None:
    source, _ = _source(db_session)

    result = source.list_filings(
        _identity(),
        kinds=[DocumentKind.OTHER_RELEVANT],
        from_date=date(2025, 1, 1),
        to_date=date(2025, 12, 31),
        limit=1,
    )

    assert isinstance(result, FilingsArchive)
    assert [item.upstream_id for item in result.filings] == ["115936"]
    assert result.coverage_start == result.coverage_end == date(2025, 12, 19)
    assert result.pages_exhausted is False
    assert result.warnings == [
        "BME does not publish an archive completeness boundary; coverage bounds reflect "
        "returned filings only.",
        "BME results were truncated at the requested limit; additional filings exist.",
    ]


def test_bme_requires_exact_persisted_identity_before_http(db_session: Session) -> None:
    source, http = _source(db_session)
    wrong = _identity().model_copy(update={"canonical_ticker": "LAB.MC"})

    result = source.list_filings(wrong)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"
    assert result.stage == "identity"
    assert result.source == "security_master"
    assert http.calls == []


def test_bme_rejects_share_details_identity_drift(db_session: Session) -> None:
    responses = _responses()
    details_payload = responses[BMEGrowthFilingsSource.DETAILS_URL][0]
    assert isinstance(details_payload, str)
    details = json.loads(details_payload)
    details["ticker"] = "LAB"
    responses[BMEGrowthFilingsSource.DETAILS_URL][0] = json.dumps(details)
    source, http = _source(db_session, responses)

    result = source.list_filings(_identity())

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
    assert result.stage == "identity"
    assert len(http.calls) == 1


def test_bme_returns_typed_archive_errors(db_session: Session) -> None:
    responses = _responses()
    responses[BMEGrowthFilingsSource.DOCUMENTS_URL] = [
        DataSourceError(error_code="rate_limit", message="slow down")
    ]
    source, _ = _source(db_session, responses)

    result = source.list_filings(_identity())

    assert isinstance(result, DataSourceError)
    assert result.error_code == "rate_limit"
    assert result.stage == "discovery"
    assert result.source == "bme"


def test_bme_rejects_archive_and_record_identity_drift(db_session: Session) -> None:
    for target in ("params", "record_company", "record_segment"):
        db_session.rollback()
        responses = _responses()
        raw = responses[BMEGrowthFilingsSource.DOCUMENTS_URL][0]
        assert isinstance(raw, str)
        payload = json.loads(raw)
        if target == "params":
            payload["params"]["companyKey"] = "EVIL"
        elif target == "record_company":
            payload["data"][0]["companyKey"] = "EVIL"
        else:
            payload["data"][0]["segment"] = "SIBE"
        responses[BMEGrowthFilingsSource.DOCUMENTS_URL][0] = json.dumps(payload)
        source, _ = _source(db_session, responses)

        result = source.list_filings(_identity())

        assert isinstance(result, DataSourceError)
        assert result.error_code == "parse"
        assert result.stage == "discovery"


def test_bme_rejects_never_ending_or_excessive_pagination(db_session: Session) -> None:
    responses = _responses()
    raw = responses[BMEGrowthFilingsSource.DOCUMENTS_URL][1]
    assert isinstance(raw, str)
    payload = json.loads(raw)
    payload["hasMoreResults"] = True
    responses[BMEGrowthFilingsSource.DOCUMENTS_URL][1] = json.dumps(payload)
    source, _ = _source(db_session, responses)

    result = source.list_filings(_identity())

    assert isinstance(result, DataSourceError)
    assert "contradicts" in result.message

    db_session.rollback()
    responses = _responses()
    raw = responses[BMEGrowthFilingsSource.DOCUMENTS_URL][0]
    assert isinstance(raw, str)
    payload = json.loads(raw)
    payload["totalResults"] = 10_000
    responses[BMEGrowthFilingsSource.DOCUMENTS_URL][0] = json.dumps(payload)
    source, _ = _source(db_session, responses)
    result = source.list_filings(_identity())
    assert isinstance(result, DataSourceError)
    assert "hard page limit" in result.message


def test_bme_rejects_non_document_archive_relative_path(db_session: Session) -> None:
    responses = _responses()
    raw = responses[BMEGrowthFilingsSource.DOCUMENTS_URL][0]
    assert isinstance(raw, str)
    payload = json.loads(raw)
    payload["data"][0]["url"] = "/admin/private"
    responses[BMEGrowthFilingsSource.DOCUMENTS_URL][0] = json.dumps(payload)
    source, _ = _source(db_session, responses)

    result = source.list_filings(_identity())

    assert isinstance(result, DataSourceError)
    assert "/Market/MTFDocuments/" in result.message
