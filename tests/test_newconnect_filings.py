"""Fail-closed contracts for the Biznes PAP NewConnect boundary."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from data_sources.errors import DataSourceError
from data_sources.filing_models import FilingsSource, IssuerIdentity
from data_sources.newconnect_filings import (
    NewConnectFilingsSource,
    parse_pap_periodic_rows,
)
from data_sources.regional_http import HttpDocument
from data_sources.security_master import SecurityMaster
from storage.models import SecurityIdentityRecord

FIXTURES = Path(__file__).parents[1] / "eval" / "fixtures" / "NEWCONNECT" / "pap"
FETCHED_AT = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)
ISIN = "PLCRFRG00016"


class _FixtureHttp:
    def __init__(self, *, waf: bool = False) -> None:
        self.calls: list[tuple[str, Mapping[str, str] | None]] = []
        self.waf = waf

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        cache_ttl_hours: float = 6.0,
    ) -> HttpDocument | DataSourceError:
        del cache_ttl_hours
        self.calls.append((url, params))
        text = (
            "<script src='/_Incapsula_Resource?x=1'></script>"
            if self.waf
            else (FIXTURES / "observed_2023_03_16.ajax.json").read_text()
        )
        return HttpDocument(
            url=url,
            text=text,
            fetched_at=FETCHED_AT,
            etag='"pap-v1"',
            last_modified="Wed, 05 Aug 2026 15:00:00 GMT",
            status_code=200,
        )


def _identity(db_session: Session) -> IssuerIdentity:
    db_session.add(
        SecurityIdentityRecord(
            venue="newconnect",
            isin=ISIN,
            canonical_ticker="CFG.WA",
            mic="XNEW",
            exchange_symbol="CFG",
            legal_name="CreativeForge Games SA",
            identity_source_url="https://newconnect.pl/company/cfg",
            resolved_at=FETCHED_AT,
            is_active=True,
        )
    )
    db_session.commit()
    return IssuerIdentity(
        canonical_ticker="CFG.WA",
        venue="newconnect",
        isin=ISIN,
        legal_name="CreativeForge Games SA",
    )


def _source(db_session: Session, http: _FixtureHttp) -> NewConnectFilingsSource:
    source = NewConnectFilingsSource(
        http=http,  # type: ignore[arg-type]
        security_master=SecurityMaster(db_session),
    )
    contract: FilingsSource = source
    assert contract is source
    return source


def test_incapsula_challenge_is_explicit_typed_network_error(db_session: Session) -> None:
    identity = _identity(db_session)
    source = _source(db_session, _FixtureHttp(waf=True))

    result = source.list_filings(identity)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"
    assert result.stage == "discovery"
    assert result.source == "biznes_pap"


def test_non_waf_content_fails_closed_without_verified_channel_contract(
    db_session: Session,
) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp()
    source = _source(db_session, http)

    result = source.list_filings(identity, to_date=date(2023, 3, 16))

    assert isinstance(result, DataSourceError)
    assert result.error_code == "stale_data"
    assert "EBI/ESPI channel" in result.message
    assert "browser-capable verified transport" in result.message
    assert len(http.calls) == 1
    assert http.calls[0][0].endswith("/articles/periodic/2023/3/16")


def test_observed_periodic_row_parses_only_neutral_raw_fields() -> None:
    payload = (FIXTURES / "observed_2023_03_16.ajax.json").read_text()

    rows = parse_pap_periodic_rows(payload, company_id="1490")

    assert len(rows) == 1
    assert rows[0].company_id == "1490"
    assert rows[0].report_number == "3/2023"
    assert rows[0].publication_date == date(2023, 3, 16)
    assert rows[0].detail_url == (
        "https://biznes.pap.pl/wiadomosci/firmy/creativeforge-games-sa-32023-"
        "skonsolidowany-raport-okresowy-spolki-za-2022-rok"
    )
    assert not hasattr(rows[0], "source_system")
    assert not hasattr(rows[0], "filing_id")


def test_unknown_provider_id_returns_identity_error_without_http(
    db_session: Session,
) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp()
    source = NewConnectFilingsSource(
        http=http,  # type: ignore[arg-type]
        security_master=SecurityMaster(db_session),
        company_ids={},
    )

    result = source.list_filings(identity)

    assert isinstance(result, DataSourceError)
    assert result.stage == "identity"
    assert "PAP company ID" in result.message
    assert http.calls == []


def test_rejects_invalid_date_range_without_http(db_session: Session) -> None:
    identity = _identity(db_session)
    http = _FixtureHttp()
    source = _source(db_session, http)

    result = source.list_filings(
        identity,
        from_date=date(2026, 2, 1),
        to_date=date(2026, 1, 1),
    )

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
    assert http.calls == []
