from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from data_sources.errors import DataSourceError
from data_sources.security_master import (
    ResolvedSecurityIdentity,
    SecurityMaster,
)
from storage.models import SecurityIdentityRecord

MILAN = "euronext_growth_milan"
MADRID = "bme_growth"
WARSAW = "newconnect"


def _identity(
    *,
    venue: str = MILAN,
    isin: str = "IT0003132476",
    ticker: str = "ALFA.MI",
    mic: str | None = "EXGM",
    symbol: str = "ALFA",
    source: str = "https://example.test/current",
    day: int = 5,
    active: bool = True,
    superseded_by: str | None = None,
) -> SecurityIdentityRecord:
    return SecurityIdentityRecord(
        venue=venue,
        isin=isin,
        canonical_ticker=ticker,
        mic=mic,
        exchange_symbol=symbol,
        legal_name=f"{symbol} S.p.A.",
        identity_source_url=source,
        resolved_at=datetime(2026, 8, day, tzinfo=timezone.utc),
        is_active=active,
        superseded_by_isin=superseded_by,
    )


def test_resolves_active_identity_with_all_strict_identifiers(db_session: Session) -> None:
    db_session.add(_identity())
    db_session.commit()

    result = SecurityMaster(db_session).resolve(
        ticker=" alfa.mi ",
        venue=MILAN,
        isin="it0003132476",
        mic="exgm",
        exchange_symbol="alfa",
    )

    assert isinstance(result, ResolvedSecurityIdentity)
    assert result.canonical_ticker == "ALFA.MI"
    assert result.isin == "IT0003132476"
    assert result.aliases == ()
    assert [(item.canonical_ticker, item.is_active) for item in result.provenance] == [
        ("ALFA.MI", True)
    ]


@pytest.mark.parametrize(
    ("ticker", "venue", "symbol"),
    [
        ("480S.MC", MADRID, "480S"),
        ("4MB.WA", WARSAW, "4MB"),
        ("WAMI28.MI", MILAN, "WAMI28"),
    ],
)
def test_resolves_actual_g12_alphanumeric_ticker_space(
    db_session: Session, ticker: str, venue: str, symbol: str
) -> None:
    db_session.add(_identity(ticker=ticker, venue=venue, symbol=symbol))
    db_session.commit()

    result = SecurityMaster(db_session).resolve(ticker=ticker.lower())

    assert isinstance(result, ResolvedSecurityIdentity)
    assert result.canonical_ticker == ticker


@pytest.mark.parametrize(
    "ticker",
    ["123.MC", "123456.WA", "ABC1234.MI", "A_BC.WA", "ABC!.MI", "480S.L"],
)
def test_rejects_symbols_outside_bounded_g12_canonical_space(
    db_session: Session, ticker: str
) -> None:
    result = SecurityMaster(db_session).resolve(ticker=ticker)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
    assert result.stage == "identity"


def test_superseded_ticker_resolves_as_alias_and_keeps_source_history(
    db_session: Session,
) -> None:
    old = _identity(
        isin="IT0003856405",
        ticker="OLD.MI",
        symbol="OLD",
        source="https://example.test/old-name",
        day=1,
        active=False,
        superseded_by="IT0003132476",
    )
    current = _identity(source="https://example.test/new-name")
    db_session.add_all([old, current])
    db_session.commit()

    result = SecurityMaster(db_session).resolve(ticker="OLD.MI")

    assert isinstance(result, ResolvedSecurityIdentity)
    assert result.canonical_ticker == "ALFA.MI"
    assert result.aliases == ("OLD.MI",)
    assert [item.identity_source_url for item in result.provenance] == [
        "https://example.test/old-name",
        "https://example.test/new-name",
    ]
    assert [item.is_active for item in result.provenance] == [False, True]


def test_historical_identifiers_on_same_alias_path_resolve_current_identity(
    db_session: Session,
) -> None:
    db_session.add_all(
        [
            _identity(
                isin="IT0003856405",
                ticker="OLD.MI",
                mic="XMIL",
                symbol="OLD",
                source="https://example.test/old-name",
                day=1,
                active=False,
                superseded_by="IT0003132476",
            ),
            _identity(source="https://example.test/new-name"),
        ]
    )
    db_session.commit()

    result = SecurityMaster(db_session).resolve(
        ticker="OLD.MI",
        isin="IT0003856405",
        mic="XMIL",
        exchange_symbol="OLD",
    )

    assert isinstance(result, ResolvedSecurityIdentity)
    assert result.canonical_ticker == "ALFA.MI"
    assert result.isin == "IT0003132476"
    assert result.aliases == ("OLD.MI",)
    assert [item.isin for item in result.provenance] == ["IT0003856405", "IT0003132476"]

    contradiction = SecurityMaster(db_session).resolve(
        ticker="OLD.MI",
        isin="IT0003856405",
        exchange_symbol="OTHER",
    )
    assert isinstance(contradiction, DataSourceError)
    assert contradiction.error_code == "not_found"


def test_dual_listing_requires_a_listing_identifier(db_session: Session) -> None:
    db_session.add_all(
        [
            _identity(),
            _identity(
                venue=MADRID,
                ticker="ALFA.MC",
                mic="XESM",
                symbol="ALFA",
            ),
        ]
    )
    db_session.commit()
    master = SecurityMaster(db_session)

    ambiguous = master.resolve(isin="IT0003132476")
    assert isinstance(ambiguous, DataSourceError)
    assert ambiguous.error_code == "parse"
    assert ambiguous.stage == "identity"
    assert ambiguous.source == "security_master"

    madrid = master.resolve(isin="IT0003132476", venue=MADRID)
    milan = master.resolve(isin="IT0003132476", mic="EXGM")
    assert isinstance(madrid, ResolvedSecurityIdentity)
    assert isinstance(milan, ResolvedSecurityIdentity)
    assert madrid.canonical_ticker == "ALFA.MC"
    assert milan.canonical_ticker == "ALFA.MI"


def test_suffix_venue_and_isin_are_validated_before_lookup(db_session: Session) -> None:
    master = SecurityMaster(db_session)

    errors = [
        master.resolve(ticker="ALFA.MI", venue=MADRID),
        master.resolve(ticker="ALFA.L"),
        master.resolve(isin="IT0003132475"),
        master.resolve(mic="XM"),
    ]
    assert all(isinstance(error, DataSourceError) for error in errors)
    assert all(
        error.error_code == "parse" for error in errors if isinstance(error, DataSourceError)
    )
    assert all(error.stage == "identity" for error in errors if isinstance(error, DataSourceError))
    assert all(
        error.source == "security_master" for error in errors if isinstance(error, DataSourceError)
    )


def test_conflicting_valid_identifiers_do_not_fall_back_to_guessing(db_session: Session) -> None:
    db_session.add(_identity())
    db_session.commit()

    result = SecurityMaster(db_session).resolve(ticker="ALFA.MI", exchange_symbol="OTHER")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"
    assert result.stage == "identity"
    assert result.source == "security_master"


def test_malformed_persisted_identity_returns_parse_error(db_session: Session) -> None:
    db_session.add(_identity(ticker="ALFA.MC"))
    db_session.commit()

    result = SecurityMaster(db_session).resolve(isin="IT0003132476", venue=MILAN)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
    assert result.stage == "identity"
    assert result.source == "security_master"


def test_current_lookup_includes_all_historical_provenance(db_session: Session) -> None:
    db_session.add_all(
        [
            _identity(
                isin="IT0003856405",
                ticker="OLD.MI",
                symbol="OLD",
                source="https://example.test/historical",
                day=1,
                active=False,
                superseded_by="IT0003132476",
            ),
            _identity(source="https://example.test/current"),
        ]
    )
    db_session.commit()

    result = SecurityMaster(db_session).resolve(ticker="ALFA.MI")

    assert isinstance(result, ResolvedSecurityIdentity)
    assert result.aliases == ("OLD.MI",)
    assert [(item.isin, item.superseded_by_isin) for item in result.provenance] == [
        ("IT0003856405", "IT0003132476"),
        ("IT0003132476", None),
    ]
