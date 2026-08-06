"""Strict, offline resolution of persisted junior-market security identities.

The security master is deliberately a database lookup, not a best-effort symbol
guesser.  Callers must handle missing and ambiguous identities explicitly before
routing a filing request to an exchange adapter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from data_sources.errors import DataSourceError
from data_sources.symbols import TICKER_PATTERN, canonical_symbol
from storage.models import SecurityIdentityRecord

VENUE_SUFFIXES: dict[str, str] = {
    "euronext_growth_milan": "MI",
    "bme_growth": "MC",
    "newconnect": "WA",
}


class SecurityMasterError(ValueError):
    """Base error for strict security-master resolution."""


class InvalidSecurityIdentifier(SecurityMasterError):
    """An identifier is malformed or contradicts another supplied identifier."""


class SecurityIdentityNotFound(SecurityMasterError):
    """No active persisted identity satisfies the supplied identifiers."""


class AmbiguousSecurityIdentity(SecurityMasterError):
    """More than one active listing satisfies the supplied identifiers."""


@dataclass(frozen=True)
class IdentityProvenance:
    """Immutable source trail for one current or superseded persisted mapping."""

    venue: str
    isin: str
    canonical_ticker: str
    identity_source_url: str
    resolved_at: datetime
    is_active: bool
    superseded_by_isin: str | None


@dataclass(frozen=True)
class ResolvedSecurityIdentity:
    """An unambiguous active listing plus the history that led to it."""

    venue: str
    isin: str
    canonical_ticker: str
    mic: str | None
    exchange_symbol: str
    legal_name: str
    identity_source_url: str
    resolved_at: datetime
    aliases: tuple[str, ...]
    provenance: tuple[IdentityProvenance, ...]


def is_valid_isin(value: str) -> bool:
    """Return whether ``value`` has valid ISO 6166 shape and Luhn check digit."""
    isin = value.strip().upper()
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", isin) is None:
        return False
    digits = "".join(
        str(ord(character) - 55) if character.isalpha() else character for character in isin
    )
    total = 0
    for index, character in enumerate(reversed(digits)):
        number = int(character)
        if index % 2 == 1:
            number *= 2
        total += number // 10 + number % 10
    return total % 10 == 0


def _normalise_venue(venue: str | None) -> str | None:
    if venue is None:
        return None
    result = venue.strip().lower()
    if result not in VENUE_SUFFIXES:
        raise InvalidSecurityIdentifier(f"unsupported security-master venue: {venue!r}")
    return result


def _normalise_ticker(ticker: str | None, venue: str | None) -> tuple[str | None, str | None]:
    if ticker is None:
        return None, venue
    result = canonical_symbol(ticker)
    if re.fullmatch(TICKER_PATTERN, result) is None:
        raise InvalidSecurityIdentifier(f"invalid canonical ticker: {ticker!r}")
    suffix = result.rpartition(".")[2] if "." in result else None
    if suffix is not None:
        matching_venues = [key for key, expected in VENUE_SUFFIXES.items() if suffix == expected]
        if not matching_venues:
            raise InvalidSecurityIdentifier(f"unsupported exchange suffix in ticker: {ticker!r}")
        inferred_venue = matching_venues[0]
        if venue is not None and venue != inferred_venue:
            raise InvalidSecurityIdentifier(
                f"ticker {result!r} belongs to {inferred_venue!r}, not {venue!r}"
            )
        venue = inferred_venue
    return result, venue


def _provenance(row: SecurityIdentityRecord) -> IdentityProvenance:
    return IdentityProvenance(
        venue=row.venue,
        isin=row.isin,
        canonical_ticker=row.canonical_ticker,
        identity_source_url=row.identity_source_url,
        resolved_at=row.resolved_at,
        is_active=row.is_active,
        superseded_by_isin=row.superseded_by_isin,
    )


class SecurityMaster:
    """Resolve a persisted identity without network calls or fuzzy matching."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def resolve(
        self,
        *,
        ticker: str | None = None,
        venue: str | None = None,
        isin: str | None = None,
        mic: str | None = None,
        exchange_symbol: str | None = None,
    ) -> ResolvedSecurityIdentity | DataSourceError:
        """Resolve one identity, returning typed data-source failures at the boundary."""
        try:
            return self._resolve_or_raise(
                ticker=ticker,
                venue=venue,
                isin=isin,
                mic=mic,
                exchange_symbol=exchange_symbol,
            )
        except SecurityIdentityNotFound as exc:
            return DataSourceError(
                error_code="not_found",
                message=str(exc),
                stage="identity",
                source="security_master",
            )
        except (InvalidSecurityIdentifier, AmbiguousSecurityIdentity) as exc:
            return DataSourceError(
                error_code="parse",
                message=str(exc),
                stage="identity",
                source="security_master",
            )

    def _resolve_or_raise(
        self,
        *,
        ticker: str | None = None,
        venue: str | None = None,
        isin: str | None = None,
        mic: str | None = None,
        exchange_symbol: str | None = None,
    ) -> ResolvedSecurityIdentity:
        """Return exactly one active identity matching all supplied identifiers.

        A superseded ticker may resolve as a historical alias when its persisted
        row points to the current ISIN.  A bare ISIN shared by dual listings is
        intentionally ambiguous; callers must add a venue, ticker, or MIC.
        """
        normal_venue = _normalise_venue(venue)
        normal_ticker, normal_venue = _normalise_ticker(ticker, normal_venue)
        normal_isin = isin.strip().upper() if isin is not None else None
        if normal_isin is not None and not is_valid_isin(normal_isin):
            raise InvalidSecurityIdentifier(f"invalid ISIN: {isin!r}")
        normal_mic = mic.strip().upper() if mic is not None else None
        if normal_mic is not None and re.fullmatch(r"[A-Z0-9]{4}", normal_mic) is None:
            raise InvalidSecurityIdentifier(f"invalid MIC: {mic!r}")
        normal_symbol = exchange_symbol.strip().upper() if exchange_symbol is not None else None
        if normal_symbol == "":
            raise InvalidSecurityIdentifier("exchange_symbol cannot be empty")
        if not any((normal_ticker, normal_venue, normal_isin, normal_mic, normal_symbol)):
            raise InvalidSecurityIdentifier("at least one security identifier is required")

        rows = list(self._session.scalars(select(SecurityIdentityRecord)))
        active_rows = [row for row in rows if row.is_active]
        paths: dict[tuple[str, str], tuple[SecurityIdentityRecord, ...]] = {}

        for row in active_rows:
            if self._matches(
                row,
                ticker=normal_ticker,
                venue=normal_venue,
                isin=normal_isin,
                mic=normal_mic,
                exchange_symbol=normal_symbol,
            ):
                paths[(row.venue, row.isin)] = self._history_for(row, rows)

        if normal_ticker is not None:
            for alias in rows:
                if alias.is_active or alias.canonical_ticker != normal_ticker:
                    continue
                target, alias_path = self._follow_alias(alias, rows)
                if target is None:
                    continue
                if self._path_matches(
                    alias_path,
                    venue=normal_venue,
                    isin=normal_isin,
                    mic=normal_mic,
                    exchange_symbol=normal_symbol,
                ):
                    history = self._history_for(target, rows)
                    combined = {item for item in (*history, *alias_path)}
                    paths[(target.venue, target.isin)] = tuple(
                        sorted(combined, key=lambda item: item.resolved_at)
                    )

        if not paths:
            raise SecurityIdentityNotFound("no active security identity matched all identifiers")
        if len(paths) > 1:
            matches = ", ".join(f"{venue}/{item_isin}" for venue, item_isin in sorted(paths))
            raise AmbiguousSecurityIdentity(
                f"security identifiers matched multiple listings: {matches}"
            )

        ((key, history),) = paths.items()
        current = next(row for row in active_rows if (row.venue, row.isin) == key)
        self._validate_persisted_identity(current)
        aliases = tuple(
            sorted(
                {
                    row.canonical_ticker
                    for row in history
                    if row.canonical_ticker != current.canonical_ticker
                }
            )
        )
        return ResolvedSecurityIdentity(
            venue=current.venue,
            isin=current.isin,
            canonical_ticker=current.canonical_ticker,
            mic=current.mic,
            exchange_symbol=current.exchange_symbol,
            legal_name=current.legal_name,
            identity_source_url=current.identity_source_url,
            resolved_at=current.resolved_at,
            aliases=aliases,
            provenance=tuple(_provenance(row) for row in history),
        )

    @staticmethod
    def _matches(
        row: SecurityIdentityRecord,
        *,
        ticker: str | None,
        venue: str | None,
        isin: str | None,
        mic: str | None,
        exchange_symbol: str | None,
    ) -> bool:
        return all(
            (
                ticker is None or row.canonical_ticker == ticker,
                venue is None or row.venue == venue,
                isin is None or row.isin == isin,
                mic is None or row.mic == mic,
                exchange_symbol is None or row.exchange_symbol.upper() == exchange_symbol,
            )
        )

    @staticmethod
    def _path_matches(
        path: tuple[SecurityIdentityRecord, ...],
        *,
        venue: str | None,
        isin: str | None,
        mic: str | None,
        exchange_symbol: str | None,
    ) -> bool:
        """Match historical identifiers only within one ticker's supersession path."""
        return all(
            (
                venue is None or any(row.venue == venue for row in path),
                isin is None or any(row.isin == isin for row in path),
                mic is None or any(row.mic == mic for row in path),
                exchange_symbol is None
                or any(row.exchange_symbol.upper() == exchange_symbol for row in path),
            )
        )

    @staticmethod
    def _follow_alias(
        first: SecurityIdentityRecord, rows: list[SecurityIdentityRecord]
    ) -> tuple[SecurityIdentityRecord | None, tuple[SecurityIdentityRecord, ...]]:
        path = [first]
        seen = {(first.venue, first.isin)}
        current = first
        while not current.is_active and current.superseded_by_isin is not None:
            key = (current.venue, current.superseded_by_isin)
            if key in seen:
                return None, tuple(path)
            seen.add(key)
            target = next(
                (row for row in rows if (row.venue, row.isin) == key),
                None,
            )
            if target is None:
                return None, tuple(path)
            path.append(target)
            current = target
        return (current if current.is_active else None), tuple(path)

    @staticmethod
    def _history_for(
        current: SecurityIdentityRecord, rows: list[SecurityIdentityRecord]
    ) -> tuple[SecurityIdentityRecord, ...]:
        history = [current]
        frontier = [current.isin]
        seen = {(current.venue, current.isin)}
        while frontier:
            target_isin = frontier.pop()
            parents = [
                row
                for row in rows
                if row.venue == current.venue and row.superseded_by_isin == target_isin
            ]
            for parent in parents:
                key = (parent.venue, parent.isin)
                if key in seen:
                    continue
                seen.add(key)
                history.append(parent)
                frontier.append(parent.isin)
        return tuple(sorted(history, key=lambda row: row.resolved_at))

    @staticmethod
    def _validate_persisted_identity(row: SecurityIdentityRecord) -> None:
        suffix = VENUE_SUFFIXES.get(row.venue)
        if suffix is None or not row.canonical_ticker.endswith(f".{suffix}"):
            raise InvalidSecurityIdentifier(
                f"persisted ticker {row.canonical_ticker!r} contradicts venue {row.venue!r}"
            )
        if not is_valid_isin(row.isin):
            raise InvalidSecurityIdentifier(f"persisted identity has invalid ISIN: {row.isin!r}")
