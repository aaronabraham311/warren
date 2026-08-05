"""Typed exchange identities shared by junior-market constituent sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import requests

from data_sources.errors import DataSourceError


@dataclass(frozen=True)
class SecurityIdentity:
    """A source-grounded mapping from an exchange listing to Warren's ticker."""

    canonical_ticker: str
    venue: str
    mic: str | None
    exchange_symbol: str
    isin: str | None
    legal_name: str | None
    identity_source_url: str
    resolved_at: datetime
    aliases: tuple[str, ...] = ()
    source_ids: Mapping[str, str] = field(default_factory=dict)


class ConstituentSource(Protocol):
    """Fetch typed identities from one exchange-specific public source."""

    def fetch(self, session: requests.Session) -> list[SecurityIdentity] | DataSourceError: ...
