"""Canonical symbol convention for the whole Warren stack.

The **canonical form** of a ticker — what portfolio CSVs, eval YAMLs, and every
caller uses — is the **Yahoo-style ``TICKER.SUFFIX`` with a dot**, uppercased.
Examples: ``AAPL``, ``BRK.B`` (US share class), ``DIR.MI`` (Milan exchange
suffix). This module is the single place that maps that canonical form into each
data source's native symbol spelling, so yfinance and Finnhub always derive their
request symbol the same way.

The subtlety a dot carries two meanings:

* **US share class** — a single trailing single-letter segment (``BRK.B``,
  ``BF.B``). Yahoo spells these with a **dash** (``BRK-B``).
* **Exchange suffix** — an ISO-style market code (``DIR.MI`` Milan, ``CIRSA.MC``
  Madrid, ``KPL.WA`` Warsaw). Yahoo keeps the **dot** here (``DIR.MI``).

Blindly replacing ``.`` with ``-`` (the old ``_yf_symbol`` behaviour) mangled
every exchange suffix into an unresolvable symbol. We disambiguate with an
explicit, extensible allow-list of known exchange suffixes.
"""

# Known Yahoo exchange suffixes that must be preserved verbatim (dot kept).
# Extend this set as new exchanges enter the universe. The current gem-hunt
# slice covers Milan (MI), Madrid (MC), and Warsaw (WA).
KNOWN_EXCHANGE_SUFFIXES: frozenset[str] = frozenset({"MI", "MC", "WA"})

# The single canonical ticker-format regex, shared by every pydantic model that
# validates a ticker (agent.models.AnalysisOutput, agent.portfolio.Holding /
# WatchlistEntry, eval.golden_set.EvalExample). Keeping one source of truth stops
# the three regexes from drifting apart and rejecting valid non-US symbols.
#
# Base: 1–5 uppercase letters, plus an OPTIONAL ``.``/``-`` suffix of 1–2 letters.
# This accepts both US share classes (``BRK.B``, ``BRK-B``) and the slice's
# exchange suffixes (``DIR.MI`` Milan, ``CIRSA.MC`` Madrid, ``KPL.WA`` Warsaw).
# Deliberately permissive — validation must not reject valid foreign symbols.
TICKER_PATTERN: str = r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$"


def canonical_symbol(ticker: str) -> str:
    """Normalise any caller input to the canonical form: uppercased, trimmed,
    Yahoo-style dots preserved. This is the shared entry point both data-source
    symbol mappers build on, so they agree on the base form."""
    return ticker.strip().upper()


def to_yahoo_symbol(ticker: str) -> str:
    """Map the canonical ``TICKER.SUFFIX`` form to Yahoo Finance's symbol spelling.

    * No dot → unchanged (``AAPL`` → ``AAPL``).
    * Dot + known exchange suffix → dot preserved (``DIR.MI`` → ``DIR.MI``).
    * Dot + single trailing letter (US share class) → dash (``BRK.B`` → ``BRK-B``).
    * Anything else with a dot → left as-is (nothing safe to rewrite).
    """
    symbol = canonical_symbol(ticker)
    if "." not in symbol:
        return symbol
    head, _, tail = symbol.rpartition(".")
    if tail in KNOWN_EXCHANGE_SUFFIXES:
        return symbol
    if head and len(tail) == 1 and tail.isalpha():
        return f"{head}-{tail}"
    return symbol


def to_finnhub_symbol(ticker: str) -> str:
    """Map the canonical form to Finnhub's symbol spelling.

    Finnhub uses the plain uppercased canonical symbol (dots preserved for both
    share classes and exchange suffixes), so this is just the canonical form. It
    exists as a distinct function so every Finnhub call site routes through the
    same shared canonicalization rather than ``.upper()``-ing inline.
    """
    return canonical_symbol(ticker)
