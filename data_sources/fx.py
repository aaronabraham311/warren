"""FX normalization — native reporting currency → USD.

yfinance returns prices, market caps and NCAV in each security's *native*
currency (EUR for Milan/Madrid, PLN for Warsaw). Warren's ``_usd``-suffixed
fields and every cross-currency threshold assume USD, so a native figure must be
converted before it is compared across names.

Design:

* **Scope** — the current gem-hunt slice covers USD, EUR and PLN only.
* **Spot rate** goes through the yfinance client boundary
  (:meth:`data_sources.yfinance_client.YFinanceClient.get_fx_rate`), which reads
  ``yf.Ticker("<BASE>USD=X")`` and caches into the shared ``CacheStore`` — so the
  ``_no_live_network`` test guard is respected and tests can mock the upstream.
* **Fallback** — :data:`FALLBACK_FX_RATES` is a committed constant table (mirrors
  the ``data/sp500.csv`` fallback pattern). A live-fetch failure degrades to it,
  and if a currency is missing there too, the native value is returned
  unconverted rather than raising. Errors are always data, never exceptions.
* **Identity** — an unknown or ``None`` currency is treated as USD (rate 1.0); we
  never convert what we cannot price, and never crash.
"""

# Currencies the slice supports. Anything outside this set is treated as USD
# (identity) so an unrecognised currency degrades gracefully instead of crashing.
SUPPORTED_CURRENCIES: frozenset[str] = frozenset({"USD", "EUR", "PLN"})

# Committed fallback spot rates: USD per 1 unit of the base currency. Used when a
# live fetch fails. Approximate is acceptable — a slightly-stale USD cap still
# beats a several-fold-wrong native-currency cap. Refresh manually as needed.
FALLBACK_FX_RATES: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "PLN": 0.25,
}


def normalize_currency(currency: str | None) -> str | None:
    """Uppercase/trim a raw currency code; empty/whitespace → ``None``."""
    if currency is None:
        return None
    normalized = currency.strip().upper()
    return normalized or None


def to_usd(
    amount: float | int | None,
    currency: str | None,
    rate: float | None = None,
) -> float | None:
    """Convert ``amount`` in ``currency`` to USD.

    * ``amount is None`` → ``None``.
    * USD, unknown, or ``None`` currency → identity (``float(amount)``, no convert).
    * EUR/PLN → multiply by ``rate`` (USD per unit) when supplied, else the
      committed :data:`FALLBACK_FX_RATES` entry.

    ``rate`` is the live spot rate resolved through the client boundary; omit it
    to fall back to the committed table.
    """
    if amount is None:
        return None
    cur = normalize_currency(currency)
    if cur is None or cur == "USD" or cur not in SUPPORTED_CURRENCIES:
        return float(amount)
    effective = rate if rate is not None else FALLBACK_FX_RATES[cur]
    return float(amount) * effective
