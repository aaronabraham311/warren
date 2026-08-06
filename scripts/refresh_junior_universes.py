"""Refresh committed junior-market fallback CSVs from their configured sources."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_sources.errors import DataSourceError  # noqa: E402
from data_sources.exchange_client import EXCHANGE_SPECS, ExchangeClient  # noqa: E402

_FIELDS = (
    "ticker",
    "venue",
    "mic",
    "exchange_symbol",
    "isin",
    "legal_name",
    "identity_source_url",
    "resolved_at",
)


def refresh(output_dir: Path) -> dict[str, int]:
    """Fetch every configured venue and atomically replace its fallback CSV."""
    counts: dict[str, int] = {}
    for key, spec in EXCHANGE_SPECS.items():
        result = ExchangeClient(spec).get_constituents()
        if isinstance(result, DataSourceError):
            raise RuntimeError(f"{key}: {result.error_code}: {result.message}")
        if not result:
            raise RuntimeError(f"{key}: source returned no identities")

        destination = output_dir / f"{key}.csv"
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDS, lineterminator="\n")
            writer.writeheader()
            for identity in sorted(result, key=lambda item: item.canonical_ticker):
                writer.writerow(
                    {
                        "ticker": identity.canonical_ticker,
                        "venue": identity.venue,
                        "mic": identity.mic or "",
                        "exchange_symbol": identity.exchange_symbol,
                        "isin": identity.isin or "",
                        "legal_name": identity.legal_name or "",
                        "identity_source_url": identity.identity_source_url,
                        "resolved_at": identity.resolved_at.isoformat(),
                    }
                )
        temporary.replace(destination)
        counts[key] = len(result)

    if sum(counts.values()) < 500:
        raise RuntimeError(f"junior universe unexpectedly small: {counts}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    counts = refresh(args.output_dir)
    print(" ".join(f"{key}={count}" for key, count in counts.items()))


if __name__ == "__main__":
    main()
