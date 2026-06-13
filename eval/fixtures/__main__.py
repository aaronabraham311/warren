"""CLI entry point: python -m eval.fixtures --record AAPL MSFT GOOG"""

import argparse
from pathlib import Path

from eval.fixtures import record_fixtures

parser = argparse.ArgumentParser(description="Record yfinance fixtures from live API")
parser.add_argument("tickers", nargs="+", help="Ticker symbols to record")
parser.add_argument(
    "--record", action="store_true", required=True, help="Must pass --record explicitly"
)
parser.add_argument(
    "--output-dir",
    default=str(Path(__file__).parent),
    help="Root fixtures directory (default: eval/fixtures/)",
)
args = parser.parse_args()

for ticker in args.tickers:
    record_fixtures(ticker.upper(), Path(args.output_dir))
