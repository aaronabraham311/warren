"""Small, deterministic recent-context buffer for the interactive terminal."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.models import AnalysisOutput

if TYPE_CHECKING:
    from agent.requests import RecentContext as ParserRecentContext
    from agent.service import RunResult


@dataclass(frozen=True, slots=True)
class ChatTurn:
    """One completed terminal interaction, without prompts or model reasoning."""

    text: str
    run_id: str | None
    tickers: tuple[str, ...]


class RecentContext:
    """Bounded ticker/run context used to resolve short follow-up requests.

    Only the user's text and public run identifiers are retained. Tool payloads,
    prompts, API configuration, and model reasoning never enter this buffer.
    """

    def __init__(self, max_turns: int = 8) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self._turns: deque[ChatTurn] = deque(maxlen=max_turns)
        self._latest_result: RunResult | None = None
        self._selected_ticker: str | None = None

    @property
    def turns(self) -> tuple[ChatTurn, ...]:
        return tuple(self._turns)

    @property
    def recent_tickers(self) -> tuple[str, ...]:
        """Most recently mentioned tickers, de-duplicated in recency order."""

        seen: set[str] = set()
        tickers: list[str] = []
        for turn in reversed(self._turns):
            for ticker in reversed(turn.tickers):
                if ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)
        return tuple(tickers)

    @property
    def latest_run_id(self) -> str | None:
        for turn in reversed(self._turns):
            if turn.run_id is not None:
                return turn.run_id
        return None

    def record_result(self, text: str, result: RunResult) -> None:
        tickers = tuple(item.ticker for item in result.ticker_results)
        self._turns.append(ChatTurn(text=text, run_id=result.run_id, tickers=tickers))
        self._latest_result = result
        self._selected_ticker = next(
            (item.ticker for item in result.ticker_results if item.analysis is not None),
            None,
        )

    def record(
        self,
        text: str,
        *,
        tickers: tuple[str, ...] = (),
        run_id: str | None = None,
    ) -> None:
        self._turns.append(ChatTurn(text=text, run_id=run_id, tickers=tickers))

    def clear(self) -> None:
        self._turns.clear()
        self._latest_result = None
        self._selected_ticker = None

    def parser_context(self) -> ParserRecentContext | None:
        """Return the pure parser's minimal context without creating an import cycle."""

        if self._latest_result is None:
            return None
        from agent.requests import RecentContext as ParserRecentContext

        tickers = tuple(item.ticker for item in self._latest_result.ticker_results)
        return ParserRecentContext(tickers=tickers, selected_ticker=self._selected_ticker)

    def select_ticker(self, ticker: str | None = None) -> AnalysisOutput | None:
        if self._latest_result is None:
            return None
        analyses = {analysis.ticker: analysis for analysis in self._latest_result.analyses}
        selected = ticker or self._selected_ticker
        if selected is None and analyses:
            selected = sorted(analyses)[0]
        if selected is None:
            return None
        analysis = analyses.get(selected)
        if analysis is not None:
            self._selected_ticker = selected
        return analysis


def build_parser() -> argparse.ArgumentParser:
    """Build the console-script parser without creating terminal state."""

    return argparse.ArgumentParser(
        prog="warren",
        description="Interactive Warren stock-analysis terminal",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script target; parse help before loading the interactive app."""

    build_parser().parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()  # must precede app/service/storage imports so WARREN_DB is applied once

    from agent.terminal.app import main as terminal_main

    return terminal_main()


if __name__ == "__main__":
    raise SystemExit(main())
