"""Pure typed parser for Warren's slash-command surface."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from data_sources.symbols import TICKER_PATTERN, canonical_symbol

PersonaName: TypeAlias = Literal["default", "dirt"]


@dataclass(frozen=True, slots=True)
class HelpCommand:
    pass


@dataclass(frozen=True, slots=True)
class NewCommand:
    pass


@dataclass(frozen=True, slots=True)
class HistoryCommand:
    ticker: str | None = None


@dataclass(frozen=True, slots=True)
class ShowCommand:
    run_id: str


@dataclass(frozen=True, slots=True)
class TraceCommand:
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioCommand:
    pass


@dataclass(frozen=True, slots=True)
class WatchlistCommand:
    pass


@dataclass(frozen=True, slots=True)
class DiscoverCommand:
    pass


@dataclass(frozen=True, slots=True)
class GemHuntCommand:
    pass


@dataclass(frozen=True, slots=True)
class PersonaCommand:
    persona: PersonaName | None = None


@dataclass(frozen=True, slots=True)
class BudgetCommand:
    max_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class ToolsCommand:
    pass


@dataclass(frozen=True, slots=True)
class QuitCommand:
    pass


@dataclass(frozen=True, slots=True)
class CommandError:
    message: str
    usage: str | None = None


Command: TypeAlias = (
    HelpCommand
    | NewCommand
    | HistoryCommand
    | ShowCommand
    | TraceCommand
    | PortfolioCommand
    | WatchlistCommand
    | DiscoverCommand
    | GemHuntCommand
    | PersonaCommand
    | BudgetCommand
    | ToolsCommand
    | QuitCommand
)
CommandOutcome: TypeAlias = Command | CommandError

COMMAND_NAMES: tuple[str, ...] = (
    "/help",
    "/new",
    "/history",
    "/show",
    "/trace",
    "/portfolio",
    "/watchlist",
    "/discover",
    "/gem-hunt",
    "/persona",
    "/budget",
    "/tools",
    "/quit",
)
_NO_ARG_COMMANDS: dict[str, Callable[[], Command]] = {
    "/help": HelpCommand,
    "/new": NewCommand,
    "/portfolio": PortfolioCommand,
    "/watchlist": WatchlistCommand,
    "/discover": DiscoverCommand,
    "/gem-hunt": GemHuntCommand,
    "/tools": ToolsCommand,
    "/quit": QuitCommand,
}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _ticker(raw: str) -> str | None:
    ticker = canonical_symbol(raw)
    return ticker if re.fullmatch(TICKER_PATTERN, ticker) is not None else None


def _wrong_arity(command: str, usage: str) -> CommandError:
    return CommandError(f"Invalid arguments for {command}.", usage=usage)


def parse_command(text: str) -> CommandOutcome:
    """Parse user input into a typed command; expected user errors are values."""
    try:
        parts = shlex.split(text.strip())
    except ValueError:
        return CommandError("Command contains an unmatched quote.")
    if not parts or not parts[0].startswith("/"):
        return CommandError("Slash commands start with '/'.", usage="/help")
    name = parts[0].casefold()
    args = parts[1:]
    if name in _NO_ARG_COMMANDS:
        if args:
            return _wrong_arity(name, name)
        return _NO_ARG_COMMANDS[name]()
    if name == "/history":
        if len(args) > 1:
            return _wrong_arity(name, "/history [ticker]")
        if not args:
            return HistoryCommand()
        ticker = _ticker(args[0])
        return (
            HistoryCommand(ticker)
            if ticker is not None
            else CommandError(f"Invalid ticker: {args[0]!r}.", usage="/history [ticker]")
        )
    if name == "/show":
        if len(args) != 1:
            return _wrong_arity(name, "/show RUN_ID")
        return (
            ShowCommand(args[0])
            if _RUN_ID_RE.fullmatch(args[0]) is not None
            else CommandError("Invalid run ID.", usage="/show RUN_ID")
        )
    if name == "/trace":
        if len(args) > 1:
            return _wrong_arity(name, "/trace [RUN_ID]")
        if not args:
            return TraceCommand()
        return (
            TraceCommand(args[0])
            if _RUN_ID_RE.fullmatch(args[0]) is not None
            else CommandError("Invalid run ID.", usage="/trace [RUN_ID]")
        )
    if name == "/persona":
        if len(args) > 1:
            return _wrong_arity(name, "/persona [default|dirt]")
        if not args:
            return PersonaCommand()
        persona = args[0].casefold()
        if persona not in ("default", "dirt"):
            return CommandError(
                "Persona must be 'default' or 'dirt'.", usage="/persona [default|dirt]"
            )
        return PersonaCommand(persona=cast(PersonaName, persona))
    if name == "/budget":
        if len(args) > 1:
            return _wrong_arity(name, "/budget [USD]")
        if not args:
            return BudgetCommand()
        try:
            amount = float(args[0])
        except ValueError:
            return CommandError("Budget must be a number in USD.", usage="/budget [USD]")
        if not 0.0 < amount <= 10.0:
            return CommandError(
                "Budget must be greater than $0 and at most $10.", usage="/budget [USD]"
            )
        return BudgetCommand(amount)
    return CommandError(f"Unknown command: {parts[0]}. Type /help for commands.")
