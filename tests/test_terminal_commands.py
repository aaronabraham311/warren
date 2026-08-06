import pytest

from agent.terminal.commands import (
    BudgetCommand,
    CommandError,
    DiscoverCommand,
    GemHuntCommand,
    HelpCommand,
    HistoryCommand,
    NewCommand,
    PersonaCommand,
    PortfolioCommand,
    QuitCommand,
    ShowCommand,
    ToolsCommand,
    TraceCommand,
    WatchlistCommand,
    parse_command,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/help", HelpCommand()),
        ("/new", NewCommand()),
        ("/history", HistoryCommand()),
        ("/history brk.b", HistoryCommand("BRK.B")),
        (
            "/show 123e4567-e89b-12d3-a456-426614174000",
            ShowCommand("123e4567-e89b-12d3-a456-426614174000"),
        ),
        ("/trace", TraceCommand()),
        ("/trace run_1", TraceCommand("run_1")),
        ("/portfolio", PortfolioCommand()),
        ("/watchlist", WatchlistCommand()),
        ("/discover", DiscoverCommand()),
        ("/gem-hunt", GemHuntCommand()),
        ("/persona", PersonaCommand()),
        ("/persona DIRT", PersonaCommand("dirt")),
        ("/budget", BudgetCommand()),
        ("/budget 2.50", BudgetCommand(2.5)),
        ("/tools", ToolsCommand()),
        ("/quit", QuitCommand()),
    ],
)
def test_all_ticket_commands_parse_to_typed_values(text: str, expected: object) -> None:
    assert parse_command(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "help",
        "/unknown",
        "/help now",
        "/history AAPL MSFT",
        "/history NOT$",
        "/show",
        "/show bad/id",
        "/trace one two",
        "/persona value",
        "/budget free",
        "/budget 0",
        "/budget -1",
        "/budget 10.01",
        '/show "unterminated',
    ],
)
def test_user_command_errors_are_values_not_exceptions(text: str) -> None:
    result = parse_command(text)
    assert isinstance(result, CommandError)
    assert result.message


def test_command_names_are_case_insensitive_but_identifiers_are_preserved() -> None:
    assert parse_command("/HELP") == HelpCommand()
    assert parse_command("/SHOW Run.Mixed-Case") == ShowCommand("Run.Mixed-Case")
