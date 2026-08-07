"""Local-only prompt completion for slash commands and persona values."""

from __future__ import annotations

from collections.abc import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from agent.terminal.commands import COMMAND_NAMES


class WarrenCompleter(Completer):
    """Complete known local syntax without querying symbols or external services."""

    def get_completions(self, document: Document, complete_event: object) -> Iterable[Completion]:
        del complete_event
        before = document.text_before_cursor
        if before.startswith("/") and " " not in before:
            prefix = before.casefold()
            for command in COMMAND_NAMES:
                if command.startswith(prefix):
                    yield Completion(command, start_position=-len(before))
            return
        persona_prefix = before.casefold()
        if persona_prefix.startswith("/persona "):
            value = before[len("/persona ") :]
            if " " in value:
                return
            for persona in ("default", "dirt"):
                if persona.startswith(value.casefold()):
                    yield Completion(persona, start_position=-len(value))


SlashCommandCompleter = WarrenCompleter
