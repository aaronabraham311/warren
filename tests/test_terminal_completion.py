from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from agent.terminal.completion import WarrenCompleter


def _completions(text: str) -> list[str]:
    return [
        completion.text
        for completion in WarrenCompleter().get_completions(
            Document(text, cursor_position=len(text)), CompleteEvent()
        )
    ]


def test_completion_is_local_and_limited_to_commands_and_personas() -> None:
    assert "/history" in _completions("/h")
    assert _completions("/persona d") == ["default", "dirt"]
    assert _completions("Analyze AA") == []
    assert _completions("/show run") == []


def test_completion_does_not_offer_second_persona_argument() -> None:
    assert _completions("/persona dirt ") == []
