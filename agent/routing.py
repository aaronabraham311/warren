from typing import Any


class HardcodedSonnetRouting:
    """Always routes to Sonnet 4.6. Will be replaced by PhaseBasedRouting in W3."""

    def select(self, iteration: int, messages: list[Any], ticker: str | None) -> str:
        return "claude-sonnet-4-6"
