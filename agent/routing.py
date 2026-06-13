import anthropic


class HardcodedSonnetRouting:
    """Always routes to Sonnet 4.6. Will be replaced by PhaseBasedRouting in W3."""

    def select(
        self, iteration: int, messages: list[anthropic.types.MessageParam], ticker: str | None
    ) -> str:
        return "claude-sonnet-4-6"
