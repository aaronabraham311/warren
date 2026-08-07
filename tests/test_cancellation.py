from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agent.budget import Budget, RunContext
from agent.cancellation import CancellationToken, NeverCancelToken, RunCancelledError
from agent.events import RunEvent, ToolCallStarted
from agent.loop import ToolRunner, analyze_ticker
from agent.persona import DefaultPersona
from agent.routing import HardcodedSonnetRouting
from agent.tools.base import Tool, ToolResult
from storage.logger import RunLogger
from tests.conftest import make_tool_use


def test_cancellation_token_is_idempotent_and_raises_at_checkpoint() -> None:
    assert not CancellationToken().is_cancelled
    token = CancellationToken()
    token.cancel()
    token.cancel()
    assert token.is_cancelled
    with pytest.raises(RunCancelledError, match="Run cancelled"):
        token.raise_if_cancelled()


def test_never_cancel_token_preserves_batch_behavior() -> None:
    token = NeverCancelToken()
    token.cancel()
    assert not token.is_cancelled
    token.raise_if_cancelled()


def test_cancelled_before_llm_does_not_call_provider(tmp_path: Path) -> None:
    token = CancellationToken()
    token.cancel()
    ctx = RunContext(
        "cancel-before-llm",
        Budget(),
        RunLogger("cancel-before-llm", tmp_path),
        cancellation=token,
    )
    client = MagicMock()

    with pytest.raises(RunCancelledError):
        analyze_ticker("AAPL", DefaultPersona(), HardcodedSonnetRouting(), ctx, client=client)

    client.messages.create.assert_not_called()


class _CancelOnToolStart:
    def __init__(self, token: CancellationToken) -> None:
        self.token = token

    def emit(self, event: RunEvent) -> None:
        if isinstance(event, ToolCallStarted):
            self.token.cancel()


class _NeverCalledRunner(ToolRunner):
    def __init__(self) -> None:
        self.called = False

    def run(self, tool: Tool, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        del tool, tool_input, ctx
        self.called = True
        raise AssertionError("tool must not run after cancellation")


def test_cancelled_at_tool_boundary_does_not_dispatch(
    tmp_path: Path, mock_claude: MagicMock
) -> None:
    mock_claude([make_tool_use("get_quote", {"ticker": "AAPL"})])
    token = CancellationToken()
    runner = _NeverCalledRunner()
    ctx = RunContext(
        "cancel-before-tool",
        Budget(),
        RunLogger("cancel-before-tool", tmp_path),
        cancellation=token,
        event_sink=_CancelOnToolStart(token),
    )

    with pytest.raises(RunCancelledError):
        analyze_ticker(
            "AAPL",
            DefaultPersona(),
            HardcodedSonnetRouting(),
            ctx,
            tool_runner=runner,
        )

    assert not runner.called
