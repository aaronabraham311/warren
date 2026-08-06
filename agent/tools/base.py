"""Tool contract (Tech Spec §5.1).

The single most important decision in the harness: **errors are returned as
data**, never raised to the agent loop. A tool's ``run`` returns a ``ToolResult``
discriminated union — ``ToolResultOk`` carries the typed payload, ``ToolResultError``
carries a structured ``error_code`` the agent can reason about ("yfinance returned
``stale_data`` — let me try Finnhub") and the loop can act on (retry vs. proceed).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, SkipValidation

from data_sources.errors import DataSourceError

# Re-exported (redundant alias marks the intentional re-export) so the ~15 tool input
# schemas that already do `from agent.tools.base import TICKER_PATTERN` inherit the single
# shared pattern. It admits both US share classes (BRK.B) and the exchange suffixes gem-hunt
# mode needs (DIR.MI, CIRSA.MC, KPL.WA); data sources normalise the suffix via
# `to_yahoo_symbol`. Single source of truth: `data_sources.symbols.TICKER_PATTERN`.
from data_sources.symbols import TICKER_PATTERN as TICKER_PATTERN

if TYPE_CHECKING:
    from agent.budget import RunContext

ErrorCode = Literal["rate_limit", "not_found", "stale_data", "network", "parse", "unknown"]


class ToolResultOk(BaseModel):
    status: Literal["ok"] = "ok"
    # SkipValidation keeps the concrete BaseModel subclass instance intact rather
    # than coercing it down to the bare BaseModel base during validation.
    data: SkipValidation[BaseModel]
    cached: bool = False
    latency_ms: int = 0


class ToolResultError(BaseModel):
    status: Literal["error"] = "error"
    error_code: ErrorCode
    message: str
    retryable: bool
    stage: str | None = None
    source: str | None = None


ToolResult = ToolResultOk | ToolResultError


# DataSourceError uses a narrower vocabulary ("not_found"/"network"/"parse", plus a
# forward-compatible "rate_limit"); map it onto the agent-facing ErrorCode + retryable.
_DATA_SOURCE_ERROR_MAP: dict[str, tuple[ErrorCode, bool]] = {
    "not_found": ("not_found", False),
    "network": ("network", True),
    "rate_limit": ("rate_limit", True),
    "stale_data": ("stale_data", False),
    "parse": ("parse", False),
}


def error_from_data_source(dse: DataSourceError) -> ToolResultError:
    code, retryable = _DATA_SOURCE_ERROR_MAP.get(dse.error_code, ("unknown", False))
    return ToolResultError(
        error_code=code,
        message=dse.message,
        retryable=retryable,
        stage=dse.stage,
        source=dse.source,
    )


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, tool_input: BaseModel, ctx: "RunContext") -> ToolResult: ...

    def to_api_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }
