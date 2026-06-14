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

if TYPE_CHECKING:
    from agent.budget import RunContext

ErrorCode = Literal["rate_limit", "not_found", "stale_data", "network", "unknown"]


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


ToolResult = ToolResultOk | ToolResultError


# DataSourceError uses a narrower vocabulary ("not_found"/"network"/"parse", plus a
# forward-compatible "rate_limit"); map it onto the agent-facing ErrorCode + retryable.
_DATA_SOURCE_ERROR_MAP: dict[str, tuple[ErrorCode, bool]] = {
    "not_found": ("not_found", False),
    "network": ("network", True),
    "rate_limit": ("rate_limit", True),
    "stale_data": ("stale_data", False),
    "parse": ("unknown", False),
}


def error_from_data_source(dse: DataSourceError) -> ToolResultError:
    code, retryable = _DATA_SOURCE_ERROR_MAP.get(dse.error_code, ("unknown", False))
    return ToolResultError(error_code=code, message=dse.message, retryable=retryable)


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
