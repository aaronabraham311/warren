from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.budget import RunContext


@dataclass
class ToolResultOk:
    content: str


@dataclass
class ToolResultError:
    error: str


ToolResult = ToolResultOk | ToolResultError


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def run(self, input: dict[str, Any], run_context: "RunContext") -> ToolResult: ...

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
