from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

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
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, object]]

    @abstractmethod
    def run(self, tool_input: dict[str, object], run_context: "RunContext") -> ToolResult: ...

    def to_api_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
