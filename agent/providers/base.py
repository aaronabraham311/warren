"""Provider-neutral message, response, and tool-schema contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast

if TYPE_CHECKING:
    from agent.tools.base import ToolDefinition

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]

Role: TypeAlias = Literal["user", "assistant"]
ReasoningEffort: TypeAlias = Literal["none", "minimal", "low", "medium", "high"]
ServiceTier: TypeAlias = Literal["auto", "default", "flex"]


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ReasoningBlock:
    """Visible reasoning summary only; opaque reasoning stays in replay."""

    text: str


@dataclass(frozen=True)
class ToolCallBlock:
    id: str
    name: str
    arguments: JSONObject


@dataclass(frozen=True)
class ToolResultBlock:
    call_id: str
    name: str
    content: str
    is_error: bool = False


ProviderBlock: TypeAlias = TextBlock | ReasoningBlock | ToolCallBlock | ToolResultBlock


@dataclass(frozen=True)
class Message:
    role: Role
    blocks: tuple[ProviderBlock, ...]
    # Exact provider output objects serialized to JSON-compatible dictionaries.
    # Adapters replay these preferentially instead of reconstructing model turns.
    replay: tuple[JSONObject, ...] = ()

    @classmethod
    def text(cls, role: Role, text: str) -> Message:
        return cls(role=role, blocks=(TextBlock(text),))


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int | None = None
    tool_use_tokens: int = 0
    total_tokens: int = 0
    raw: JSONObject = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResponse:
    blocks: tuple[TextBlock | ReasoningBlock | ToolCallBlock, ...]
    stop_reason: str
    usage: Usage
    model_id: str
    replay: tuple[JSONObject, ...] = ()

    def assistant_message(self) -> Message:
        return Message(role="assistant", blocks=self.blocks, replay=self.replay)


class Provider(Protocol):
    name: str

    def complete(
        self,
        *,
        model: str,
        system_prompt: str,
        portfolio_context: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        max_tokens: int,
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort = "none",
        service_tier: ServiceTier = "auto",
    ) -> ProviderResponse: ...

    def tool_result_turn(self, results: list[ToolResultBlock]) -> list[Message]: ...


def sanitize_json_schema(schema: dict[str, object], *, strict: bool = False) -> JSONObject:
    """Inline local refs and remove Pydantic-only JSON Schema metadata.

    In strict mode every object is closed and every property is required, as
    required by OpenAI structured function tools. Formerly optional properties
    are made nullable so callers can preserve their default semantics.
    """

    raw = cast(JSONObject, deepcopy(schema))
    definitions_value = raw.pop("$defs", {})
    definitions = definitions_value if isinstance(definitions_value, dict) else {}

    def clean(value: JSONValue) -> JSONValue:
        if isinstance(value, list):
            return [clean(item) for item in value]
        if not isinstance(value, dict):
            return value

        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = definitions.get(ref.removeprefix("#/$defs/"))
            if isinstance(target, dict):
                merged = dict(target)
                merged.update({key: item for key, item in value.items() if key != "$ref"})
                return clean(merged)

        result: JSONObject = {}
        for key, item in value.items():
            if key in {"title", "default", "examples"}:
                continue
            result[key] = clean(item)

        properties_value = result.get("properties")
        if strict and isinstance(properties_value, dict):
            properties = properties_value
            old_required_value = result.get("required", [])
            old_required = (
                {item for item in old_required_value if isinstance(item, str)}
                if isinstance(old_required_value, list)
                else set()
            )
            for name, property_schema in list(properties.items()):
                if name not in old_required and isinstance(property_schema, dict):
                    properties[name] = _make_nullable(property_schema)
            result["required"] = list(properties)
            result["additionalProperties"] = False
        return result

    cleaned = clean(raw)
    if not isinstance(cleaned, dict):
        raise TypeError("tool schema must be a JSON object")
    return cleaned


def _make_nullable(schema: JSONObject) -> JSONObject:
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        if not any(isinstance(item, dict) and item.get("type") == "null" for item in any_of):
            schema["anyOf"] = [*any_of, {"type": "null"}]
        return schema
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        schema["type"] = [schema_type, "null"]
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


def strip_null_values(value: JSONValue) -> JSONValue:
    """Drop null object fields emitted for strict schemas before validation."""

    if isinstance(value, list):
        return [strip_null_values(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_null_values(item) for key, item in value.items() if item is not None}
    return value


def combined_system_prompt(system_prompt: str, portfolio_context: str) -> str:
    if not portfolio_context:
        return system_prompt
    return f"{system_prompt}\n\n{portfolio_context}"
