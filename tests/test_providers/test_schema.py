from pydantic import BaseModel

from agent.providers.base import sanitize_json_schema, strip_null_values
from agent.tools.screen import ScreenUniverseInput


class _Defaults(BaseModel):  # type: ignore[explicit-any]
    required: str
    count: int = 3


def test_openai_strict_schema_closes_maps_and_nullable_defaults() -> None:
    schema = sanitize_json_schema(_Defaults.model_json_schema(), strict=True)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["required", "count"]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert properties["count"] == {"type": ["integer", "null"]}
    assert "title" not in schema


def test_screen_criteria_is_a_closed_named_shape_for_strict_providers() -> None:
    schema = sanitize_json_schema(ScreenUniverseInput.model_json_schema(), strict=True)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    criteria = properties["criteria"]
    assert isinstance(criteria, dict)
    assert criteria["additionalProperties"] is False
    criteria_properties = criteria["properties"]
    assert isinstance(criteria_properties, dict)
    assert "pe_ratio_max" in criteria_properties


def test_null_fields_from_strict_tool_calls_are_removed_before_validation() -> None:
    assert strip_null_values({"required": "x", "count": None}) == {"required": "x"}
