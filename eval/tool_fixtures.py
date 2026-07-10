"""Tool-output fixtures — the replay layer that makes the eval command deterministic.

Where ``eval/fixtures/{TICKER}/{client}/{method}/`` stores *raw upstream payloads* for the
data-fetcher tests (so those tests exercise the real parsing path), this module stores the
*serialized tool result* the agent loop feeds back to Claude:

    eval/fixtures/{TICKER}/tools/{tool_name}/{input_hash}.json

``input_hash`` is ``sha256(json.dumps(tool_input, sort_keys=True))[:8]``, matching the
convention in ``eval.fixtures``. A file holds one of:

    {"status": "ok",    "data": {...}}                                  → ToolResultOk
    {"status": "error", "error_code": "...", "message": "...", "retryable": false}

``FixtureToolRunner`` satisfies ``agent.loop.ToolRunner``. It never calls ``tool.run()``,
so no data-source client is ever constructed and replay cannot reach the network *by
construction* — not merely by a socket guard.

A tool call with no recorded fixture is a miss: the runner records it on ``.misses`` and
returns ``ToolResultError(error_code="not_found", retryable=False)``. Non-retryable is
load-bearing — a retryable miss would send the loop into exponential backoff on every
unrecorded call.
"""

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel

from agent.budget import RunContext
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def tool_input_hash(tool_input: dict[str, object]) -> str:
    """First 8 hex chars of sha256 over the canonical input JSON."""
    return sha256(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:8]


def tool_fixture_path(
    ticker: str,
    tool_name: str,
    tool_input: dict[str, object],
    root: Path = FIXTURES_DIR,
) -> Path:
    return root / ticker / "tools" / tool_name / f"{tool_input_hash(tool_input)}.json"


def has_tool_fixtures(ticker: str, root: Path = FIXTURES_DIR) -> bool:
    """True when *ticker* has at least one recorded tool fixture.

    The runner uses this to skip tickers wholesale rather than spend a live LLM call
    producing an analysis grounded in nothing but ``not_found`` errors.
    """
    tools_dir = root / ticker / "tools"
    return tools_dir.is_dir() and any(tools_dir.glob("*/*.json"))


@dataclass(frozen=True)
class FixtureMiss:
    tool_name: str
    input_hash: str


@dataclass
class FixtureToolRunner:
    """Serves recorded tool results from disk instead of calling the real tool."""

    ticker: str
    root: Path = FIXTURES_DIR
    misses: list[FixtureMiss] = field(default_factory=list)

    def run(self, tool: Tool, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        raw_input: dict[str, object] = tool_input.model_dump(mode="json")
        path = tool_fixture_path(self.ticker, tool.name, raw_input, self.root)
        if not path.exists():
            self.misses.append(FixtureMiss(tool.name, tool_input_hash(raw_input)))
            return ToolResultError(
                error_code="not_found",
                message=(
                    f"no recorded fixture for {tool.name}({self.ticker}); "
                    f"expected {path}. Record it with the eval fixture recorder."
                ),
                retryable=False,
            )
        return _deserialize(json.loads(path.read_text()), tool)


def _deserialize(payload: dict[str, object], tool: Tool) -> ToolResult:
    if payload.get("status") == "error":
        return ToolResultError.model_validate(payload)
    # cached=True: a replayed result did no I/O, and the flag is what the WAL records.
    data = tool.output_schema.model_validate(payload["data"])
    return ToolResultOk(data=data, cached=True)


def record_tool_result(
    ticker: str,
    tool_name: str,
    tool_input: dict[str, object],
    result: ToolResult,
    root: Path = FIXTURES_DIR,
) -> Path:
    """Write *result* to its fixture path and return it.

    Kept here so the on-disk format has exactly one owner; the fixture-recording ticket
    calls this rather than re-deriving the layout.
    """
    path = tool_fixture_path(ticker, tool_name, tool_input, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, ToolResultOk):
        payload: dict[str, object] = {"status": "ok", "data": result.data.model_dump(mode="json")}
    else:
        payload = result.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
