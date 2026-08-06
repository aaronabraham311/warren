"""Tool-output fixtures — the replay layer that makes the eval command deterministic.

Where ``eval/fixtures/{TICKER}/{client}/{method}/`` stores *raw upstream payloads* for the
data-fetcher tests (so those tests exercise the real parsing path), this module stores the
*serialized tool result* the agent loop feeds back to Claude:

    eval/fixtures/{TICKER}/tools/{tool_name}/{input_hash}.json

``input_hash`` is ``sha256(json.dumps(tool_input, sort_keys=True))[:8]``, matching the
convention in ``eval.fixtures``. A file holds one of:

    {"recorded_at": "...", "status": "ok",    "data": {...}}            → ToolResultOk
    {"recorded_at": "...", "status": "error", "error_code": "...", ...} → ToolResultError

``FixtureToolRunner`` satisfies ``agent.loop.ToolRunner``. It never calls ``tool.run()``,
so no data-source client is ever constructed and replay cannot reach the network *by
construction* — not merely by a socket guard.

A tool call with no recorded fixture is a miss: the runner records it on ``.misses`` and
returns ``ToolResultError(error_code="not_found", retryable=False)``. Non-retryable is
load-bearing — a retryable miss would send the loop into exponential backoff on every
unrecorded call.

Fixtures rot: yfinance schemas drift, filing dates advance, news windows slide. Every file
carries a ``recorded_at`` stamp, and loading one older than :data:`STALE_AFTER` warns.
Refresh with ``python -m eval.fixtures.recorder`` — see ``eval/fixtures/README.md``.
"""

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel

from agent.budget import RunContext
from agent.tools.base import Tool, ToolResult, ToolResultError, ToolResultOk
from agent.tools.dirt_scenarios import ModelDirtScenariosInput, model_dirt_scenarios
from eval.fixture_evidence import validate_fixture_result

FIXTURES_DIR = Path(__file__).parent / "fixtures"

STALE_AFTER = timedelta(days=90)

# ``EstimateIntrinsicValueTool`` converts these explicit values and ``None`` to the same
# behavioral assumptions.  Keep the fixture key equally semantic without importing the
# tool module (which would couple the offline replay layer to a concrete implementation).
_DCF_DEFAULTS: dict[str, object] = {
    "growth_rate": 0.08,
    "discount_rate": 0.10,
    "terminal_growth_rate": 0.025,
    "projection_years": 10,
}


def canonical_tool_input(tool_name: str, tool_input: Mapping[str, object]) -> dict[str, object]:
    """Return the semantic input used to key a tool fixture.

    Pydantic validation already supplies ordinary schema defaults and canonical scalar
    types before this function is called.  The DCF is the exceptional case: its schema
    uses ``None`` to mean a behavioral default, so an explicit default must collapse to
    the same key.  True overrides remain distinct.  News windows deliberately receive no
    special treatment because seven and thirty days are different requests.
    """
    canonical = dict(tool_input)
    if tool_name == "estimate_intrinsic_value":
        for field_name, default in _DCF_DEFAULTS.items():
            if canonical.get(field_name) == default:
                canonical[field_name] = None
    return canonical


def tool_input_hash(tool_input: Mapping[str, object]) -> str:
    """First 8 hex chars of sha256 over the canonical input JSON."""
    return sha256(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:8]


def tool_fixture_path(
    ticker: str,
    tool_name: str,
    tool_input: Mapping[str, object],
    root: Path = FIXTURES_DIR,
) -> Path:
    canonical = canonical_tool_input(tool_name, tool_input)
    return root / ticker / "tools" / tool_name / f"{tool_input_hash(canonical)}.json"


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


@dataclass(frozen=True)
class FixtureEvidenceIssue:
    tool_name: str
    input_hash: str
    reason: str


@dataclass(frozen=True)
class FixtureObservation:
    """The exact validated input/result pair exposed to the agent during replay."""

    tool_name: str
    canonical_input: dict[str, object]
    input_hash: str
    result: ToolResult


@dataclass
class FixtureToolRunner:
    """Serves recorded tool results from disk instead of calling the real tool."""

    ticker: str
    root: Path = FIXTURES_DIR
    misses: list[FixtureMiss] = field(default_factory=list)
    served: dict[str, ToolResultOk] = field(default_factory=dict)
    evidence_issues: list[FixtureEvidenceIssue] = field(default_factory=list)
    observations: list[FixtureObservation] = field(default_factory=list)

    def run(self, tool: Tool, tool_input: BaseModel, ctx: RunContext) -> ToolResult:
        # Recompute the one deterministic arithmetic contract directly through its pure
        # function. Deliberately do not dispatch Tool.run(): fixture replay's guarantee that
        # it never invokes a live tool remains load-bearing for every Tool implementation.
        if tool.name == "model_dirt_scenarios":
            if not isinstance(tool_input, ModelDirtScenariosInput):
                return ToolResultError(
                    error_code="parse",
                    message="model_dirt_scenarios received an unexpected input schema",
                    retryable=False,
                )
            try:
                pure_result = ToolResultOk(data=model_dirt_scenarios(tool_input), cached=True)
            except ValueError as exc:
                return ToolResultError(
                    error_code="parse",
                    message=f"invalid DIRT decision contract: {exc}",
                    retryable=False,
                )
            self.served[tool.name] = pure_result
            return pure_result
        raw_input: dict[str, object] = tool_input.model_dump(mode="json")
        canonical_input = canonical_tool_input(tool.name, raw_input)
        path = tool_fixture_path(self.ticker, tool.name, canonical_input, self.root)
        if not path.exists():
            self.misses.append(FixtureMiss(tool.name, tool_input_hash(canonical_input)))
            result: ToolResult = ToolResultError(
                error_code="not_found",
                message=(
                    f"no recorded fixture for {tool.name}({self.ticker}); "
                    f"expected {path}. Record it with the eval fixture recorder."
                ),
                retryable=False,
            )
            self.observations.append(
                FixtureObservation(
                    tool.name,
                    canonical_input,
                    tool_input_hash(canonical_input),
                    result,
                )
            )
            return result
        payload: dict[str, object] = json.loads(path.read_text())
        warn_if_stale(payload, path)
        result = _deserialize(payload, tool)
        try:
            validate_fixture_result(self.ticker, tool, tool_input, result)
        except ValueError as exc:
            self.evidence_issues.append(
                FixtureEvidenceIssue(tool.name, tool_input_hash(canonical_input), str(exc))
            )
            result = ToolResultError(
                error_code="not_found",
                message=f"recorded fixture is unusable: {exc}",
                retryable=False,
            )
            self.observations.append(
                FixtureObservation(
                    tool.name,
                    canonical_input,
                    tool_input_hash(canonical_input),
                    result,
                )
            )
            return result
        if isinstance(result, ToolResultOk):
            self.served[tool.name] = result
        self.observations.append(
            FixtureObservation(
                tool.name,
                canonical_input,
                tool_input_hash(canonical_input),
                result,
            )
        )
        return result


def warn_if_stale(payload: dict[str, object], path: Path) -> None:
    """Warn when a fixture was recorded more than :data:`STALE_AFTER` ago.

    Fixtures predating the ``recorded_at`` stamp are silent: their age is unknown, and a
    warning we cannot substantiate would train the reader to ignore the real ones.
    """
    stamp = payload.get("recorded_at")
    if not isinstance(stamp, str):
        return
    recorded_at = datetime.fromisoformat(stamp)
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - recorded_at
    if age > STALE_AFTER:
        warnings.warn(
            f"Fixture {path.parent.parent.parent.name}/{path.parent.name} is {age.days} days "
            f"old — refresh with: python -m eval.fixtures.recorder",
            stacklevel=3,
        )


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
    recorded_at: datetime | None = None,
) -> Path:
    """Write *result* to its fixture path and return it. Overwrites any existing fixture.

    Kept here so the on-disk format has exactly one owner; the fixture recorder calls this
    rather than re-deriving the layout.
    """
    path = tool_fixture_path(ticker, tool_name, tool_input, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, ToolResultOk):
        payload: dict[str, object] = {"status": "ok", "data": result.data.model_dump(mode="json")}
    else:
        payload = result.model_dump(mode="json")
    payload["recorded_at"] = (recorded_at or datetime.now(timezone.utc)).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
