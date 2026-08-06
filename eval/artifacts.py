"""Private, append-only audit artifacts for model evaluation runs.

The public ``--output`` file intentionally remains a list of ``EvalGrade`` objects.  This
module writes the evidence needed to audit those grades to a separate, gitignored JSONL
companion.  Each ticker is flushed independently so an interrupted run still leaves useful
evidence, and files are owner-readable/writable only.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import anthropic

from agent.models import AnalysisOutput
from eval.grader import EvalGrade
from eval.tool_fixtures import FixtureEvidenceIssue, FixtureMiss

ARTIFACT_SCHEMA_VERSION = 1


def content_hash(text: str) -> str:
    """Return a stable identifier without persisting private prompt text."""

    return sha256(text.encode("utf-8")).hexdigest()


def fixture_set_id(ticker: str, root: Path) -> str:
    """Hash the exact committed fixture bytes available to one ticker."""

    digest = sha256()
    ticker_root = root / ticker / "tools"
    for path in sorted(ticker_root.glob("*/*.json")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class CapturedResponse:
    model_id: str
    stop_reason: str
    blocks: list[dict[str, object]]
    usage: dict[str, object]

    @classmethod
    def from_response(cls, response: anthropic.types.Message) -> CapturedResponse:
        blocks = [block.model_dump(mode="json") for block in response.content]
        usage: dict[str, object] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_tokens": response.usage.cache_read_input_tokens or 0,
            "cache_write_tokens": response.usage.cache_creation_input_tokens or 0,
        }
        return cls(
            model_id=response.model,
            stop_reason=str(response.stop_reason),
            blocks=blocks,
            usage=usage,
        )

    @property
    def final_text(self) -> str | None:
        for block in reversed(self.blocks):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                return str(block["text"])
        return None


@dataclass(frozen=True)
class EvalArtifactRecord:
    run_id: str
    ticker: str
    provider: str
    model: str
    service_tier: str
    reasoning_effort: str
    persona: str
    prompt_hash: str
    fixture_set_id: str
    tool_trace: str
    started_at: str
    completed_at: str
    fixture_misses: list[dict[str, str]]
    fixture_evidence_issues: list[dict[str, str]]
    responses: list[CapturedResponse]
    analysis_output: dict[str, object] | None
    raw_final_content: str | None
    failure: dict[str, str] | None
    grade: dict[str, object]
    judge_model: str | None = None
    judge_prompt_version: str | None = None

    def to_json(self) -> str:
        payload = asdict(self)
        payload["schema_version"] = ARTIFACT_SCHEMA_VERSION
        return json.dumps(payload, sort_keys=True)


class EvalArtifactWriter:
    """Append ticker records to an owner-only JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self._file = os.fdopen(descriptor, "w", encoding="utf-8")

    def write(self, record: EvalArtifactRecord) -> None:
        self._file.write(record.to_json() + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> EvalArtifactWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def make_artifact_record(
    *,
    run_id: str,
    ticker: str,
    provider: str,
    model: str,
    service_tier: str,
    reasoning_effort: str,
    persona: str,
    persona_prompt: str,
    fixtures_root: Path,
    log_path: Path,
    started_at: datetime,
    responses: list[CapturedResponse],
    fixture_misses: list[FixtureMiss],
    fixture_evidence_issues: list[FixtureEvidenceIssue],
    grade: EvalGrade,
    result: AnalysisOutput | None,
    failure: BaseException | None,
    judge_model: str | None = None,
    judge_prompt_version: str | None = None,
) -> EvalArtifactRecord:
    raw_final = next(
        (text for response in reversed(responses) if (text := response.final_text) is not None),
        None,
    )
    return EvalArtifactRecord(
        run_id=run_id,
        ticker=ticker,
        provider=provider,
        model=model,
        service_tier=service_tier,
        reasoning_effort=reasoning_effort,
        persona=persona,
        prompt_hash=content_hash(persona_prompt),
        fixture_set_id=fixture_set_id(ticker, fixtures_root),
        tool_trace=str(log_path),
        started_at=started_at.isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        fixture_misses=[
            {"tool_name": miss.tool_name, "input_hash": miss.input_hash} for miss in fixture_misses
        ],
        fixture_evidence_issues=[
            {
                "tool_name": issue.tool_name,
                "input_hash": issue.input_hash,
                "reason": issue.reason,
            }
            for issue in fixture_evidence_issues
        ],
        responses=responses,
        analysis_output=None if result is None else result.model_dump(mode="json"),
        raw_final_content=raw_final,
        failure=(
            None if failure is None else {"type": type(failure).__name__, "message": str(failure)}
        ),
        grade=grade.model_dump(mode="json"),
        judge_model=judge_model,
        judge_prompt_version=judge_prompt_version,
    )
