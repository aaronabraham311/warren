import hashlib
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol

from sqlalchemy import create_engine, delete, event
from sqlalchemy.orm import Session

from storage.models import Analysis, AnalysisData, Run, RunStatus, ToolCall


class _DBAPICursor(Protocol):
    def execute(self, statement: str) -> object: ...
    def close(self) -> None: ...


class _DBAPIConnection(Protocol):
    def cursor(self) -> _DBAPICursor: ...


_DB_URL = f"sqlite:///{os.environ.get('WARREN_DB', 'warren.db')}"
engine = create_engine(_DB_URL, echo=False)


@event.listens_for(engine, "connect")
def _set_wal_mode(dbapi_connection: _DBAPIConnection, connection_record: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def migrate() -> None:
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


_TOOL_OUTPUT_MAX_BYTES = 8192


def upsert_analysis(run_id: str, ticker: str, data: AnalysisData) -> None:
    with Session(engine) as session:
        session.execute(
            delete(Analysis).where(Analysis.run_id == run_id, Analysis.ticker == ticker)
        )
        session.add(Analysis(run_id=run_id, ticker=ticker, **data))
        session.commit()


def write_run_start(run_id: str, started_at: datetime) -> None:
    with Session(engine) as session:
        session.add(Run(id=run_id, started_at=started_at, status="running"))
        session.commit()


def write_run_end(
    run_id: str,
    status: RunStatus,
    total_input_tokens: int,
    total_output_tokens: int,
    total_cost_usd: float,
    num_tool_calls: int,
    completed_at: datetime,
    error_msg: str | None = None,
) -> None:
    with Session(engine) as session:
        run = session.get(Run, run_id)
        if run is None:
            raise RuntimeError(f"Run {run_id!r} not found; was write_run_start called?")
        run.status = status
        run.total_input_tokens = total_input_tokens
        run.total_output_tokens = total_output_tokens
        run.total_cost_usd = total_cost_usd
        run.num_tool_calls = num_tool_calls
        run.completed_at = completed_at
        run.error_msg = error_msg
        session.commit()


def write_tool_call(
    run_id: str,
    tool_name: str,
    input_json: str,
    raw_output: str,
    latency_ms: int,
    cached: bool,
    error_msg: str | None,
    seq: int,
) -> None:
    truncated = truncate_tool_output(raw_output, run_id, seq)
    output_file_path: str | None = None
    if truncated != raw_output:
        try:
            marker = json.loads(truncated)
            output_file_path = marker.get("path")
        except (json.JSONDecodeError, AttributeError):
            pass
    with Session(engine) as session:
        session.add(
            ToolCall(
                run_id=run_id,
                tool_name=tool_name,
                input_json=input_json,
                output_json=truncated,
                output_file_path=output_file_path,
                latency_ms=latency_ms,
                cached=cached,
                error_msg=error_msg,
            )
        )
        session.commit()


def truncate_tool_output(output_json: str, run_id: str, tool_call_id: int) -> str:
    if len(output_json.encode()) <= _TOOL_OUTPUT_MAX_BYTES:
        return output_json

    out_dir = Path(f"logs/runs/{run_id}/tool_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tool_call_id}.json"
    out_path.write_text(output_json)

    sha256 = hashlib.sha256(output_json.encode()).hexdigest()
    return json.dumps({"truncated": True, "path": str(out_path), "sha256": sha256})
