import hashlib
import json
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol

from sqlalchemy import Engine, create_engine, delete, event, text
from sqlalchemy.orm import Session

from storage.models import (
    Analysis,
    AnalysisData,
    EvalRun,
    PromptVersion,
    Run,
    RunStatus,
    ToolCall,
)

log = logging.getLogger(__name__)


class _DBAPICursor(Protocol):
    def execute(self, statement: str) -> object: ...
    def close(self) -> None: ...


class _DBAPIConnection(Protocol):
    def cursor(self) -> _DBAPICursor: ...


# None until first call to get_engine(); allows monkeypatching in tests and
# ensures WARREN_DB is read after load_dotenv() has run.
engine: Engine | None = None


def _set_wal_mode(dbapi_connection: _DBAPIConnection, connection_record: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def _set_fk_enforcement(dbapi_connection: _DBAPIConnection, connection_record: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine() -> Engine:
    global engine
    if engine is None:
        db_url = f"sqlite:///{os.environ.get('WARREN_DB', 'warren.db')}"
        engine = create_engine(db_url, echo=False)
        event.listen(engine, "connect", _set_wal_mode)
        event.listen(engine, "connect", _set_fk_enforcement)
    return engine


def migrate() -> None:
    from alembic import command
    from alembic.config import Config

    # Drop any _alembic_tmp_* tables left by a previously crashed batch-alter migration.
    # SQLite DDL can't be rolled back, so alembic leaves these on failure; sweeping them
    # here before every upgrade makes all batch-alter migrations idempotent on retry.
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '_alembic_tmp_%'")
        ).fetchall()
        for (name,) in rows:
            conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
        conn.commit()

    alembic_ini = Path(__file__).parent.parent / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini))
    command.upgrade(alembic_cfg, "head")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session


_TOOL_OUTPUT_MAX_BYTES = 8192


def upsert_analysis(run_id: str, ticker: str, data: AnalysisData) -> None:
    with Session(get_engine()) as session:
        session.execute(
            delete(Analysis).where(Analysis.run_id == run_id, Analysis.ticker == ticker)
        )
        session.add(Analysis(run_id=run_id, ticker=ticker, **data))
        session.commit()


def ensure_run_started(
    run_id: str, started_at: datetime, prompt_version_id: int | None = None
) -> None:
    """Insert a ``runs`` row, or reset an existing one to a fresh 'running' state.

    ``write_run_start`` raises on a duplicate id, which is the right guard for the nightly
    run (a UUID collision is a bug). The eval command deliberately reuses a pinned
    ``--eval-run-id`` across runs so the grades can be diffed, so it needs upsert
    semantics instead.
    """
    with Session(get_engine()) as session:
        run = session.get(Run, run_id)
        if run is None:
            session.add(
                Run(
                    id=run_id,
                    started_at=started_at,
                    status="running",
                    prompt_version_id=prompt_version_id,
                )
            )
        else:
            run.started_at = started_at
            run.status = "running"
            run.prompt_version_id = prompt_version_id
            run.completed_at = None
            run.error_msg = None
        session.commit()


def write_eval_run(
    run_id: str,
    example_ticker: str,
    passed: bool,
    check_results: str,
    diff_notes: str | None = None,
) -> None:
    """Persist one ticker's eval grade, idempotently.

    Delete-then-insert on (run_id, example_ticker) so re-running an eval under a fixed
    ``--eval-run-id`` overwrites rather than duplicates. ``eval_runs.run_id`` is an FK to
    ``runs.id``, so ``write_run_start`` must have run first.
    """
    with Session(get_engine()) as session:
        session.execute(
            delete(EvalRun).where(
                EvalRun.run_id == run_id, EvalRun.example_ticker == example_ticker
            )
        )
        session.add(
            EvalRun(
                run_id=run_id,
                example_ticker=example_ticker,
                passed=passed,
                check_results=check_results,
                diff_notes=diff_notes,
            )
        )
        session.commit()


def ensure_prompt_version(
    version_tag: str,
    persona_system_prompt: str,
    routing_policy_name: str,
) -> int:
    """Return the id of the matching PromptVersion row, inserting one if absent."""
    with Session(get_engine()) as session:
        existing = (
            session.query(PromptVersion)
            .filter_by(
                version_tag=version_tag,
                persona_system_prompt=persona_system_prompt,
                routing_policy_name=routing_policy_name,
            )
            .first()
        )
        if existing is not None:
            return int(existing.id)
        pv = PromptVersion(
            version_tag=version_tag,
            persona_system_prompt=persona_system_prompt,
            routing_policy_name=routing_policy_name,
        )
        session.add(pv)
        session.commit()
        return int(pv.id)


def write_run_start(
    run_id: str, started_at: datetime, prompt_version_id: int | None = None
) -> None:
    with Session(get_engine()) as session:
        session.add(
            Run(
                id=run_id,
                started_at=started_at,
                status="running",
                prompt_version_id=prompt_version_id,
            )
        )
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
    with Session(get_engine()) as session:
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
