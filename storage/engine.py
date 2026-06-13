import hashlib
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, delete, event
from sqlalchemy.orm import Session

from storage.models import Analysis

_DB_URL = f"sqlite:///{os.environ.get('WARREN_DB', 'warren.db')}"
engine = create_engine(_DB_URL, echo=False)


@event.listens_for(engine, "connect")
def _set_wal_mode(dbapi_connection: Any, connection_record: Any) -> None:
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


def upsert_analysis(run_id: str, ticker: str, data: dict[str, Any]) -> None:
    with Session(engine) as session:
        session.execute(
            delete(Analysis).where(Analysis.run_id == run_id, Analysis.ticker == ticker)
        )
        session.add(Analysis(run_id=run_id, ticker=ticker, **data))
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
