import hashlib
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

_DB_URL = f"sqlite:///{os.environ.get('WARREN_DB', 'warren.db')}"
engine = create_engine(_DB_URL, echo=False)


@event.listens_for(engine, "connect")
def _set_wal_mode(dbapi_connection: Any, connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class Base(DeclarativeBase):
    pass


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_tag: Mapped[str] = mapped_column(Text, nullable=False)
    persona_system_prompt: Mapped[str | None] = mapped_column(Text)
    routing_policy_name: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str | None] = mapped_column(Text)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    prompt_version_id: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str | None] = mapped_column(Text)
    total_input_tokens: Mapped[int | None] = mapped_column(Integer)
    total_output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_cost_usd: Mapped[float | None] = mapped_column(Float)
    num_tool_calls: Mapped[int | None] = mapped_column(Integer)
    error_msg: Mapped[str | None] = mapped_column(Text)


class Holding(Base):
    __tablename__ = "holdings"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    shares: Mapped[float | None] = mapped_column(Float)
    cost_basis: Mapped[float | None] = mapped_column(Float)
    purchase_date: Mapped[date | None] = mapped_column(Date)
    current_price: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime)


class Watchlist(Base):
    __tablename__ = "watchlist"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    notes: Mapped[str | None] = mapped_column(Text)
    added_at: Mapped[datetime | None] = mapped_column(DateTime, default=datetime.utcnow)


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (UniqueConstraint("run_id", "ticker", name="uq_analyses_run_ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(Text)
    ticker: Mapped[str | None] = mapped_column(Text)
    analysis_type: Mapped[str | None] = mapped_column(Text)
    recommendation: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    thesis: Mapped[str | None] = mapped_column(Text)
    lynch_signals: Mapped[str | None] = mapped_column(Text)
    buffett_signals: Mapped[str | None] = mapped_column(Text)
    key_risks: Mapped[str | None] = mapped_column(Text)
    data_quality_notes: Mapped[str | None] = mapped_column(Text)
    tool_calls_made: Mapped[int | None] = mapped_column(Integer)
    tokens_used: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=datetime.utcnow)


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(Text)
    tool_name: Mapped[str | None] = mapped_column(Text)
    input_json: Mapped[str | None] = mapped_column(Text)
    output_json: Mapped[str | None] = mapped_column(Text)
    output_file_path: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cached: Mapped[int] = mapped_column(Integer, default=0)
    error_msg: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=datetime.utcnow)


class EvalExample(Base):
    __tablename__ = "eval_examples"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    expected_recommendation: Mapped[str | None] = mapped_column(Text)
    expected_thesis_keywords: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    last_curated: Mapped[date | None] = mapped_column(Date)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(Text)
    example_ticker: Mapped[str | None] = mapped_column(Text)
    passed: Mapped[int | None] = mapped_column(Integer)
    check_results: Mapped[str | None] = mapped_column(Text)
    diff_notes: Mapped[str | None] = mapped_column(Text)


class DiscoveryCooldown(Base):
    __tablename__ = "discovery_cooldown"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    flagged_at: Mapped[datetime | None] = mapped_column(DateTime)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    suppression_reason: Mapped[str | None] = mapped_column(Text)


# Performance indexes (Tech Spec §7.5) — defined after models so column refs resolve
Index("idx_analyses_ticker_created", Analysis.ticker, Analysis.created_at.desc())
Index("idx_analyses_run", Analysis.run_id)
Index("idx_tool_calls_run", ToolCall.run_id)
Index("idx_runs_started", Run.started_at.desc())
Index("idx_eval_runs_run", EvalRun.run_id)


def migrate() -> None:
    Base.metadata.create_all(engine)


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
