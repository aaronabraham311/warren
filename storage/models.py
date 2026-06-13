from datetime import date, datetime, timezone
from typing import Literal, TypedDict

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

RunStatus = Literal["running", "success", "cost_aborted", "failed"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_tag: Mapped[str] = mapped_column(Text, nullable=False)
    persona_system_prompt: Mapped[str | None] = mapped_column(Text)
    routing_policy_name: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=_utcnow)
    notes: Mapped[str | None] = mapped_column(Text)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    prompt_version_id: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[RunStatus | None] = mapped_column(Text)
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
    added_at: Mapped[datetime | None] = mapped_column(DateTime, default=_utcnow)


class AnalysisData(TypedDict):
    analysis_type: str | None
    recommendation: str | None
    confidence: float | None
    thesis: str | None
    lynch_signals: list[str]
    buffett_signals: list[str]
    key_risks: list[str]
    data_quality_notes: list[str]
    tool_calls_made: int | None
    tokens_used: int | None


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (UniqueConstraint("run_id", "ticker", name="uq_analyses_run_ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
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
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=_utcnow)


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(Text)
    tool_name: Mapped[str | None] = mapped_column(Text)
    input_json: Mapped[str | None] = mapped_column(Text)
    output_json: Mapped[str | None] = mapped_column(Text)
    output_file_path: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    error_msg: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=_utcnow)


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
    passed: Mapped[bool | None] = mapped_column(Boolean)
    check_results: Mapped[str | None] = mapped_column(Text)
    diff_notes: Mapped[str | None] = mapped_column(Text)


class DiscoveryCooldown(Base):
    __tablename__ = "discovery_cooldown"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    flagged_at: Mapped[datetime | None] = mapped_column(DateTime)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime)
    suppression_reason: Mapped[str | None] = mapped_column(Text)


Index("idx_analyses_ticker_created", Analysis.ticker, Analysis.created_at.desc())
Index("idx_analyses_run", Analysis.run_id)
Index("idx_tool_calls_run", ToolCall.run_id)
Index("idx_runs_started", Run.started_at.desc())
Index("idx_eval_runs_run", EvalRun.run_id)
