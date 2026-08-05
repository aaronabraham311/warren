from datetime import date, datetime, timezone
from typing import Literal, NotRequired, TypedDict

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
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
    lynch_signals: dict[str, list[str]]
    buffett_signals: dict[str, list[str]]
    key_risks: list[str]
    data_quality_notes: list[str]
    tool_calls_made: int | None
    tokens_used: int | None
    termination_reason: NotRequired[str | None]


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (UniqueConstraint("run_id", "ticker", name="uq_analyses_run_ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(Text, ForeignKey("runs.id"), nullable=False)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_type: Mapped[str | None] = mapped_column(Text)
    recommendation: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    thesis: Mapped[str | None] = mapped_column(Text)
    lynch_signals: Mapped[dict[str, list[str]] | None] = mapped_column(JSON)
    buffett_signals: Mapped[dict[str, list[str]] | None] = mapped_column(JSON)
    key_risks: Mapped[list[str] | None] = mapped_column(JSON)
    data_quality_notes: Mapped[list[str] | None] = mapped_column(JSON)
    tool_calls_made: Mapped[int | None] = mapped_column(Integer)
    tokens_used: Mapped[int | None] = mapped_column(Integer)
    termination_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, default=_utcnow)


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(Text, ForeignKey("runs.id"))
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
    run_id: Mapped[str | None] = mapped_column(Text, ForeignKey("runs.id"))
    example_ticker: Mapped[str | None] = mapped_column(Text)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    check_results: Mapped[str | None] = mapped_column(Text)
    diff_notes: Mapped[str | None] = mapped_column(Text)


class UniverseSnapshot(Base):
    """Weekly-refreshed snapshot of a sorted screening universe, keyed by ``kind``.

    One row per universe kind (``"sp500"`` for the default nightly US universe,
    ``"gem_hunt"`` for the global 3-exchange universe). ``kind`` is the primary key,
    so an upsert-on-``kind`` keeps at most one row per kind — no cleanup job needed.
    ``tickers_json`` is a sorted JSON array; the weekly cadence exists purely to avoid
    a re-scrape of the constituent source, not for any prompt-prefix reason (the
    universe never enters an LLM prompt — screening is deterministic per-ticker Python
    in ``agent.screening``).
    """

    __tablename__ = "universe_snapshots"

    kind: Mapped[str] = mapped_column(Text, primary_key=True)
    tickers_json: Mapped[str] = mapped_column(Text, nullable=False)
    refreshed_at: Mapped[date] = mapped_column(Date, nullable=False)


class SecurityIdentityRecord(Base):
    """Stable ISIN-backed identity, including superseded historical mappings."""

    __tablename__ = "security_identities"

    venue: Mapped[str] = mapped_column(Text, primary_key=True)
    isin: Mapped[str] = mapped_column(Text, primary_key=True)
    canonical_ticker: Mapped[str] = mapped_column(Text, nullable=False)
    mic: Mapped[str | None] = mapped_column(Text)
    exchange_symbol: Mapped[str] = mapped_column(Text, nullable=False)
    legal_name: Mapped[str] = mapped_column(Text, nullable=False)
    identity_source_url: Mapped[str] = mapped_column(Text, nullable=False)
    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    superseded_by_isin: Mapped[str | None] = mapped_column(Text)


class FilingManifest(Base):
    """One immutable version of a discovered primary-filing artifact.

    ``filing_id`` is the source-stable document identity; ``checksum`` makes versions
    append-only when an upstream URL changes. Identical checksums may be referenced by
    multiple filing IDs so mirrored announcements deduplicate in ArtifactStore without
    losing discovery provenance.
    """

    __tablename__ = "filing_manifests"
    __table_args__ = (
        CheckConstraint("length(checksum) = 64", name="ck_filing_manifests_checksum_length"),
        CheckConstraint("byte_length >= 0", name="ck_filing_manifests_byte_length"),
        ForeignKeyConstraint(
            ["filing_id", "supersedes_checksum"],
            ["filing_manifests.filing_id", "filing_manifests.checksum"],
            name="fk_filing_manifests_supersedes",
        ),
    )

    filing_id: Mapped[str] = mapped_column(Text, primary_key=True)
    checksum: Mapped[str] = mapped_column(Text, primary_key=True)
    issuer_isin: Mapped[str | None] = mapped_column(Text)
    venue: Mapped[str] = mapped_column(Text, nullable=False)
    source_system: Mapped[str] = mapped_column(Text, nullable=False)
    upstream_id: Mapped[str | None] = mapped_column(Text)
    document_kind: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    publication_date: Mapped[date | None] = mapped_column(Date)
    reporting_period_end: Mapped[date | None] = mapped_column(Date)
    landing_page_url: Mapped[str | None] = mapped_column(Text)
    direct_document_url: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    byte_length: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    etag: Mapped[str | None] = mapped_column(Text)
    last_modified: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    source_language: Mapped[str | None] = mapped_column(Text)
    parser_version: Mapped[str | None] = mapped_column(Text)
    extraction_version: Mapped[str | None] = mapped_column(Text)
    translation_version: Mapped[str | None] = mapped_column(Text)
    extracted_text_checksum: Mapped[str | None] = mapped_column(Text)
    extracted_text_artifact_key: Mapped[str | None] = mapped_column(Text)
    translated_text_checksum: Mapped[str | None] = mapped_column(Text)
    translated_text_artifact_key: Mapped[str | None] = mapped_column(Text)
    artifact_key: Mapped[str] = mapped_column(Text, nullable=False)
    supersedes_checksum: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ForensicSnapshot(Base):
    """Reusable forensic extraction for one immutable filing-corpus version."""

    __tablename__ = "forensic_snapshots"

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    as_of: Mapped[date] = mapped_column(Date, primary_key=True)
    lookback_start: Mapped[date] = mapped_column(Date, primary_key=True)
    extractor_version: Mapped[str] = mapped_column(Text, primary_key=True)
    corpus_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    venue: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    evidence_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    coverage_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    warnings_json: Mapped[list[object]] = mapped_column(JSON, nullable=False)


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
Index(
    "idx_security_identities_current_ticker",
    SecurityIdentityRecord.canonical_ticker,
    SecurityIdentityRecord.is_active,
)
Index("idx_eval_runs_run", EvalRun.run_id)
Index("idx_security_identities_isin", SecurityIdentityRecord.isin)
Index("idx_filing_manifests_checksum", FilingManifest.checksum)
Index(
    "idx_filing_manifests_issuer_date",
    FilingManifest.issuer_isin,
    FilingManifest.retrieved_at.desc(),
)
Index(
    "idx_filing_manifests_document_versions",
    FilingManifest.filing_id,
    FilingManifest.retrieved_at.desc(),
)
Index(
    "idx_filing_manifests_selection",
    FilingManifest.issuer_isin,
    FilingManifest.document_kind,
    FilingManifest.reporting_period_end.desc(),
    FilingManifest.publication_date.desc(),
)
Index(
    "idx_forensic_snapshots_ticker_as_of",
    ForensicSnapshot.ticker,
    ForensicSnapshot.as_of.desc(),
    ForensicSnapshot.generated_at.desc(),
)
