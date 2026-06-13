"""initial_schema

Revision ID: 403609d94106
Revises:
Create Date: 2026-06-12 22:37:01.320934

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "403609d94106"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version_tag", sa.Text(), nullable=False),
        sa.Column("persona_system_prompt", sa.Text(), nullable=True),
        sa.Column("routing_policy_name", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("prompt_version_id", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("total_input_tokens", sa.Integer(), nullable=True),
        sa.Column("total_output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_cost_usd", sa.Float(), nullable=True),
        sa.Column("num_tool_calls", sa.Integer(), nullable=True),
        sa.Column("error_msg", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_runs_started", "runs", [sa.text("started_at DESC")])
    op.create_table(
        "holdings",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("shares", sa.Float(), nullable=True),
        sa.Column("cost_basis", sa.Float(), nullable=True),
        sa.Column("purchase_date", sa.Date(), nullable=True),
        sa.Column("current_price", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("ticker"),
    )
    op.create_table(
        "watchlist",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("added_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("ticker"),
    )
    op.create_table(
        "analyses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("ticker", sa.Text(), nullable=True),
        sa.Column("analysis_type", sa.Text(), nullable=True),
        sa.Column("recommendation", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("thesis", sa.Text(), nullable=True),
        sa.Column("lynch_signals", sa.Text(), nullable=True),
        sa.Column("buffett_signals", sa.Text(), nullable=True),
        sa.Column("key_risks", sa.Text(), nullable=True),
        sa.Column("data_quality_notes", sa.Text(), nullable=True),
        sa.Column("tool_calls_made", sa.Integer(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "ticker", name="uq_analyses_run_ticker"),
    )
    op.create_index("idx_analyses_run", "analyses", ["run_id"])
    op.create_index(
        "idx_analyses_ticker_created", "analyses", [sa.text("ticker"), sa.text("created_at DESC")]
    )
    op.create_table(
        "tool_calls",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("tool_name", sa.Text(), nullable=True),
        sa.Column("input_json", sa.Text(), nullable=True),
        sa.Column("output_json", sa.Text(), nullable=True),
        sa.Column("output_file_path", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("cached", sa.Integer(), nullable=False),
        sa.Column("error_msg", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_tool_calls_run", "tool_calls", ["run_id"])
    op.create_table(
        "eval_examples",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("expected_recommendation", sa.Text(), nullable=True),
        sa.Column("expected_thesis_keywords", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("last_curated", sa.Date(), nullable=True),
        sa.PrimaryKeyConstraint("ticker"),
    )
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("example_ticker", sa.Text(), nullable=True),
        sa.Column("passed", sa.Integer(), nullable=True),
        sa.Column("check_results", sa.Text(), nullable=True),
        sa.Column("diff_notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_eval_runs_run", "eval_runs", ["run_id"])
    op.create_table(
        "discovery_cooldown",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("flagged_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("suppression_reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("ticker"),
    )


def downgrade() -> None:
    op.drop_table("discovery_cooldown")
    op.drop_index("idx_eval_runs_run", table_name="eval_runs")
    op.drop_table("eval_runs")
    op.drop_table("eval_examples")
    op.drop_index("idx_tool_calls_run", table_name="tool_calls")
    op.drop_table("tool_calls")
    op.drop_index("idx_analyses_ticker_created", table_name="analyses")
    op.drop_index("idx_analyses_run", table_name="analyses")
    op.drop_table("analyses")
    op.drop_table("watchlist")
    op.drop_table("holdings")
    op.drop_index("idx_runs_started", table_name="runs")
    op.drop_table("runs")
    op.drop_table("prompt_versions")
