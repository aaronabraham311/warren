"""Persist versioned forensic evidence snapshots for the regional filing corpus.

Revision ID: d7a4c8e2f913
Revises: c4aac1e13582
Create Date: 2026-08-05 22:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d7a4c8e2f913"
down_revision: Union[str, Sequence[str], None] = "c4aac1e13582"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A snapshot is reusable only for the exact source corpus and extractor version.
    # Full filing artifacts stay in ArtifactStore; SQLite contains bounded evidence,
    # coverage, warning JSON, and immutable source IDs/hashes only.
    op.create_table(
        "forensic_snapshots",
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("issuer_isin", sa.Text(), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("lookback_start", sa.Date(), nullable=False),
        sa.Column("extractor_version", sa.Text(), nullable=False),
        sa.Column("corpus_hash", sa.Text(), nullable=False),
        sa.Column("venue", sa.Text(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("coverage_json", sa.JSON(), nullable=False),
        sa.Column("warnings_json", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint(
            "ticker",
            "issuer_isin",
            "venue",
            "as_of",
            "lookback_start",
            "extractor_version",
            "corpus_hash",
        ),
    )
    op.create_index(
        "idx_forensic_snapshots_ticker_as_of",
        "forensic_snapshots",
        ["ticker", sa.text("as_of DESC"), sa.text("generated_at DESC")],
        unique=False,
    )


def downgrade() -> None:
    # No batch-table guard is needed: downgrade drops the new table as a whole.
    op.drop_index("idx_forensic_snapshots_ticker_as_of", table_name="forensic_snapshots")
    op.drop_table("forensic_snapshots")
