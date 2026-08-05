"""Persist immutable primary-filing artifact manifests and provenance.

Revision ID: a5e0b1e349aa
Revises: 4d6f8a1b2c3d
Create Date: 2026-08-05 18:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a5e0b1e349aa"
down_revision: Union[str, Sequence[str], None] = "4d6f8a1b2c3d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Source PDFs and derived text are too large for SQLite. This table keeps only
    # append-only version/provenance metadata pointing into the content-addressed store.
    # A new table needs no SQLite batch operation.
    op.create_table(
        "filing_manifests",
        sa.Column("filing_id", sa.Text(), nullable=False),
        sa.Column("checksum", sa.Text(), nullable=False),
        sa.Column("issuer_isin", sa.Text(), nullable=True),
        sa.Column("venue", sa.Text(), nullable=False),
        sa.Column("source_system", sa.Text(), nullable=False),
        sa.Column("landing_page_url", sa.Text(), nullable=True),
        sa.Column("direct_document_url", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=False),
        sa.Column("byte_length", sa.Integer(), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("etag", sa.Text(), nullable=True),
        sa.Column("last_modified", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("source_language", sa.Text(), nullable=True),
        sa.Column("parser_version", sa.Text(), nullable=True),
        sa.Column("extraction_version", sa.Text(), nullable=True),
        sa.Column("translation_version", sa.Text(), nullable=True),
        sa.Column("artifact_key", sa.Text(), nullable=False),
        sa.Column("supersedes_checksum", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("filing_id", "checksum"),
    )
    op.create_index("idx_filing_manifests_checksum", "filing_manifests", ["checksum"], unique=False)
    op.create_index(
        "idx_filing_manifests_issuer_date",
        "filing_manifests",
        ["issuer_isin", sa.text("retrieved_at DESC")],
        unique=False,
    )
    op.create_index(
        "idx_filing_manifests_document_versions",
        "filing_manifests",
        ["filing_id", sa.text("retrieved_at DESC")],
        unique=False,
    )


def downgrade() -> None:
    # Whole-table removal uses native SQLite DROP TABLE; no batch temp table is created.
    op.drop_index("idx_filing_manifests_document_versions", table_name="filing_manifests")
    op.drop_index("idx_filing_manifests_issuer_date", table_name="filing_manifests")
    op.drop_index("idx_filing_manifests_checksum", table_name="filing_manifests")
    op.drop_table("filing_manifests")
