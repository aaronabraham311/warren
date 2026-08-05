"""Add filing selection metadata and derived-text artifact references.

Revision ID: c4aac1e13582
Revises: a5e0b1e349aa
Create Date: 2026-08-05 21:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c4aac1e13582"
down_revision: Union[str, Sequence[str], None] = "a5e0b1e349aa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Regional read_filing selection must use discovery kind/reporting dates rather
    # than arbitrary retrieval order. Derived text remains content-addressed outside
    # SQLite; the manifest stores only checksums and relative artifact keys.
    with op.batch_alter_table("filing_manifests", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("upstream_id", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("document_kind", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("title", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("publication_date", sa.Date(), nullable=True))
        batch_op.add_column(sa.Column("reporting_period_end", sa.Date(), nullable=True))
        batch_op.add_column(sa.Column("extracted_text_checksum", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("extracted_text_artifact_key", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("translated_text_checksum", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("translated_text_artifact_key", sa.Text(), nullable=True))
    # Batch reflection loses SQLite DESC index terms, so restore both pre-existing
    # provenance indexes explicitly after the table copy.
    op.drop_index("idx_filing_manifests_issuer_date", table_name="filing_manifests")
    op.drop_index("idx_filing_manifests_document_versions", table_name="filing_manifests")
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
    op.create_index(
        "idx_filing_manifests_selection",
        "filing_manifests",
        [
            "issuer_isin",
            "document_kind",
            sa.text("reporting_period_end DESC"),
            sa.text("publication_date DESC"),
        ],
        unique=False,
    )


def downgrade() -> None:
    # Alembic CLI bypasses storage.engine.migrate()'s stale temporary-table sweep.
    op.execute(sa.text('DROP TABLE IF EXISTS "_alembic_tmp_filing_manifests"'))
    op.drop_index("idx_filing_manifests_selection", table_name="filing_manifests")
    with op.batch_alter_table("filing_manifests", recreate="always") as batch_op:
        batch_op.drop_column("translated_text_artifact_key")
        batch_op.drop_column("translated_text_checksum")
        batch_op.drop_column("extracted_text_artifact_key")
        batch_op.drop_column("extracted_text_checksum")
        batch_op.drop_column("reporting_period_end")
        batch_op.drop_column("publication_date")
        batch_op.drop_column("title")
        batch_op.drop_column("document_kind")
        batch_op.drop_column("upstream_id")
    op.drop_index("idx_filing_manifests_issuer_date", table_name="filing_manifests")
    op.drop_index("idx_filing_manifests_document_versions", table_name="filing_manifests")
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
