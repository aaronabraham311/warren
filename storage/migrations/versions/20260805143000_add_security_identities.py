"""Persist ISIN-backed junior-market identities for primary-filing routing.

Revision ID: 4d6f8a1b2c3d
Revises: 58a9dfef1fe7
Create Date: 2026-08-05 14:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "4d6f8a1b2c3d"
down_revision: Union[str, Sequence[str], None] = "58a9dfef1fe7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Universe refreshes already resolve stable exchange identifiers. Persisting
    # them prevents later regional-filing work from guessing an ISIN from Yahoo's
    # transport ticker. This is a new table, so SQLite batch mode is unnecessary.
    op.create_table(
        "security_identities",
        sa.Column("venue", sa.Text(), nullable=False),
        sa.Column("isin", sa.Text(), nullable=False),
        sa.Column("canonical_ticker", sa.Text(), nullable=False),
        sa.Column("mic", sa.Text(), nullable=True),
        sa.Column("exchange_symbol", sa.Text(), nullable=False),
        sa.Column("legal_name", sa.Text(), nullable=False),
        sa.Column("identity_source_url", sa.Text(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("superseded_by_isin", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("venue", "isin"),
    )
    op.create_index("idx_security_identities_isin", "security_identities", ["isin"], unique=False)
    op.create_index(
        "idx_security_identities_current_ticker",
        "security_identities",
        ["canonical_ticker", "is_active"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_security_identities_current_ticker", table_name="security_identities")
    op.drop_index("idx_security_identities_isin", table_name="security_identities")
    op.drop_table("security_identities")
