"""add_universe_snapshots

Revision ID: f73924f4bb90
Revises: 423a829bf2d5
Create Date: 2026-06-28 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f73924f4bb90"
down_revision: Union[str, Sequence[str], None] = "423a829bf2d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "universe_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tickers_json", sa.Text(), nullable=False),
        sa.Column("refreshed_at", sa.Date(), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_universe_single_row"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("universe_snapshots")
