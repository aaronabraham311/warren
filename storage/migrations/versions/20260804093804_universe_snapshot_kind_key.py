"""universe_snapshots keyed by kind (sp500 | gem_hunt)

Gem-hunt mode needs its own weekly-cached universe alongside the default S&P 500 one.
The table was a single-row cache (``CHECK (id = 1)``), which cannot hold two universes.
This migration re-keys it on a ``kind`` primary key so each universe kind gets its own
row. ``universe_snapshots`` is a rebuildable weekly cache (``agent.universe`` re-fetches
on the next run), so dropping and recreating the table — rather than an in-place data
migration — is safe and simpler; any existing sp500 snapshot is simply re-fetched.

Revision ID: 58a9dfef1fe7
Revises: 9c1e7a4d2b08
Create Date: 2026-08-04 09:38:04.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "58a9dfef1fe7"
down_revision: Union[str, Sequence[str], None] = "9c1e7a4d2b08"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Re-key universe_snapshots on ``kind``. This is a whole-table drop+create (no
    # batch_alter_table needed), safe because the table is a rebuildable weekly cache.
    #
    # NOTE: storage.engine.migrate() sweeps all _alembic_tmp_* tables before calling
    # alembic upgrade, so upgrade() needs no stale-tmp guard here.
    op.drop_table("universe_snapshots")
    op.create_table(
        "universe_snapshots",
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("tickers_json", sa.Text(), nullable=False),
        sa.Column("refreshed_at", sa.Date(), nullable=False),
        sa.PrimaryKeyConstraint("kind"),
    )


def downgrade() -> None:
    # Restore the original single-row (id = 1) shape.
    op.drop_table("universe_snapshots")
    op.create_table(
        "universe_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tickers_json", sa.Text(), nullable=False),
        sa.Column("refreshed_at", sa.Date(), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_universe_single_row"),
        sa.PrimaryKeyConstraint("id"),
    )
