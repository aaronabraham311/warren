"""Persist the G18 DIRT decision contract and queryable outcome projections.

Revision ID: 6a95860c76eb
Revises: d7a4c8e2f913
Create Date: 2026-08-05 23:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "6a95860c76eb"
down_revision: Union[str, Sequence[str], None] = "d7a4c8e2f913"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Keep complete typed decision payloads for audit/replay while projecting the two
    # dashboard sort/filter fields into scalar columns. All columns remain nullable so
    # pre-G18 analyses and non-DIRT personas migrate without synthetic defaults.
    with op.batch_alter_table("analyses", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("dirt_signals", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("dirt_decision", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("decision_outcome", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("probability_weighted_irr", sa.Float(), nullable=True))


def downgrade() -> None:
    # Alembic CLI bypasses storage.engine.migrate()'s generic stale-table sweep.
    # Remove a table left by an interrupted prior downgrade before batch recreation.
    op.execute(sa.text('DROP TABLE IF EXISTS "_alembic_tmp_analyses"'))

    with op.batch_alter_table("analyses", recreate="always") as batch_op:
        batch_op.drop_column("probability_weighted_irr")
        batch_op.drop_column("decision_outcome")
        batch_op.drop_column("dirt_decision")
        batch_op.drop_column("dirt_signals")
