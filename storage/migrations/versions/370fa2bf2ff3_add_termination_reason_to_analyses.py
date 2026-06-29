"""add_termination_reason_to_analyses

Revision ID: 370fa2bf2ff3
Revises: 423a829bf2d5
Create Date: 2026-06-28 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "370fa2bf2ff3"
down_revision: Union[str, Sequence[str], None] = "423a829bf2d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("analyses", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("termination_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    for tmp in ("_alembic_tmp_analyses",):
        op.execute(f'DROP TABLE IF EXISTS "{tmp}"')

    with op.batch_alter_table("analyses", recreate="always") as batch_op:
        batch_op.drop_column("termination_reason")
