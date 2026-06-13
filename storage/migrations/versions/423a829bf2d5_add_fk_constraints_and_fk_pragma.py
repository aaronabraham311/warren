"""add_fk_constraints_and_fk_pragma

Revision ID: 423a829bf2d5
Revises: 403609d94106
Create Date: 2026-06-13 10:10:03.311354

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "423a829bf2d5"
down_revision: Union[str, Sequence[str], None] = "403609d94106"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite can't add FK constraints via ALTER TABLE; use batch mode to recreate tables.
    # Stale _alembic_tmp_* tables from a prior crashed run are swept generically by
    # storage.engine.migrate() before alembic upgrade is called.
    with op.batch_alter_table("analyses", recreate="always") as batch_op:
        batch_op.create_foreign_key("fk_analyses_run_id", "runs", ["run_id"], ["id"])

    with op.batch_alter_table("tool_calls", recreate="always") as batch_op:
        batch_op.create_foreign_key("fk_tool_calls_run_id", "runs", ["run_id"], ["id"])

    with op.batch_alter_table("eval_runs", recreate="always") as batch_op:
        batch_op.create_foreign_key("fk_eval_runs_run_id", "runs", ["run_id"], ["id"])


def downgrade() -> None:
    # Guard against stale tmp tables from a previously crashed downgrade (alembic CLI
    # bypasses storage.engine.migrate(), so we sweep here for the three tables we touch).
    for tmp in ("_alembic_tmp_eval_runs", "_alembic_tmp_tool_calls", "_alembic_tmp_analyses"):
        op.execute(f'DROP TABLE IF EXISTS "{tmp}"')

    with op.batch_alter_table("eval_runs", recreate="always") as batch_op:
        batch_op.drop_constraint("fk_eval_runs_run_id", type_="foreignkey")

    with op.batch_alter_table("tool_calls", recreate="always") as batch_op:
        batch_op.drop_constraint("fk_tool_calls_run_id", type_="foreignkey")

    with op.batch_alter_table("analyses", recreate="always") as batch_op:
        batch_op.drop_constraint("fk_analyses_run_id", type_="foreignkey")
