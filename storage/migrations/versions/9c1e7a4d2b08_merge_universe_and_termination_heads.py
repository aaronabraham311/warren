"""merge universe_snapshots and termination_reason heads

Two migrations were authored independently off 423a829bf2d5 — f73924f4bb90
(universe_snapshots) and 370fa2bf2ff3 (analyses.termination_reason) — leaving the
revision graph with two heads. Alembic then refuses to resolve ``upgrade head``, so
``storage.engine.migrate()`` raised for every entrypoint that calls it. This is an
empty merge revision: it reconciles the graph and applies no schema change.

Revision ID: 9c1e7a4d2b08
Revises: 370fa2bf2ff3, f73924f4bb90
Create Date: 2026-07-09 00:00:00.000000

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "9c1e7a4d2b08"
down_revision: Union[str, Sequence[str], None] = ("370fa2bf2ff3", "f73924f4bb90")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
