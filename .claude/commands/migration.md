# Warren migration skill

Warren uses SQLite + Alembic. SQLite has constraints that make migrations subtler
than Postgres — this skill scaffolds new migrations correctly and reviews existing
ones for the class of bugs we've already been burned by.

## How to invoke

- `/migration` — review the most-recently modified file in `storage/migrations/versions/`
- `/migration <path>` — review that specific migration file
- `/migration new <description>` — scaffold a new migration with best practices baked in
- `/migration new <description> --tables t1 t2` — scaffold touching specific tables

---

## Mode A — Review an existing migration

Determine the target file:
- If a path was given, use it.
- Otherwise, find the most-recently modified `.py` in `storage/migrations/versions/`
  (exclude `__pycache__`).

Read the file in full, then check every item below and report findings grouped by
severity (Bug / Warning / Info). For each finding quote the relevant line(s) and
give a one-sentence fix.

### Checklist

**Bug — will break on SQLite**
1. `op.alter_column()` or `op.add_column()` outside a `batch_alter_table` block —
   SQLite does not support `ALTER TABLE … ALTER COLUMN` or `ALTER TABLE … ADD NOT NULL
   COLUMN`. Any such call must be wrapped in `with op.batch_alter_table(…, recreate="always")`.
2. `op.drop_constraint()` outside batch — same restriction.
3. `op.add_constraint()` / `op.create_foreign_key()` outside batch — same.
4. Raw `sa.text("ALTER TABLE …")` used directly instead of Alembic ops.

**Bug — crash-retry will deadlock**
5. `downgrade()` uses `batch_alter_table` but has no stale-tmp-table guard at the
   top. Required pattern (alembic CLI bypasses `engine.migrate()`'s generic sweep):
   ```python
   for tmp in ("_alembic_tmp_tbl1", "_alembic_tmp_tbl2"):
       op.execute(f'DROP TABLE IF EXISTS "{tmp}"')
   ```
   The list must name every table that `downgrade()` calls `batch_alter_table` on.
   (`upgrade()` does NOT need this guard — `storage.engine.migrate()` sweeps all
   `_alembic_tmp_*` tables generically before calling `command.upgrade()`.)

**Bug — FK migration will fail on non-empty DB**
6. Migration adds a FK constraint on a table that may contain rows whose `run_id`
   (or other FK column) doesn't exist in the parent table. Adding a FK via batch
   mode copies rows into the new tmp table with the constraint active — orphaned
   rows cause `FOREIGN KEY constraint failed`. Fix: add a pre-flight DELETE of
   orphaned rows before the `batch_alter_table` block, or note that the operator
   must clean data first.

**Warning — autogenerate will produce broken output**
7. Read `storage/migrations/env.py`. If either `context.configure()` call is missing
   `render_as_batch=True`, flag it. Without it, `alembic revision --autogenerate`
   emits bare `ALTER TABLE` statements that fail on SQLite.

**Warning — fragile SQL**
8. `op.execute(f"… {var} …")` where `var` is not a compile-time string literal —
   identifier should be quoted: `f'… "{var}" …'`.

**Info — style**
9. `upgrade()` and `downgrade()` don't mirror each other's table list or operation
   order (downgrade should be the exact reverse of upgrade for FK/constraint ops).
10. Migration has no docstring comment explaining *why* the schema is changing (not
    just what — the what is in the ops).

After the checklist, print a summary line: `✓ No issues` or `N issue(s) found`.
If there are bugs, offer to apply the fixes.

---

## Mode B — Scaffold a new migration

When the user says `/migration new <description>`, do the following:

### Step 1 — gather context
- Run `uv run alembic heads` (or read the latest revision from the most-recent
  migration file) to get the current head revision ID.
- Read `storage/models.py` to understand the current ORM models.
- If `--tables` was given, read only those table sections; otherwise infer from the
  description which tables are affected.

### Step 2 — generate the file

Create `storage/migrations/versions/<timestamp>_<slug>.py` where:
- `<timestamp>` is 14 digits: `YYYYMMDDHHMMSS`
- `<slug>` is the description lowercased, spaces→underscores, max 40 chars

Use this exact template, filling in the blanks:

```python
"""<description>

Revision ID: <leave blank — alembic will fill this in; use a placeholder like 'REPLACE_ME'>
Revises: <current head revision id>
Create Date: <ISO datetime>

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "REPLACE_ME"
down_revision: Union[str, Sequence[str], None] = "<current head>"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # <describe what this migration does and why>
    #
    # NOTE: storage.engine.migrate() sweeps all _alembic_tmp_* tables before
    # calling alembic upgrade, so upgrade() needs no stale-tmp guard here.
    #
    # <generated ops go here>
    # Remember: all ALTER TABLE / DROP CONSTRAINT / ADD CONSTRAINT ops on SQLite
    # MUST be inside `with op.batch_alter_table("tbl", recreate="always") as batch_op:`


def downgrade() -> None:
    # <describe what this undoes>
    #
    # Drop stale tmp tables from a previously crashed downgrade (alembic CLI
    # bypasses engine.migrate(), so we guard here for every table we batch-alter).
    for tmp in (<quoted list of "_alembic_tmp_<tbl>" for each table downgrade touches>):
        op.execute(f'DROP TABLE IF EXISTS "{tmp}"')

    # <reverse of upgrade() ops, in reverse order>
```

### Step 3 — generate the actual ops

Apply these rules when writing the ops:

- **Adding/removing columns, indexes, constraints**: always use `batch_alter_table`.
- **Creating new tables from scratch** (`op.create_table`): no batch needed.
- **Dropping whole tables** (`op.drop_table`): no batch needed.
- **Adding a FK** to an existing table: use `batch_op.create_foreign_key(…)` inside
  batch. Also add a pre-flight orphan-row check comment:
  ```python
  # PRE-FLIGHT: ensure no orphaned rows exist before adding FK or migration will fail.
  # op.execute("DELETE FROM <child> WHERE <fk_col> NOT IN (SELECT id FROM <parent>)")
  ```
  Leave the DELETE commented out — the operator decides whether to run it.
- **Dropping a FK**: use `batch_op.drop_constraint(name, type_="foreignkey")` inside batch.
- Use `sa.text()` for any raw SQL inside migrations, not bare strings.

### Step 4 — generate the revision ID

Run `python -c "import uuid; print(uuid.uuid4().hex[:12])"` to get a 12-char hex
string and substitute it for `REPLACE_ME` in both `revision` and the filename.

### Step 5 — confirm

Print the full file content, then ask: "Does this look right? I'll write it to disk
once you confirm." Write the file only after the user says yes (or equivalent).
Then remind them to run `ruff format storage/migrations/versions/<file>.py` and
`uv run alembic upgrade head` to apply it.
