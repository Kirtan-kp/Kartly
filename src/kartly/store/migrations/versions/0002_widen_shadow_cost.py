"""widen shadow cost columns

A single small LLM call costs a few millionths of a dollar at reference prices, so six decimal
places rounded away up to half of it. Ten places keep per-call costs and their sums accurate.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-26 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("runs", "run_steps")


def _alter(old: tuple[int, int], new: tuple[int, int]) -> None:
    for table in TABLES:
        with op.batch_alter_table(table) as batch:
            batch.alter_column(
                "shadow_cost_usd",
                existing_type=sa.Numeric(*old),
                type_=sa.Numeric(*new),
                existing_nullable=False,
                existing_server_default=sa.text("0"),
            )


def upgrade() -> None:
    _alter(old=(12, 6), new=(16, 10))


def downgrade() -> None:
    _alter(old=(16, 10), new=(12, 6))
