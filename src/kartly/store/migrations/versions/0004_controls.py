"""controls

Operator switches (the agent, writes) that take effect without a deploy.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-11 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('controls',
    sa.Column('name', sa.String(length=40), nullable=False),
    sa.Column('value', sa.String(length=20), nullable=False),
    sa.Column('reason', sa.String(length=300), nullable=True),
    sa.Column('updated_by', sa.String(length=64), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('name', name=op.f('pk_controls'))
    )


def downgrade() -> None:
    op.drop_table('controls')
