"""alert_states

Where each alert rule stands, so a rule that stays bad is announced once.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-14 09:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('alert_states',
    sa.Column('name', sa.String(length=60), nullable=False),
    sa.Column('status', sa.String(length=10), nullable=False),
    sa.Column('severity', sa.String(length=10), nullable=False),
    sa.Column('message', sa.String(length=300), nullable=False),
    sa.Column('details', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=True),
    sa.Column('since', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_checked', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('name', name=op.f('pk_alert_states'))
    )


def downgrade() -> None:
    op.drop_table('alert_states')
