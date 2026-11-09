"""approvals

One row per proposed action that needs a human decision, bound to a hash of the exact action.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-08 18:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('approvals',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('action_id', sa.Uuid(), nullable=False),
    sa.Column('ticket_id', sa.Uuid(), nullable=False),
    sa.Column('payload_hash', sa.String(length=64), nullable=False),
    sa.Column('reason', sa.String(length=60), nullable=False),
    sa.Column('status', sa.Enum('pending', 'approved', 'rejected', 'expired', name='approval_status', native_enum=False, create_constraint=True, length=32), server_default='pending', nullable=False),
    sa.Column('requested_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('decided_by', sa.String(length=64), nullable=True),
    sa.Column('decided_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('decision_note', sa.String(length=300), nullable=True),
    sa.ForeignKeyConstraint(['action_id'], ['actions.id'], name=op.f('fk_approvals_action_id_actions'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['ticket_id'], ['tickets.id'], name=op.f('fk_approvals_ticket_id_tickets'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_approvals')),
    sa.UniqueConstraint('action_id', name=op.f('uq_approvals_action_id'))
    )
    op.create_index(op.f('ix_approvals_status'), 'approvals', ['status'], unique=False)
    op.create_index(op.f('ix_approvals_ticket_id'), 'approvals', ['ticket_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_approvals_ticket_id'), table_name='approvals')
    op.drop_index(op.f('ix_approvals_status'), table_name='approvals')
    op.drop_table('approvals')
