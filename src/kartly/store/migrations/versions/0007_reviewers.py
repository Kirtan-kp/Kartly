"""reviewers and admin sessions

Accounts for the admin interface, and the browsers signed in with them.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-15 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('reviewers',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('username', sa.String(length=32), nullable=False),
    sa.Column('password_hash', sa.String(length=200), nullable=False),
    sa.Column('role', sa.String(length=10), server_default='reviewer', nullable=False),
    sa.Column('active', sa.Boolean(), server_default=sa.true(), nullable=False),
    sa.Column('failed_attempts', sa.Integer(), server_default='0', nullable=False),
    sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_reviewers')),
    sa.UniqueConstraint('username', name=op.f('uq_reviewers_username'))
    )
    op.create_table('admin_sessions',
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('reviewer_id', sa.Uuid(), nullable=False),
    sa.Column('csrf', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['reviewer_id'], ['reviewers.id'], name=op.f('fk_admin_sessions_reviewer_id_reviewers'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('token_hash', name=op.f('pk_admin_sessions'))
    )
    op.create_index(op.f('ix_admin_sessions_reviewer_id'), 'admin_sessions', ['reviewer_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_admin_sessions_reviewer_id'), table_name='admin_sessions')
    op.drop_table('admin_sessions')
    op.drop_table('reviewers')
