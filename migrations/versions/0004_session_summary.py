"""Store incremental conversation summaries without modifying original messages."""

import sqlalchemy as sa
from alembic import op

revision = "0004_session_summary"
down_revision = "0003_research_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sessions", sa.Column("context_summary", sa.Text(), nullable=False, server_default="")
    )
    op.add_column(
        "sessions",
        sa.Column("summary_message_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_check_constraint("ck_session_summary_count", "sessions", "summary_message_count >= 0")


def downgrade() -> None:
    op.drop_constraint("ck_session_summary_count", "sessions", type_="check")
    op.drop_column("sessions", "summary_message_count")
    op.drop_column("sessions", "context_summary")
