"""Add durable research jobs."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003_research_jobs"
down_revision = "0002_embedding_model"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("lease_token", sa.String(32), nullable=True),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_research_job_status",
        ),
    )
    op.create_index("ix_research_jobs_status_created", "research_jobs", ["status", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_research_jobs_status_created", table_name="research_jobs")
    op.drop_table("research_jobs")
