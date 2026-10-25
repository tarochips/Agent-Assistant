"""Track which embedding model produced each stored chunk vector."""

import sqlalchemy as sa
from alembic import op

revision = "0002_embedding_model"
down_revision = "0001_postgres_pgvector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column(
            "embedding_model",
            sa.String(length=256),
            server_default="legacy-char-hash-384-v1",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("chunks", "embedding_model")
