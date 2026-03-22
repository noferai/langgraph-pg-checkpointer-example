import os
from contextlib import contextmanager
from typing import Generator
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not found in environment variables. Please set it in your .env file.")
    return url


@contextmanager
def get_checkpointer(
    setup: bool = False,
) -> Generator[PostgresSaver, None, None]:
    """Context manager for PostgreSQL checkpointer.

    Args:
        setup: If True, initialize database tables

    Yields:
        PostgresSaver instance
    """
    db_url = get_database_url()

    with PostgresSaver.from_conn_string(db_url) as checkpointer:
        if setup:
            checkpointer.setup()
        yield checkpointer


def initialize_database() -> None:
    """Initialize database tables for checkpointing."""
    print("Initializing PostgreSQL checkpointer database...")
    with get_checkpointer(setup=True) as checkpointer:
        print("✓ Database tables created successfully")
        print(f"  - Checkpoint table ready")
        print(f"  - Using database: {get_database_url()}")
