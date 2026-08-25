"""Async SQLAlchemy engine/session setup, backed by Supabase Postgres.

Connects through Supabase's Supavisor pooler in transaction mode (port 6543),
which is what makes Postgres reachable from Render's free tier (IPv4-only)
and keeps connection counts low across cold starts. Transaction-mode pooling
doesn't support server-side prepared statements, so asyncpg's statement
cache is disabled below -- without that, queries intermittently fail with
"prepared statement already exists" errors.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

load_dotenv(Path(__file__).resolve().parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "")


def _unique_statement_name() -> str:
    # asyncpg's default auto-generated names are short sequential counters
    # per connection object, so under Supavisor's transaction-mode pooling
    # (many client connections multiplexed onto few backend server
    # connections) two different clients can generate the same name and
    # collide on a backend that still has an old statement of that name
    # sitting around -- "prepared statement already exists". Unique names
    # sidestep that; see https://github.com/MagicStack/asyncpg/issues/837.
    return f"__asyncpg_{uuid.uuid4().hex}__"


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


engine = create_async_engine(
    _to_asyncpg_url(DATABASE_URL),
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_name_func": _unique_statement_name,
    },
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
