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
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = os.getenv("DATABASE_URL", "")


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


engine = create_async_engine(
    _to_asyncpg_url(DATABASE_URL),
    connect_args={"statement_cache_size": 0},
    pool_pre_ping=True,
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
