"""Conversation/message persistence helpers, used by the /chat and
/conversations endpoints in main.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Conversation, Message

TITLE_MAX_LEN = 60

# Tune based on real OpenRouter free-tier headroom once multiple users are
# live -- the shared upstream limit (a few hundred requests/day across ALL
# users) is the real ceiling; this per-user cap just keeps one user from
# eating the whole thing.
DAILY_MESSAGE_LIMIT = 20


def make_title(first_message: str) -> str:
    text = " ".join(first_message.split())
    if not text:
        return "New conversation"
    if len(text) <= TITLE_MAX_LEN:
        return text
    return text[: TITLE_MAX_LEN - 1].rstrip() + "…"


async def get_owned_conversation(
    db: AsyncSession, conversation_id: str, user_id: str
) -> Conversation | None:
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError:
        return None

    conversation = await db.get(Conversation, conversation_uuid)
    if conversation is None or str(conversation.user_id) != user_id:
        return None
    return conversation


async def create_conversation(db: AsyncSession, user_id: str, first_message: str) -> Conversation:
    conversation = Conversation(user_id=uuid.UUID(user_id), title=make_title(first_message))
    db.add(conversation)
    await db.flush()  # assigns conversation.id without committing yet
    return conversation


async def load_history(db: AsyncSession, conversation_id: uuid.UUID) -> list[dict[str, str]]:
    result = await db.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    )
    return [{"role": message.role, "content": message.content} for message in result.scalars().all()]


def save_message(db: AsyncSession, conversation_id: uuid.UUID, role: str, content: str) -> None:
    db.add(Message(conversation_id=conversation_id, role=role, content=content))


async def count_user_messages_today(db: AsyncSession, user_id: str) -> int:
    """Count this user's sent (role='user') messages since UTC midnight.

    Reuses the existing messages table as the counter instead of adding a
    separate rate-limit table -- one less piece of state to keep in sync,
    and it naturally resets at midnight with no cron/cleanup job needed.
    """
    # Message.created_at is TIMESTAMP WITHOUT TIME ZONE (naive, stored as
    # UTC by Postgres's `now()`), so the comparison value must be naive too
    # -- an aware datetime here raises "can't compare offset-naive and
    # offset-aware datetimes" against asyncpg.
    start_of_day = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    result = await db.execute(
        select(func.count(Message.id))
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Conversation.user_id == uuid.UUID(user_id),
            Message.role == "user",
            Message.created_at >= start_of_day,
        )
    )
    return result.scalar_one()
