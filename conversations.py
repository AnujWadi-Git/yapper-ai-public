"""Conversation/message persistence helpers, used by the /chat and
/conversations endpoints in main.py.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Conversation, Message

TITLE_MAX_LEN = 60


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
