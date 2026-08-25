import asyncio
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import auth
import conversations
import rag
from db import get_db
from models import Conversation, Message, User
from tools import TOOL_SPECS, execute_tool_call

# Voice output is browser SpeechSynthesis only in this build (see README) --
# ElevenLabs and SadTalker are deliberately not part of the public product;
# both exist in the original single-user project for local-only use.

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="Yapper AI")

# Cookie-based auth means credentialed requests, which browsers refuse to
# pair with a wildcard origin -- so this must be a real allowlist, not "*".
_default_origins = "http://localhost:8000,http://127.0.0.1:8000,https://yapper-ai-public.onrender.com"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warn_if_model_may_not_support_tools() -> None:
    # The tool-use layer only works if OPENROUTER_MODEL can call functions.
    # We can't detect that reliably, so just surface the configured model.
    model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    print(
        f"[tools] {len(TOOL_SPECS)} tool(s) registered; using model '{model}'. "
        "If the model does not support tool calling, tools are silently ignored. "
        "See https://openrouter.ai/models?supported_parameters=tools"
    )


class ChatRequest(BaseModel):
    text: str
    conversation_id: str | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime


class SignupRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    email: str


SYSTEM_PROMPT = (
    "You are Yapper AI, a helpful assistant that talks naturally like a thoughtful human. "
    "Your default style should feel like a calm, smart person speaking out loud, not a stiff chatbot. "
    "Use contractions like I'm, you're, that's, and let's. "
    "For casual messages, reply in 1 to 3 natural sentences. "
    "For detailed questions, answer like ChatGPT: clear, useful, step by step, and structured only when helpful. "
    "Do not overuse bullets unless the user asks for an explanation, plan, list, or code. "
    "React to the user's tone briefly before helping, but do not be fake or dramatic. "
    "Ask a short follow-up question when it would make the conversation feel natural. "
    "Do not use emojis unless the user asks for them. "
    "If you are unsure, say so clearly instead of making things up."
)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.post("/auth/signup", response_model=AuthResponse)
async def signup(payload: SignupRequest, response: Response, db: AsyncSession = Depends(get_db)):
    email = payload.email.strip().lower()

    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status_code=400, detail="An account with that email already exists.")

    user = User(email=email, password_hash=auth.hash_password(payload.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)

    auth.set_session_cookie(response, auth.create_access_token(str(user.id)))
    return AuthResponse(email=user.email)


@app.post("/auth/login", response_model=AuthResponse)
async def login(
    payload: LoginRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db)
):
    auth.enforce_login_rate_limit(request)

    email = payload.email.strip().lower()
    user = await db.scalar(select(User).where(User.email == email))
    if not user or not auth.verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    auth.set_session_cookie(response, auth.create_access_token(str(user.id)))
    return AuthResponse(email=user.email)


@app.post("/auth/logout")
def logout(response: Response):
    auth.clear_session_cookie(response)
    return {"ok": True}


@app.get("/auth/me", response_model=AuthResponse)
async def me(user_id: str = Depends(auth.get_current_user_id), db: AsyncSession = Depends(get_db)):
    user = await db.get(User, uuid.UUID(user_id))
    if not user:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    return AuthResponse(email=user.email)


@app.post("/documents/upload")
async def upload_document(file: UploadFile, user_id: str = Depends(auth.get_current_user_id)):
    content = await file.read()
    try:
        result = await asyncio.to_thread(
            rag.document_store.add_document, user_id, file.filename or "document", content
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return result


@app.get("/documents")
def list_documents(user_id: str = Depends(auth.get_current_user_id)):
    return {"documents": rag.document_store.list_documents(user_id)}


@app.post("/chat")
async def chat(
    request: ChatRequest,
    user_id: str = Depends(auth.get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    user_text = request.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Text is required.")

    if request.conversation_id:
        conversation = await conversations.get_owned_conversation(db, request.conversation_id, user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        history = await conversations.load_history(db, conversation.id)
    else:
        conversation = await conversations.create_conversation(db, user_id, user_text)
        history = []

    reply, tools_used = await safe_call_openrouter(user_text, history, user_id)

    conversations.save_message(db, conversation.id, "user", user_text)
    conversations.save_message(db, conversation.id, "assistant", reply)
    await db.commit()

    return {
        "reply": reply,
        "text": reply,
        "tools_used": tools_used,
        "conversation_id": str(conversation.id),
    }


@app.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    user_id: str = Depends(auth.get_current_user_id), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == uuid.UUID(user_id))
        .order_by(Conversation.created_at.desc())
    )
    return [
        ConversationSummary(id=str(c.id), title=c.title, created_at=c.created_at)
        for c in result.scalars().all()
    ]


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def get_conversation_messages(
    conversation_id: str,
    user_id: str = Depends(auth.get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    conversation = await conversations.get_owned_conversation(db, conversation_id, user_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    result = await db.execute(
        select(Message).where(Message.conversation_id == conversation.id).order_by(Message.created_at)
    )
    return [
        MessageOut(role=m.role, content=m.content, created_at=m.created_at)
        for m in result.scalars().all()
    ]


async def safe_call_openrouter(
    user_text: str, history: list[dict[str, str]], user_id: str
) -> tuple[str, list[str]]:
    try:
        return await call_openrouter(user_text, history, user_id)
    except Exception as error:
        print(f"OpenRouter failed: {error}")
        return (
            "Sorry, I had trouble reaching my AI brain just now. Try again in a moment.",
            [],
        )


# Safety cap on LLM -> tool -> LLM round-trips within a single /chat turn.
MAX_TOOL_ROUNDS = 5


async def call_openrouter(
    user_text: str, history: list[dict[str, str]], user_id: str
) -> tuple[str, list[str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    context_block = await asyncio.to_thread(rag.build_context_block, user_id, user_text)
    if context_block:
        messages.append({"role": "system", "content": context_block})

    messages.extend(clean_history(history))
    messages.append({"role": "user", "content": user_text})

    tools_used: list[str] = []  # names of tools called this turn, for the UI

    # LLM -> optional tool call(s) -> tool result(s) -> LLM, until a plain reply.
    for _ in range(MAX_TOOL_ROUNDS):
        message = await _post_openrouter(messages, use_tools=True)
        tool_calls = message.get("tool_calls")

        if not tool_calls:
            return clean_ai_text(message.get("content") or ""), tools_used

        # Record the assistant turn that asked for tools, then answer each call.
        messages.append(message)
        for tool_call in tool_calls:
            tools_used.append(tool_call.get("function", {}).get("name", "unknown"))
            result = await asyncio.to_thread(execute_tool_call, tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": result,
            })

    # Exhausted tool rounds; force one final text-only reply so TTS gets something.
    message = await _post_openrouter(messages, use_tools=False)
    return clean_ai_text(message.get("content") or ""), tools_used


async def _post_openrouter(messages: list[dict], use_tools: bool) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise RuntimeError("Missing real OPENROUTER_API_KEY.")

    model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
        "X-Title": os.getenv("APP_NAME", "Yapper AI"),
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1200,
    }
    if use_tools:
        payload["tools"] = TOOL_SPECS

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
    except httpx.TimeoutException as error:
        raise RuntimeError("OpenRouter took too long to answer.") from error
    except httpx.RequestError as error:
        raise RuntimeError(f"Could not reach OpenRouter: {error}") from error

    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter error: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]


def clean_ai_text(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F1E6-\U0001F1FF"
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text).strip()


def clean_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned = []

    for message in history[-10:]:
        role = message.get("role")
        content = message.get("content", "").strip()

        if role not in {"user", "assistant"} or not content:
            continue

        cleaned.append({"role": role, "content": content[:1200]})

    return cleaned

