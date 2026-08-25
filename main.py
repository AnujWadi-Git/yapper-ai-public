import asyncio
import os
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import edge_tts
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import auth
import rag
from db import get_db
from models import User
from tools import TOOL_SPECS, execute_tool_call

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
VIDEO_DIR = STATIC_DIR / "videos"

load_dotenv(BASE_DIR / ".env")

STATIC_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)

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

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    history: list[dict[str, str]] = Field(default_factory=list)


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

# Simple in-memory job store. This resets when the server restarts.
VIDEO_JOBS: dict[str, dict[str, str | None]] = {}
VIDEO_JOBS_LOCK = threading.Lock()


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
async def chat(request: ChatRequest, user_id: str = Depends(auth.get_current_user_id)):
    user_text = request.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Text is required.")

    reply, tools_used = await safe_call_openrouter(user_text, request.history, user_id)
    audio_path = await safe_generate_voice_audio(reply)
    audio_url = static_url(audio_path) if audio_path else None
    job_id = None

    # SadTalker is slow, so we only queue it and return immediately.
    if sadtalker_is_enabled():
        if not audio_path:
            audio_path = await safe_text_to_speech(reply)

        if audio_path:
            job_id = start_video_job(audio_path)

    return {
        "reply": reply,
        "text": reply,
        "audio_url": audio_url,
        "video_url": None,
        "job_id": job_id,
        "tools_used": tools_used,
    }


@app.get("/video-status/{job_id}")
def video_status(job_id: str):
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Video job not found.")

    return job


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


async def safe_generate_voice_audio(text: str) -> Path | None:
    try:
        return await generate_voice_audio(text)
    except Exception as error:
        print(f"Voice generation failed: {error}")
        return None


async def safe_text_to_speech(text: str) -> Path | None:
    try:
        return await text_to_speech(text)
    except Exception as error:
        print(f"Fallback TTS for SadTalker failed: {error}")
        return None


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


def static_url(path: Path) -> str:
    return f"/static/{path.relative_to(STATIC_DIR)}"


async def generate_voice_audio(text: str) -> Path | None:
    provider = os.getenv("TTS_PROVIDER", "browser").lower()

    if provider == "elevenlabs":
        return await elevenlabs_text_to_speech(text)

    return None


async def elevenlabs_text_to_speech(text: str) -> Path:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key or api_key == "your_elevenlabs_api_key_here":
        raise RuntimeError("TTS_PROVIDER is elevenlabs, but ELEVENLABS_API_KEY is missing.")

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    audio_id = uuid.uuid4().hex
    audio_path = AUDIO_DIR / f"{audio_id}.mp3"

    payload = {
        "text": text[:4500],
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.8,
            "style": 0.35,
            "use_speaker_boost": True,
        },
    }

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    params = {"output_format": "mp3_44100_128"}
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, params=params, headers=headers, json=payload)
    except httpx.RequestError as error:
        raise RuntimeError(f"Could not reach ElevenLabs: {error}") from error

    if response.status_code >= 400:
        raise RuntimeError(f"ElevenLabs error: {response.text}")

    audio_path.write_bytes(response.content)
    return audio_path


async def text_to_speech(text: str) -> Path:
    audio_id = uuid.uuid4().hex
    audio_path = AUDIO_DIR / f"{audio_id}.mp3"

    if os.getenv("TTS_PROVIDER", "browser").lower() == "elevenlabs":
        generated_path = await elevenlabs_text_to_speech(text)
        shutil.copyfile(generated_path, audio_path)
        return audio_path

    try:
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save(str(audio_path))
    except Exception as error:
        raise RuntimeError(f"Text-to-speech failed: {error}") from error

    return audio_path


def sadtalker_is_enabled() -> bool:
    return os.getenv("SADTALKER_ENABLED", "false").lower() == "true"


def start_video_job(audio_path: Path) -> str:
    job_id = uuid.uuid4().hex

    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "video_url": None,
            "error": None,
        }

    thread = threading.Thread(
        target=process_video_job,
        args=(job_id, audio_path),
        daemon=True,
    )
    thread.start()

    return job_id


def process_video_job(job_id: str, audio_path: Path) -> None:
    set_video_job_status(job_id, "processing")

    try:
        video_url = run_sadtalker(audio_path)
        set_video_job_status(job_id, "done", video_url=video_url)
    except Exception as error:
        print(f"SadTalker job {job_id} failed: {error}")
        set_video_job_status(job_id, "failed", error=str(error))


def set_video_job_status(
    job_id: str,
    status: str,
    video_url: str | None = None,
    error: str | None = None,
) -> None:
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if not job:
            return

        job["status"] = status
        if video_url is not None:
            job["video_url"] = video_url
        if error is not None:
            job["error"] = error


def run_sadtalker(audio_path: Path) -> str:
    sadtalker_path = Path(os.getenv("SADTALKER_PATH", "")).expanduser()
    sadtalker_python = os.getenv("SADTALKER_PYTHON", "python")
    face_image = Path(os.getenv("SADTALKER_FACE_IMAGE", "static/face.png"))

    if not sadtalker_path.exists():
        raise RuntimeError("SADTALKER_PATH does not exist.")

    if not face_image.is_absolute():
        face_image = BASE_DIR / face_image

    if not face_image.exists():
        raise RuntimeError("Face image not found. Put a face image at static/face.png.")

    result_dir = VIDEO_DIR / audio_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sadtalker_python,
        "inference.py",
        "--driven_audio",
        str(audio_path),
        "--source_image",
        str(face_image),
        "--result_dir",
        str(result_dir),
        "--still",
        "--preprocess",
        "full",
    ]

    result = subprocess.run(
        command,
        cwd=sadtalker_path,
        capture_output=True,
        text=True,
        timeout=1200,
    )

    if result.returncode != 0:
        raise RuntimeError(f"SadTalker failed:\n{result.stderr or result.stdout}")

    generated_video = find_latest_video(result_dir)
    if not generated_video:
        raise RuntimeError("SadTalker did not create a video.")

    final_video = VIDEO_DIR / f"{audio_path.stem}.mp4"
    shutil.copyfile(generated_video, final_video)

    return f"/static/videos/{final_video.name}"


async def generate_talking_face(audio_path: Path) -> str | None:
    if not sadtalker_is_enabled():
        return None

    return await asyncio.to_thread(run_sadtalker, audio_path)


def find_latest_video(folder: Path) -> Path | None:
    videos = list(folder.rglob("*.mp4"))
    if not videos:
        return None
    return max(videos, key=lambda path: path.stat().st_mtime)
