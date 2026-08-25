"""Auth layer: password hashing, JWT session tokens, login rate limiting.

Session state lives entirely in a signed JWT held in an HttpOnly cookie
(not localStorage, to keep it out of reach of XSS). There is no server-side
session store or refresh-token rotation -- fine for this MVP's scope; if that
changes, revisit.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Cookie, HTTPException, Request

load_dotenv(Path(__file__).resolve().parent / ".env")

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30
COOKIE_NAME = "yapper_session"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() == "true"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(user_id: str) -> str:
    if not JWT_SECRET:
        raise RuntimeError("Missing JWT_SECRET.")
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str | None:
    if not JWT_SECRET:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    return payload.get("sub")


async def get_current_user_id(
    session_token: str | None = Cookie(default=None, alias=COOKIE_NAME),
) -> str:
    user_id = decode_access_token(session_token) if session_token else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    return user_id


def set_session_cookie(response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=JWT_EXPIRE_DAYS * 24 * 3600,
        path="/",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


# --- Login rate limiting: brute-force guard, not a precision limiter.
# In-memory and per-process -- fine at this scale (no Redis needed); resets
# on restart/redeploy, which just means attempt counts reset too.

LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_ATTEMPTS = 5

_login_attempts: dict[str, list[float]] = {}
_login_lock = Lock()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_login_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    now = time.monotonic()
    with _login_lock:
        attempts = [t for t in _login_attempts.get(ip, []) if now - t < LOGIN_WINDOW_SECONDS]
        if len(attempts) >= LOGIN_MAX_ATTEMPTS:
            _login_attempts[ip] = attempts
            raise HTTPException(
                status_code=429,
                detail="Too many login attempts. Please wait a few minutes and try again.",
            )
        attempts.append(now)
        _login_attempts[ip] = attempts
