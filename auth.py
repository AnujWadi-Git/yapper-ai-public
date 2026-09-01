"""Auth layer: password hashing, JWT session tokens, login rate limiting.

Session state lives entirely in a signed JWT held in an HttpOnly cookie
(not localStorage, to keep it out of reach of XSS). There is no server-side
session store or refresh-token rotation -- fine for this MVP's scope; if that
changes, revisit.
"""

from __future__ import annotations

import os
import time
import uuid
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


# Revoked-token guard for logout. Keyed by the token's own "jti" claim rather
# than the raw token so revoked entries don't hold the full JWT in memory.
# In-memory and per-process, same tradeoff as the rate limiter above: it
# resets on restart/redeploy, meaning any pre-restart logout stops being
# enforced -- fine for this MVP's traffic, revisit if that changes.
_revoked_jtis: set[str] = set()
_revoked_lock = Lock()


def create_access_token(user_id: str) -> str:
    if not JWT_SECRET:
        raise RuntimeError("Missing JWT_SECRET.")
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {"sub": user_id, "exp": expire, "jti": uuid.uuid4().hex}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str | None:
    if not JWT_SECRET:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    with _revoked_lock:
        if payload.get("jti") in _revoked_jtis:
            return None
    return payload.get("sub")


def revoke_token(token: str) -> None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return
    jti = payload.get("jti")
    if not jti:
        return
    with _revoked_lock:
        _revoked_jtis.add(jti)


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


# --- Rate limiting: brute-force / mass-signup guard, not a precision limiter.
# In-memory and per-process -- fine at this scale (no Redis needed); resets
# on restart/redeploy, which just means attempt counts reset too.

LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_ATTEMPTS = 5

# Looser than login (real users rarely sign up more than once), but present --
# an unlimited signup endpoint lets one IP mint accounts to multiply its
# effective /chat quota past the per-user daily message cap.
SIGNUP_WINDOW_SECONDS = 60 * 60
SIGNUP_MAX_ATTEMPTS = 5

_attempts: dict[tuple[str, str], list[float]] = {}
_attempts_lock = Lock()


def _client_ip(request: Request) -> str:
    # Render's own community reports disagree on whether X-Forwarded-For is
    # trustworthy or client-spoofable on their platform (there's an open
    # Render feedback thread titled "Send the correct X-Forwarded-For" about
    # exactly this). Cloudflare -- which sits in front of Render's edge --
    # sets CF-Connecting-IP itself and overwrites any client-supplied copy,
    # so prefer that when present; it's a stronger guarantee than XFF here.
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(
    request: Request, bucket: str, window_seconds: int, max_attempts: int, message: str
) -> None:
    key = (bucket, _client_ip(request))
    now = time.monotonic()
    with _attempts_lock:
        attempts = [t for t in _attempts.get(key, []) if now - t < window_seconds]
        if len(attempts) >= max_attempts:
            _attempts[key] = attempts
            raise HTTPException(status_code=429, detail=message)
        attempts.append(now)
        _attempts[key] = attempts


def enforce_login_rate_limit(request: Request) -> None:
    _enforce_rate_limit(
        request,
        "login",
        LOGIN_WINDOW_SECONDS,
        LOGIN_MAX_ATTEMPTS,
        "Too many login attempts. Please wait a few minutes and try again.",
    )


def enforce_signup_rate_limit(request: Request) -> None:
    _enforce_rate_limit(
        request,
        "signup",
        SIGNUP_WINDOW_SECONDS,
        SIGNUP_MAX_ATTEMPTS,
        "Too many accounts created from this network. Please wait an hour and try again.",
    )
