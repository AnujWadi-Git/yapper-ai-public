# Yapper AI — Public Multi-User Build

Work in progress. This folder is a separate project from the original
single-user `Yapper-AI-main` app, with its own git history, so a future
public deploy never risks pulling in unrelated files.

## Live

**https://yapper-ai-public.onrender.com** — Render free web service, auto-deploys
from `main` on every push. Cold-starts after ~15 min idle (free-tier behavior,
not a bug).

Gotcha hit once already: if you push code and change env vars close together,
the code-triggered deploy can start before the env var change lands, so the
live instance boots without the new value (saw this as `RuntimeError: Missing
JWT_SECRET` right after adding auth). Fix is just triggering one more manual
deploy after env vars are confirmed set. Worth a glance at `/auth/me` or
similar after any deploy that adds a new required env var.

## Status

- [x] Step 0: reviewed existing single-user codebase
- [x] RAG layer (`rag.py`) — local embeddings via `fastembed` (ONNX runtime,
      no PyTorch, fits the free 512MB instance), in-memory per-user chunk
      store, `/documents/upload` and `/documents` endpoints, retrieval
      injected into `/chat`.
- [x] Database (Postgres via Supabase free tier) + SQLAlchemy models +
      Alembic migrations — `users`, `conversations`, `messages` tables live.
      `pgvector` extension enabled for the later RAG persistence swap.
- [x] Deploy: Render (free web service) + Supabase (free Postgres)
- [x] Auth (signup/login, bcrypt, JWT in HttpOnly cookie, login rate limiting)
      — `/auth/signup`, `/auth/login`, `/auth/logout`, `/auth/me`. `/chat` and
      `/documents/*` now require a signed-in session; the `X-User-Id` stub is gone.
- [x] Conversation persistence — `/chat` now saves every user/assistant
      message to `conversations`/`messages`, keyed to the signed-in user.
      `GET /conversations` lists a user's threads, `GET
      /conversations/{id}/messages` loads one. History for a turn is loaded
      from the DB (not the client) once a `conversation_id` is passed, so a
      conversation now has real memory across requests, not just within one
      client-held array.
- [x] Disable ElevenLabs / SadTalker in this build, browser `SpeechSynthesis`
      only — both removed entirely from `main.py` (not just default-off);
      see Voice/video notes below.
- [x] Frontend: login/signup, conversation history sidebar, cold-start
      "waking up" banner. Verified the full signup → chat (with a real tool
      call) → new-chat → switch-conversation → logout flow in a real browser.
- [x] Per-user + shared rate limiting on `/chat` — 20 messages/user/day
      (DB-backed, resets at UTC midnight, no separate counter table); a
      distinct friendly message when OpenRouter's *shared* free-tier limit
      is hit, separate from hitting your own daily cap. See notes below.
- [x] End-to-end verification on free tiers, including hitting OpenRouter's
      shared rate limit on purpose — actually reproduced it (not simulated)
      and confirmed the graceful message end-to-end. See notes below.

## RAG layer notes

`rag.py` is process-local, in-memory storage keyed by `user_id`. It is not
persisted across restarts (and won't survive Render free-tier spin-downs).
Its `add_document` / `search` interface is designed to be swapped for a
`pgvector`-backed one in the same Supabase Postgres instance (extension
already enabled) — nothing in `main.py` needs to change beyond the import.
Not done yet; tracked above.

`user_id` now comes from the authenticated session (`auth.get_current_user_id`,
reading the `yapper_session` JWT cookie) — the old `X-User-Id` header stub is gone.

Embeddings run through `fastembed` (ONNX runtime, `BAAI/bge-small-en-v1.5`,
~50MB) instead of `sentence-transformers`/PyTorch — the original choice blew
past Render's free-tier 512MB RAM cap. First request that touches RAG
downloads the model from Hugging Face and caches it locally (one-time, free,
no ongoing API cost or rate limit).

## Database notes

Postgres is Supabase's free tier, reached through the Supavisor **transaction
pooler** (port 6543) rather than a direct connection — Supabase's direct
connections are IPv6-only on new projects, and Render's free tier is
IPv4-only.

Transaction-mode pooling doesn't support server-side prepared statements the
way a normal Postgres connection does. `statement_cache_size: 0` in
`db.py` alone isn't enough to fix this: asyncpg still auto-generates short,
sequential prepared-statement names (`__asyncpg_stmt_N__`) per connection
object, and since the pooler multiplexes many client connections onto few
backend server connections, two different clients can land on a backend
that already has a statement by that same auto-generated name from someone
else's session -- `DuplicatePreparedStatementError`. Fixed by also passing
`prepared_statement_name_func` so every prepared statement gets a globally
unique (uuid-based) name instead. Verified by hammering `/auth/login`
repeatedly with different failure paths in a row -- this reproduces reliably
without the fix and disappears with it.

Migrations: `alembic revision --autogenerate -m "..."` then `alembic upgrade
head`. `migrations/env.py` reads `DATABASE_URL` from `.env` and runs through
the sync `psycopg2` driver (the app itself uses `asyncpg`); both go through
the same pooler. psycopg2 doesn't hit the prepared-statement issue above
(different protocol), so migrations don't need the same workaround.

Models (`users`, `conversations`, `messages`) exist, are migrated, and are
now all in active use (see Conversation persistence notes below).

## Conversation persistence notes

`/chat` takes an optional `conversation_id`. Omit it to start a new
conversation (title auto-generated from the first message, truncated to 60
chars); pass one back to continue an existing thread. Ownership is checked
on every read/write in `conversations.py` (`get_owned_conversation`) --
a conversation that doesn't belong to the caller (or doesn't exist) returns
404, verified by testing one user against another's conversation ID directly.

Multi-turn history is now loaded from the database on each request instead
of trusted from the client -- previously the frontend held the whole
conversation array and sent it back every time, which meant history could
be edited/spoofed client-side and didn't survive a page reload. The `/chat`
request schema dropped the old client-supplied `history` field as a result.

## Rate limiting notes

Two independent limits, for two different problems:

**Per-user daily cap** (`conversations.DAILY_MESSAGE_LIMIT = 20`): counts
each user's own `role='user'` rows in `messages` since UTC midnight --
reuses the existing table instead of a separate counter, so it needs no
cron/cleanup job and survives restarts for free. `/chat` checks this before
doing any OpenRouter work, so a user who's already over their limit doesn't
cost a wasted upstream call. Verified by seeding 20 messages for a test user
directly in Postgres and confirming the 21st `/chat` call returns 429 with
`"You've reached today's limit of 20 messages..."`, while a second,
fresh user is unaffected (isolation, not a global counter).

**Shared OpenRouter free-tier limit**: this is a different failure and gets
different handling. `_post_openrouter` now raises a distinct
`OpenRouterRateLimited` when OpenRouter returns HTTP 429 *or* embeds a
`{"error": {"code": 429}}` in an HTTP 200 body (both happen in practice --
verified by reading a live OpenRouter response so this isn't a guess).
`safe_call_openrouter` catches that specifically and returns "Yapper is busy
right now, try again in a minute" as the assistant's reply -- a normal 200
chat response, not an HTTP error, so the frontend doesn't need special
handling beyond just showing it.

This was verified against the real shared limit, not simulated: fired a
burst of concurrent requests at a free OpenRouter model until it started
returning 429, then pointed `/chat` at that same model and confirmed the
"Yapper is busy" message came back end-to-end instead of a crash or raw
error. Reverted `OPENROUTER_MODEL` back to a working free model afterward.

## Voice/video notes

ElevenLabs and SadTalker are not just disabled by config in this build --
their code is gone from `main.py` (no `edge-tts`/`httpx`-to-ElevenLabs calls,
no `/video-status` endpoint, no background video-job thread, no `static/`
audio/video dirs). Both still exist in the original single-user
`Yapper-AI-main` project (and in this repo's git history before this
commit) for local-only use, per the cost math in the original brief: neither
has a free tier that survives real multi-user traffic. `index.html` calls
the browser's `SpeechSynthesis` API directly -- genuinely free, runs
client-side, no server involvement or rate limit regardless of traffic.

## Frontend notes

`index.html` is a single-page app now: an auth screen (login/signup toggle)
gates a chat screen (conversation sidebar + chat + avatar), switching on
whether `GET /auth/me` succeeds on load. All `fetch` calls use
`credentials: "same-origin"` so the session cookie rides along automatically
-- no manual token handling in JS, consistent with keeping the JWT out of
reach of XSS.

Cold start: on boot, if `/auth/me` hasn't resolved within ~2.5s, a banner
explains the free-tier server may be waking up from sleep, rather than the
page just looking broken/stuck.

`/chat`'s 401 (session expired) and 429 (rate limited, once Step 5 lands)
responses are both handled with a plain-language message instead of a raw
error or a crash -- per the "must be handled gracefully in the UI, not
hidden" requirement in the original brief.

## Known gap found during setup

The `.env` copied over from the original project has
`OPENROUTER_MODEL=openai/gpt-oss-20b:free`, which OpenRouter has since
deprecated (`This model is unavailable for free`). Pick a currently-free
model from https://openrouter.ai/models?max_price=0 before running `/chat`.
This is unrelated to the RAG work above — the retrieval/upload endpoints
were verified independently of the OpenRouter call.

## Explicitly out of scope for this build

Billing/monetization, admin dashboard, email verification, password reset,
SadTalker in the public build, ElevenLabs in the public build. Revisit only
if paying for infrastructure becomes an option.
