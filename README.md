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
- [ ] Conversation persistence (wiring `/chat` to actually write to the DB)
- [ ] Disable ElevenLabs / SadTalker in this build, browser `SpeechSynthesis` only
- [ ] Per-user + global rate limiting on `/chat`
- [ ] Frontend: login/signup, conversation history, rate-limit UI, cold-start UI
- [ ] End-to-end verification on free tiers, including hitting OpenRouter's
      shared rate limit on purpose

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

Models (`users`, `conversations`, `messages`) exist and are migrated;
`/auth/signup` and `/auth/login` write to and read from `users`, but `/chat`
still doesn't persist conversations/messages yet -- next up.

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
