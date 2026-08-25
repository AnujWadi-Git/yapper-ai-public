# Yapper AI — Public Multi-User Build

Work in progress. This folder is a separate project from the original
single-user `Yapper-AI-main` app, with its own git history, so a future
public deploy never risks pulling in unrelated files.

## Live

**https://yapper-ai-public.onrender.com** — Render free web service, auto-deploys
from `main` on every push. Cold-starts after ~15 min idle (free-tier behavior,
not a bug).

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
- [ ] Auth (signup/login, bcrypt, JWT in HttpOnly cookie)
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

`user_id` is currently read from an `X-User-Id` request header
(`get_user_id` in `main.py`) as a placeholder — **not real auth**. It will
be replaced by the authenticated user's id from the JWT cookie once the
auth step lands.

Embeddings run through `fastembed` (ONNX runtime, `BAAI/bge-small-en-v1.5`,
~50MB) instead of `sentence-transformers`/PyTorch — the original choice blew
past Render's free-tier 512MB RAM cap. First request that touches RAG
downloads the model from Hugging Face and caches it locally (one-time, free,
no ongoing API cost or rate limit).

## Database notes

Postgres is Supabase's free tier, reached through the Supavisor **transaction
pooler** (port 6543) rather than a direct connection — Supabase's direct
connections are IPv6-only on new projects, and Render's free tier is
IPv4-only. Transaction-mode pooling doesn't support server-side prepared
statements, so `db.py` disables asyncpg's statement cache
(`statement_cache_size: 0`); leaving that out causes intermittent "prepared
statement already exists" errors under load.

Migrations: `alembic revision --autogenerate -m "..."` then `alembic upgrade
head`. `migrations/env.py` reads `DATABASE_URL` from `.env` and runs through
the sync `psycopg2` driver (the app itself uses `asyncpg`); both go through
the same pooler.

Models (`users`, `conversations`, `messages`) exist and are migrated, but
`/chat` doesn't write to them yet — that lands with the auth step, once there's
a real `user_id` to attach conversations to instead of the `X-User-Id` stub.

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
