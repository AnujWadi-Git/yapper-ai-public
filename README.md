# Yapper AI — Public Multi-User Build

Work in progress. This folder is a separate project from the original
single-user `Yapper-AI-main` app, with its own git history, so a future
public deploy never risks pulling in unrelated files.

## Status

- [x] Step 0: reviewed existing single-user codebase
- [x] RAG layer (`rag.py`) — local embeddings (sentence-transformers,
      `all-MiniLM-L6-v2`), in-memory per-user chunk store, `/documents/upload`
      and `/documents` endpoints, retrieval injected into `/chat`.
- [ ] Database (Postgres via Supabase free tier) + SQLAlchemy models
- [ ] Auth (signup/login, bcrypt, JWT in HttpOnly cookie)
- [ ] Conversation persistence
- [ ] Disable ElevenLabs / SadTalker in this build, browser `SpeechSynthesis` only
- [ ] Per-user + global rate limiting on `/chat`
- [ ] Frontend: login/signup, conversation history, rate-limit UI, cold-start UI
- [ ] Deploy: Render (free web service) + Supabase (free Postgres)
- [ ] End-to-end verification on free tiers, including hitting OpenRouter's
      shared rate limit on purpose

## RAG layer notes

`rag.py` is process-local, in-memory storage keyed by `user_id`. It is not
persisted across restarts. Once the database step lands, its `add_document`
/ `search` interface should be backed by a `pgvector` table in the same
Postgres instance instead — nothing in `main.py` needs to change beyond the
import.

`user_id` is currently read from an `X-User-Id` request header
(`get_user_id` in `main.py`) as a placeholder — **not real auth**. It will
be replaced by the authenticated user's id from the JWT cookie once the
auth step lands.

First request that touches RAG downloads the ~80MB `all-MiniLM-L6-v2` model
from Hugging Face and caches it locally (one-time, free, no ongoing API
cost or rate limit — this is the reason we didn't use a hosted embeddings API).

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

<!-- auto-deploy verification: this line will be removed once confirmed -->
