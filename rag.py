"""RAG (retrieval-augmented generation) layer for Yapper AI.

Lets a user upload a text/PDF document; it gets chunked and embedded locally
(no external API, so it stays free regardless of traffic) using fastembed
(ONNX runtime, no PyTorch) so it fits within a 512MB free-tier web service,
and relevant chunks are retrieved and injected into the chat prompt as context.

Storage here is in-memory, keyed by user_id, and process-local (lost on
restart). That's the correct MVP for a single-process free-tier deploy.
When the DB step lands, swap `DocumentStore` for a pgvector-backed one
behind the same `add_document` / `search` interface — nothing else in
main.py needs to change.
"""

from __future__ import annotations

import io
import threading
import uuid
from dataclasses import dataclass, field

import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

# ONNX-runtime based, no PyTorch -- fits comfortably in a 512MB free-tier
# web service. ~50MB model, downloaded once on first use and cached locally.
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: TextEmbedding | None = None
_model_lock = threading.Lock()

CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50
TOP_K = 4
MAX_DOCUMENT_CHARS = 200_000  # guards embedding time on huge uploads


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TextEmbedding(model_name=_MODEL_NAME)
    return _model


def _embed(texts: list[str]) -> list[np.ndarray]:
    model = _get_model()
    vectors = [np.asarray(v, dtype=np.float32) for v in model.embed(texts)]
    return [v / np.linalg.norm(v) for v in vectors]


def extract_text(filename: str, content: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return content.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


@dataclass
class _Chunk:
    doc_id: str
    filename: str
    text: str
    embedding: np.ndarray


@dataclass
class _UserDocs:
    chunks: list[_Chunk] = field(default_factory=list)
    filenames: set[str] = field(default_factory=set)


class DocumentStore:
    """In-memory per-user document chunk store with cosine-similarity search."""

    def __init__(self) -> None:
        self._by_user: dict[str, _UserDocs] = {}
        self._lock = threading.Lock()

    def add_document(self, user_id: str, filename: str, content: bytes) -> dict:
        text = extract_text(filename, content)[:MAX_DOCUMENT_CHARS]
        pieces = chunk_text(text)
        if not pieces:
            raise ValueError("No extractable text found in document.")

        embeddings = _embed(pieces)

        doc_id = uuid.uuid4().hex
        chunks = [
            _Chunk(doc_id=doc_id, filename=filename, text=piece, embedding=emb)
            for piece, emb in zip(pieces, embeddings)
        ]

        with self._lock:
            user_docs = self._by_user.setdefault(user_id, _UserDocs())
            user_docs.chunks.extend(chunks)
            user_docs.filenames.add(filename)

        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}

    def list_documents(self, user_id: str) -> list[str]:
        with self._lock:
            user_docs = self._by_user.get(user_id)
            return sorted(user_docs.filenames) if user_docs else []

    def search(self, user_id: str, query: str, top_k: int = TOP_K) -> list[str]:
        with self._lock:
            user_docs = self._by_user.get(user_id)
            chunks = list(user_docs.chunks) if user_docs else []

        if not chunks:
            return []

        query_embedding = _embed([query])[0]

        scored = [
            (float(np.dot(query_embedding, chunk.embedding)), chunk)
            for chunk in chunks
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        return [chunk.text for _score, chunk in scored[:top_k]]

    def clear_user(self, user_id: str) -> None:
        with self._lock:
            self._by_user.pop(user_id, None)


# Single process-wide store, imported by main.py.
document_store = DocumentStore()


def build_context_block(user_id: str, query: str) -> str | None:
    """Return a system-message-ready context block from the user's docs, or None."""
    chunks = document_store.search(user_id, query)
    if not chunks:
        return None

    joined = "\n\n---\n\n".join(chunks)
    return (
        "The user has uploaded document(s). Use the following excerpts to answer "
        "if relevant; ignore them if the question is unrelated:\n\n" + joined
    )
