"""
Memory system for SourceMind AI.

Three layers:
  1. Short-term  — last N chat turns from PostgreSQL (exact recall)
  2. Summary     — rolling compressed summary from PostgreSQL (medium-term context)
  3. Vector      — semantic memory embeddings in Qdrant (long-term retrieval)
"""
import uuid
from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete as sql_delete

from app.models.memory import ChatMessage, ChatSummary

# Qdrant collection for memory vectors
MEMORY_COLLECTION = "sourcemind_memory"

# How many recent turns to keep as short-term memory
SHORT_TERM_TURNS = 6

# Summarise after every N turns
SUMMARISE_EVERY = 10


# ── Short-term memory ─────────────────────────────────────────────────────────

async def load_short_term(
    db: AsyncSession,
    notebook_id: str,
    user_id: str,
    conversation_id: str,
) -> List[dict]:
    """Return last SHORT_TERM_TURNS messages as [{"role": ..., "content": ...}]."""
    stmt = (
        select(ChatMessage)
        .where(
            ChatMessage.notebook_id == uuid.UUID(notebook_id),
            ChatMessage.user_id == uuid.UUID(user_id),
            ChatMessage.conversation_id == conversation_id,
        )
        .order_by(ChatMessage.turn_index.desc())
        .limit(SHORT_TERM_TURNS)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    rows = list(reversed(rows))  # oldest first
    return [{"role": r.role, "content": r.content} for r in rows]


async def save_turn(
    db: AsyncSession,
    notebook_id: str,
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
) -> int:
    """Append one turn and return its turn_index."""
    # Get current max turn_index
    stmt = (
        select(ChatMessage.turn_index)
        .where(
            ChatMessage.notebook_id == uuid.UUID(notebook_id),
            ChatMessage.conversation_id == conversation_id,
        )
        .order_by(ChatMessage.turn_index.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    last_idx = result.scalar_one_or_none()
    next_idx = (last_idx or 0) + 1

    msg = ChatMessage(
        notebook_id=uuid.UUID(notebook_id),
        user_id=uuid.UUID(user_id),
        conversation_id=conversation_id,
        role=role,
        content=content,
        turn_index=next_idx,
    )
    db.add(msg)
    await db.commit()
    return next_idx


async def get_turn_count(
    db: AsyncSession,
    notebook_id: str,
    conversation_id: str,
) -> int:
    """Return total number of turns in this conversation."""
    stmt = (
        select(ChatMessage.turn_index)
        .where(
            ChatMessage.notebook_id == uuid.UUID(notebook_id),
            ChatMessage.conversation_id == conversation_id,
        )
        .order_by(ChatMessage.turn_index.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    last = result.scalar_one_or_none()
    return last or 0


# ── Summary memory ────────────────────────────────────────────────────────────

async def load_summary(
    db: AsyncSession,
    notebook_id: str,
    conversation_id: str,
) -> Optional[str]:
    """Load the latest rolling summary for this conversation."""
    stmt = (
        select(ChatSummary)
        .where(
            ChatSummary.notebook_id == uuid.UUID(notebook_id),
            ChatSummary.conversation_id == conversation_id,
        )
        .order_by(ChatSummary.updated_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.scalars().first()
    return row.summary if row else None


async def save_summary(
    db: AsyncSession,
    notebook_id: str,
    user_id: str,
    conversation_id: str,
    summary: str,
    turns_covered: int,
) -> None:
    """Upsert the rolling summary for this conversation."""
    # Delete old summary first
    await db.execute(
        sql_delete(ChatSummary).where(
            ChatSummary.notebook_id == uuid.UUID(notebook_id),
            ChatSummary.conversation_id == conversation_id,
        )
    )
    new_summary = ChatSummary(
        notebook_id=uuid.UUID(notebook_id),
        user_id=uuid.UUID(user_id),
        conversation_id=conversation_id,
        summary=summary,
        turns_covered=turns_covered,
    )
    db.add(new_summary)
    await db.commit()


# ── Vector memory (Qdrant) ────────────────────────────────────────────────────

def _ensure_memory_collection(qdrant_client, embed_dim: int = 768) -> None:
    """Create the memory Qdrant collection if it doesn't exist."""
    from qdrant_client.models import VectorParams, Distance
    collections = qdrant_client.get_collections().collections
    if not any(c.name == MEMORY_COLLECTION for c in collections):
        qdrant_client.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )
        print(f"[Memory] Created Qdrant collection '{MEMORY_COLLECTION}'")


def store_memory_vector(
    qdrant_client,
    embed_model,
    notebook_id: str,
    conversation_id: str,
    text: str,
    memory_type: str = "turn",  # "turn" | "summary" | "fact"
) -> None:
    """Embed and store a memory vector in Qdrant."""
    try:
        from qdrant_client.models import PointStruct
        _ensure_memory_collection(qdrant_client)
        vector = embed_model.encode(text, convert_to_numpy=True, normalize_embeddings=True).tolist()
        qdrant_client.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "notebook_id": notebook_id,
                        "conversation_id": conversation_id,
                        "text": text,
                        "memory_type": memory_type,
                    }
                )
            ]
        )
    except Exception as e:
        print(f"[Memory] Vector store error: {e}")


def retrieve_memory_vectors(
    qdrant_client,
    embed_model,
    notebook_id: str,
    query: str,
    top_k: int = 5,
) -> List[str]:
    """Retrieve semantically relevant memories for the current query."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        _ensure_memory_collection(qdrant_client)
        vector = embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()
        results = qdrant_client.search(
            collection_name=MEMORY_COLLECTION,
            query_vector=vector,
            query_filter=Filter(
                must=[FieldCondition(key="notebook_id", match=MatchValue(value=notebook_id))]
            ),
            limit=top_k,
        )
        return [r.payload.get("text", "") for r in results if r.payload]
    except Exception as e:
        print(f"[Memory] Vector retrieval error: {e}")
        return []


# ── Summarisation helper (called from chat.py) ────────────────────────────────

async def maybe_summarise(
    db: AsyncSession,
    notebook_id: str,
    user_id: str,
    conversation_id: str,
    turn_count: int,
    groq_keys: List[str],
    groq_model: str = "llama-3.1-8b-instant",
) -> Optional[str]:
    """
    If turn_count is a multiple of SUMMARISE_EVERY, generate a new rolling summary.
    Returns the new summary text or None.
    """
    if turn_count % SUMMARISE_EVERY != 0:
        return None

    # Load all turns for this conversation
    stmt = (
        select(ChatMessage)
        .where(
            ChatMessage.notebook_id == uuid.UUID(notebook_id),
            ChatMessage.conversation_id == conversation_id,
        )
        .order_by(ChatMessage.turn_index.asc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        return None

    conversation_text = "\n".join(
        f"{r.role.upper()}: {r.content}" for r in rows
    )

    # Load existing summary to incorporate
    existing_summary = await load_summary(db, notebook_id, conversation_id)
    existing_part = f"\nExisting summary to update:\n{existing_summary}" if existing_summary else ""

    prompt = (
        f"You are a memory summariser. Compress the following conversation into a concise summary "
        f"(max 300 words) capturing: main topics discussed, key facts stated, decisions made, "
        f"and user goals. Preserve names, numbers, and specific details.{existing_part}\n\n"
        f"Conversation:\n{conversation_text[-6000:]}\n\nSummary:"
    )

    summary = None
    from groq import Groq as GroqClient
    for key in groq_keys:
        try:
            client = GroqClient(api_key=key)
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=groq_model,
                temperature=0.1,
                max_tokens=400,
            )
            summary = resp.choices[0].message.content
            break
        except Exception as e:
            err = str(e)
            if "rate_limit" in err or "429" in err:
                continue
            break

    if summary:
        await save_summary(db, notebook_id, user_id, conversation_id, summary, turn_count)
        print(f"[Memory] Summary updated at turn {turn_count} for conversation {conversation_id}")

    return summary
