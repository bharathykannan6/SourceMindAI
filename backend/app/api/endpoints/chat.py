import uuid
import math
import re
import threading
from typing import Any, List, Set, Dict, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from groq import Groq
import httpx

from app.api import deps
from app.models.user import User
from app.models.notebook import Notebook
from app.models.workspace import Workspace
from app.models.document import Document
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.core.config import settings
from app.rag.ingestion import embed_model, qdrant_client, VECTOR_COLLECTION
from app.rag.memory import (
    load_short_term, save_turn, get_turn_count,
    load_summary, store_memory_vector, retrieve_memory_vectors,
    maybe_summarise, SHORT_TERM_TURNS
)
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder

router = APIRouter()

# Load reranker once at startup
print("[Chat] Loading reranker model...")
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("[Chat] Reranker loaded.")
except Exception as e:
    print(f"[Chat] Reranker failed to load: {e}. Reranking disabled.")
    reranker = None

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

def tokenize(text: str) -> List[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]

def check_toc_similarity(text: str) -> float:
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return 0.0
    toc_lines_count = 0
    for line in lines:
        if re.search(r'\s\d{1,3}$', line):
            toc_lines_count += 1
            continue
        if re.match(r'^\d+[\.\s]', line):
            toc_lines_count += 1
            continue
        if "..." in line or "···" in line or ". ." in line:
            toc_lines_count += 1
            continue
    return toc_lines_count / len(lines)


@router.post("/", response_model=ChatResponse)
async def chat_with_notebook(
    request: ChatRequest,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> Any:
    """
    RAG Chat endpoint. Searches documents in the specified notebook and
    synthesizes a response using the Groq or Ollama LLM.
    """
    # 1. Verify notebook and workspace ownership
    stmt = select(Notebook, Workspace).join(Workspace).where(
        Notebook.id == request.notebook_id,
        Workspace.owner_id == current_user.id
    )
    result = await db.execute(stmt)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Notebook not found")

    notebook_name = row[0].name
    notebook_id_str = str(request.notebook_id)
    user_id_str = str(current_user.id)

    # ── Memory: resolve or create conversation_id ─────────────────────────────
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Layer 1: Short-term memory (last N turns)
    short_term = await load_short_term(db, notebook_id_str, user_id_str, conversation_id)

    # Layer 2: Summary memory
    summary_memory = await load_summary(db, notebook_id_str, conversation_id)

    # Layer 3: Vector memory (semantically relevant past exchanges)
    vector_memories = retrieve_memory_vectors(
        qdrant_client, embed_model, notebook_id_str, request.message, top_k=3
    )

    # Save user turn immediately
    turn_count = await save_turn(
        db, notebook_id_str, user_id_str, conversation_id, "user", request.message
    )

    # Build Groq key pool early — needed by query classifier before the main LLM call
    groq_keys = [
        k for k in [
            settings.GROQ_API_KEY,
            settings.GROQ_API_KEY_2,
            settings.GROQ_API_KEY_3,
            settings.GROQ_API_KEY_4,
            settings.GROQ_API_KEY_5,
        ] if k and k.strip()
    ]
    GROQ_MODEL_CASCADE = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
    ]
    groq_last_error = ""

    # 2. Embed the user's message
    try:
        query_vector = embed_model.encode(request.message, convert_to_numpy=True).tolist()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation error: {str(e)}"
        )

    # 3. Build Qdrant filter
    notebook_filter_conditions = [
        FieldCondition(key="notebook_id", match=MatchValue(value=str(request.notebook_id)))
    ]
    selected_doc_ids = request.document_ids or []
    if selected_doc_ids:
        from qdrant_client.models import Filter as QFilter, MatchAny
        doc_filter = Filter(
            must=[
                FieldCondition(key="notebook_id", match=MatchValue(value=str(request.notebook_id))),
                FieldCondition(key="document_id", match=MatchAny(any=selected_doc_ids))
            ]
        )
    else:
        doc_filter = Filter(must=notebook_filter_conditions)

    # 4. Scroll chunks for BM25 (capped at 3000)
    chunks_res = []
    try:
        all_chunks = []
        next_offset = None
        SCROLL_BATCH = 500
        SCROLL_MAX = 3000
        while len(all_chunks) < SCROLL_MAX:
            batch, next_offset = qdrant_client.scroll(
                collection_name=VECTOR_COLLECTION,
                scroll_filter=doc_filter,
                limit=SCROLL_BATCH,
                offset=next_offset,
                with_payload=True
            )
            all_chunks.extend(batch)
            if next_offset is None or len(batch) < SCROLL_BATCH:
                break
        chunks_res = all_chunks
    except Exception as scroll_err:
        print(f"Qdrant scroll error: {scroll_err}")

    # 5. BM25 scoring
    sorted_bm25 = []
    doc_count = len(chunks_res)
    if doc_count > 0:
        chunk_lengths = []
        inverted_index = {}
        for idx, chunk in enumerate(chunks_res):
            text_content = chunk.payload.get("text", "") if chunk.payload else ""
            tokens = tokenize(text_content)
            chunk_lengths.append(len(tokens))
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            for term, freq in freqs.items():
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((idx, freq))
        avgdl = sum(chunk_lengths) / doc_count if doc_count > 0 else 0
        query_terms = tokenize(request.message)
        bm25_scores = {}
        k1 = 1.2
        b = 0.75
        for term in query_terms:
            if term in inverted_index:
                postings = inverted_index[term]
                n_q = len(postings)
                idf = math.log(1.0 + (doc_count - n_q + 0.5) / (n_q + 0.5))
                for chunk_idx, tf in postings:
                    doc_len = chunk_lengths[chunk_idx]
                    denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl if avgdl > 0 else 1.0))
                    num = tf * (k1 + 1.0)
                    term_score = idf * (num / denom)
                    bm25_scores[chunk_idx] = bm25_scores.get(chunk_idx, 0.0) + term_score
        sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)

    # 6. Detect query type — LLM classification replaces fragile keyword matching.
    # A tiny llama-3.1-8b-instant call (~50 tokens) correctly handles phrasing
    # that keyword lists miss: "give me a rundown", "break it down", "elaborate", etc.
    # Falls back to keyword matching if the classification call fails.
    query_lower = request.message.lower()

    # TOC detection stays keyword-based — it's a structural query with clear signals
    is_toc_query = any(term in query_lower for term in [
        "table of content", "table of contents", "toc", "index of", "chapters", "list the contents"
    ])

    # ── LLM query classification ──────────────────────────────────────────────
    is_broad_query = False  # default; set by classifier below
    _classification_done = False

    if not is_toc_query and groq_keys:
        _classify_prompt = (
            "Classify the following user query into exactly one category.\n"
            "Reply with ONLY the single letter — no explanation, no punctuation, nothing else.\n\n"
            "Categories:\n"
            "B = broad/overview query (summary, overview, explain all, what is this about, "
            "give a rundown, break it down, elaborate on, what does this cover, "
            "topics, themes, main ideas, comprehensive view, tell me about, journey, history)\n"
            "S = specific query (a precise question, definition lookup, who is X, "
            "comparison, list of items, a single fact, step-by-step instruction)\n\n"
            f"Query: {request.message}\n\n"
            "Reply with B or S only."
        )
        for _groq_key in groq_keys:
            try:
                _clf_client = Groq(api_key=_groq_key)
                _clf_resp = _clf_client.chat.completions.create(
                    messages=[{"role": "user", "content": _classify_prompt}],
                    model="llama-3.1-8b-instant",   # fastest/cheapest — classification only
                    temperature=0.0,
                    max_tokens=2,
                )
                _label = _clf_resp.choices[0].message.content.strip().upper()
                if _label.startswith("B"):
                    is_broad_query = True
                    print(f"[QueryClassifier] → BROAD (LLM classified)")
                else:
                    is_broad_query = False
                    print(f"[QueryClassifier] → SPECIFIC (LLM classified)")
                _classification_done = True
                break
            except Exception as _clf_err:
                print(f"[QueryClassifier] LLM call failed: {_clf_err} — will use keyword fallback")
                continue

    # Keyword fallback if LLM classification failed or no keys available
    if not _classification_done:
        is_broad_query = any(term in query_lower for term in [
            "summarize", "summary", "overview", "explain", "describe", "tell me about",
            "what is", "what are", "journey", "history", "all", "entire", "whole",
            "everything", "comprehensive", "detailed", "full",
            "about", "topics", "themes", "issues", "problems", "common", "main",
            "key", "important", "agents", "users", "conversations", "discuss",
            "cover", "contains", "file", "document", "report"
        ])
        print(f"[QueryClassifier] → {'BROAD' if is_broad_query else 'SPECIFIC'} (keyword fallback)")

    dense_limit = 60 if (is_broad_query or is_toc_query) else 20
    max_context_chunks = 60 if is_toc_query else (50 if is_broad_query else 20)

    # 7. Dense semantic search (original query vector)
    dense_results = []
    try:
        dense_results = qdrant_client.search(
            collection_name=VECTOR_COLLECTION,
            query_vector=query_vector,
            query_filter=doc_filter,
            limit=dense_limit
        )
    except Exception as e:
        print(f"Qdrant dense search error: {e}")

    # 7b. HyDE — Hypothetical Document Embedding
    #
    # HALLUCINATION GUARD DESIGN (addresses the core HyDE risk):
    #
    # Problem: if the LLM generates a wild hypothetical, its vector points
    # in the wrong direction and pollutes retrieval.
    #
    # Four-layer defence implemented here:
    #   1. temperature=0.0  — deterministic, no creative drift
    #   2. max_tokens=120   — forces a short, dense factual phrase; no room to hallucinate
    #   3. Strict prompt     — instructs the model to use ONLY terms likely in the document;
    #                          explicitly forbids inventing facts
    #   4. Cosine similarity gate — the HyDE vector is only used if it is
    #                          sufficiently similar to the original query vector
    #                          (dot product >= 0.25). If the LLM drifted far from
    #                          the query topic, the cosine similarity drops below
    #                          the threshold and HyDE is silently skipped.
    #                          Original query vector always runs regardless.
    #
    # HyDE is skipped entirely for broad/summary queries — those go to
    # Map-Reduce which reads all chunks directly, so HyDE adds no value there.

    hyde_dense_results = []
    hyde_vector = None

    if not is_broad_query and not is_toc_query and groq_keys and doc_count > 0:
        _hyde_prompt = (
            "Write a single short paragraph (3-5 sentences) that directly answers "
            "the following question, as if it were an excerpt from a relevant document. "
            "Use only terminology and concepts that would realistically appear in a "
            "document on this topic. Do NOT invent specific names, numbers, dates, or "
            "facts — use realistic placeholder language instead. "
            "Output only the paragraph, nothing else.\n\n"
            f"Question: {request.message}"
        )
        for _groq_key in groq_keys:
            try:
                _hyde_client = Groq(api_key=_groq_key)
                _hyde_resp = _hyde_client.chat.completions.create(
                    messages=[{"role": "user", "content": _hyde_prompt}],
                    model="llama-3.1-8b-instant",  # fast + cheap — generation only
                    temperature=0.0,               # Guard 1: fully deterministic
                    max_tokens=120,                # Guard 2: no room to hallucinate
                )
                _hypo_text = _hyde_resp.choices[0].message.content.strip()

                if _hypo_text:
                    _hypo_vector = embed_model.encode(
                        _hypo_text, convert_to_numpy=True
                    ).tolist()

                    # Guard 4: cosine similarity gate
                    # Both vectors are L2-normalised by bge-base, so dot product == cosine sim
                    import numpy as np
                    _q = np.array(query_vector)
                    _h = np.array(_hypo_vector)
                    _sim = float(np.dot(_q, _h) / (np.linalg.norm(_q) * np.linalg.norm(_h) + 1e-9))

                    if _sim >= 0.25:
                        hyde_vector = _hypo_vector
                        hyde_dense_results = qdrant_client.search(
                            collection_name=VECTOR_COLLECTION,
                            query_vector=hyde_vector,
                            query_filter=doc_filter,
                            limit=dense_limit
                        )
                        print(f"[HyDE] Generated hypothetical (sim={_sim:.3f}) — "
                              f"{len(hyde_dense_results)} results")
                    else:
                        print(f"[HyDE] Hypothetical rejected: cosine sim {_sim:.3f} < 0.25 "
                              f"— skipping to avoid polluting retrieval")
                break
            except Exception as _hyde_err:
                print(f"[HyDE] Failed: {_hyde_err} — skipping")
                break

    # 8. RRF fusion — up to three ranked lists:
    #   - dense_results      : original query vector (always present)
    #   - hyde_dense_results : HyDE hypothetical vector (only if sim gate passed)
    #   - sorted_bm25        : BM25 keyword scores
    #
    # RRF weight for HyDE is 0.7x vs original dense — it is a supporting signal,
    # not an equal vote. This limits the blast radius if HyDE drifts slightly.
    HYDE_RRF_WEIGHT = 0.7  # original dense = 1.0, HyDE = 0.7, BM25 = 1.0

    rrf_scores = {}
    point_map = {}

    # Original dense results — full weight
    for rank, hit in enumerate(dense_results):
        p_id = str(hit.id)
        point_map[p_id] = hit
        rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + 1.0 / (60.0 + (rank + 1))

    # HyDE dense results — reduced weight (0.7x)
    for rank, hit in enumerate(hyde_dense_results):
        p_id = str(hit.id)
        if p_id not in point_map:
            point_map[p_id] = hit
        rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + HYDE_RRF_WEIGHT / (60.0 + (rank + 1))

    # BM25 results — full weight
    for rank, (chunk_idx, score) in enumerate(sorted_bm25[:60]):
        chunk = chunks_res[chunk_idx]
        p_id = str(chunk.id)
        point_map[p_id] = chunk
        rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + 1.0 / (60.0 + (rank + 1))

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 9. Reranker — top 50 candidates (was 30).
    # Broader coverage ensures relevant chunks ranked 31-50 in RRF
    # (due to BM25/dense disagreement) are properly evaluated.
    RERANK_TOP_K = 50
    rerank_candidates = sorted_rrf[:RERANK_TOP_K]
    if reranker and rerank_candidates:
        try:
            candidate_texts = [
                point_map[p_id].payload.get("text", "") if point_map[p_id].payload else ""
                for p_id, _ in rerank_candidates
            ]
            pairs = [(request.message, text) for text in candidate_texts]
            rerank_scores = reranker.predict(pairs)
            reranked = sorted(
                zip([p_id for p_id, _ in rerank_candidates], rerank_scores),
                key=lambda x: x[1], reverse=True
            )
            remaining = sorted_rrf[RERANK_TOP_K:]
            sorted_rrf = [(p_id, float(score)) for p_id, score in reranked] + remaining
            print(f"[Reranker] Reranked {len(rerank_candidates)} candidates (top {RERANK_TOP_K})")
        except Exception as rerank_err:
            print(f"[Reranker] Error: {rerank_err} — skipping reranking")

    # 10. Build context and citations
    context_str = ""
    citations_list: List[Citation] = []
    doc_cache = {}
    final_hits = []
    seen_chunk_keys = set()

    # TOC chunk prepending
    toc_chunks_to_prepend = []
    if is_toc_query and doc_count > 0:
        candidate_doc_ids = set()
        for p_id, _ in sorted_rrf[:15]:
            hit = point_map[p_id]
            payload = hit.payload if hit.payload else {}
            doc_id = payload.get("document_id")
            if doc_id:
                candidate_doc_ids.add(str(doc_id))
        for doc_id_str in candidate_doc_ids:
            doc_chunks = [c for c in chunks_res if c.payload and str(c.payload.get("document_id")) == doc_id_str]
            start_chunk = None
            for chunk in doc_chunks:
                payload = chunk.payload
                text_lower = payload.get("text", "").lower()
                if "table of contents" in text_lower or "contents" in text_lower or "index" in text_lower:
                    if check_toc_similarity(payload.get("text", "")) > 0.4:
                        start_chunk = chunk
                        break
            if not start_chunk:
                for p_id, _ in sorted_rrf[:15]:
                    hit = point_map[p_id]
                    payload = hit.payload if hit.payload else {}
                    if str(payload.get("document_id")) == doc_id_str:
                        text_lower = payload.get("text", "").lower()
                        if "table of contents" in text_lower or "contents" in text_lower or "index" in text_lower:
                            start_chunk = hit
                            break
            if start_chunk:
                start_idx = start_chunk.payload.get("chunk_index", 0)
                toc_chunks_to_prepend.append(start_chunk)
                current_idx = start_idx + 1
                for _ in range(15):
                    next_chunk = next((c for c in doc_chunks if c.payload.get("chunk_index") == current_idx), None)
                    if next_chunk:
                        text = next_chunk.payload.get("text", "")
                        if check_toc_similarity(text) > 0.15:
                            toc_chunks_to_prepend.append(next_chunk)
                            current_idx += 1
                        else:
                            next_next_chunk = next((c for c in doc_chunks if c.payload.get("chunk_index") == current_idx + 1), None)
                            if next_next_chunk and check_toc_similarity(next_next_chunk.payload.get("text", "")) > 0.15:
                                toc_chunks_to_prepend.append(next_chunk)
                                toc_chunks_to_prepend.append(next_next_chunk)
                                current_idx += 2
                            else:
                                break
                    else:
                        break

    base_toc_score = 0.5
    for hit in toc_chunks_to_prepend:
        payload = hit.payload if hit.payload else {}
        doc_id = payload.get("document_id")
        chunk_idx = payload.get("chunk_index")
        if doc_id and chunk_idx is not None:
            chunk_key = (doc_id, chunk_idx)
            if chunk_key not in seen_chunk_keys:
                seen_chunk_keys.add(chunk_key)
                final_hits.append((hit, base_toc_score))
                base_toc_score -= 0.001

    for p_id, score in sorted_rrf[:max_context_chunks]:
        hit = point_map[p_id]
        payload = hit.payload if hit.payload else {}
        doc_id = payload.get("document_id")
        chunk_idx = payload.get("chunk_index")
        if doc_id and chunk_idx is not None:
            chunk_key = (doc_id, chunk_idx)
            if chunk_key in seen_chunk_keys:
                continue
            seen_chunk_keys.add(chunk_key)
        final_hits.append((hit, score))

    for idx, (hit, score) in enumerate(final_hits[:max_context_chunks]):
        payload = hit.payload if hit.payload else {}
        doc_id = payload.get("document_id")
        chunk_text = payload.get("text")
        file_name = "Source Document"
        if doc_id:
            if doc_id in doc_cache:
                file_name = doc_cache[doc_id]
            else:
                try:
                    doc_uuid = uuid.UUID(doc_id)
                    doc_stmt = select(Document).where(Document.id == doc_uuid)
                    doc_res = await db.execute(doc_stmt)
                    doc_model = doc_res.scalars().first()
                    if doc_model:
                        file_name = doc_model.title
                        doc_cache[doc_id] = file_name
                except Exception as db_err:
                    print(f"Error fetching document metadata: {db_err}")
        source_num = idx + 1
        context_str += f"Source [{source_num}] (Document: {file_name}):\n{chunk_text}\n\n"
        citations_list.append(Citation(
            document_id=doc_id or "",
            file_name=file_name,
            text=chunk_text or "",
            score=float(score)
        ))

    # ── SUMMARY PATH: Map-Reduce over ALL chunks ─────────────────────────────
    if is_broad_query and not is_toc_query:
        print(f"[Chat] Summary query detected — using Map-Reduce over {len(chunks_res)} chunks")

        from collections import defaultdict
        doc_chunks_map: dict = defaultdict(list)
        for chunk in chunks_res:
            payload = chunk.payload if chunk.payload else {}
            doc_id = payload.get("document_id", "unknown")
            chunk_idx = payload.get("chunk_index", 0)
            text = payload.get("text", "")
            doc_chunks_map[doc_id].append((chunk_idx, text))

        for doc_id in doc_chunks_map:
            doc_chunks_map[doc_id].sort(key=lambda x: x[0])

        doc_name_map = {}
        for doc_id in doc_chunks_map:
            if doc_id in doc_cache:
                doc_name_map[doc_id] = doc_cache[doc_id]
            else:
                try:
                    doc_uuid = uuid.UUID(doc_id)
                    doc_stmt = select(Document).where(Document.id == doc_uuid)
                    doc_res = await db.execute(doc_stmt)
                    doc_model = doc_res.scalars().first()
                    if doc_model:
                        doc_name_map[doc_id] = doc_model.title
                        doc_cache[doc_id] = doc_model.title
                    else:
                        doc_name_map[doc_id] = "Unknown Document"
                except Exception:
                    doc_name_map[doc_id] = "Unknown Document"

        MAX_MAP_BATCHES = 15
        BATCH_CHARS = 6000
        partial_summaries = []

        map_system = (
            "You are an elite document intelligence engine operating inside OpenNotebook — "
            "an enterprise-grade AI research platform. "
            "Your task is to extract high-signal information from the provided document excerpt. "
            "Identify and capture: core topics, critical facts, key figures, quantitative data, "
            "conclusions, decisions, and any notable insights. "
            "Be thorough yet precise. Eliminate noise. Output structured extracted information only — "
            "no preamble, no commentary, no filler."
        )

        for doc_id, indexed_chunks in doc_chunks_map.items():
            file_name = doc_name_map.get(doc_id, "Document")
            all_text = "\n\n".join(text for _, text in indexed_chunks)
            total_chars = len(all_text)

            all_batches = [all_text[i:i + BATCH_CHARS] for i in range(0, total_chars, BATCH_CHARS)]
            total_batches = len(all_batches)

            if total_batches <= MAX_MAP_BATCHES:
                sampled_batches = list(enumerate(all_batches))
            else:
                # Smart sampling: always include start + end + evenly spaced middle.
                # Previous uniform sampling could skip the introduction and conclusions
                # entirely if they happened to fall between sampled indices.
                # Conclusions and key findings almost always appear at the end;
                # context and background at the start — both are critical for quality.
                HEAD = min(3, MAX_MAP_BATCHES // 3)          # first N batches
                TAIL = min(3, MAX_MAP_BATCHES // 3)          # last N batches
                MIDDLE_BUDGET = MAX_MAP_BATCHES - HEAD - TAIL  # remaining slots

                head_indices = list(range(HEAD))
                tail_indices = list(range(total_batches - TAIL, total_batches))

                # Middle: evenly spaced from the region between head and tail
                middle_start = HEAD
                middle_end = total_batches - TAIL
                if MIDDLE_BUDGET > 0 and middle_end > middle_start:
                    step = (middle_end - middle_start) / MIDDLE_BUDGET
                    middle_indices = [
                        int(middle_start + i * step) for i in range(MIDDLE_BUDGET)
                    ]
                else:
                    middle_indices = []

                # Deduplicate and preserve document order
                seen_idx: set = set()
                sampled_indices = []
                for idx in head_indices + middle_indices + tail_indices:
                    if idx not in seen_idx and 0 <= idx < total_batches:
                        seen_idx.add(idx)
                        sampled_indices.append(idx)
                sampled_indices.sort()
                sampled_batches = [(i, all_batches[i]) for i in sampled_indices]

            print(f"[Chat] Map phase: '{file_name}' — {len(indexed_chunks)} chunks, {total_chars} chars, {len(sampled_batches)}/{total_batches} batches")

            doc_partial_summaries = []
            for batch_num, (original_idx, batch_text) in enumerate(sampled_batches):
                batch_summary = ""
                position_pct = int((original_idx / max(total_batches - 1, 1)) * 100)
                map_user = f"Document: {file_name} (Section ~{position_pct}% through document)\n\n{batch_text}"

                for model_name in GROQ_MODEL_CASCADE:
                    if batch_summary:
                        break
                    for groq_key in groq_keys:
                        try:
                            groq_client_map = Groq(api_key=groq_key)
                            resp = groq_client_map.chat.completions.create(
                                messages=[
                                    {"role": "system", "content": map_system},
                                    {"role": "user", "content": map_user}
                                ],
                                model=model_name,
                                temperature=0.1,
                                max_tokens=400,
                            )
                            batch_summary = resp.choices[0].message.content
                            print(f"[Chat] Batch {batch_num+1}/{len(sampled_batches)} done")
                            break
                        except Exception as e:
                            err_str = str(e)
                            if "rate_limit_exceeded" in err_str or "429" in err_str:
                                continue
                            else:
                                break

                if batch_summary:
                    doc_partial_summaries.append(batch_summary)
                else:
                    doc_partial_summaries.append(batch_text[:300] + "...")

            if doc_partial_summaries:
                partial_summaries.append(
                    f"### Document: {file_name}\n" + "\n\n".join(doc_partial_summaries)
                )

        if partial_summaries:
            combined = "\n\n".join(partial_summaries)
            MAX_REDUCE_CHARS = 16000
            if len(combined) > MAX_REDUCE_CHARS:
                combined = combined[:MAX_REDUCE_CHARS] + "\n\n[Partial summaries trimmed]"

            reduce_system = (
                "You are OpenNotebook's senior research synthesis engine — an authoritative, "
                "enterprise-grade AI that produces citation-grounded, content-driven summaries.\n\n"

                "## YOUR TASK\n"
                "You have been given extracted intelligence from one or more source documents. "
                "Read the content carefully. Identify what kind of document this is "
                "(e.g. academic paper, business report, legal document, transcript, technical manual, "
                "news article, interview, dataset, etc.) and what topics and themes it actually contains. "
                "Then write a comprehensive summary that reflects the ACTUAL structure and content "
                "of the source material — not a generic template.\n\n"

                "## STRUCTURING RULES\n"
                "- Do NOT use a fixed template or pre-defined section names.\n"
                "- Let the content determine the sections. "
                "If the document is about a product launch, your sections should reflect that. "
                "If it is a legal case, structure around parties, arguments, and rulings. "
                "If it is a research paper, follow its natural flow: problem, methodology, results, conclusions. "
                "If it is a transcript, surface speakers, key discussion points, and outcomes.\n"
                "- Use Markdown headers (##) for each major section you derive from the content.\n"
                "- Use bullet points within sections when listing multiple items or findings.\n"
                "- Be specific: use actual names, figures, dates, decisions, and technical terms "
                "exactly as they appear in the source.\n\n"

                "## CITATION MANDATE (NON-NEGOTIABLE)\n"
                "Every sentence or factual claim MUST end with an inline source marker [1], [2], etc. "
                "The numbers correspond to the order documents were provided. "
                "No sentence may exist without a citation. "
                "Do NOT append a references section — only inline [N] markers within the prose.\n\n"

                "## QUALITY STANDARDS\n"
                "- Never hallucinate or infer beyond what the source material explicitly states.\n"
                "- If information on a topic is absent, state: "
                "'The source documents do not address [topic].' — do not fill gaps with assumptions.\n"
                "- Maintain an authoritative, precise, and information-dense tone throughout.\n"
                "- Surface the most important and actionable content first within each section."
            )
            reduce_user = (
                f"Partial summaries from notebook '{notebook_name}':\n\n{combined}\n\n"
                f"User request: {request.message}"
            )

            llm_response = ""
            for model_name in GROQ_MODEL_CASCADE:
                if llm_response:
                    break
                for groq_key in groq_keys:
                    try:
                        groq_client_reduce = Groq(api_key=groq_key)
                        completion = groq_client_reduce.chat.completions.create(
                            messages=[
                                {"role": "system", "content": reduce_system},
                                {"role": "user", "content": reduce_user}
                            ],
                            model=model_name,
                            temperature=0.2,
                            max_tokens=2048,
                        )
                        llm_response = completion.choices[0].message.content
                        key_index = groq_keys.index(groq_key) + 1
                        print(f"[Chat] Reduce phase done using Groq key #{key_index}, model: {model_name}")
                        break
                    except Exception as groq_err:
                        err_str = str(groq_err)
                        groq_last_error = err_str
                        if "rate_limit_exceeded" in err_str or "429" in err_str:
                            continue
                        else:
                            break

            if llm_response:
                # Rebuild citations from final_hits for the map-reduce path
                citations_list = []
                for idx, (hit, score) in enumerate(final_hits[:10]):
                    payload = hit.payload if hit.payload else {}
                    doc_id = payload.get("document_id")
                    chunk_text = payload.get("text", "")
                    file_name = doc_name_map.get(doc_id, "Source Document")
                    citations_list.append(Citation(
                        document_id=doc_id or "",
                        file_name=file_name,
                        text=chunk_text,
                        score=float(score)
                    ))

                # Save assistant turn for memory
                await save_turn(
                    db, notebook_id_str, user_id_str, conversation_id, "assistant", llm_response
                )
                exchange_text = f"User: {request.message}\nAssistant: {llm_response[:500]}"
                threading.Thread(
                    target=store_memory_vector,
                    args=(qdrant_client, embed_model, notebook_id_str, conversation_id, exchange_text, "turn"),
                    daemon=True
                ).start()

                return ChatResponse(
                    response=llm_response,
                    citations=citations_list,
                    conversation_id=conversation_id
                )

    # ── NON-SUMMARY PATH: standard RAG (specific questions, TOC, etc.) ─────────
    # Build memory context block
    memory_block = ""
    if summary_memory:
        memory_block += f"## Conversation Summary So Far\n{summary_memory}\n\n"
    if vector_memories:
        memory_block += "## Relevant Past Context\n"
        for m in vector_memories:
            memory_block += f"- {m[:200]}\n"
        memory_block += "\n"
    if short_term:
        memory_block += "## Recent Conversation\n"
        for turn in short_term:
            prefix = "User" if turn["role"] == "user" else "Assistant"
            memory_block += f"{prefix}: {turn['content'][:300]}\n"
        memory_block += "\n"

    system_prompt = (
        "You are OpenNotebook's premium AI research assistant — an authoritative, enterprise-grade "
        "intelligence engine designed to deliver structured, citation-grounded, and information-dense "
        f"responses. The user is working inside a notebook named '{notebook_name}'.\n\n"

        "## CORE MANDATE\n"
        "Answer exclusively from the Source Context provided. Never introduce outside knowledge, "
        "assumptions, or hallucinated content. Every factual claim must be traceable to a source.\n\n"

        "## RESPONSE FORMAT — MATCH THE QUERY TYPE\n\n"

        "**1. LIST / ENUMERATION** (e.g. 'list the questions', 'what are the steps', 'give me the topics')\n"
        "   → Output a clean, numbered list. No intro paragraph. No headers. No commentary.\n"
        "   Example output:\n"
        "   1. Define data communication [1]\n"
        "   2. Compare MAC and IP address [2]\n"
        "   3. Explain error correction [1]\n\n"

        "**2. PERSON / ENTITY LOOKUP** (e.g. 'who is X', 'tell me about X')\n"
        "   → If found: one precise paragraph — name, role, context, key details. Cite every sentence.\n"
        "   → If not found after careful search (including partial name matches):\n"
        "     Reply ONLY: 'There is no mention of [X] in the uploaded documents.' Then stop.\n\n"

        "**3. DEFINITION / EXPLANATION** (e.g. 'what is X', 'explain X', 'how does X work')\n"
        "   → Lead with a crisp one-sentence definition [citation].\n"
        "   → Follow with bullet points if multiple aspects or mechanisms are involved.\n"
        "   → Use bold for key terms. Keep explanation precise and technically accurate.\n\n"

        "**4. SUMMARY / OVERVIEW** (e.g. 'summarize', 'what is this about', 'give me an overview')\n"
        "   → Structure:\n"
        "     ## Executive Summary\n"
        "     ## Key Topics & Themes\n"
        "     ## Notable Details (names, dates, figures, decisions)\n"
        "     ## People Involved (if applicable)\n"
        "     ## Common Issues / Outcomes (if applicable)\n"
        "   → Use bullet points within each section. Be specific — no vague generalities.\n\n"

        "**5. COMPARISON** (e.g. 'compare X and Y', 'difference between X and Y')\n"
        "   → Use a structured side-by-side bullet format or clearly separated sections per item.\n"
        "   → Bold the differentiating attributes. Cite each point.\n\n"

        "**6. ALL OTHER QUERIES**\n"
        "   → Answer directly, concisely, and with high information density.\n"
        "   → If the topic is not in the source context:\n"
        "     'The uploaded documents do not contain information about [topic].'\n\n"

        "## CITATION MANDATE (NON-NEGOTIABLE)\n"
        "- Place an inline citation marker [N] after EVERY sentence that states a fact.\n"
        "- [N] corresponds to Source [1], Source [2], etc. from the context provided.\n"
        "- No sentence may exist without a citation marker.\n"
        "- Do NOT group citations at the end — embed [N] directly after each sentence.\n"
        "  Correct: 'The system supports 5 concurrent users [1]. Login is blocked beyond this limit [2].'\n"
        "  Incorrect: 'The system supports 5 users and blocks logins beyond that limit. [1][2]'\n\n"

        "## PROFESSIONAL STANDARDS\n"
        "- Tone: authoritative, precise, and information-dense — never casual, generic, or verbose\n"
        "- Formatting: use Markdown (bold, headers, bullets) only when it enhances clarity\n"
        "- Scale: keep the response proportional to the complexity of the query\n"
        "- Accuracy: preserve exact names, figures, dates, and technical terms from the source material\n"
        "- Gaps: if source material is incomplete, state it explicitly rather than inferring\n"
        "- Concepts: if the document is technical, explain precisely while preserving correctness\n"
        "- Priority: surface the most important and actionable information first\n"
    )

    user_content = (
        f"{memory_block}"
        f"Source Context:\n{context_str or 'No sources available in this notebook.'}\n\n"
        f"User Question: {request.message}"
    )

    MAX_CONTEXT_CHARS = 18000 if (is_broad_query or is_toc_query) else 8000
    output_tokens = 2048 if (is_broad_query or is_toc_query) else 1024

    if len(context_str) > MAX_CONTEXT_CHARS:
        context_str = context_str[:MAX_CONTEXT_CHARS] + "\n\n[Context trimmed to fit token limit]"
        print(f"[Chat] Context trimmed to {MAX_CONTEXT_CHARS} chars to stay within TPM limit")

    user_content = (
        f"Source Context:\n{context_str or 'No sources available in this notebook.'}\n\n"
        f"User Question: {request.message}"
    )

    llm_response = ""

    for model_name in GROQ_MODEL_CASCADE:
        if llm_response:
            break
        for groq_key in groq_keys:
            try:
                groq_client = Groq(api_key=groq_key)
                completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    model=model_name,
                    temperature=0.2,
                    max_tokens=output_tokens,
                )
                llm_response = completion.choices[0].message.content
                key_index = groq_keys.index(groq_key) + 1
                print(f"[Chat] Responded using Groq key #{key_index}, model: {model_name}")
                break
            except Exception as groq_err:
                err_str = str(groq_err)
                groq_last_error = err_str
                key_index = groq_keys.index(groq_key) + 1
                if "rate_limit_exceeded" in err_str or "429" in err_str:
                    print(f"[Chat] Groq key #{key_index} rate limited on {model_name}, trying next key...")
                    continue
                else:
                    print(f"[Chat] Groq key #{key_index} error on {model_name}: {groq_err}")
                    break

    # 14. Ollama fallback
    if not llm_response and settings.OLLAMA_BASE_URL:
        try:
            async with httpx.AsyncClient() as client:
                models_res = await client.get(
                    f"{settings.OLLAMA_BASE_URL}/api/tags",
                    timeout=5.0
                )
                available_model = "llama3.1:8b"
                if models_res.status_code == 200:
                    models = models_res.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    print(f"[Ollama] Available models: {model_names}")
                    preferred = [
                        "llama3.3:70b",
                        "llama3.1:70b",
                        "qwen2.5:32b",
                        "phi4:14b",
                        "llama3.1:8b",
                        "llama3.1",
                        "llama3:latest",
                        "llama3",
                        "mistral",
                        "phi3",
                    ]
                    matched = next((p for p in preferred if p in model_names), None)
                    if matched:
                        available_model = matched
                    elif model_names:
                        available_model = model_names[0]

                print(f"[Ollama] Using model: {available_model}")

                ollama_citations = citations_list[:3]
                ollama_context = ""
                for i, c in enumerate(ollama_citations):
                    ollama_context += f"[{i+1}] {c.file_name}:\n{c.text[:300]}\n\n"

                ollama_system = (
                    "You are OpenNotebook's AI research assistant — authoritative, precise, and citation-grounded. "
                    "Answer the user's question using ONLY the source context provided. "
                    "Place an inline citation [N] after every factual sentence. "
                    "If the answer is not in the context, state: "
                    "'The uploaded documents do not contain information about [topic].' "
                    "Use structured formatting (numbered lists, bold, bullets) when it aids clarity. "
                    "Never hallucinate. Never use outside knowledge."
                )
                ollama_user_content = (
                    f"Source Context:\n{ollama_context or 'No sources available.'}\n"
                    f"Question: {request.message}"
                )

                print(f"[Ollama] Prompt size: system={len(ollama_system)} chars, user={len(ollama_user_content)} chars")

                ollama_res = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": available_model,
                        "messages": [
                            {"role": "system", "content": ollama_system},
                            {"role": "user", "content": ollama_user_content}
                        ],
                        "options": {"temperature": 0.3, "num_predict": 512},
                        "stream": False
                    },
                    timeout=300.0
                )
                if ollama_res.status_code == 200:
                    ollama_json = ollama_res.json()
                    llm_response = ollama_json.get("message", {}).get("content", "")
                    print(f"[Ollama] Response received successfully")
                else:
                    print(f"[Ollama] Error status: {ollama_res.status_code} - {ollama_res.text}")
        except Exception as ollama_err:
            import traceback
            print(f"[Ollama] API error: {type(ollama_err).__name__}: {ollama_err}")
            print(f"[Ollama] Traceback: {traceback.format_exc()}")

    # 15. Final fallback
    if not llm_response:
        if not context_str:
            llm_response = (
                "It looks like there are no documents in this notebook yet. "
                "Please upload a PDF, Word, PowerPoint, audio file, or paste text using the panel on the left. "
                "Once uploaded, I will index and parse your documents for semantic search."
            )
        else:
            retry_hint = ""
            if "rate_limit_exceeded" in groq_last_error and "Please try again in" in groq_last_error:
                try:
                    retry_hint = " (" + groq_last_error.split("Please try again in ")[1].split(".")[0] + " remaining on Groq free tier)"
                except Exception:
                    retry_hint = ""

            llm_response = (
                f"**AI model temporarily unavailable{retry_hint}.**\n\n"
                "All configured AI models are either rate-limited or unreachable right now.\n\n"
                "**What you can do:**\n"
                "- Wait a few minutes and try again\n"
                "- Start Ollama locally (`ollama serve`) as a free unlimited fallback\n"
                "- Upgrade your Groq plan at https://console.groq.com/settings/billing\n\n"
                "---\n"
                f"**Retrieved {len(citations_list)} relevant source excerpts** while waiting:\n\n"
                + "\n\n".join([
                    f"**[{i+1}] {c.file_name}:**\n> {c.text[:300]}..."
                    for i, c in enumerate(citations_list[:5])
                ])
            )

    # ── Save assistant turn + async memory updates ──────────────────────────
    if llm_response:
        # Save assistant turn to short-term memory
        asst_turn_count = await save_turn(
            db, notebook_id_str, user_id_str, conversation_id, "assistant", llm_response
        )

        # Store vector memory for this exchange (background thread)
        exchange_text = f"User: {request.message}\nAssistant: {llm_response[:500]}"
        threading.Thread(
            target=store_memory_vector,
            args=(qdrant_client, embed_model, notebook_id_str, conversation_id, exchange_text, "turn"),
            daemon=True
        ).start()

        # Maybe summarise (background thread — every SUMMARISE_EVERY turns)
        async def _bg_summarise():
            try:
                await maybe_summarise(
                    db, notebook_id_str, user_id_str,
                    conversation_id, asst_turn_count, groq_keys
                )
            except Exception as e:
                print(f"[Memory] Background summarise error: {e}")

        import asyncio
        asyncio.ensure_future(_bg_summarise())

    return ChatResponse(
        response=llm_response,
        citations=citations_list,
        conversation_id=conversation_id
    )