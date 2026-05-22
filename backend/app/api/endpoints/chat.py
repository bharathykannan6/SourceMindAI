import uuid
import math
import re
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

    # 6. Detect query type
    query_lower = request.message.lower()
    is_toc_query = any(term in query_lower for term in [
        "table of content", "table of contents", "toc", "index of", "chapters", "list the contents"
    ])
    is_broad_query = any(term in query_lower for term in [
        "summarize", "summary", "overview", "explain", "describe", "tell me about",
        "what is", "what are", "journey", "history", "all", "entire", "whole",
        "everything", "comprehensive", "detailed", "full",
        "about", "topics", "themes", "issues", "problems", "common", "main",
        "key", "important", "agents", "users", "conversations", "discuss",
        "cover", "contains", "file", "document", "report"
    ])

    dense_limit = 60 if (is_broad_query or is_toc_query) else 20
    max_context_chunks = 60 if is_toc_query else (50 if is_broad_query else 20)

    # 7. Dense semantic search
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

    # 8. RRF fusion
    rrf_scores = {}
    point_map = {}
    for rank, hit in enumerate(dense_results):
        p_id = str(hit.id)
        point_map[p_id] = hit
        rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + 1.0 / (60.0 + (rank + 1))
    for rank, (chunk_idx, score) in enumerate(sorted_bm25[:60]):
        chunk = chunks_res[chunk_idx]
        p_id = str(chunk.id)
        point_map[p_id] = chunk
        rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + 1.0 / (60.0 + (rank + 1))
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 9. Reranker
    rerank_candidates = sorted_rrf[:30]
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
            remaining = sorted_rrf[30:]
            sorted_rrf = [(p_id, float(score)) for p_id, score in reranked] + remaining
            print(f"[Reranker] Reranked {len(rerank_candidates)} candidates")
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

    # Build Groq key pool and model cascade early — used in both summary and RAG paths
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

    # ── SUMMARY PATH: Map-Reduce over ALL chunks ─────────────────────────────
    # For summary/overview queries, instead of retrieving top-N chunks,
    # we read ALL chunks for the document(s), split into batches,
    # summarize each batch, then combine into a final summary.
    # This covers 100% of the document instead of ~2%.
    if is_broad_query and not is_toc_query:
        print(f"[Chat] Summary query detected — using Map-Reduce over {len(chunks_res)} chunks")

        # Group chunks by document, sorted by chunk_index for reading order
        from collections import defaultdict
        doc_chunks_map: dict = defaultdict(list)
        for chunk in chunks_res:
            payload = chunk.payload if chunk.payload else {}
            doc_id = payload.get("document_id", "unknown")
            chunk_idx = payload.get("chunk_index", 0)
            text = payload.get("text", "")
            doc_chunks_map[doc_id].append((chunk_idx, text))

        # Sort each doc's chunks by index (reading order)
        for doc_id in doc_chunks_map:
            doc_chunks_map[doc_id].sort(key=lambda x: x[0])

        # Resolve doc_id -> file_name
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

        # MAP phase: summarize each document in batches
        # Cap at MAX_MAP_BATCHES to keep total response time under ~60 seconds.
        # Chunks are evenly sampled across the full document to maintain
        # coverage of all sections (start, middle, end) not just top chunks.
        MAX_MAP_BATCHES = 15  # 15 batches x ~2s each = ~30s map phase
        BATCH_CHARS = 6000
        partial_summaries = []

        map_system = (
            "You are a document analysis assistant. "
            "Read the following excerpt and extract key information: "
            "main topics, important facts, key figures, conclusions, notable details. "
            "Be concise but thorough. Output only extracted information, no preamble."
        )

        for doc_id, indexed_chunks in doc_chunks_map.items():
            file_name = doc_name_map.get(doc_id, "Document")
            all_text = "\n\n".join(text for _, text in indexed_chunks)
            total_chars = len(all_text)

            # Split full text into batches
            all_batches = [all_text[i:i + BATCH_CHARS] for i in range(0, total_chars, BATCH_CHARS)]
            total_batches = len(all_batches)

            # Evenly sample up to MAX_MAP_BATCHES from across the document
            if total_batches <= MAX_MAP_BATCHES:
                sampled_batches = list(enumerate(all_batches))
            else:
                step = total_batches / MAX_MAP_BATCHES
                sampled_indices = [int(i * step) for i in range(MAX_MAP_BATCHES)]
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

        # REDUCE phase: combine all partial summaries into final answer
        if partial_summaries:
            combined = "\n\n".join(partial_summaries)
            # Trim combined to fit within token budget for reduce step
            MAX_REDUCE_CHARS = 16000
            if len(combined) > MAX_REDUCE_CHARS:
                combined = combined[:MAX_REDUCE_CHARS] + "\n\n[Partial summaries trimmed]"

            reduce_system = (
                "You are an expert research assistant. "
                "You have been given partial summaries of one or more documents. "
                "Synthesize them into a single, comprehensive, well-structured final summary.\n\n"
                "Use this structure (include only sections that have relevant content):\n"
                "## Overview\n"
                "## Key Topics Covered\n"
                "## Executive Summary\n"
                "## Key Findings / Important Insights\n"
                "## Important Metrics / Numbers (if applicable)\n"
                "## Section-wise Summary (if document is large)\n"
                "## People / Organizations / Technologies Mentioned\n"
                "## Conclusion\n\n"
                "Use markdown formatting. Be specific — use actual names, numbers, and facts from the summaries. "
                "Do not hallucinate. If information is not in the summaries, do not include it."
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
                # Build citations from all chunks (first 10 for display)
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
                return ChatResponse(response=llm_response, citations=citations_list)

    # ── NON-SUMMARY PATH: standard RAG (specific questions, TOC, etc.) ─────────
    system_prompt = (
        "You are a helpful AI research assistant inside OpenNotebookLM.\n"
        f"The user is asking questions about documents in a notebook named '{notebook_name}'.\n\n"
        "INSTRUCTIONS:\n"
        "Answer the user's question directly using ONLY the Source Context provided. "
        "Never use outside knowledge.\n\n"
        "FORMAT based on what the user asked:\n\n"
        "1. LIST REQUEST (e.g. 'list the questions', 'give me the topics', 'what are the steps'):\n"
        "   Output a clean numbered list. No intro paragraph. No section headers. No commentary.\n"
        "   Example:\n"
        "   1. Define data communication\n"
        "   2. Compare MAC and IP address\n"
        "   3. Explain error correction\n\n"
        "2. PERSON / IDENTITY (e.g. 'who is X', 'tell me about X'):\n"
        "   If found in context: one short paragraph with name, role, and details.\n"
        "   If NOT found: reply ONLY with 'There is no mention of [X] in the uploaded documents.' Then stop.\n"
        "   Important: search carefully including partial name matches before saying not found.\n\n"
        "3. EXPLAIN / DEFINE (e.g. 'what is X', 'explain X', 'how does X work'):\n"
        "   Give a clear direct explanation. Use bullets if listing multiple aspects.\n\n"
        "4. SUMMARY / OVERVIEW (e.g. 'summarize', 'what is this about', 'overview'):\n"
        "   Write a structured summary \n"
        """Follow this exact structure whenever applicable:



2. Document Overview
- Provide a short 2–5 line overview explaining what the document is about.

3. Key Topics Covered
- List the major topics, concepts, or sections discussed in the document using bullet points.

4. Executive Summary
- Generate a concise but information-dense summary covering the core ideas, objectives, and conclusions of the document.

5. Important Insights / Key Findings
- Extract the most meaningful insights, discoveries, arguments, or conclusions.
- Use numbered bullet points.


7. Important Metrics / Numbers (if applicable)
- Extract statistics, percentages, benchmarks, measurements, financial figures, dates, or performance metrics.

8. Section-wise Summary (if the document is large)
- Summarize major sections individually.

9. Conclusion
- Provide a final concise conclusion about the overall purpose and outcome of the document.

10. Citations / References
- When possible, include page numbers, section references, timestamps, or source references for important claims.

11. Entities and Concepts
- Extract important people, organizations, technologies, products, research areas, or concepts mentioned in the document.\n"""

        "   For chat/support logs also add: ## People Involved, ## Common Issues\n"
        "   Use bullet points. Be specific with names, dates, numbers from context.\n\n"
        "5. COMPARISON (e.g. 'compare X and Y', 'difference between X and Y'):\n"
        "   Use clear bullet points per item or side-by-side structure.\n\n"
        "6. ANY OTHER QUESTION:\n"
        "   Answer directly and concisely using only the Source Context.\n"
        "   If not found: 'The uploaded documents do not contain information about [topic].'\n\n"
        "ALWAYS:\n"
        "- Never add sections or content the user did not ask for.\n"
        "- Never pad the response with unrelated content.\n"
        "- Cite sources inline with [1], [2] when quoting or paraphrasing.\n"
        "- Use Markdown (bold, bullets, headers) only when it improves clarity.\n"
        "- Keep the response proportional to what was asked.\n"

        """Behavior Rules:
- Maintain clarity and readability.
- Use markdown formatting.
- Prefer concise but high-information responses.
- Avoid hallucinating missing information.
- Clearly state when information is not available in the document.
- Preserve factual accuracy.
- If the document is technical, explain difficult concepts in simpler terms while preserving correctness.
- If the document is very large, prioritize the most important and actionable information.
- If the user asks for a short summary, provide:
  - TL;DR
  - Key insights
  - Final conclusion only.
- If the user asks for a detailed summary, include all sections.
- Always maintain professional formatting suitable for research assistants like NotebookLM or enterprise AI knowledge systems.\n"""
    )

    user_content = (
        f"Source Context:\n{context_str or 'No sources available in this notebook.'}\n\n"
        f"User Question: {request.message}"
    )

    # 12. Token budget
    # TPM limits per model: llama-3.3-70b = 6k TPM, llama-3.1-8b = 20k TPM
    # Keep total prompt (context + system + question) under 5000 tokens
    # to avoid hitting TPM limit across all keys simultaneously.
    # 1 token ~ 4 chars, so 5000 tokens ~ 20000 chars of context
    MAX_CONTEXT_CHARS = 18000 if (is_broad_query or is_toc_query) else 8000
    output_tokens = 2048 if (is_broad_query or is_toc_query) else 1024

    # Trim context_str to stay within token budget
    if len(context_str) > MAX_CONTEXT_CHARS:
        context_str = context_str[:MAX_CONTEXT_CHARS] + "\n\n[Context trimmed to fit token limit]"
        print(f"[Chat] Context trimmed to {MAX_CONTEXT_CHARS} chars to stay within TPM limit")

    # Rebuild user_content after trimming
    user_content = (
        f"Source Context:\n{context_str or 'No sources available in this notebook.'}\n\n"
        f"User Question: {request.message}"
    )

    # 13. Groq key pool + model cascade (used for non-summary RAG path)
    # Strategy:
    #   Round 1: try llama-3.3-70b across ALL 5 keys
    #   Round 2: try llama-3.1-8b across ALL 5 keys
    #   Round 3: try qwen/qwen3-32b across ALL 5 keys
    #   All exhausted → Ollama fallback

    llm_response = ""

    # Outer loop: model first, then keys
    # So all keys are tried with model A before falling back to model B
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
                break  # success — stop trying keys for this model
            except Exception as groq_err:
                err_str = str(groq_err)
                groq_last_error = err_str
                key_index = groq_keys.index(groq_key) + 1
                if "rate_limit_exceeded" in err_str or "429" in err_str:
                    print(f"[Chat] Groq key #{key_index} rate limited on {model_name}, trying next key...")
                    continue  # try next key with same model
                else:
                    print(f"[Chat] Groq key #{key_index} error on {model_name}: {groq_err}")
                    break  # non-rate-limit error, skip remaining keys for this model

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
                        "llama3.3:70b",    # same model as Groq — best quality
                        "llama3.1:70b",    # close second
                        "qwen2.5:32b",     # excellent mid-size
                        "phi4:14b",        # best small model for RAG
                        "llama3.1:8b",     # fast fallback
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

                # Short prompt for Ollama — keeps CPU inference fast
                ollama_citations = citations_list[:3]
                ollama_context = ""
                for i, c in enumerate(ollama_citations):
                    ollama_context += f"[{i+1}] {c.file_name}:\n{c.text[:300]}\n\n"

                ollama_system = (
                    "You are a helpful assistant. Answer the user's question using ONLY the source context. "
                    "Be concise and direct. Format as a numbered list if the user asks for a list. "
                    "If the answer is not in the context, say so."
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

    return ChatResponse(
        response=llm_response,
        citations=citations_list
    )
