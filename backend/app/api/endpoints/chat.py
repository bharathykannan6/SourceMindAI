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

router = APIRouter()

# English stop words for fast keyword tokenization in BM25
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
    """Tokenize text by lowercasing, extracting alphanumeric words, and filtering out stopwords."""
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]

def check_toc_similarity(text: str) -> float:
    """Helper to detect if a text chunk is part of a Table of Contents (TOC)."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return 0.0
    
    toc_lines_count = 0
    for line in lines:
        # Ends with a page number (1-3 digits) optionally followed by some punctuation
        if re.search(r'\s\d{1,3}$', line):
            toc_lines_count += 1
            continue
        # Starts with digits followed by dot/space
        if re.match(r'^\d+[\.\s]', line):
            toc_lines_count += 1
            continue
        # Contains dot leaders
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

    # 3. Retrieve all chunks belonging to this notebook from Qdrant for BM25 Scoring
    # If specific document_ids are provided, filter to those only
    notebook_filter_conditions = [
        FieldCondition(
            key="notebook_id",
            match=MatchValue(value=str(request.notebook_id))
        )
    ]

    # Add document_id filter if specific docs are selected
    selected_doc_ids = request.document_ids or []
    if selected_doc_ids:
        from qdrant_client.models import Filter as QFilter, MatchAny
        doc_filter = Filter(
            must=[
                FieldCondition(
                    key="notebook_id",
                    match=MatchValue(value=str(request.notebook_id))
                ),
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=selected_doc_ids)
                )
            ]
        )
    else:
        doc_filter = Filter(must=notebook_filter_conditions)

    chunks_res = []
    try:
        chunks_res, _ = qdrant_client.scroll(
            collection_name=VECTOR_COLLECTION,
            scroll_filter=doc_filter,
            limit=10000,
            with_payload=True
        )
    except Exception as scroll_err:
        print(f"Qdrant scroll error (payloads search fallback): {scroll_err}")

    # Build dynamic inverted index and calculate BM25 scores
    sorted_bm25 = []
    doc_count = len(chunks_res)
    if doc_count > 0:
        chunk_lengths = []
        inverted_index = {} # term -> list of (chunk_idx, term_frequency)
        
        for idx, chunk in enumerate(chunks_res):
            text_content = chunk.payload.get("text", "") if chunk.payload else ""
            tokens = tokenize(text_content)
            chunk_lengths.append(len(tokens))
            
            # Compute term frequencies
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
                
            # Populate inverted index
            for term, freq in freqs.items():
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((idx, freq))
        
        avgdl = sum(chunk_lengths) / doc_count if doc_count > 0 else 0
        
        # Score query terms against candidate chunks containing them
        query_terms = tokenize(request.message)
        bm25_scores = {} # chunk_idx -> float score
        k1 = 1.2
        b = 0.75
        
        for term in query_terms:
            if term in inverted_index:
                postings = inverted_index[term]
                n_q = len(postings)
                
                # BM25 standard IDF formulation
                idf = math.log(1.0 + (doc_count - n_q + 0.5) / (n_q + 0.5))
                
                for chunk_idx, tf in postings:
                    doc_len = chunk_lengths[chunk_idx]
                    denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl if avgdl > 0 else 1.0))
                    num = tf * (k1 + 1.0)
                    term_score = idf * (num / denom)
                    bm25_scores[chunk_idx] = bm25_scores.get(chunk_idx, 0.0) + term_score
                    
        sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)

    # 4. Retrieve dense semantic search results (top 15) from Qdrant
    dense_results = []
    try:
        dense_results = qdrant_client.search(
            collection_name=VECTOR_COLLECTION,
            query_vector=query_vector,
            query_filter=doc_filter,
            limit=15
        )
    except Exception as e:
        print(f"Qdrant dense search error: {e}")

    # 5. Fuse dense and sparse results using Reciprocal Rank Fusion (RRF)
    rrf_scores = {} # point_id -> float score
    point_map = {}  # point_id -> Point/Record object
    
    # Add dense ranks to RRF
    for rank, hit in enumerate(dense_results):
        p_id = str(hit.id)
        point_map[p_id] = hit
        rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + 1.0 / (60.0 + (rank + 1))
        
    # Add sparse BM25 ranks to RRF (limiting to top 60 candidates)
    for rank, (chunk_idx, score) in enumerate(sorted_bm25[:60]):
        chunk = chunks_res[chunk_idx]
        p_id = str(chunk.id)
        point_map[p_id] = chunk
        rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + 1.0 / (60.0 + (rank + 1))
        
    # Sort by descending RRF score
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 6. Construct context and citations list
    context_str = ""
    citations_list: List[Citation] = []
    doc_cache = {}

    is_toc_query = any(term in request.message.lower() for term in ["table of content", "table of contents", "toc", "index of", "chapters", "list the contents"])

    final_hits = []
    seen_chunk_keys = set()

    # If it's a TOC query, let's proactively identify and group TOC chunks by document
    toc_chunks_to_prepend = []
    if is_toc_query and doc_count > 0:
        # Find doc_ids that appear in the top search hits to ensure we only retrieve TOC for relevant documents
        candidate_doc_ids = set()
        for p_id, _ in sorted_rrf[:15]:
            hit = point_map[p_id]
            payload = hit.payload if hit.payload else {}
            doc_id = payload.get("document_id")
            if doc_id:
                candidate_doc_ids.add(str(doc_id))

        for doc_id_str in candidate_doc_ids:
            # Find all chunks for this document
            doc_chunks = [c for c in chunks_res if c.payload and str(c.payload.get("document_id")) == doc_id_str]
            # Find the starting TOC chunk
            start_chunk = None
            for chunk in doc_chunks:
                payload = chunk.payload
                text_lower = payload.get("text", "").lower()
                # Check if this chunk is a likely start of the TOC
                if "table of contents" in text_lower or "contents" in text_lower or "index" in text_lower:
                    if check_toc_similarity(payload.get("text", "")) > 0.4:
                        start_chunk = chunk
                        break
            
            # If we didn't find one with score > 0.4, try any chunk with the TOC keyword in the top hits
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
                # Fetch subsequent chunks sequentially as long as they look like TOC
                toc_chunks_to_prepend.append(start_chunk)
                current_idx = start_idx + 1
                max_toc_expansion = 15 # safety limit
                for _ in range(max_toc_expansion):
                    next_chunk = next((c for c in doc_chunks if c.payload.get("chunk_index") == current_idx), None)
                    if next_chunk:
                        text = next_chunk.payload.get("text", "")
                        if check_toc_similarity(text) > 0.15:
                            toc_chunks_to_prepend.append(next_chunk)
                            current_idx += 1
                        else:
                            # Let's check one more chunk in case of a single transition/page-boundary gap
                            next_next_chunk = next((c for c in doc_chunks if c.payload.get("chunk_index") == current_idx + 1), None)
                            if next_next_chunk and check_toc_similarity(next_next_chunk.payload.get("text", "")) > 0.15:
                                toc_chunks_to_prepend.append(next_chunk)
                                toc_chunks_to_prepend.append(next_next_chunk)
                                current_idx += 2
                            else:
                                break
                    else:
                        break

    # Add the identified TOC chunks to final_hits first
    # We assign them a higher pseudo RRF score to ensure they stay at the front
    base_toc_score = 0.5 # higher than any normal RRF score
    for hit in toc_chunks_to_prepend:
        payload = hit.payload if hit.payload else {}
        doc_id = payload.get("document_id")
        chunk_idx = payload.get("chunk_index")
        if doc_id and chunk_idx is not None:
            chunk_key = (doc_id, chunk_idx)
            if chunk_key not in seen_chunk_keys:
                seen_chunk_keys.add(chunk_key)
                final_hits.append((hit, base_toc_score))
                base_toc_score -= 0.001 # slightly degrade to maintain order

    # Add the remaining top RRF search hits
    for p_id, score in sorted_rrf[:15]:
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

    # Determine how many context chunks to send to the LLM
    max_context_chunks = 30 if is_toc_query else 15
    for idx, (hit, score) in enumerate(final_hits[:max_context_chunks]):
        payload = hit.payload if hit.payload else {}
        doc_id = payload.get("document_id")
        chunk_text = payload.get("text")

        # Get the actual filename/title of the document
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


    # 5. Synthesize LLM Response
    system_prompt = (
        "You are a strict document-only AI research assistant inside OpenNotebookLM.\n"
        f"You are helping the user with their notebook named '{notebook_name}'.\n"
        "CRITICAL RULES - you MUST follow these without exception:\n"
        "1. You may ONLY answer using the Source Context chunks provided below. Nothing else.\n"
        "2. You must NEVER use your pre-trained knowledge, general knowledge, or any information outside the provided Source Context.\n"
        "3. If the Source Context does not contain information to answer the query, you MUST respond EXACTLY with:\n"
        "   \"I'm sorry, but the information about '[topic]' is not present in any of the documents uploaded to this notebook. Please upload a document containing this information and try again.\"\n"
        "4. Do NOT add any general information, background knowledge, or helpful tips from outside the documents. Silence is better than hallucination.\n"
        "5. Use inline citations like [1], [2], etc., corresponding to the numbered sources provided. Never invent citations.\n"
        "6. Format your response clearly in Markdown.\n"
    )

    user_content = f"Source Context:\n{context_str or 'No sources available in this notebook.'}\n\nUser Query: {request.message}"

    llm_response = ""
    # Try Groq first
    if settings.GROQ_API_KEY:
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
            )
            llm_response = completion.choices[0].message.content
        except Exception as groq_err:
            print(f"Groq API error: {groq_err}")

    # Fallback to Ollama if Groq fails or is not configured
    if not llm_response and settings.OLLAMA_BASE_URL:
        try:
            async with httpx.AsyncClient() as client:
                ollama_res = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": "llama3",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        "options": {"temperature": 0.3},
                        "stream": False
                    },
                    timeout=30.0
                )
                if ollama_res.status_code == 200:
                    ollama_json = ollama_res.json()
                    llm_response = ollama_json.get("message", {}).get("content", "")
        except Exception as ollama_err:
            print(f"Ollama API error: {ollama_err}")

    # Final backup if everything is down or empty
    if not llm_response:
        if not context_str:
            llm_response = (
                "It looks like there are no documents in this notebook yet, or the database is starting up. "
                "Please upload a PDF, Word, PowerPoint, Audio file, or Web Link using the 'Drop files or click to add' "
                "button in the side explorer. Once uploaded, I will automatically index and parse your documents for semantic search!"
            )
        else:
            llm_response = (
                f"I've found {len(citations_list)} relevant excerpts in your notebook documents, "
                "but I couldn't reach the AI language generation server to synthesize a full summary. "
                "Here are the direct source snippets I found:\n\n"
                + "\n\n".join([f"**From {c.file_name}:**\n> {c.text}" for c in citations_list])
            )

    return ChatResponse(
        response=llm_response,
        citations=citations_list
    )
