import asyncio
import sys
import math
import re
import uuid
from typing import List
from groq import Groq
import httpx

sys.path.append("c:\\PROJECTS\\Dsignz Media\\OpenNotebookLM\\backend")

from app.core.config import settings
from app.rag.ingestion import embed_model, qdrant_client, VECTOR_COLLECTION
from qdrant_client.models import Filter, FieldCondition, MatchValue

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

async def test_llm():
    notebook_id = "9fee9916-3780-4339-ae90-8283c5b472f2"
    notebook_name = "third"
    message = "list out the table of content"
    
    print(f"Testing RAG + LLM for Notebook: {notebook_name} ({notebook_id})")
    
    # 1. Embed
    query_vector = embed_model.encode(message, convert_to_numpy=True).tolist()
    
    # 2. Scroll all
    chunks_res, _ = qdrant_client.scroll(
        collection_name=VECTOR_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="notebook_id", match=MatchValue(value=str(notebook_id)))
            ]
        ),
        limit=10000,
        with_payload=True
    )
    
    # 3. BM25
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
        query_terms = tokenize(message)
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

    # 4. Dense Search
    dense_results = qdrant_client.search(
        collection_name=VECTOR_COLLECTION,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="notebook_id", match=MatchValue(value=str(notebook_id)))
            ]
        ),
        limit=15
    )

    # 5. Fuse
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

    # 6. TOC logic
    is_toc_query = any(term in message.lower() for term in ["table of content", "table of contents", "toc", "index of", "chapters", "list the contents"])
    final_hits = []
    seen_chunk_keys = set()
    
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
        
        if is_toc_query and doc_id and chunk_idx is not None:
            text_lower = payload.get("text", "").lower()
            if "table of contents" in text_lower or "contents" in text_lower or "index" in text_lower:
                for next_idx in range(chunk_idx + 1, chunk_idx + 9):
                    next_chunk = next((c for c in chunks_res if c.payload and str(c.payload.get("document_id")) == str(doc_id) and c.payload.get("chunk_index") == next_idx), None)
                    if next_chunk:
                        next_key = (doc_id, next_idx)
                        if next_key not in seen_chunk_keys:
                            seen_chunk_keys.add(next_key)
                            final_hits.append((next_chunk, score * 0.9))

    context_str = ""
    for idx, (hit, score) in enumerate(final_hits[:25]):
        payload = hit.payload if hit.payload else {}
        chunk_text = payload.get("text")
        context_str += f"Source [{idx+1}] (Document: ai in india):\n{chunk_text}\n\n"

    # 7. LLM
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

    user_content = f"Source Context:\n{context_str}\n\nUser Query: {message}"

    print("Sending request to Groq...")
    if settings.GROQ_API_KEY:
        client = Groq(api_key=settings.GROQ_API_KEY)
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )
        print("Response:\n", completion.choices[0].message.content)
    else:
        print("Groq API Key not configured.")

if __name__ == "__main__":
    asyncio.run(test_llm())
