import asyncio
import sys
import math
import re
import uuid
from typing import List, Tuple
from sqlalchemy.future import select

sys.path.append("c:\\PROJECTS\\Dsignz Media\\OpenNotebookLM\\backend")

from app.db.database import AsyncSessionLocal
from app.models.notebook import Notebook
from app.models.workspace import Workspace
from app.models.document import Document
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

async def test_retrieval():
    notebook_id = "b909a821-9f1a-49a5-8779-077cfb7482f2"
    message = "list out the table of content"
    
    print(f"Testing retrieval for Notebook: {notebook_id}")
    print(f"Message: {message}")
    
    # 1. Embed the user's message
    query_vector = embed_model.encode(message, convert_to_numpy=True).tolist()
    
    # 2. Retrieve all chunks
    chunks_res, _ = qdrant_client.scroll(
        collection_name=VECTOR_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="notebook_id",
                    match=MatchValue(value=str(notebook_id))
                )
            ]
        ),
        limit=10000,
        with_payload=True
    )
    print(f"Total chunks in notebook: {len(chunks_res)}")
    
    # BM25 Scoring
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
        
    print(f"Top 5 BM25 matches:")
    for idx, score in sorted_bm25[:5]:
        print(f"  Chunk {chunks_res[idx].payload.get('chunk_index')}: score {score:.4f} | Snippet: {chunks_res[idx].payload.get('text')[:100]!r}")

    # Dense Semantic Search
    dense_results = qdrant_client.search(
        collection_name=VECTOR_COLLECTION,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="notebook_id",
                    match=MatchValue(value=str(notebook_id))
                )
            ]
        ),
        limit=15
    )
    print(f"Top 5 Dense matches:")
    for rank, hit in enumerate(dense_results[:5]):
        print(f"  Chunk {hit.payload.get('chunk_index')}: score {hit.score:.4f} | Snippet: {hit.payload.get('text')[:100]!r}")

    # Fuse
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
    print(f"Top 5 RRF matches:")
    for rank, (p_id, score) in enumerate(sorted_rrf[:5]):
        hit = point_map[p_id]
        print(f"  Chunk {hit.payload.get('chunk_index')}: score {score:.6f} | Snippet: {hit.payload.get('text')[:100]!r}")

    # TOC logic
    is_toc_query = any(term in message.lower() for term in ["table of content", "table of contents", "toc", "index of", "chapters", "list the contents"])
    print(f"is_toc_query: {is_toc_query}")
    
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
                print(f"TOC match found in Chunk {chunk_idx}! Expanding...")
                for next_idx in range(chunk_idx + 1, chunk_idx + 9):
                    next_chunk = next((c for c in chunks_res if c.payload and str(c.payload.get("document_id")) == str(doc_id) and c.payload.get("chunk_index") == next_idx), None)
                    if next_chunk:
                        next_key = (doc_id, next_idx)
                        if next_key not in seen_chunk_keys:
                            seen_chunk_keys.add(next_key)
                            final_hits.append((next_chunk, score * 0.9))
                            print(f"  Expanded chunk {next_idx}")
                            
    print(f"Total final hits: {len(final_hits)}")
    print(f"Final hits chunk indexes: {[h.payload.get('chunk_index') for h, s in final_hits]}")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
