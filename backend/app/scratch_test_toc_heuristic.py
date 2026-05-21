import asyncio
import sys
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

sys.path.append("c:\\PROJECTS\\Dsignz Media\\OpenNotebookLM\\backend")
from app.core.config import settings

qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
VECTOR_COLLECTION = "sourcemind_documents"

def check_toc_similarity(text: str) -> float:
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

async def main():
    doc_id = "6a0fcb2e-7b0f-4444-a305-5b5d39896feb"
    
    res, _ = qdrant_client.scroll(
        collection_name=VECTOR_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=doc_id))
            ]
        ),
        limit=517,
    )
    
    sorted_chunks = sorted(res, key=lambda x: x.payload.get("chunk_index", 0))
    
    print("Testing TOC similarity heuristic on chunks 0 to 20:")
    for i in range(25):
        if i < len(sorted_chunks):
            hit = sorted_chunks[i]
            text = hit.payload.get("text", "")
            score = check_toc_similarity(text)
            snippet = text[:80].replace("\n", " ")
            print(f"Chunk {i}: TOC score = {score:.2f} | Snippet: {snippet}")

if __name__ == "__main__":
    asyncio.run(main())
