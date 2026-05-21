import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.core.config import settings

qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
VECTOR_COLLECTION = "sourcemind_documents"

async def main():
    doc_id = "6a0fcb2e-7b0f-4444-a305-5b5d39896feb"
    
    # Scroll through points of this document
    offset = None
    all_chunks = []
    
    while True:
        res, next_offset = qdrant_client.scroll(
            collection_name=VECTOR_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=100,
            offset=offset
        )
        all_chunks.extend(res)
        if not next_offset:
            break
        offset = next_offset
        
    print(f"Total chunks found for document: {len(all_chunks)}")
    
    # Search for "table of contents", "contents", or numbered items in TOC format
    toc_chunks = []
    for hit in all_chunks:
        text = hit.payload.get("text", "")
        if "table of contents" in text.lower() or "contents" in text.lower():
            toc_chunks.append(hit)
            
    print(f"Found {len(toc_chunks)} chunks containing 'contents' or 'table of contents':")
    for hit in toc_chunks:
        print(f"\nChunk Index: {hit.payload.get('chunk_index')}")
        print(f"Text Snippet:\n{hit.payload.get('text')[:400]}")

if __name__ == "__main__":
    asyncio.run(main())
