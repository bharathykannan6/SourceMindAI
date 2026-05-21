import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.core.config import settings

qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
VECTOR_COLLECTION = "sourcemind_documents"

async def main():
    doc_id = "6a0fcb2e-7b0f-4444-a305-5b5d39896feb"
    
    # Scroll and retrieve chunk index 0 to 15
    res, _ = qdrant_client.scroll(
        collection_name=VECTOR_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=doc_id))
            ]
        ),
        limit=517,
    )
    
    # Sort by chunk_index
    sorted_chunks = sorted(res, key=lambda x: x.payload.get("chunk_index", 0))
    
    for i in range(9):
        if i < len(sorted_chunks):
            hit = sorted_chunks[i]
            print(f"\n================ CHUNK INDEX: {hit.payload.get('chunk_index')} ================")
            print(hit.payload.get("text"))

if __name__ == "__main__":
    asyncio.run(main())

