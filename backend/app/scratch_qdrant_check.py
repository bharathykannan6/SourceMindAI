import asyncio
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.core.config import settings

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
VECTOR_COLLECTION = "sourcemind_documents"

async def main():
    query = "table of content"
    query_vector = embed_model.encode(query).tolist()
    
    # Semantic search in Qdrant
    results = qdrant_client.search(
        collection_name=VECTOR_COLLECTION,
        query_vector=query_vector,
        limit=10
    )
    
    print("Semantic Search Results:")
    for i, hit in enumerate(results):
        payload = hit.payload
        print(f"\n[{i+1}] Score: {hit.score:.4f}")
        print(f"Doc ID: {payload.get('document_id')}")
        print(f"Chunk Index: {payload.get('chunk_index')}")
        print(f"Text Snippet:\n{payload.get('text')[:300]}")

if __name__ == "__main__":
    asyncio.run(main())
