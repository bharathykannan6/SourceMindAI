import asyncio
from app.db.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy.future import select
from app.core.storage import get_minio_client
import fitz

async def main():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document))
        docs = result.scalars().all()
        print("Documents in DB:")
        for doc in docs:
            print(f"ID: {doc.id}, Title: {doc.title}, Path: {doc.file_path}, Type: {doc.file_type}")
            if "ai" in doc.title.lower() or "journey" in doc.title.lower():
                print("Found target document!")
                # Download from MinIO
                minio_client = get_minio_client()
                bucket_name = "sourcemind-documents"
                object_name = doc.file_path.split(f"{bucket_name}/")[1]
                response = minio_client.get_object(bucket_name, object_name)
                try:
                    file_bytes = response.read()
                    print(f"Downloaded {len(file_bytes)} bytes.")
                    pdf = fitz.open(stream=file_bytes, filetype="pdf")
                    print(f"PDF Pages: {len(pdf)}")
                    # Let's inspect the first 10 pages for TOC
                    for page_num in range(min(15, len(pdf))):
                        text = pdf[page_num].get_text()
                        if "CONTENTS" in text or "Contents" in text or "Table of" in text:
                            print(f"\n--- Page {page_num+1} contains TOC indicators ---")
                            print(text[:1500])
                finally:
                    response.close()
                    response.release_conn()

if __name__ == "__main__":
    asyncio.run(main())
