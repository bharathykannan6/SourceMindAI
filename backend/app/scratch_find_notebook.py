import asyncio
from app.db.database import AsyncSessionLocal
from app.models.notebook import Notebook
from app.models.document import Document
from sqlalchemy.future import select

async def main():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Notebook))
        notebooks = res.scalars().all()
        print("All Notebooks:")
        for nb in notebooks:
            print(f"  ID: {nb.id}, Name: {nb.name}, Workspace: {nb.workspace_id}")
        
        print("\nAll Documents:")
        res = await db.execute(select(Document))
        docs = res.scalars().all()
        for doc in docs:
            print(f"  ID: {doc.id}, Title: {doc.title}, Notebook: {doc.notebook_id}")

if __name__ == "__main__":
    asyncio.run(main())
