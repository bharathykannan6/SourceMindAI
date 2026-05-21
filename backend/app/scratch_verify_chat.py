import asyncio
import sys
import uuid

sys.path.append("c:\\PROJECTS\\Dsignz Media\\OpenNotebookLM\\backend")

from app.api.endpoints.chat import chat_with_notebook
from app.schemas.chat import ChatRequest
from app.db.database import AsyncSessionLocal
from app.models.user import User

async def main():
    print("Verifying updated chat.py retrieval and TOC expansion...")
    
    async with AsyncSessionLocal() as db:
        mock_user = User(
            id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
            email="test@sourcemind.ai",
            is_active=True
        )
        
        request = ChatRequest(
            message="list out the table of content",
            notebook_id=uuid.UUID("9fee9916-3780-4339-ae90-8283c5b472f2")
        )
        
        try:
            res = await chat_with_notebook(request=request, db=db, current_user=mock_user)
            print("\n--- CHAT RESPONSE RECEIVED ---")
            print(f"Response: {res.response[:200]}...")
            print(f"\nTotal Citations Returned: {len(res.citations)}")
            print("Citations List:")
            for idx, cit in enumerate(res.citations):
                snippet = cit.text[:80].replace('\n', ' ')
                print(f"  {idx+1}. File: {cit.file_name} | Score: {cit.score:.4f} | Snippet: {snippet}")
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
