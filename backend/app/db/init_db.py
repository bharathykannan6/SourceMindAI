import asyncio
import uuid
from sqlalchemy.future import select
from app.db.database import engine, Base, AsyncSessionLocal

# Import all models to ensure they register on Base.metadata
from app.models.user import User
from app.models.workspace import Workspace
from app.models.notebook import Notebook
from app.models.document import Document

async def init_db():
    print("Connecting to database and initializing tables...")
    async with engine.begin() as conn:
        # Create all tables defined in models
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables initialized successfully!")

    print("Checking if mock test user exists...")
    mock_user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    
    async with AsyncSessionLocal() as session:
        # Check if user exists
        stmt = select(User).where(User.id == mock_user_id)
        result = await session.execute(stmt)
        user = result.scalars().first()
        
        if not user:
            print("Mock user not found. Inserting mock user to satisfy foreign key constraints...")
            mock_user = User(
                id=mock_user_id,
                email="test@sourcemind.ai",
                hashed_password="mocked_hashed_password",
                full_name="Mock User",
                is_active=True
            )
            session.add(mock_user)
            await session.commit()
            print("Mock user inserted successfully!")
        else:
            print("Mock user already exists!")

if __name__ == "__main__":
    asyncio.run(init_db())
