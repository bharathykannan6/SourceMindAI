from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.api.main import api_router
from app.db.database import engine, Base
from app.models.user import User
from app.models.workspace import Workspace
from app.models.notebook import Notebook
from app.models.document import Document

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Automatically create database tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Automatically insert mock test user to satisfy foreign key constraints
    import uuid
    from sqlalchemy.future import select
    from app.db.database import AsyncSessionLocal
    mock_user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    async with AsyncSessionLocal() as session:
        stmt = select(User).where(User.id == mock_user_id)
        result = await session.execute(stmt)
        if not result.scalars().first():
            mock_user = User(
                id=mock_user_id,
                email="test@sourcemind.ai",
                hashed_password="mocked_hashed_password",
                full_name="Mock User",
                is_active=True
            )
            session.add(mock_user)
            await session.commit()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="SourceMind AI Backend",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)


# ── Enforce upload size limit ─────────────────────────────────────────────────
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"File too large. Maximum allowed size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
                },
            )
    return await call_next(request)


# ── CORS ──────────────────────────────────────────────────────────────────────
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).rstrip("/") for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
