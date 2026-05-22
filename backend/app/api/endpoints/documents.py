import uuid
import threading
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.models.user import User
from app.models.notebook import Notebook
from app.models.workspace import Workspace
from app.models.document import Document, DocumentStatus
from app.schemas.document import Document as DocumentSchema, DocumentURLCreate, DocumentTextCreate
from app.core.storage import get_minio_client, upload_file_to_minio
from app.rag.ingestion import process_document

router = APIRouter()
MINIO_BUCKET = "sourcemind-documents"


def run_ingestion(document_id: str, file_path: str, file_type: str, notebook_id: str):
    """Run ingestion in background thread. Status updated inside process_document."""
    process_document(document_id, file_path, file_type, notebook_id)


# ── POST / ── upload file ────────────────────────────────────────────────────

@router.post("/", response_model=DocumentSchema, status_code=status.HTTP_201_CREATED)
async def upload_document(
    notebook_id: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> Any:
    stmt = select(Notebook, Workspace).join(Workspace).where(
        Notebook.id == notebook_id,
        Workspace.owner_id == current_user.id
    )
    result = await db.execute(stmt)
    if not result.first():
        raise HTTPException(status_code=404, detail="Notebook not found")

    minio_client = get_minio_client()
    file_content = await file.read()
    file_ext = file.filename.split(".")[-1] if "." in file.filename else ""
    object_name = f"{current_user.id}/{notebook_id}/{uuid.uuid4()}.{file_ext}"

    try:
        upload_file_to_minio(
            client=minio_client,
            bucket_name=MINIO_BUCKET,
            object_name=object_name,
            data=file_content,
            content_type=file.content_type or "application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")

    document = Document(
        title=file.filename,
        notebook_id=notebook_id,
        file_path=f"{MINIO_BUCKET}/{object_name}",
        file_type=file_ext,
        status="pending"
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    threading.Thread(
        target=run_ingestion,
        args=(str(document.id), document.file_path, file_ext, str(notebook_id)),
        daemon=True
    ).start()

    return document


# ── GET /notebook/{notebook_id} ─────────────────────────────────────────────

@router.get("/notebook/{notebook_id}", response_model=List[DocumentSchema])
async def read_documents(
    notebook_id: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> Any:
    stmt = select(Notebook, Workspace).join(Workspace).where(
        Notebook.id == notebook_id,
        Workspace.owner_id == current_user.id
    )
    result = await db.execute(stmt)
    if not result.first():
        raise HTTPException(status_code=404, detail="Notebook not found")

    doc_stmt = select(Document).where(Document.notebook_id == notebook_id)
    doc_result = await db.execute(doc_stmt)
    return doc_result.scalars().all()


# ── POST /url ────────────────────────────────────────────────────────────────

@router.post("/url", response_model=DocumentSchema, status_code=status.HTTP_201_CREATED)
async def ingest_url(
    *,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
    document_in: DocumentURLCreate
) -> Any:
    stmt = select(Notebook, Workspace).join(Workspace).where(
        Notebook.id == document_in.notebook_id,
        Workspace.owner_id == current_user.id
    )
    result = await db.execute(stmt)
    if not result.first():
        raise HTTPException(status_code=404, detail="Notebook not found")

    url = document_in.url
    is_youtube = "youtube.com" in url or "youtu.be" in url
    file_type = "youtube" if is_youtube else "url"

    title = document_in.title or url.split("/")[-1] or "Web Link"
    if is_youtube:
        if "v=" in url:
            title = f"YouTube: {url.split('v=')[1].split('&')[0]}"
        elif "youtu.be/" in url:
            title = f"YouTube: {url.split('youtu.be/')[-1].split('?')[0]}"

    document = Document(
        title=title,
        notebook_id=document_in.notebook_id,
        file_path=url,
        file_type=file_type,
        status="pending"
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    threading.Thread(
        target=run_ingestion,
        args=(str(document.id), url, file_type, str(document_in.notebook_id)),
        daemon=True
    ).start()

    return document


# ── POST /text ───────────────────────────────────────────────────────────────

@router.post("/text", response_model=DocumentSchema, status_code=status.HTTP_201_CREATED)
async def ingest_raw_text(
    *,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
    document_in: DocumentTextCreate
) -> Any:
    from datetime import datetime

    stmt = select(Notebook, Workspace).join(Workspace).where(
        Notebook.id == document_in.notebook_id,
        Workspace.owner_id == current_user.id
    )
    result = await db.execute(stmt)
    if not result.first():
        raise HTTPException(status_code=404, detail="Notebook not found")

    title = document_in.title or f"Pasted Text ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
    text_content = document_in.text.encode("utf-8")

    minio_client = get_minio_client()
    object_name = f"{current_user.id}/{document_in.notebook_id}/{uuid.uuid4()}.txt"

    try:
        upload_file_to_minio(
            client=minio_client,
            bucket_name=MINIO_BUCKET,
            object_name=object_name,
            data=text_content,
            content_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")

    document = Document(
        title=title,
        notebook_id=document_in.notebook_id,
        file_path=f"{MINIO_BUCKET}/{object_name}",
        file_type="txt",
        status="pending"
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    threading.Thread(
        target=run_ingestion,
        args=(str(document.id), document.file_path, "txt", str(document_in.notebook_id)),
        daemon=True
    ).start()

    return document


# ── DELETE /{document_id} ── must be LAST to avoid catching /url /text ───────

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> None:
    stmt = select(Document, Notebook, Workspace).join(
        Notebook, Document.notebook_id == Notebook.id
    ).join(
        Workspace, Notebook.workspace_id == Workspace.id
    ).where(
        Document.id == document_id,
        Workspace.owner_id == current_user.id
    )
    result = await db.execute(stmt)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    document = row[0]
    file_path = document.file_path
    doc_id = str(document.id)

    await db.delete(document)
    await db.commit()

    def delete_vectors(did: str):
        try:
            from app.rag.ingestion import qdrant_client, VECTOR_COLLECTION
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            # Check collection exists before attempting delete
            collections = qdrant_client.get_collections().collections
            if not any(c.name == VECTOR_COLLECTION for c in collections):
                return  # nothing to delete
            qdrant_client.delete(
                collection_name=VECTOR_COLLECTION,
                points_selector=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=did))]
                )
            )
        except Exception as e:
            print(f"[Delete Vectors Error] {e}")

    def delete_from_minio(fp: str):
        try:
            if fp.startswith(MINIO_BUCKET):
                object_name = fp[len(MINIO_BUCKET) + 1:]
                get_minio_client().remove_object(MINIO_BUCKET, object_name)
        except Exception as e:
            print(f"[Delete MinIO Error] {e}")

    threading.Thread(target=delete_vectors, args=(doc_id,), daemon=True).start()
    threading.Thread(target=delete_from_minio, args=(file_path,), daemon=True).start()
