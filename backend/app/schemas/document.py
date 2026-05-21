from typing import Optional
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime
from app.models.document import DocumentStatus

class DocumentBase(BaseModel):
    title: str

class DocumentCreate(DocumentBase):
    notebook_id: UUID
    file_type: str
    file_path: str

class DocumentURLCreate(BaseModel):
    notebook_id: UUID
    url: str
    title: Optional[str] = None

class DocumentTextCreate(BaseModel):
    notebook_id: UUID
    text: str
    title: Optional[str] = None


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[DocumentStatus] = None

class DocumentInDBBase(DocumentBase):
    id: UUID
    notebook_id: UUID
    file_path: str
    file_type: str
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}

class Document(DocumentInDBBase):
    pass
