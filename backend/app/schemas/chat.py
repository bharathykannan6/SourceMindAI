from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    notebook_id: UUID
    document_ids: Optional[List[str]] = None  # If provided, filter to these documents only

class Citation(BaseModel):
    document_id: str
    file_name: str
    text: str
    score: float

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
