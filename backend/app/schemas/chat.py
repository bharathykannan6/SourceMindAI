from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    notebook_id: UUID
    document_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None  # client sends this to persist memory across turns

class Citation(BaseModel):
    document_id: str
    file_name: str
    text: str
    score: float

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    conversation_id: str  # always returned so client can track it
