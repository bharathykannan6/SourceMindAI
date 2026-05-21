from typing import Optional
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime

class NotebookBase(BaseModel):
    name: str

class NotebookCreate(NotebookBase):
    workspace_id: UUID

class NotebookUpdate(BaseModel):
    name: Optional[str] = None

class NotebookInDBBase(NotebookBase):
    id: UUID
    workspace_id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}

class Notebook(NotebookInDBBase):
    pass
