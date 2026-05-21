from typing import Optional
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime

class WorkspaceBase(BaseModel):
    name: str

class WorkspaceCreate(WorkspaceBase):
    pass

class WorkspaceUpdate(BaseModel):
    name: Optional[str] = None

class WorkspaceInDBBase(WorkspaceBase):
    id: UUID
    owner_id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}

class Workspace(WorkspaceInDBBase):
    pass
