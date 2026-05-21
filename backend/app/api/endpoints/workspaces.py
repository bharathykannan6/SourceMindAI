from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.models.user import User
from app.models.workspace import Workspace
from app.schemas.workspace import Workspace as WorkspaceSchema, WorkspaceCreate, WorkspaceUpdate

router = APIRouter()

@router.get("/", response_model=List[WorkspaceSchema])
async def read_workspaces(
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> Any:
    """Retrieve workspaces for current user."""
    stmt = select(Workspace).where(Workspace.owner_id == current_user.id).order_by(Workspace.created_at.asc())
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("/", response_model=WorkspaceSchema, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    workspace_in: WorkspaceCreate,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> Any:
    """Create new workspace."""
    workspace = Workspace(
        name=workspace_in.name,
        owner_id=current_user.id
    )
    db.add(workspace)
    await db.commit()
    await db.refresh(workspace)
    return workspace

@router.get("/{workspace_id}", response_model=WorkspaceSchema)
async def read_workspace(
    workspace_id: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> Any:
    """Get workspace by ID."""
    stmt = select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    result = await db.execute(stmt)
    workspace = result.scalars().first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace

@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_workspace(
    workspace_id: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> None:
    """Delete a workspace."""
    stmt = select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    result = await db.execute(stmt)
    workspace = result.scalars().first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    await db.delete(workspace)
    await db.commit()
