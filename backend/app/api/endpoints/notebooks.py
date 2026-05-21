from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.models.user import User
from app.models.workspace import Workspace
from app.models.notebook import Notebook
from app.schemas.notebook import Notebook as NotebookSchema, NotebookCreate, NotebookUpdate

router = APIRouter()

@router.get("/workspace/{workspace_id}", response_model=List[NotebookSchema])
async def read_notebooks(
    workspace_id: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> Any:
    """Retrieve notebooks for a specific workspace."""
    # First verify the user owns this workspace
    workspace_stmt = select(Workspace).where(Workspace.id == workspace_id, Workspace.owner_id == current_user.id)
    workspace_result = await db.execute(workspace_stmt)
    if not workspace_result.scalars().first():
        raise HTTPException(status_code=404, detail="Workspace not found")

    stmt = select(Notebook).where(Notebook.workspace_id == workspace_id).order_by(Notebook.created_at.asc())
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("/", response_model=NotebookSchema, status_code=status.HTTP_201_CREATED)
async def create_notebook(
    notebook_in: NotebookCreate,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> Any:
    """Create new notebook."""
    # Verify workspace ownership
    workspace_stmt = select(Workspace).where(Workspace.id == notebook_in.workspace_id, Workspace.owner_id == current_user.id)
    workspace_result = await db.execute(workspace_stmt)
    if not workspace_result.scalars().first():
        raise HTTPException(status_code=404, detail="Workspace not found")

    notebook = Notebook(
        name=notebook_in.name,
        workspace_id=notebook_in.workspace_id
    )
    db.add(notebook)
    await db.commit()
    await db.refresh(notebook)
    return notebook

@router.delete("/{notebook_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_notebook(
    notebook_id: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> None:
    """Delete a notebook."""
    # Join with Workspace to verify ownership
    stmt = select(Notebook, Workspace).join(Workspace).where(
        Notebook.id == notebook_id,
        Workspace.owner_id == current_user.id
    )
    result = await db.execute(stmt)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Notebook not found")
    
    notebook = row[0]
    await db.delete(notebook)
    await db.commit()
