import asyncio
import httpx

async def test_creation():
    async with httpx.AsyncClient() as client:
        # 1. Fetch workspaces
        res = await client.get("http://127.0.0.1:8080/api/v1/workspaces/")
        print("Workspaces status:", res.status_code)
        workspaces = res.json()
        print("Workspaces:", workspaces)
        
        if not workspaces:
            # Create workspace
            res_ws = await client.post("http://127.0.0.1:8080/api/v1/workspaces/", json={"name": "Test Workspace"})
            print("Create Workspace status:", res_ws.status_code)
            workspace = res_ws.json()
        else:
            workspace = workspaces[0]
            
        workspace_id = workspace["id"]
        print("Using Workspace ID:", workspace_id)
        
        # 2. Try to create notebook
        res_nb = await client.post(
            "http://127.0.0.1:8080/api/v1/notebooks/",
            json={"name": "Test Notebook", "workspace_id": workspace_id}
        )
        print("Create Notebook status:", res_nb.status_code)
        print("Create Notebook response:", res_nb.text)

if __name__ == "__main__":
    asyncio.run(test_creation())
