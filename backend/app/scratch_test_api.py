import asyncio
import httpx

async def main():
    notebook_id = "9fee9916-3780-4339-ae90-8283c5b472f2"
    payload = {"notebook_id": notebook_id, "message": "list out the table of content"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post("http://127.0.0.1:8080/api/v1/chat/", json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Response:", data.get("response", ""))
            print("Citations:", len(data.get("citations", [])))
        else:
            print("Error:", response.text[:200])

if __name__ == "__main__":
    asyncio.run(main())
