# OpenNotebookLM — Bug Fixes

## Issue 1: File Upload Not Working

### Root Cause (Backend)

`main.py` defines `MAX_UPLOAD_SIZE = 500 * 1024 * 1024` but **never wires it into FastAPI**.
FastAPI + uvicorn default to a very small body limit. Large files silently fail or get cut off.

**Fix — `backend/app/main.py`:** Add a size-limit middleware and increase uvicorn body limits.

```python
# backend/app/main.py  ← replace the entire file with this

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.api.main import api_router
from app.db.database import engine, Base
from app.models.user import User
from app.models.workspace import Workspace
from app.models.notebook import Notebook
from app.models.document import Document

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    import uuid
    from sqlalchemy.future import select
    from app.db.database import AsyncSessionLocal
    mock_user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    async with AsyncSessionLocal() as session:
        stmt = select(User).where(User.id == mock_user_id)
        result = await session.execute(stmt)
        if not result.scalars().first():
            mock_user = User(
                id=mock_user_id,
                email="test@sourcemind.ai",
                hashed_password="mocked_hashed_password",
                full_name="Mock User",
                is_active=True
            )
            session.add(mock_user)
            await session.commit()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="SourceMind AI Backend",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# ── Enforce upload size limit ─────────────────────────────────────────────────
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"File too large. Maximum allowed size is {MAX_UPLOAD_SIZE // (1024*1024)} MB."}
            )
    return await call_next(request)

# ── CORS ──────────────────────────────────────────────────────────────────────
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).rstrip("/") for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
```

**Fix — `start-backend.bat`:** Pass `--limit-max-requests` and large body limit to uvicorn.

```bat
:: backend/start-backend.bat  ← replace with this
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload --h11-max-incomplete-event-size 536870912
```

> `--h11-max-incomplete-event-size 536870912` = 512 MB, large enough for the 500 MB cap.

---

### Root Cause (Frontend)

`api.ts → uploadDocument` uses plain `axios.post` with no:
- Upload progress feedback
- Timeout override (axios default 0 = no timeout, but proxy/network can cut it)
- File size validation before sending

**Fix — `frontend/src/lib/api.ts`:** Add pre-flight size check + progress-friendly config.

```typescript
// Replace the uploadDocument function in api.ts

const MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024; // 500 MB — must match backend

export async function uploadDocument(
  file: File,
  notebookId?: string,
  onProgress?: (percent: number) => void
): Promise<Document> {
  // ── Pre-flight size check ───────────────────────────────────────────────
  if (file.size > MAX_FILE_SIZE_BYTES) {
    throw new Error(
      `File "${file.name}" is ${(file.size / (1024 * 1024)).toFixed(1)} MB, ` +
      `which exceeds the 500 MB limit.`
    );
  }

  let nId = notebookId;
  if (!nId || nId === "default_workspace_id") {
    const wId = await ensureWorkspace();
    nId = await ensureNotebook(wId);
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("notebook_id", nId);

  const response = await apiClient.post<any>("/documents/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 0, // No timeout — large files take time
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(percent);
      }
    },
  });

  const doc = response.data;
  return {
    id: doc.id,
    file_name: doc.title,
    status: doc.status,
    notebook_id: doc.notebook_id,
    file_path: doc.file_path,
    file_type: doc.file_type,
    created_at: doc.created_at,
  };
}
```

**Fix — `frontend/src/app/(dashboard)/workspace/page.tsx`:** Show upload progress %.

In the state section, add:
```typescript
const [uploadProgress, setUploadProgress] = useState<number | null>(null);
```

Update the `uploadMutation`:
```typescript
const uploadMutation = useMutation({
  mutationFn: (file: File) =>
    uploadDocument(file, selectedNotebookId || undefined, (pct) =>
      setUploadProgress(pct)
    ),
  onSuccess: () => {
    setUploadError(null);
    setUploadProgress(null);
    queryClient.invalidateQueries({ queryKey: ['documents', selectedNotebookId] });
  },
  onError: (error: any) => {
    setUploadProgress(null);
    const msg =
      error?.response?.data?.detail ||
      error?.message ||
      "Upload failed. Check that the backend is running and MinIO is accessible.";
    setUploadError(msg);
    queryClient.invalidateQueries({ queryKey: ['documents', selectedNotebookId] });
  },
});
```

In the upload zone JSX, replace the spinner text with:
```tsx
<span className="text-xs text-muted-foreground block">
  {uploadMutation.isPending
    ? uploadProgress !== null
      ? `Uploading… ${uploadProgress}%`
      : "Uploading…"
    : "Drop files or click to add"}
</span>
{/* Progress bar */}
{uploadMutation.isPending && uploadProgress !== null && (
  <div className="w-full h-1 bg-white/10 rounded-full mt-2 overflow-hidden">
    <div
      className="h-full bg-primary transition-all duration-200 rounded-full"
      style={{ width: `${uploadProgress}%` }}
    />
  </div>
)}
```

---

## Issue 2: OCR Not Working

### Root Cause

The OCR code in `ingestion.py` is already written correctly. The issue is that:

1. **Tesseract binary is not installed** on the Windows machine.
2. **`pytesseract` doesn't know where Tesseract is** unless you configure the path.
3. **`poppler` is not installed**, which `pdf2image` requires.

### Fix — Install Tesseract on Windows

1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - Pick `tesseract-ocr-w64-setup-5.x.x.exe` (64-bit)
2. Install to default path: `C:\Program Files\Tesseract-OCR\`
3. During install, select **"Additional language data"** → tick **English** (already default)

### Fix — Install Poppler on Windows

1. Download from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\poppler\`
3. Add `C:\poppler\Library\bin` to your system PATH

### Fix — Tell pytesseract where Tesseract is

Add this to the top of `backend/app/rag/ingestion.py`, after the imports:

```python
# ── Tesseract path for Windows ────────────────────────────────────────────────
import pytesseract
import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Fix — Pass `poppler_path` to `pdf2image` on Windows

Update the `extract_text_from_pdf_ocr` function in `ingestion.py`:

```python
def extract_text_from_pdf_ocr(file_bytes: bytes) -> str:
    """OCR fallback for scanned/image-based PDFs using pdf2image + pytesseract."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        import platform

        # Windows: tell pdf2image where poppler is
        poppler_path = None
        if platform.system() == "Windows":
            poppler_path = r"C:\poppler\Library\bin"

        pages = convert_from_bytes(file_bytes, dpi=200, poppler_path=poppler_path)
        text_parts = []
        for i, page_img in enumerate(pages):
            page_text = pytesseract.image_to_string(page_img, lang="eng")
            text_parts.append(f"--- Page {i+1} ---\n{page_text}")
            print(f"[OCR] Page {i+1}/{len(pages)} done")
        return "\n\n".join(text_parts)

    except ImportError:
        raise ValueError(
            "OCR requires: pip install pytesseract pdf2image Pillow "
            "and Tesseract installed at C:\\Program Files\\Tesseract-OCR\\"
        )
    except Exception as e:
        raise ValueError(f"OCR failed: {e}")
```

### Fix — Add OCR support for image files (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`)

Currently the backend ignores image files — they fall through to `file_bytes.decode()` which produces garbage. Add explicit image OCR support in `process_document`:

In `ingestion.py`, add this constant near the top:
```python
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp", "gif", "webp"}
```

In the `process_document` function, inside the `else` block where files are read from MinIO, add a branch for images **before** the final `else`:

```python
elif file_type_lower in IMAGE_EXTENSIONS:
    text = extract_text_from_image(file_bytes)
```

And add this new function to `ingestion.py`:

```python
def extract_text_from_image(file_bytes: bytes) -> str:
    """OCR an uploaded image file directly."""
    try:
        import pytesseract
        from PIL import Image
        import platform
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img, lang="eng")
    except Exception as e:
        raise ValueError(f"Image OCR failed: {e}")
```

Also update the frontend `accept` attribute in `workspace/page.tsx` to include image types:
```tsx
accept=".pdf,.doc,.docx,.ppt,.pptx,.txt,.md,.mp3,.wav,.m4a,.ogg,.flac,.webm,.aac,.png,.jpg,.jpeg,.tiff,.bmp"
```

---

## Quick Checklist

| Step | What to do |
|------|-----------|
| 1 | Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki |
| 2 | Install Poppler, add `C:\poppler\Library\bin` to PATH |
| 3 | Edit `backend/app/main.py` — add the size-limit middleware |
| 4 | Edit `backend/start-backend.bat` — add `--h11-max-incomplete-event-size` |
| 5 | Edit `backend/app/rag/ingestion.py` — add pytesseract path + poppler_path + image OCR |
| 6 | Edit `frontend/src/lib/api.ts` — add pre-flight size check + onUploadProgress |
| 7 | Edit `frontend/.../workspace/page.tsx` — add progress state + progress bar UI |
| 8 | Restart backend with `start-backend.bat` |
