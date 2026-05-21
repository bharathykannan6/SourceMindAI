import io
import uuid
import platform
import psycopg2
from typing import List
import fitz  # PyMuPDF
import docx
from pptx import Presentation
import httpx
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.storage import get_minio_client

# ── Tesseract path for Windows ────────────────────────────────────────────────
try:
    import pytesseract
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    pass  # pytesseract is optional; only needed for scanned PDFs / image files

# ── Load model ONCE at startup, reuse across all threads ─────────────────────
print("[Ingestion] Loading embedding model... (one-time startup cost)")
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("[Ingestion] Embedding model loaded.")

qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

VECTOR_COLLECTION = "sourcemind_documents"
AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac", "webm", "aac"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "tif", "bmp", "gif", "webp"}


def get_sync_db_conn():
    """Get a synchronous psycopg2 connection for use inside background threads."""
    return psycopg2.connect(
        host=settings.POSTGRES_SERVER,
        port=settings.POSTGRES_PORT,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        dbname=settings.POSTGRES_DB
    )


def update_document_status_sync(document_id: str, new_status: str, error_message: str = None):
    """Update document status using a plain sync DB connection — safe to call from any thread."""
    try:
        conn = get_sync_db_conn()
        cur = conn.cursor()
        if error_message:
            cur.execute(
                "UPDATE documents SET status = %s, error_message = %s WHERE id = %s",
                (new_status, error_message[:500], document_id)  # cap at 500 chars
            )
        else:
            cur.execute(
                "UPDATE documents SET status = %s WHERE id = %s",
                (new_status, document_id)
            )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB Update Error] document_id={document_id} status={new_status} error={e}")


def init_qdrant():
    """Ensure the Qdrant collection exists."""
    collections = qdrant_client.get_collections().collections
    if not any(c.name == VECTOR_COLLECTION for c in collections):
        qdrant_client.create_collection(
            collection_name=VECTOR_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF. Falls back to OCR if the PDF is image-based (scanned)."""
    # First try PyMuPDF (fast, works for text-based PDFs)
    text = ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text() + "\n\n"

    # If text is too sparse, the PDF is likely scanned — fall back to OCR
    if len(text.strip()) < 100:
        print("[Ingestion] PDF has little/no selectable text — attempting OCR fallback")
        text = extract_text_from_pdf_ocr(file_bytes)

    return text


def extract_text_from_pdf_ocr(file_bytes: bytes) -> str:
    """OCR fallback for scanned/image-based PDFs using pdf2image + pytesseract."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        # Windows: tell pdf2image where poppler is
        poppler_path = None
        if platform.system() == "Windows":
            poppler_path = r"C:\poppler\poppler-24.02.0\Library\bin"
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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


def extract_text_from_image(file_bytes: bytes) -> str:
    """OCR an uploaded image file directly using pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img, lang="eng")
    except ImportError:
        raise ValueError(
            "Image OCR requires: pip install pytesseract Pillow "
            "and Tesseract installed at C:\\Program Files\\Tesseract-OCR\\"
        )
    except Exception as e:
        raise ValueError(f"Image OCR failed: {e}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                full_text.append(cell.text)
    return "\n".join(full_text)


def extract_text_from_pptx(file_bytes: bytes) -> str:
    prs = Presentation(io.BytesIO(file_bytes))
    full_text = []
    for i, slide in enumerate(prs.slides):
        full_text.append(f"--- Slide {i+1} ---")
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                full_text.append(shape.text)
    return "\n".join(full_text)


def extract_text_from_url(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    with httpx.Client(headers=headers, timeout=15.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "head", "nav", "footer", "iframe", "aside"]):
        element.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(phrase for phrase in lines if phrase)


def extract_text_from_youtube(url: str) -> str:
    video_id = None
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    return "\n".join(entry["text"] for entry in transcript_list)


def extract_text_from_audio(file_bytes: bytes, file_ext: str) -> str:
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured")
    client = Groq(api_key=settings.GROQ_API_KEY)
    filename = f"audio.{file_ext}" if file_ext else "audio.mp3"
    audio_file = io.BytesIO(file_bytes)
    audio_file.name = filename
    return client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        response_format="text"
    )


def process_document(document_id: str, file_path: str, file_type: str, notebook_id: str):
    """
    Parse → chunk → embed → store in Qdrant.
    Updates document status to done or error directly via sync psycopg2.
    """
    print(f"[Ingestion] Starting: document_id={document_id} file_type={file_type}")

    try:
        file_type_lower = file_type.lower()
        text = ""

        # ── 1. Extract text ───────────────────────────────────────────────────
        if file_type_lower in ["url", "youtube"]:
            if file_type_lower == "url":
                text = extract_text_from_url(file_path)
            else:
                text = extract_text_from_youtube(file_path)
        else:
            minio_client = get_minio_client()
            bucket_name = "sourcemind-documents"
            object_name = file_path.split(f"{bucket_name}/")[1]
            response = minio_client.get_object(bucket_name, object_name)
            try:
                file_bytes = response.read()
            finally:
                response.close()
                response.release_conn()

            if file_type_lower == "pdf":
                text = extract_text_from_pdf(file_bytes)
            elif file_type_lower in ["docx", "doc"]:
                text = extract_text_from_docx(file_bytes)
            elif file_type_lower in ["pptx", "ppt"]:
                text = extract_text_from_pptx(file_bytes)
            elif file_type_lower in AUDIO_EXTENSIONS:
                text = extract_text_from_audio(file_bytes, file_type_lower)
            elif file_type_lower in IMAGE_EXTENSIONS:
                text = extract_text_from_image(file_bytes)
            else:
                text = file_bytes.decode("utf-8", errors="ignore")

        if not text.strip():
            raise ValueError("Extracted text is empty")

        print(f"[Ingestion] Extracted {len(text)} chars from document_id={document_id}")

        # ── 2. Chunk ──────────────────────────────────────────────────────────
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        if not chunks:
            raise ValueError("No chunks generated from text")

        print(f"[Ingestion] Created {len(chunks)} chunks for document_id={document_id}")

        # ── 3. Embed (model already loaded at startup — fast) ─────────────────
        embeddings = embed_model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )

        print(f"[Ingestion] Embeddings done for document_id={document_id}")

        # ── 4. Store in Qdrant ────────────────────────────────────────────────
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "document_id": str(document_id),
                    "notebook_id": str(notebook_id),
                    "text": chunk_text,
                    "chunk_index": i
                }
            )
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings))
        ]

        init_qdrant()
        qdrant_client.upsert(collection_name=VECTOR_COLLECTION, points=points)

        print(f"[Ingestion] Stored {len(points)} vectors for document_id={document_id}")

        # ── 5. Mark as done ───────────────────────────────────────────────────
        update_document_status_sync(document_id, "done")
        print(f"[Ingestion] DONE: document_id={document_id}")

    except Exception as e:
        print(f"[Ingestion] ERROR: document_id={document_id} error={e}")
        update_document_status_sync(document_id, "error")
