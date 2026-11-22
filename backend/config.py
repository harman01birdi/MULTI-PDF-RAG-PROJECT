import os
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR=Path(__file__).resolve().parent
ROOT_DIR=BACKEND_DIR.parent

load_dotenv(ROOT_DIR / ".env")

UPLOAD_DIR=Path(os.getenv("UPLOAD_DIR", BACKEND_DIR / "uploads"))
CHROMA_DIR=Path(os.getenv("CHROMA_DIR", BACKEND_DIR / "chroma_db"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


MAX_UPLOAD_BYTES=20 *1024 *1024



LLM_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



CHROMA_HOST=os.getenv("CHROMA_HOST")
CHROMA_PORT=int(os.getenv("CHROMA_PORT", "8000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")



INCOMING_DIR=UPLOAD_DIR/ "incoming"
INCOMING_DIR.mkdir(parents=True, exist_ok=True)