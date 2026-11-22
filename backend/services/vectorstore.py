# the api and the worker both build their vectore store here
import time
import chromadb 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_HOST, CHROMA_DIR, CHROMA_PORT, EMBEDDING_MODEL

COLLECTION="pdfs"

def build_embedding()-> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _connect_chroma(retries: int = 30, delay: float = 2.0):
    # containers start in no particular order , so wait for the Chroma server
    last_error= None
    for _ in range(retries):
        try:
            client= chromadb.HttpClient(host= CHROMA_HOST, port= CHROMA_PORT)
            client.heartbeat()
            return client
        except Exception as exc:
            last_error=exc
            time.sleep(delay)
    raise RuntimeError(f"could not reach chroma at {CHROMA_HOST}: {CHROMA_PORT}") from last_error


def build_vectorstore(embeddings) -> Chroma:
    if CHROMA_HOST:
        return Chroma(
            client=_connect_chroma(),
            collection_name=COLLECTION,
            embedding_function= embeddings,
        )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function= embeddings,
        persist_directory=str(CHROMA_DIR),
    )