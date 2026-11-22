from pathlib import Path

from services .ingestion import ingest_pdf
from services.vectorstore import build_embedding, build_vectorstore
from workers.celery_app import celery_app

from config import UPLOAD_DIR
from services.documents import prune_stale_chunks

_vectorstore=None


def get_vectorstore():
    # created lazily, once per worker process
    global _vectorstore
    if _vectorstore is None:
        _vectorstore=build_vectorstore(build_embedding())
    return _vectorstore



@celery_app.task(name="ingest_pdf")
def ingest_pdf_task(file_path: str, filename: str)-> dict:
    path=Path(file_path)
    try:
        result= ingest_pdf(get_vectorstore(), path.read_bytes(), filename)
    finally:
        path.unlink(missing_ok=True)

    if result is None:
        raise ValueError("No extractable text found in the PDF. ")
    
    return result


@celery_app.task(name="reindex_pdf")
def reindex_pdf_task(doc_id: str, filename: str) -> dict:
    path= UPLOAD_DIR / f"{doc_id}.pdf"
    if not path.exists():
        raise ValueError("the original pdf is no longer stored. ")


    vectorstore= get_vectorstore()
    # ingest pdf, then prune: the document never disappears if processing fails halfway

    result= ingest_pdf(vectorstore, path.read_bytes(), filename)
    if result is None:
        raise ValueError("No extractable text found in the PDF")
    result["removed_stale_chunks"]= prune_stale_chunks(vectorstore, doc_id, result["chunks"])
    return result



    
    