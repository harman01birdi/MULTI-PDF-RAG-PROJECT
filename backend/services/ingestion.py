import hashlib
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import UPLOAD_DIR

def ingest_pdf(vectorstore, data:bytes, filename: str) ->dict | None:
    doc_id=hashlib.sha256(data).hexdigest()[:16]

    dest=UPLOAD_DIR/ f"{doc_id}.pdf"
    dest.write_bytes(data)

    pages=PyPDFLoader(str(dest)).load()
    uploaded_at= datetime.now(timezone.utc).isoformat()
    for page in pages:
        page.metadata["source_file"]=filename
        page.metadata["doc_id"]=doc_id
        page.metadata["page"]=page.metadata.get("page", 0)+1
        page.metadata["page_count"]= len(pages)
        page.metadata["size_bytes"]= len(data)
        page.metadata["uploaded_at"]= uploaded_at
    
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks=splitter.split_documents(pages)
    if not chunks:
        dest.unlink(missing_ok=True)
        return None

    ids=[f"{doc_id}-{i}" for i in range(len(chunks))]
    vectorstore.add_documents(chunks, ids=ids)

    return {"doc_id": doc_id, "file": filename, "chunks": len(chunks)}