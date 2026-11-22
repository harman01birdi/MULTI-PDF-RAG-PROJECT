from config import UPLOAD_DIR


BATCH_SIZE = 500 # keep delete calls small



def _summarize(doc_id: str, metadatas: list[dict]) -> dict:
    first= metadatas[0]
    pages= first.get("page_count") or max ((m.get("page", 0) for m in metadatas), default= 0)
    return {
        "doc_id": doc_id,
        "file": first.get("source_file", "unknown"),
        "chunks" : len(metadatas),
        "pages": pages,
        "size_bytes": first.get("size_bytes"),
        "uploaded_at": first.get("uploaded_at"),
    }
    

def list_documents(vectorstore)  -> list[dict]:
    data=vectorstore.get(include= ["metadatas"])
    grouped: dict[str, list[dict]] = {}
    for meta in (data.get("metadatas") or []):
        if not meta or "doc_id" not in meta:
            continue
        grouped.setdefault(meta["doc_id"], []).append(meta)

    return [_summarize(doc_id, metas) for doc_id, metas in grouped.items()]


def get_document(vectorstore, doc_id: str) -> dict | None:
    data= vectorstore.get(where={"doc_id": doc_id}, include=["metadatas"])
    if not data["ids"]:
        return None
    return _summarize(doc_id, data["metadatas"])


def _delete_ids(vectorstore, ids: list[str]) -> None:
    for i in range(0, len(ids), BATCH_SIZE):
        vectorstore.delete(ids=ids[i:i + BATCH_SIZE])


def delete_document(vectorstore, doc_id: str) -> int:
    ids= vectorstore.get(where={"doc_id": doc_id}, include=["metadatas"])["ids"]
    _delete_ids(vectorstore, ids)
    (UPLOAD_DIR / f"{doc_id}.pdf").unlink(missing_ok=True)
    return len(ids)


def prune_stale_chunks(vectorstore, doc_id: str, keep: int) -> int:
    # after re-indexing, drop chunks whose ids are no longer produced
    valid= {f"{doc_id}-{i}" for i in range(keep)}
    ids= vectorstore.get(where={"doc_id": doc_id}, include=["metadatas"])["ids"]
    stale= [i for i in ids if i not in valid]
    _delete_ids(vectorstore, stale)
    return len(stale)