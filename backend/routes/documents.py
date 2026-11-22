import logging 
from pathlib import Path
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from celery.result import AsyncResult
from fastapi import Path as PathParam


from config import MAX_UPLOAD_BYTES, INCOMING_DIR, UPLOAD_DIR  
from deps import get_vectorstore, get_redis
from workers.celery_app import celery_app
from services import documents as doc_services

router=APIRouter()
logger=logging.getLogger("uvicorn.error")


JOB_TTL_SECONDS=24 * 3600
STATE_MAP={
    "PENDING": "queued",
    "RECEIVED": "queued",
    "STARTED": "processing",
    "SUCCESS": "done",
    "FAILURE": "failed"
}
# doc ids are 16 hex characters (a content hash). this validation matters:
# the id ends up in a file path, so "../../x" must never get through. 
DOC_ID= Annotated[str, PathParam(pattern=r"^[0-9a-f]{16}$")]

def safe_display_name(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name



def enqueue_ingestion(redis_client, data: bytes, name: str) -> str:
    job_id=uuid.uuid4().hex
    tmp_path=INCOMING_DIR/ f"{job_id}.pdf"
    tmp_path.write_bytes(data)

    try:
        #remember the job, so unknown ids can return 404 (celery reports pending for any id.)
        redis_client.setex(f"job:{job_id}", JOB_TTL_SECONDS, name)
        celery_app.send_task("ingest_pdf", args=[str(tmp_path), name], task_id=job_id)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return job_id


def enqueue_reindex(redis_client, doc_id: str, filename: str) -> str:
    job_id= uuid.uuid4().hex
    redis_client.setex(f"job:{job_id}", JOB_TTL_SECONDS, filename)
    celery_app.send_task("reindex_pdf", args=[doc_id, filename], task_id=job_id)
    return job_id



@router.post("/upload", status_code=202)
async def upload(
    files: list[UploadFile] = File(...),
    redis_client=Depends(get_redis),
):
    pdfs = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    if not pdfs:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF.")

    validated = []
    for pdf in pdfs:
        name = safe_display_name(pdf.filename)

        data = await pdf.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{name} is larger than 20 MB.")
        if not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail=f"{name} is not a valid PDF.")
        validated.append((name, data))

    jobs=[]
    for name, data in validated:
        try:
            job_id= await run_in_threadpool(enqueue_ingestion, redis_client, data, name)
        except Exception:
            logger.exception("could not enqueue %s", name)
            raise HTTPException(status_code=503, detail=f"job queue unavailable. try again shortly")
        jobs.append({"job_id":job_id, "file":name, "status":"queued"})

    return {"message": f"{len(jobs)} PDF(s) queued for processing.", "jobs": jobs}




@router.get("/jobs/{job_id}")
def job_status(job_id: str, redis_client=Depends(get_redis)):
    filename= redis_client.get(f"job:{job_id}")
    if filename is None:
        raise HTTPException(status_code= 404, detail="job not found")

    task= AsyncResult(job_id, app=celery_app)
    status= STATE_MAP.get(task.state, "processing")

    response= {"job_id": job_id, "file": filename, "status":status, "result": None}
    if status== "done":
        response["result"]= task.result
    elif status == "failed":
        task_error = task.result
        logger.error(
            "Celery job %s failed: %s",
            job_id,
            task_error,
            exc_info=False,
        )
        response["error"] = (
            str(task_error)
            if isinstance(task_error, ValueError)
            else "Could not process the PDF. It may be corrupted or encrypted."
        )
    return response



@router.get("/documents")
def list_documents(vectorstore=Depends(get_vectorstore)):
    return doc_services.list_documents(vectorstore)


@router.get("/documents/{doc_id}")
def get_document(doc_id: DOC_ID, vectorstore=Depends(get_vectorstore)):
    doc= doc_services.get_document(vectorstore, doc_id)
    if doc is None:
        raise HTTPException(status_code= 404, detail="document not found. ")
    doc["pdf_stored"] = (UPLOAD_DIR / f"{doc_id}.pdf").exists()
    return doc



@router.delete("/documents/{doc_id}")
def delete_document(doc_id: DOC_ID, vectorstore=Depends(get_vectorstore)):
    deleted = doc_services.delete_document(vectorstore, doc_id)
    if deleted==0:
        raise HTTPException(status_code=404, detail= "document not found. ")
    return {"doc_id": doc_id, "deleted_chunks": deleted}


@router.post("/documents/{doc_id}/reindex", status_code= 202)
def reindex_document(
    doc_id: DOC_ID,
    vectorstore= Depends(get_vectorstore),
    redis_client= Depends(get_redis)
):
    doc= doc_services.get_document(vectorstore, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found. ")
    if not (UPLOAD_DIR / f"{doc_id}.pdf").exists():
        raise HTTPException(status_code= 409, detail="the original pdf is no longer stored. upload it again to reindex. ")

    try:
        job_id= enqueue_reindex(redis_client, doc_id, doc["file"])
    except Exception:
        logger.exception("could not enqueue reindex for %s", doc_id)
        raise HTTPException(status_code=503, detail="job queue unavailable. ")
    return {"job_id": job_id, "file": doc["file"], "status": "queued"}


