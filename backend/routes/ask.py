import json 
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from deps import get_llm, get_vectorstore
from schemas import AskRequest
from services.rag import DocumentNotFoundError, answer_question, build_prompt, retrieve, stream_token, unique_source    

router = APIRouter()
logger=logging.getLogger("uvicorn.error")


def sse(event: str, data: dict) -> str:
    #json.dumps keeps newlines inside tokens from breaking the SSE format
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/ask")
def ask(
    body: AskRequest,
    vectorstore=Depends(get_vectorstore),
    llm=Depends(get_llm),
):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if vectorstore._collection.count() == 0:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    try:
        return answer_question(vectorstore, llm, question, body.doc_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=400, detail="Document ID not found.")



@router.post("/ask/stream")
def ask_stream(
    body: AskRequest,
    vectorstore=Depends(get_vectorstore),
    llm=Depends(get_llm)
):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if vectorstore._collection.count() == 0:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    try:
        docs=retrieve(vectorstore, question, body.doc_id)
    except DocumentNotFoundError:
        raise HTTPException(status_code=400, detail="Document ID not found.")

    prompt=build_prompt(docs, question)
    sources=unique_source(docs)

    def event_stream():
        yield sse("sources", {"sources": sources})
        try:
            for token in stream_token(llm, prompt):
                yield sse("token", {"text": token})
            yield sse("done", {})
        except Exception:
            logger.exception("Streaming error")
            yield sse("error", {"detail": "Generation failed"})
    return StreamingResponse(
        event_stream(), 
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})