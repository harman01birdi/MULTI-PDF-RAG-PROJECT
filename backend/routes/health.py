import os
from fastapi import APIRouter, Depends
from deps import get_vectorstore

router=APIRouter()

@router.get("/health")
def health(vectorstore= Depends(get_vectorstore)):
    document_count=vectorstore._collection.count()
    return{
        "status" : "ok",
        "ready" : True,
        "has_api_key" : bool(os.getenv("GOOGLE_API_KEY")),
        "document_count" : document_count
    }