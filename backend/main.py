from contextlib import asynccontextmanager

import redis 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI


from config import LLM_MODEL, REDIS_URL
from services.vectorstore import build_embedding, build_vectorstore
from routes import ask, documents, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm=ChatGoogleGenerativeAI(model=LLM_MODEL)
    app.state.vectorstore=build_vectorstore(build_embedding())
    app.state.redis=redis.Redis.from_url(REDIS_URL, decode_responses=True)
    yield

    app.state.redis.close()



app = FastAPI(title="PDF RAG Assistant API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(ask.router)
app.include_router(documents.router)

