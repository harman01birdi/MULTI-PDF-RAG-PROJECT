from fastapi import Request


def get_vectorstore(request: Request):
    return request.app.state.vectorstore


def get_llm(request: Request):
    return request.app.state.llm


def get_redis(request: Request):
    return request.app.state.redis