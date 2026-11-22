from langchain_classic.chains import RetrievalQA

from prompts import DOC_PROMPT, QA_PROMPT


class DocumentNotFoundError(Exception):
    pass


def answer_question(vectorstore, llm, question: str, doc_id: str | None = None, k: int | None = None) -> dict:
    search_k = k if k is not None else (4 if doc_id else 6)
    search_kwargs={"k" : search_k}

    if doc_id:
        found= vectorstore.get(where={"doc_id": doc_id}, limit=1)
        if not found["ids"]:
            raise DocumentNotFoundError(doc_id)
        search_kwargs["filter"] = {"doc_id": doc_id}
    
    retriever= vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT, "document_prompt": DOC_PROMPT},
    )

    result=qa_chain.invoke({"query": question})

    seen=set()
    sources=[]
    for doc in result.get("source_documents", []):
        key=(doc.metadata.get("source_file", "unknown"), doc.metadata.get("page", "N/A"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source_file": key[0], "page":key[1]})

    return {"answer": result["result"], "sources": sources}



def retrieve(vectorstore, question: str, doc_id: str | None = None, k: int | None = None):
    search_k = k if k is not None else (4 if doc_id else 6)
    search_kwargs={"k": search_k}

    if doc_id:
        found=vectorstore.get(where={"doc_id": doc_id}, limit =1)
        if not found["ids"]:
            raise DocumentNotFoundError(doc_id)
        search_kwargs["filter"]={"doc_id": doc_id}
    
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    return retriever.invoke(question)


def build_prompt(docs, question: str) -> str:
    context="\n\n".join(
        DOC_PROMPT.format(
            page_content=d.page_content,
            source_file=d.metadata.get("source_file", "unknown"),
            page=d.metadata.get("page", "N/A")
        )
        for d in docs
    )
    return QA_PROMPT.format(
        context=context,
        question= question
    )


def unique_source(docs) -> list[dict]:
    seen=set()
    sources=[]

    for d in docs:
        key=(d.metadata.get("source_file","unknown"),d.metadata.get("page", "N/A"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source_file" : key[0], "page" : key[1]})
    return sources


def _chunk_text(chunk) -> str:
    content= chunk.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for part in content
    )



def stream_token(llm, prompt: str):
    for chunk in llm.stream(prompt):
        text=_chunk_text(chunk)
        if text:
            yield text