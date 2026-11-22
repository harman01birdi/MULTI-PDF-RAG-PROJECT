from langchain_core.prompts import PromptTemplate

QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant answering questions based only on the provided PDF context.
Each context passage begins with a [Source: file, page N] label.
If the answer isn't in the context, say you don't know. Don't make things up.

Context:
{context}

Question: {question}

Answer clearly. After each fact, cite its source like (file.pdf, p.N):""",
    input_variables=["context", "question"],
)

DOC_PROMPT = PromptTemplate(
    template="[Source: {source_file}, page {page}]\n{page_content}",
    input_variables=["page_content", "source_file", "page"],
)