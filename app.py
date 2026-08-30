import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()  

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.3
)
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
h1 {
    color: #1e293b;
    font-size: 42px;
    font-weight: 700;
}
h2, h3 {
    color: #334155;
}
.stButton > button {
    border-radius: 10px;
    border: none;
    background-color: #2563eb;
    color: white;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #1d4ed8;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #94a3b8;
    border-radius: 12px;
    padding: 15px;
    background-color: white;
}
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #1e293b;
    color: #f8fafc;
    text-align: center;
    padding: 8px;
    font-size: 14px;
    font-weight: 500;
    z-index: 999999;
    letter-spacing: 0.3px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🤖 PDF RAG Assistant")
    st.write("AI-powered document assistant")
    st.divider()
    st.subheader("📄 Document")
    st.write("Upload a PDF and ask questions from it.")
    st.divider()
    st.subheader("⚙️ System Status")
    embed_status = st.empty()
    vector_status = st.empty()
    retriever_status = st.empty()
    embed_status.warning("🟡 Embedding Idle")
    vector_status.warning("🟡 Vector Store Idle")
    retriever_status.warning("🟡 Retriever Idle")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

st.title("📚 PDF RAG Assistant")
st.caption("Ask questions and get answers grounded in your uploaded PDF.")

pdfs = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if pdfs:
    documents = []
    for pdf in pdfs:
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())

        loader = PyPDFLoader(pdf.name)
        pdf_documents = loader.load()

        for doc in pdf_documents:
            doc.metadata["source_file"] = pdf.name

        documents.extend(pdf_documents)

    st.success(f"{len(pdfs)} PDF(s) uploaded successfully! ✅")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    st.write("Total Chunks:", len(chunks))

    # --- Embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    embed_status.success("🟢 Embedding Ready")

    # --- Vector Store ---
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vector_status.success("🟢 Vector Store Ready")

    # --- Retriever ---
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    retriever_status.success("🟢 Retriever Ready")

    # --- Custom Prompt ---
    prompt_template = """You are a helpful assistant answering questions based only on the provided PDF context.
If the answer isn't in the context, say you don't know — don't make things up.

Context:
{context}

Question: {question}

Answer clearly and cite the source file/page when possible:"""

    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # --- RetrievalQA Chain ---
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

st.divider()

# --- Chat History Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
if question := st.chat_input("Ask a question about your PDF..."):
    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"]
                sources = result.get("source_documents", [])

                st.write(answer)

                if sources:
                    with st.expander("📌 Sources"):
                        for s in sources:
                            page = s.metadata.get("page", "N/A")
                            src_file = s.metadata.get("source_file", "Unknown")
                            st.caption(f"**{src_file}** — page {page}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Footer ---
st.markdown("""
<div class="footer">
    AI RAG Assistant • Built by Harmanpreet Birdi
</div>
""", unsafe_allow_html=True)
