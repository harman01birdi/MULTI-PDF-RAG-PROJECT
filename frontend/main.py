import json
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
JOB_PATH = os.getenv("JOB_PATH", "/jobs").rstrip("/") # use "/jobs" after you apply the Stage 2 backend
JOB_TIMEOUT_S = 600
POLL_INTERVAL_S = 1.5
ALL_DOCS = "__all__"

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📚", layout="wide")

# ---- keep your existing st.markdown("""<style> ... </style>""", unsafe_allow_html=True) block here ----

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


for key, default in {
    "messages": [],
    "uploaded_sigs": [],
    "failed_sigs" : [],
    "upload_summary": [],
    "last_doc_id": None,
}.items():
    st.session_state.setdefault(key, default)


# ---------- helpers ----------
def api_error(response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text


def fetch_documents() -> tuple[list[dict], str | None]:
    try:
        r = requests.get(f"{API_URL}/documents", timeout=10)
        r.raise_for_status()    
        return r.json(), None
    except requests.RequestException as exc:
        return [], str(exc)


def wait_for_job(job_id: str) -> dict:
    deadline = time.time() + JOB_TIMEOUT_S
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_URL}{JOB_PATH}/{job_id}", timeout=10)
            if r.status_code == 404:
                detail = r.json().get("detail", "Not Found")
                return {"status": "failed", "error": f"{detail} (GET {JOB_PATH}/{job_id})"}
            r.raise_for_status()
            job = r.json()
            if job["status"] in ("done", "failed"):
                return job
        except requests.RequestException:
            pass   # transient network error: keep polling
        time.sleep(POLL_INTERVAL_S)
    return {"status": "failed", "error": "Timed out waiting for processing."}


def ingest(pdfs) -> None:
    files = [("files", (p.name, p.getvalue(), "application/pdf")) for p in pdfs]
    try:
        resp = requests.post(f"{API_URL}/upload", files=files, timeout=60)
    except requests.RequestException as exc:
        st.session_state.upload_summary = [("error", f"Could not reach backend: {exc}")]
        return False
    if resp.status_code != 202:
        st.session_state.upload_summary = [("error", api_error(resp))]
        return False

    jobs = resp.json()["jobs"]
    summary = []
    with st.status(f"Processing {len(jobs)} PDF(s)...", expanded=True) as status:
        for job in jobs:
            st.write(f"⏳ {job['file']}")
            final = wait_for_job(job["job_id"])
            if final["status"] == "done":
                result = final["result"]
                summary.append(("ok", f"{job['file']}: {result['chunks']} chunks indexed ✅"))
            else:
                summary.append(("error", f"{job['file']}: {final.get('error', 'processing failed')}"))
        status.update(label="Processing finished", state="complete")
    st.session_state.upload_summary = summary
    # Default to "All documents" after upload so questions search across the full corpus
    st.session_state.last_doc_id = ALL_DOCS

    return not any(kind == "error" for kind, _ in summary)


def stream_answer(question: str, doc_id: str | None, sources_out: list):
    body = {"question": question, "doc_id": doc_id}
    try:
        with requests.post(
            f"{API_URL}/ask/stream", json=body, stream=True, timeout=(5, 120)
        ) as r:
            if r.status_code != 200:
                yield f"⚠️ {api_error(r)}"
                return
            event = None
            for raw in r.iter_lines(chunk_size=None):
                if not raw:
                    continue
                line = raw.decode("utf-8")
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event == "token":
                        yield data["text"]
                    elif event == "sources":
                        sources_out.extend(data["sources"])
                    elif event == "error":
                        yield "\n\n⚠️ Generation failed."
    except requests.RequestException as exc:
        yield f"⚠️ Could not reach backend: {exc}"



def show_sources(sources: list, answer: str = "") -> None:
    negative_phrases = ["i don't know", "do not know", "cannot find", "not mentioned", "not provided", "does not contain"]
    if answer and any(p in answer.lower() for p in negative_phrases):
        return
    if sources:
        with st.expander("📌 Sources"):
            for src in sources:
                st.caption(f"**{src['source_file']}**, page {src['page']}")



def format_size(n) -> str:
    if not n:
        return "size unknown"
    return f"{n / 1024:.1f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"



def reindex(doc_id: str) -> None:
    try:
        r = requests.post(f"{API_URL}/documents/{doc_id}/reindex", timeout=30)
    except requests.RequestException as exc:
        st.session_state.upload_summary = [("error", f"Could not reach backend: {exc}")]
        return
    if r.status_code != 202:
        st.session_state.upload_summary = [("error", api_error(r))]
        return
    job = r.json()
    with st.spinner(f"Re-indexing {job['file']}..."):
        final = wait_for_job(job["job_id"])
    if final["status"] == "done":
        res = final["result"]
        msg = f"{job['file']}: re-indexed, {res['chunks']} chunks, {res['removed_stale_chunks']} stale removed ✅"
        st.session_state.upload_summary = [("ok", msg)]
    else:
        st.session_state.upload_summary = [("error", f"{job['file']}: {final.get('error', 'failed')}")]



def delete_document(doc_id: str) -> None:
    try:
        r = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=30)
    except requests.RequestException as exc:
        st.session_state.upload_summary = [("error", f"Could not reach backend: {exc}")]
        return
    if r.status_code != 200:
        st.session_state.upload_summary = [("error", api_error(r))]
        return
    st.session_state.upload_summary = [("ok", f"Deleted {r.json()['deleted_chunks']} chunks.")]
    st.session_state.last_doc_id = None
    st.session_state.messages = []
    st.session_state.uploaded_sigs = []   # lets you upload the same file again


# ---------- sidebar ----------
docs, docs_error = fetch_documents()

with st.sidebar:
    st.title("🤖 PDF RAG Assistant")
    st.write("AI-powered document assistant")
    st.divider()

    st.subheader("📄 Document")
    options = [ALL_DOCS] + [d["doc_id"] for d in docs]
    labels = {ALL_DOCS: "All documents"}
    labels.update({d["doc_id"]: f"{d['file']} ({d['doc_id'][:6]})" for d in docs})
    last = st.session_state.get("last_doc_id", ALL_DOCS)
    default_index = options.index(last) if last in options else 0
    choice = st.selectbox(
        "Ask about", options, index=default_index, format_func=lambda x: labels[x]
    )
    st.session_state.last_doc_id = choice
    selected_doc_id = None if choice == ALL_DOCS else choice
    scope_chunks = sum(
        d["chunks"] for d in docs if selected_doc_id in (None, d["doc_id"])
    )
    st.caption(f"{len(docs)} document(s) indexed, {scope_chunks} chunks in scope")
    if selected_doc_id:
        detail = next(d for d in docs if d["doc_id"] == selected_doc_id)
        uploaded = (detail.get("uploaded_at") or "unknown")[:10]
        st.caption(f"{detail['pages']} pages · {format_size(detail.get('size_bytes'))} · uploaded {uploaded}")
        if st.button("🔄 Re-index"):
            reindex(selected_doc_id)
            st.rerun()
        confirm = st.checkbox("Confirm deletion")
        if st.button("🗑️ Delete document", disabled=not confirm):
            delete_document(selected_doc_id)
            st.rerun()
    st.divider()

    st.subheader("⚙️ System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success("🟢 Backend Online")
        if health.get("ready"):
            st.success("🟢 Retriever Ready")
        else:
            st.warning("🟡 Retriever Idle")
        if not health.get("has_api_key"):
            st.error("Missing GOOGLE_API_KEY in backend .env")
    except requests.RequestException:
        st.error("🔴 Backend Offline")
        st.caption(f"Start API at {API_URL}")

    if docs_error:
        st.error(f"Failed to fetch documents: {docs_error}")

# ---------- main ----------
st.title("📚 PDF RAG Assistant")
st.caption("Ask questions and get answers grounded in your uploaded PDF.")

pdfs = st.file_uploader("Upload your PDF", type=["pdf"], accept_multiple_files=True)
if pdfs:
    sigs = [f"{p.name}:{p.size}" for p in pdfs]
    if sigs != st.session_state.uploaded_sigs:
        retry = sigs == st.session_state.failed_sigs and st.button("🔄 Retry upload")
        if sigs != st.session_state.failed_sigs or retry:
            if ingest(pdfs):
                st.session_state.uploaded_sigs = sigs
                st.session_state.failed_sigs = []
                st.session_state.messages = []
            else:
                st.session_state.failed_sigs = sigs
            st.rerun()

for kind, text in st.session_state.upload_summary:
    (st.success if kind == "ok" else st.error)(text)

st.divider()

if selected_doc_id:
    doc_label = labels.get(selected_doc_id, "Selected document")
    col_info, col_btn = st.columns([4, 1])
    with col_info:
        st.info(f"🎯 Scope: **{doc_label}** ({scope_chunks} chunks)")
    with col_btn:
        if st.button("🌐 Search All"):
            st.session_state.last_doc_id = ALL_DOCS
            st.rerun()
elif docs:
    st.info(f"🌐 Scope: **All documents** ({len(docs)} documents, {scope_chunks} chunks in scope)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []), msg["content"])

if question := st.chat_input("Ask a question about your PDF..."):
    if not docs:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            sources: list = []
            answer = st.write_stream(stream_answer(question, selected_doc_id, sources))
            show_sources(sources, answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

# ---- keep your existing footer st.markdown("""<div class="footer"> ... </div>""", unsafe_allow_html=True) here ----



st.markdown("""
<div class="footer">
    AI RAG Assistant • Built by Harmanpreet Birdi
</div>
""", unsafe_allow_html=True)
