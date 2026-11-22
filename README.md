# 📚 PDF RAG Assistant

An AI-powered document assistant that lets you upload PDFs and ask questions grounded in their content, using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- Upload one or multiple PDF documents
- Automatic text chunking and embedding
- Semantic search powered by a vector database
- Context-grounded answers (no hallucination — the model only answers from your document)
- Source citation for every answer (file name + page number)
- Clean, professional chat interface built with Streamlit

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **Backend/API:** FastAPI
- **LLM:** Google Gemini (via `langchain-google-genai`)
- **Embeddings:** HuggingFace Sentence Transformers
- **Vector Store:** ChromaDB
- **Framework:** LangChain

## ⚙️ Setup & Installation

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/pdf-rag-assistant.git
   cd pdf-rag-assistant
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Gemini API key
   - Copy `env.example` to `.env`
   - Add your API key from [Google AI Studio](https://aistudio.google.com)
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. Run the backend API
   ```bash
   cd backend
   uvicorn app:app --reload --port 8000
   ```

5. In a second terminal, run the frontend
   ```bash
   cd frontend
   streamlit run main.py
   ```

The UI talks to `http://127.0.0.1:8000`. Check the RAG pipeline first at `http://127.0.0.1:8000/docs` (`/upload` then `/ask`).

The original single-file app is still in `MAIN-app.py`.

## 📸 How It Works

1. Upload a PDF from the sidebar
2. The document is split into chunks and embedded into a vector store
3. Ask any question about the document in the chat box
4. Get an answer grounded in the PDF, with cited sources

## 📝 License

This project is open source and available for learning purposes.

---

Built by **Harmanpreet Birdi**
