# GovTech Policy & Exam Navigator

Offline-first Retrieval-Augmented Generation (RAG) assistant that parses government policy PDFs and GATE syllabus material, running locally with FAISS, HuggingFace embeddings, and Gemini for grounded answers. Built as a portfolio project by a Data Science & AI student at **IIIT Dharwad**.

## Features
- **Offline-first RAG**: Local FAISS vector store built from your PDFs (policies, brochures, exam syllabi).
- **Local embeddings**: Uses the offline `all-MiniLM-L6-v2` HuggingFace model from the `models/` folder—no outbound calls for embedding.
- **Gemini integration**: Streams answers via Google’s Gemini (flash variants) with context grounding to reduce hallucinations.
- **Secure by default**: Runs entirely on your machine; documents and vectors never leave local storage.
- **Streamlit UI**: Chat interface with citation-style context snippets.

## Architecture
1. **Ingestion**: `build_vector_db.py` scans `pdfs/`, extracts text with `pypdf`, chunks with `langchain-text-splitters`, and embeds using local MiniLM. Chunks + metadata are stored in `vector_store/` (FAISS).
2. **Retrieval**: At query time, the app loads FAISS and fetches the top-3 relevant chunks.
3. **Generation**: Retrieved context is passed to Gemini (`gemini-1.5-flash*` family with automatic fallbacks) via LangChain prompt to answer strictly from context.

## Local Setup
```bash
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt  # if you add one
```

Populate supporting data:
- Put PDFs into `pdfs/`.
- Place the offline embedding model at `models/all-MiniLM-L6-v2` (already structured for sentence-transformers).

Build the vector store:
```bash
.\.venv\Scripts\python build_vector_db.py --src ./pdfs --out ./vector_store --model-local-dir D:\ai_agent_policy\models\all-MiniLM-L6-v2
```

Run the app:
```bash
.\.venv\Scripts\python -m streamlit run app.py
```

## Project Structure
- `app.py` — Streamlit chat app (retrieval + Gemini).
- `build_vector_db.py` — PDF loader → chunker → embeddings → FAISS writer.
- `vector_store/` — Local FAISS index (generated; git-ignored).
- `models/` — Offline HuggingFace model (git-ignored).
- `pdfs/` — Local PDFs (git-ignored).

## Notes on Gemini Models
The app auto-tries `gemini-1.5-flash` variants then `gemini-1.0-pro-latest`. You can override the model name from the sidebar (or environment). API key is set in `app.py` and can be overridden via `GEMINI_API_KEY`.

## Portfolio Positioning
- Focuses on **secure, offline document handling** for sensitive government policy data.
- Demonstrates **RAG best practices**: local embeddings + FAISS + LLM with strict context prompts to curb hallucinations.
- Showcases **LLM tooling** (LangChain, Gemini) and practical data engineering for retrieval pipelines.

## Git Quickstart
Run these commands from the project root:
```bash
git init
git add .gitignore README.md app.py build_vector_db.py
git commit -m "Initial commit: offline-first GovTech RAG with FAISS, MiniLM embeddings, Gemini UI"
```

You can add other tracked files (e.g., requirements.txt) before committing as needed.
