"""Build a local FAISS vector database from all PDFs in the workspace.

Usage:
    python build_vector_db.py --src . --out ./vector_store --chunk-size 800 --chunk-overlap 150

Dependencies: langchain, langchain-community, pypdf, sentence-transformers, faiss-cpu
"""

import argparse
import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def find_pdfs(root: Path) -> List[Path]:
    """Return a list of PDF files under the root (non-recursive)."""
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def load_documents(pdf_paths: List[Path]):
    """Load all PDFs into LangChain Document objects."""
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        documents.extend(docs)
    return documents


def chunk_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(documents)


def build_faiss_index(chunks, output_dir: Path, model_name: str, model_local_dir: str | None = None):
    # HuggingFaceEmbeddings downloads the model on first use unless a local cache path is provided.
    # Prefer the local path if provided to avoid network fetches.
    embed_kwargs = {"model_name": model_local_dir or model_name}
    embeddings = HuggingFaceEmbeddings(**embed_kwargs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector DB from PDFs.")
    parser.add_argument("--src", default=".", help="Directory containing PDFs (non-recursive).")
    parser.add_argument("--out", default="./vector_store", help="Output directory for the FAISS index.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for splitting text.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for splitting text.")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformers model name or local path.",
    )
    parser.add_argument(
        "--model-local-dir",
        default=None,
        help="Optional local directory containing the model to avoid downloads (passed as cache_folder).",
    )
    args = parser.parse_args()

    src_dir = Path(args.src).resolve()
    out_dir = Path(args.out).resolve()

    pdf_paths = find_pdfs(src_dir)
    if not pdf_paths:
        print(f"No PDFs found in {src_dir}. Nothing to index.")
        return

    print(f"Found {len(pdf_paths)} PDF(s). Loading...")
    documents = load_documents(pdf_paths)
    print(f"Loaded {len(documents)} documents (page-level). Splitting into chunks...")

    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"Created {len(chunks)} text chunks. Building embeddings and FAISS index (this may take a minute)...")

    try:
        build_faiss_index(chunks, out_dir, model_name=args.model, model_local_dir=args.model_local_dir)
    except Exception as exc:
        print("Failed to build embeddings/index.")
        print("Error:", exc)
        print(
            "Tip: If internet is blocked, download the model locally and rerun with "
            "--model-local-dir <path-to-model>. Default model: sentence-transformers/all-MiniLM-L6-v2"
        )
        return

    print(f"Done. FAISS index saved to: {out_dir}")
    print(f"Stats -> PDFs: {len(pdf_paths)}, pages: {len(documents)}, chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
