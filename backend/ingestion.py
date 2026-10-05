"""Document ingestion: PDF parsing → chunking → embedding → ChromaDB storage."""

import os

import chromadb
from pypdf import PdfReader

from chunker import Chunk, chunk_by_fixed_size, chunk_by_paragraph
from config import settings
from embedder import embed_texts

chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection(name: str = "nuclear_docs") -> chromadb.Collection:
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text per page from PDF. Returns [(page_num, text), ...]."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def ingest_pdf(
    pdf_path: str,
    strategy: str = "fixed_size",
    collection_name: str = "nuclear_docs",
) -> dict:
    """Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB."""
    filename = os.path.basename(pdf_path)
    pages = extract_text_from_pdf(pdf_path)

    all_chunks: list[Chunk] = []
    for page_num, text in pages:
        if strategy == "paragraph":
            chunks = chunk_by_paragraph(text, source=filename, page=page_num)
        else:
            chunks = chunk_by_fixed_size(
                text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                source=filename,
                page=page_num,
            )
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"status": "error", "message": "No text extracted from PDF"}

    # Batch embed locally
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = [c.text for c in all_chunks[i : i + batch_size]]
        all_embeddings.extend(embed_texts(batch_texts))

    # Store in ChromaDB
    collection = get_collection(collection_name)
    ids = [f"{filename}_{strategy}_{i}" for i in range(len(all_chunks))]
    collection.add(
        ids=ids,
        documents=[c.text for c in all_chunks],
        embeddings=all_embeddings,
        metadatas=[c.metadata for c in all_chunks],
    )

    return {
        "status": "success",
        "filename": filename,
        "strategy": strategy,
        "total_pages": len(pages),
        "total_chunks": len(all_chunks),
    }
