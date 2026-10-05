"""Retrieval: query → embedding → ChromaDB search → ranked results."""

from dataclasses import dataclass

from config import settings
from embedder import embed_query
from ingestion import get_collection


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float
    strategy: str
    clauses: str = ""


def retrieve(
    query: str,
    top_k: int | None = None,
    collection_name: str = "nuclear_docs",
) -> list[RetrievedChunk]:
    """Retrieve most relevant chunks for a query."""
    k = top_k or settings.top_k
    collection = get_collection(collection_name)

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = 1 - distance  # cosine distance → similarity

        chunks.append(
            RetrievedChunk(
                text=results["documents"][0][i],
                source=metadata.get("source", "unknown"),
                page=metadata.get("page", 0),
                score=round(similarity, 4),
                strategy=metadata.get("strategy", "unknown"),
                clauses=metadata.get("clauses", ""),
            )
        )

    return chunks
