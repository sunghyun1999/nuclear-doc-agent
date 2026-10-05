"""Cross-encoder re-ranking for improved retrieval quality."""

from sentence_transformers import CrossEncoder

from retriever import RetrievedChunk

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    """Re-rank retrieved chunks using cross-encoder for better relevance."""
    if not chunks:
        return chunks

    reranker = get_reranker()
    pairs = [[query, chunk.text] for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score = round(float(score), 4)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[:top_k]
