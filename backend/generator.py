"""Generation: context + query → Gemini → structured answer with citations."""

import time

import httpx

from config import settings
from retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a nuclear engineering document assistant specialized in SMR (Small Modular Reactor) safety regulations and standards (IAEA SSR-2/1, NRC 10 CFR, etc.).

Your role:
1. Answer questions based ONLY on the provided context documents.
2. Always cite your sources — include the document name, page number, and regulatory clause (e.g., Requirement 42, §50.46) for every claim.
3. If the context doesn't contain enough information to answer, say so explicitly. Do NOT hallucinate.
4. Use precise technical language appropriate for nuclear engineering professionals.
5. Structure your answers clearly with key points highlighted.

Citation format: [Source: document_name, p.XX, Requirement/§ if applicable]

Remember: In the nuclear domain, accuracy is paramount. It is always better to say "I don't have enough information" than to provide uncertain information."""

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"models/{settings.gemini_model}:generateContent"
)


def _gemini_request(payload: dict, timeout: float = 60.0) -> dict:
    """Send request to Gemini API with retry logic."""
    headers = {
        "x-goog-api-key": settings.gemini_api_key,
        "Content-Type": "application/json",
    }
    for attempt in range(5):
        try:
            response = httpx.post(
                GEMINI_URL, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 503) and attempt < 4:
                time.sleep(15 * (attempt + 1))
            else:
                raise
    raise RuntimeError("Gemini API request failed after retries")


def _compute_confidence(chunks: list[RetrievedChunk]) -> dict:
    """Compute confidence level based on retrieval scores."""
    if not chunks:
        return {"level": "low", "score": 0.0}
    avg_score = sum(c.score for c in chunks) / len(chunks)
    top_score = chunks[0].score if chunks else 0.0
    # Weighted: 60% top score + 40% average
    confidence = 0.6 * top_score + 0.4 * avg_score
    if confidence >= 0.65:
        level = "high"
    elif confidence >= 0.4:
        level = "medium"
    else:
        level = "low"
    return {"level": level, "score": round(confidence, 3)}


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into context string."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        clause_info = f", Clauses: {chunk.clauses}" if chunk.clauses else ""
        context_parts.append(
            f"[Document {i}: {chunk.source}, Page {chunk.page}{clause_info}] "
            f"(relevance: {chunk.score})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
) -> dict:
    """Generate answer using Gemini with retrieved context."""
    context = format_context(chunks)

    user_message = f"""Context documents:

{context}

---

Question: {query}

Answer the question based on the context above. Cite specific documents, page numbers, and regulatory clauses."""

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_message}]}],
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 0.3,
        },
    }

    data = _gemini_request(payload)
    answer_text = data["candidates"][0]["content"]["parts"][0]["text"]

    sources = []
    seen = set()
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "document": chunk.source,
                    "page": chunk.page,
                    "relevance": chunk.score,
                    "clauses": chunk.clauses,
                }
            )

    confidence = _compute_confidence(chunks)

    return {
        "answer": answer_text,
        "sources": sources,
        "model": settings.gemini_model,
        "chunks_used": len(chunks),
        "confidence": confidence,
    }
