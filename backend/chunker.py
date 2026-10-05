"""Document chunking strategies for RAG pipeline."""

import re
from dataclasses import dataclass

# Patterns for nuclear regulatory clause detection
_CLAUSE_PATTERNS = [
    re.compile(r"Requirement\s+(\d+)", re.IGNORECASE),
    re.compile(r"§\s*(\d+(?:\.\d+)*)"),
    re.compile(r"(?:para(?:graph)?\.?\s*)(\d+(?:\.\d+)*)"),
    re.compile(r"Article\s+(\d+(?:\.\d+)*)", re.IGNORECASE),
]


def _extract_clauses(text: str) -> list[str]:
    """Extract regulatory clause references from text."""
    clauses: list[str] = []
    for pattern in _CLAUSE_PATTERNS:
        for match in pattern.finditer(text):
            clauses.append(match.group(0).strip())
    return clauses


@dataclass
class Chunk:
    text: str
    metadata: dict


def chunk_by_fixed_size(
    text: str, chunk_size: int, overlap: int, source: str, page: int
) -> list[Chunk]:
    """Fixed-size chunking with overlap."""
    if overlap >= chunk_size:
        overlap = chunk_size // 4
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "source": source,
                        "page": page,
                        "strategy": "fixed_size",
                        "char_start": start,
                        "char_end": end,
                        "clauses": ", ".join(_extract_clauses(chunk_text)),
                    },
                )
            )
        start += chunk_size - overlap
    return chunks


def chunk_by_paragraph(text: str, source: str, page: int) -> list[Chunk]:
    """Paragraph-based chunking — splits on double newlines."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        if len(para) < 50:
            continue
        chunks.append(
            Chunk(
                text=para,
                metadata={
                    "source": source,
                    "page": page,
                    "strategy": "paragraph",
                    "paragraph_index": i,
                    "clauses": ", ".join(_extract_clauses(para)),
                },
            )
        )
    return chunks
