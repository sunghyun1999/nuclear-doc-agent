---
title: Nuclear Doc Agent
emoji: ⚛️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
---

# Nuclear Doc Agent

RAG-based Q&A system for nuclear regulatory documents. Built as a proof-of-concept for AI-assisted nuclear engineering document review.

## Architecture

```
[PDF Upload] → [Chunking (fixed/paragraph)] → [Embedding (local)] → [ChromaDB]
                                                                         ↓
[User Query] → [Embedding] → [Vector Search] → [Cross-Encoder Re-ranking] → [Gemini Generation] → [Answer + Source Citations]
                                                                                                           ↓
                                                                                              [Evaluation Pipeline]
                                                                                              (Retrieval Recall + Answer Quality + Chunking Comparison)
```

## Tech Stack

- **Backend**: FastAPI (Python)
- **LLM**: Google Gemini (free tier)
- **Embedding**: sentence-transformers `all-MiniLM-L6-v2` (local, no API needed)
- **Re-ranking**: Cross-encoder `ms-marco-MiniLM-L-6-v2` (local)
- **Vector DB**: ChromaDB (persistent, local)
- **Frontend**: React + TypeScript + Vite

**Cost: $0** — all inference runs locally or on free tiers.

## Quick Start

### Backend
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add Gemini API key
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Features

### Core RAG Pipeline
- PDF upload with selectable chunking strategy
- Local embedding (no API cost)
- Vector similarity search with cosine distance
- **Cross-encoder re-ranking** for improved retrieval precision
- Gemini-powered answer generation with source citations

### Evaluation Pipeline
- **10 golden Q&A test cases** based on IAEA SSR-2/1 safety standards
- **Retrieval recall measurement**: % of expected source pages found
- **LLM-as-judge answer quality scoring**: automated 0~1 scoring
- **Re-ranking impact analysis**: quantifies how much re-ranking improves retrieval
- **Chunking strategy comparison**: fixed-size vs paragraph, measured on the same eval set

### Nuclear Domain Design
- **Source citations**: every answer references specific documents and page numbers — critical for nuclear regulatory traceability
- **Human-in-the-loop**: system provides evidence-backed suggestions, not automated decisions
- **Audit-ready**: designed with the understanding that nuclear domain requires full traceability of AI-assisted decisions

## Design Decisions

### Why Re-ranking?
Bi-encoder (embedding) search is fast but imprecise — it compares compressed representations. Cross-encoder re-ranking scores each (query, chunk) pair directly, significantly improving relevance at the cost of latency. For regulatory documents where precision matters more than speed, this tradeoff is correct.

### Why Local Embeddings?
- Zero API cost for experimentation
- No data leaves the machine (important for nuclear domain sensitivity)
- `all-MiniLM-L6-v2` provides good quality at 384 dimensions

### Why Evaluation Pipeline?
"It works" is not enough for production AI systems. The evaluation pipeline provides:
1. **Reproducible quality metrics** — not subjective "looks good"
2. **Regression detection** — catch quality drops when changing prompts/models/chunking
3. **Data-driven decisions** — choose chunking strategy based on measured recall, not intuition

### Chunking Strategy Comparison
| Strategy | Behavior | Best For |
|---|---|---|
| Fixed-size (800 chars, 200 overlap) | Consistent chunk sizes, may split sentences | Dense technical text |
| Paragraph-based | Preserves semantic boundaries | Structured documents with clear sections |

The evaluation pipeline measures which strategy produces better retrieval recall on the golden test set.
