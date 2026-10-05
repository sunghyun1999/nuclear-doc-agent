"""Evaluation pipeline: auto-evaluate RAG quality with golden Q&A dataset."""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from config import settings
from reranker import rerank
from retriever import retrieve

EVAL_DATA_PATH = Path(__file__).parent / "eval_dataset.json"


@dataclass
class EvalCase:
    question: str
    expected_answer: str
    expected_pages: list[int] = field(default_factory=list)


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    generated_answer: str
    retrieval_score: float  # % of expected pages found in retrieved chunks
    answer_score: float  # LLM-judged relevance 0~1
    rerank_helped: bool  # did reranking improve page recall?


def load_eval_dataset() -> list[EvalCase]:
    """Load evaluation Q&A pairs from JSON file."""
    if not EVAL_DATA_PATH.exists():
        return []
    with open(EVAL_DATA_PATH) as f:
        data = json.load(f)
    return [EvalCase(**item) for item in data]


def evaluate_retrieval(
    question: str,
    expected_pages: list[int],
    use_rerank: bool = False,
) -> tuple[float, list[int]]:
    """Evaluate retrieval: what % of expected pages were found?"""
    chunks = retrieve(question, top_k=10)
    if use_rerank:
        chunks = rerank(question, chunks, top_k=5)
    else:
        chunks = chunks[:5]

    retrieved_pages = [c.page for c in chunks]
    if not expected_pages:
        return 1.0, retrieved_pages

    found = sum(1 for p in expected_pages if p in retrieved_pages)
    recall = found / len(expected_pages)
    return recall, retrieved_pages


def evaluate_answer_quality(
    question: str,
    expected: str,
    generated: str,
) -> float:
    """Use Gemini to judge answer quality (0~1 score)."""
    prompt = f"""You are an evaluation judge. Score how well the generated answer matches the expected answer.

Question: {question}

Expected answer: {expected}

Generated answer: {generated}

Score from 0.0 to 1.0 where:
- 1.0 = generated answer fully covers the expected answer with correct information
- 0.5 = partially correct, some key points missing
- 0.0 = completely wrong or irrelevant

Respond with ONLY a JSON object: {{"score": 0.X, "reason": "brief explanation"}}"""

    from generator import _gemini_request

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }

    try:
        data = _gemini_request(payload)
    except Exception:
        return 0.5

    text = data["candidates"][0]["content"]["parts"][0]["text"]

    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
        return float(result.get("score", 0.5))
    return 0.5


def run_evaluation() -> dict:
    """Run full evaluation pipeline on all test cases."""
    cases = load_eval_dataset()
    if not cases:
        return {"error": "No evaluation dataset found. Create eval_dataset.json first."}

    results: list[dict] = []
    total_retrieval = 0.0
    total_answer = 0.0
    rerank_improvements = 0

    for case in cases:
        # Retrieval eval: without reranking
        recall_no_rerank, _ = evaluate_retrieval(case.question, case.expected_pages, use_rerank=False)

        # Retrieval eval: with reranking
        recall_with_rerank, retrieved_pages = evaluate_retrieval(case.question, case.expected_pages, use_rerank=True)

        rerank_helped = recall_with_rerank > recall_no_rerank
        if rerank_helped:
            rerank_improvements += 1

        # Generate answer (with reranking)
        chunks = retrieve(case.question, top_k=10)
        chunks = rerank(case.question, chunks, top_k=5)

        from generator import generate_answer

        # Rate limit delay for Gemini free tier
        time.sleep(15)
        gen_result = generate_answer(case.question, chunks)

        # Answer quality eval
        time.sleep(15)
        answer_score = evaluate_answer_quality(
            case.question, case.expected_answer, gen_result["answer"]
        )

        total_retrieval += recall_with_rerank
        total_answer += answer_score

        results.append({
            "question": case.question,
            "retrieval_recall_no_rerank": round(recall_no_rerank, 2),
            "retrieval_recall_with_rerank": round(recall_with_rerank, 2),
            "rerank_helped": rerank_helped,
            "answer_score": round(answer_score, 2),
            "retrieved_pages": retrieved_pages,
            "expected_pages": case.expected_pages,
        })

    n = len(cases)
    return {
        "total_cases": n,
        "avg_retrieval_recall": round(total_retrieval / n, 3),
        "avg_answer_score": round(total_answer / n, 3),
        "rerank_improvement_rate": f"{rerank_improvements}/{n}",
        "results": results,
    }


def run_chunking_comparison() -> dict:
    """Compare fixed_size vs paragraph chunking strategies."""
    cases = load_eval_dataset()
    if not cases:
        return {"error": "No evaluation dataset found."}

    from ingestion import get_collection, ingest_pdf
    from pathlib import Path

    # Check if both collections exist
    upload_dir = Path("./uploads")
    pdfs = list(upload_dir.glob("*.pdf"))
    if not pdfs:
        return {"error": "No PDFs uploaded. Upload a document first."}

    strategies = {}
    for strategy in ["fixed_size", "paragraph"]:
        collection_name = f"nuclear_docs_{strategy}"

        # Check if collection exists and has data
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        col = client.get_or_create_collection(name=collection_name)

        if col.count() == 0:
            # Ingest with this strategy
            for pdf in pdfs:
                ingest_pdf(str(pdf), strategy=strategy, collection_name=collection_name)

        # Evaluate retrieval for each case
        total_recall = 0.0
        for case in cases:
            chunks_raw = retrieve(case.question, top_k=10, collection_name=collection_name)
            chunks_reranked = rerank(case.question, chunks_raw, top_k=5)
            retrieved_pages = [c.page for c in chunks_reranked]

            if case.expected_pages:
                found = sum(1 for p in case.expected_pages if p in retrieved_pages)
                total_recall += found / len(case.expected_pages)
            else:
                total_recall += 1.0

        col_count = col.count()
        strategies[strategy] = {
            "total_chunks": col_count,
            "avg_retrieval_recall": round(total_recall / len(cases), 3),
        }

    return {
        "comparison": strategies,
        "total_eval_cases": len(cases),
        "recommendation": (
            "fixed_size" if strategies["fixed_size"]["avg_retrieval_recall"]
            >= strategies["paragraph"]["avg_retrieval_recall"]
            else "paragraph"
        ),
    }
