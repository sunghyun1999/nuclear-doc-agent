"""Nuclear Document Q&A Agent — FastAPI server."""

import os
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generator import generate_answer
from ingestion import get_collection, ingest_pdf
from reranker import rerank
from retriever import retrieve

app = FastAPI(
    title="Nuclear Doc Agent",
    description="AI-Assisted Regulatory Review for SMR Safety Analysis",
    version="0.3.0",
)

# Serve frontend static files if available (production / HF Spaces)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# --- Models ---


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_rerank: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
    chunks_used: int
    confidence: dict


# --- Endpoints ---


@app.get("/health")
def health():
    collection = get_collection()
    return {"status": "ok", "indexed_chunks": collection.count()}


@app.post("/documents/upload")
async def upload_document(file: UploadFile, strategy: str = "fixed_size"):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if strategy not in ("fixed_size", "paragraph"):
        raise HTTPException(
            status_code=400, detail="Strategy must be 'fixed_size' or 'paragraph'"
        )

    safe_name = Path(file.filename).name
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = UPLOAD_DIR / safe_name
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingest_pdf(str(file_path), strategy=strategy)

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])

    return result


@app.get("/documents")
def list_documents():
    collection = get_collection()
    if collection.count() == 0:
        return {"documents": [], "total_chunks": 0}

    all_metadata = collection.get()["metadatas"]
    sources = {}
    for meta in all_metadata:
        src = meta.get("source", "unknown")
        if src not in sources:
            sources[src] = {"source": src, "chunks": 0, "strategy": meta.get("strategy")}
        sources[src]["chunks"] += 1

    return {
        "documents": list(sources.values()),
        "total_chunks": collection.count(),
    }


@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    collection = get_collection()
    if collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a PDF first.",
        )

    # Retrieve more candidates, then rerank to top_k
    chunks = retrieve(req.question, top_k=req.top_k * 3)

    if req.use_rerank:
        chunks = rerank(req.question, chunks, top_k=req.top_k)
    else:
        chunks = chunks[: req.top_k]

    result = generate_answer(req.question, chunks)
    return result


_eval_state: dict = {"status": "idle", "result": None}
_chunking_state: dict = {"status": "idle", "result": None}


@app.post("/evaluate")
def run_eval(background_tasks: BackgroundTasks):
    """Start evaluation pipeline (runs in background)."""
    from evaluation import run_evaluation

    collection = get_collection()
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed.")

    if _eval_state["status"] == "running":
        return {"status": "running", "message": "Evaluation already in progress"}

    def _run():
        _eval_state["status"] = "running"
        _eval_state["result"] = None
        try:
            _eval_state["result"] = run_evaluation()
            _eval_state["status"] = "done"
        except Exception as e:
            _eval_state["result"] = {"error": str(e)}
            _eval_state["status"] = "error"

    background_tasks.add_task(_run)
    return {"status": "started"}


@app.get("/evaluate/result")
def get_eval_result():
    """Get evaluation result (poll until done)."""
    return {"status": _eval_state["status"], "result": _eval_state["result"]}


@app.post("/evaluate/chunking-comparison")
def run_chunking_eval(background_tasks: BackgroundTasks):
    """Start chunking comparison (runs in background)."""
    from evaluation import run_chunking_comparison

    if _chunking_state["status"] == "running":
        return {"status": "running"}

    def _run():
        _chunking_state["status"] = "running"
        _chunking_state["result"] = None
        try:
            _chunking_state["result"] = run_chunking_comparison()
            _chunking_state["status"] = "done"
        except Exception as e:
            _chunking_state["result"] = {"error": str(e)}
            _chunking_state["status"] = "error"

    background_tasks.add_task(_run)
    return {"status": "started"}


@app.get("/evaluate/chunking-result")
def get_chunking_result():
    return {"status": _chunking_state["status"], "result": _chunking_state["result"]}


_graph_cache: dict = {"data": None}


@app.get("/requirements/graph")
def get_requirements_graph():
    """Return requirement cross-reference graph for ontology visualization."""
    if _graph_cache["data"] is not None:
        return _graph_cache["data"]

    pdfs = list(UPLOAD_DIR.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(status_code=400, detail="No documents uploaded.")

    from ontology import build_graph

    graph = build_graph(str(pdfs[0]))
    _graph_cache["data"] = graph
    return graph


@app.delete("/documents")
def clear_documents():
    collection = get_collection()
    if collection.count() > 0:
        all_ids = collection.get()["ids"]
        collection.delete(ids=all_ids)
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            os.remove(f)
    _graph_cache["data"] = None
    return {"status": "cleared"}


# SPA fallback — serve index.html for non-API routes
if STATIC_DIR.exists():

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
