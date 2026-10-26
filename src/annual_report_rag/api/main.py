"""FastAPI backend.

uv run arr-serve            # http://localhost:8000/docs
"""

import json
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from annual_report_rag.config import get_settings
from annual_report_rag.registry import COMPANIES, REPORTS, REPORTS_BY_ID
from annual_report_rag.retrieval import store
from annual_report_rag.retrieval.filters import Filters
from annual_report_rag.retrieval.hybrid import RetrievalOptions

EXAMPLES = [
    "Compare the revenue growth of InterGlobe Aviation and Infosys in FY24.",
    "How many aircraft did IndiGo operate at the end of FY24?",
    "What was TCS's operating margin in FY24?",
    "How did Infosys's free cash flow change between FY24 and FY25?",
    "What are the key risks IndiGo highlights in its FY24 annual report?",
    "Compare employee headcount and attrition at TCS and Infosys in FY24.",
]


def _warm_up() -> None:
    """Load indexes and local models in the background so the first question is fast."""
    try:
        from annual_report_rag.retrieval.bm25 import get_bm25
        from annual_report_rag.retrieval.models import get_cross_encoder

        strategy = get_settings().chunk_strategy
        get_bm25(strategy)
        store.get_vectorstore(strategy).similarity_search("revenue", k=1)
        get_cross_encoder()
        print("warm-up done")
    except Exception as exc:
        print(f"warm-up failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


settings = get_settings()
app = FastAPI(title="Annual Report Analyst", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    """Sliding one-minute window per client IP, to protect the free LLM quota."""
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
    now, window = time.time(), _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= settings.rate_limit_per_minute:
        raise HTTPException(429, "Too many questions, please wait a minute.")
    window.append(now)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    companies: list[str] = []
    fiscal_years: list[str] = []
    strategy: str = Field("structure", pattern="^(fixed|structure)$")
    use_bm25: bool = True
    use_vector: bool = True
    use_rerank: bool = True
    use_multi_query: bool = True


class VectorlessRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    report_id: str = "indigo_FY24"


def _indexed_reports() -> set[str]:
    return {d.metadata["report_id"] for d in store.load_chunks(settings.chunk_strategy)}


def _tree_reports() -> list[str]:
    return sorted(p.stem for p in settings.tree_dir.glob("*.json"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "indexed_reports": sorted(_indexed_reports()),
        "vectorless_reports": _tree_reports(),
        "llm_configured": bool(settings.groq_api_key),
    }


@app.get("/api/reports")
def reports():
    indexed = _indexed_reports()
    return {
        "companies": [asdict(c) for c in COMPANIES.values()],
        "reports": [
            {
                "id": r.id,
                "company": r.company,
                "fiscal_year": r.fiscal_year,
                "source_url": r.url,
                "indexed": r.id in indexed,
                "has_pdf": (settings.pdf_dir / r.file).exists(),
            }
            for r in REPORTS
        ],
        "vectorless_reports": _tree_reports(),
        "examples": EXAMPLES,
    }


@app.post("/api/ask")
async def ask(body: AskRequest, request: Request):
    rate_limit(request)
    from annual_report_rag.rag.chain import answer_question

    opts = RetrievalOptions(
        strategy=body.strategy,
        use_bm25=body.use_bm25 or not body.use_vector,
        use_vector=body.use_vector,
        use_rerank=body.use_rerank,
        use_multi_query=body.use_multi_query,
    )
    filters = Filters.of(body.companies, body.fiscal_years)
    result = await run_in_threadpool(answer_question, body.question, filters, opts)
    return result.to_dict()


@app.post("/api/ask/vectorless")
async def ask_vectorless(body: VectorlessRequest, request: Request):
    rate_limit(request)
    if body.report_id not in _tree_reports():
        raise HTTPException(404, f"no vectorless index for {body.report_id}")
    from annual_report_rag.vectorless.navigator import answer_vectorless

    result = await run_in_threadpool(answer_vectorless, body.question, body.report_id)
    return result.to_dict()


@app.get("/api/eval")
def eval_results():
    results = settings.results_dir
    summary = results / "summary.json"
    experiments = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else []
    return {"experiments": experiments}


@app.get("/api/eval/{name}")
def eval_detail(name: str):
    path = settings.results_dir / f"{name}.json"
    if not path.exists() or path.parent != settings.results_dir:
        raise HTTPException(404)
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/pdf/{report_id}")
def pdf(report_id: str):
    report = REPORTS_BY_ID.get(report_id)
    path = settings.pdf_dir / report.file if report else None
    if not path or not path.exists():
        raise HTTPException(404)
    return FileResponse(
        path, media_type="application/pdf", filename=report.file, content_disposition_type="inline"
    )


def main() -> None:
    import uvicorn

    uvicorn.run("annual_report_rag.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
