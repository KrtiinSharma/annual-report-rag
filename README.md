# Annual Report Analyst

Ask questions across the annual reports of Indian listed companies and get answers with
**page-level citations**. For example: *"Compare the revenue growth of InterGlobe Aviation and
Infosys in FY24"*.

Built with LangChain, Groq (gpt-oss-120b), FastEmbed, Chroma, LangSmith and RAGAS, with a
FastAPI backend and a React frontend. See [REQUIREMENTS.md](REQUIREMENTS.md) for the full
functional and technical requirements.

| | |
|---|---|
| **Ingestion** | PyMuPDF text + tables → Markdown, two chunking strategies (fixed vs structure-aware) |
| **Retrieval** | BM25 + vector search, Reciprocal Rank Fusion, cross-encoder reranker, multi-query rewriting, company/year filters |
| **Vectorless RAG** | contents-tree index + LLM navigation for one long report, compared with vector RAG |
| **Evaluation** | LangSmith tracing, 30–50 question dataset, RAGAS faithfulness / relevancy / context precision / recall |
| **Serving** | FastAPI on Hugging Face Spaces, React UI on GitHub Pages |

## Quick start

```bash
uv sync --all-extras
cp .env.example .env        # add GROQ_API_KEY and LANGSMITH_API_KEY
uv run arr-download         # fetches what it can and lists PDFs to download by hand
uv run arr-ingest           # parse, chunk (both strategies), embed
uv run arr-tree --report indigo_FY24   # contents tree for vectorless RAG
uv run arr-serve            # API on http://localhost:8000/docs
npm --prefix frontend install && npm --prefix frontend run dev   # UI on http://localhost:5173
```

### Getting the PDFs

Company and exchange sites block scripted downloads, so `arr-download` prints the page for each
missing report. Save each PDF into `data/pdfs/` with the name it prints, for example
`indigo_FY24.pdf`. The corpus is defined in [`registry.py`](src/annual_report_rag/registry.py).

| Report | Document | Currency |
|---|---|---|
| InterGlobe Aviation (IndiGo) FY24, FY25 | Annual report | ₹ |
| TCS FY24 | Integrated annual report | ₹ |
| Infosys FY24, FY25 | Form 20-F (annual report filed with the US SEC, IFRS) | US$ |

Cross-company comparisons work on growth rates and ratios; absolute Infosys figures are in US$.

## Evaluation

```bash
uv run arr-eval upload                                   # questions -> LangSmith dataset
uv run arr-eval retrieval --strategy fixed               # chunking comparison (no LLM)
uv run arr-eval retrieval --strategy structure
uv run arr-eval run --name hybrid-structure              # RAGAS, logged as a LangSmith experiment
uv run arr-eval run --name hybrid-fixed --strategy fixed
uv run arr-eval run --name vectorless --mode vectorless --subset vectorless
uv run arr-eval run --name vector-indigo --subset vectorless --companies indigo --years FY24
uv run arr-eval report                                   # docs/results/RESULTS.md
```

Runs resume where they stopped, which helps with Groq's free-tier daily token limits.
Results: [docs/results/RESULTS.md](docs/results/RESULTS.md).

## Deployment (free)

- **Backend**: `uv run python scripts/deploy_space.py --space <hf-user>/annual-report-analyst --origins https://<github-user>.github.io`
  creates a Docker Space, uploads code + indexes + PDFs, and sets the API keys as Space secrets.
- **Frontend**: the `pages.yml` workflow builds `frontend/` and publishes it to GitHub Pages.
  Set the repository variable `API_URL` to the Space URL, and set Pages → Source to "GitHub Actions".

## Project layout

```
src/annual_report_rag/
  ingest/       PDF parsing, chunking strategies, index build
  retrieval/    BM25, vector store, RRF, reranker, query planner, filters
  rag/          answer chain and citations
  vectorless/   contents-tree builder and navigator
  eval/         dataset, RAGAS metrics, LangSmith experiments, reports
  api/          FastAPI app
frontend/       React + Vite + Tailwind UI
data/eval/      test questions (committed)
data/trees/     vectorless indexes (committed)
docs/results/   evaluation results (committed)
```
