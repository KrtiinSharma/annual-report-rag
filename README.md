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
| **Serving** | FastAPI on Google Cloud Run, React UI on GitHub Pages |

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
uv run arr-eval run --name hybrid-structure --subset sample   # RAGAS, logged to LangSmith
uv run arr-eval run --name vectorless --mode vectorless --subset vectorless
uv run arr-eval run --name vector-indigo --subset vectorless --companies indigo --years FY24
uv run arr-eval report                                   # docs/results/RESULTS.md
```

The RAGAS runs answer with Claude Haiku 5.5 (`ANSWER_MODEL=claude-haiku-5-5`) and judge with the
same model: Groq's free 200K tokens/day ran out after ~16 evaluated questions, and the whole
evaluation costs well under $1 on Claude. The live site keeps answering with gpt-oss-120b on Groq.
RAGAS runs on a fixed 20-question sample (`--subset sample`) plus the 17 vectorless questions; the
chunking comparison (`retrieval`) uses all 48. Runs resume where they stopped, and a row whose
judging failed is re-scored without regenerating its answer.

```bash
ANSWER_MODEL=claude-haiku-5-5 uv run arr-eval run --name hybrid-structure --subset sample
```

Results: [docs/results/RESULTS.md](docs/results/RESULTS.md).

## Deployment (free)

- **Backend** on Google Cloud Run (scales to zero; free within the monthly allowance):
  `uv run python scripts/deploy_cloudrun.py --project <gcp-project> --origins https://<github-user>.github.io`
  builds the Dockerfile with Cloud Build (code + indexes + PDFs, see `.gcloudignore`) and sets the
  API keys from `.env` as environment variables. Needs the gcloud CLI and billing enabled.
  On a new project, first let the default build account build from source (one-time):
  `gcloud projects add-iam-policy-binding <gcp-project> --member="serviceAccount:<project-number>-compute@developer.gserviceaccount.com" --role="roles/cloudbuild.builds.builder"`
- **Frontend**: the `pages.yml` workflow builds `frontend/` and publishes it to GitHub Pages.
  Set the repository variable `API_URL` to the Cloud Run URL, and Pages → Source to "GitHub Actions".

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
