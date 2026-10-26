# Annual Report Analyst — Requirements

A retrieval-augmented question-answering system over the annual reports of Indian listed
companies. Users ask questions such as *"Compare the revenue growth of InterGlobe Aviation and
Infosys in FY24"* and get an answer grounded in the reports, with **page-level citations**.

---

## 1. Functional requirements

### FR-1 Document corpus
| ID | Requirement |
|----|-------------|
| FR-1.1 | The corpus contains annual-report PDFs of at least 4 Indian listed companies, downloaded from company websites or BSE. InterGlobe Aviation (IndiGo) is included. |
| FR-1.2 | At least two companies have reports for two fiscal years so that the year filter is meaningful. |
| FR-1.3 | Each report is registered with: company name, ticker/BSE code, fiscal year (e.g. `FY24`), source URL, local file name. |
| FR-1.4 | Adding a new report requires only a registry entry plus re-running ingestion. |

### FR-2 Ingestion
| ID | Requirement |
|----|-------------|
| FR-2.1 | Extract text from every page and keep the **1-based physical page number** of the PDF. |
| FR-2.2 | Detect tables and convert them to Markdown so numbers stay aligned with their row/column labels. |
| FR-2.3 | Support **two chunking strategies**: (a) *fixed* — recursive character splitting within a page; (b) *structure-aware* — heading-led sections with tables kept as whole chunks and a section breadcrumb prepended. |
| FR-2.4 | Every chunk carries metadata: `company`, `fiscal_year`, `page`, `chunk_type` (text/table), `section`, `strategy`, `source_url`. |
| FR-2.5 | A comparison report shows retrieval quality (hit-rate@k, MRR) and RAGAS context scores for both strategies on the same test set. |

### FR-3 Retrieval
| ID | Requirement |
|----|-------------|
| FR-3.1 | **Keyword search** with BM25. |
| FR-3.2 | **Vector search** with a dense embedding model. |
| FR-3.3 | **Hybrid fusion** of BM25 and vector results with Reciprocal Rank Fusion (RRF). |
| FR-3.4 | A **cross-encoder reranker** reorders the fused candidates. |
| FR-3.5 | **Multi-query rewriting**: the LLM rewrites the question into several search queries and splits comparison questions into one sub-query per company/year. |
| FR-3.6 | **Filters** for company and fiscal year, applied to both BM25 and vector search. Filters can be set explicitly (UI) or inferred from the question. |
| FR-3.7 | Each pipeline stage (BM25, vector, rerank, multi-query) can be switched on/off per request for ablation. |

### FR-4 Answer generation
| ID | Requirement |
|----|-------------|
| FR-4.1 | The answer uses only retrieved context and cites sources inline as `[n]`. |
| FR-4.2 | Every citation resolves to company, fiscal year, page number, a text snippet and a link that opens the PDF at that page. |
| FR-4.3 | If the context does not contain the answer, the system says so instead of guessing. |
| FR-4.4 | Numeric answers state units (₹ crore, %, etc.) as given in the report. |

### FR-5 Vectorless RAG
| ID | Requirement |
|----|-------------|
| FR-5.1 | For one long report, build a hierarchical **table-of-contents tree** (sections → sub-sections with page ranges and LLM summaries) without embeddings. |
| FR-5.2 | At query time the LLM reasons over the tree, picks the relevant nodes, reads those pages and answers with page citations. |
| FR-5.3 | A comparison report contrasts vectorless and vector RAG on the same questions: RAGAS scores, latency and tokens used. |

### FR-6 Evaluation and observability (LangSmith)
| ID | Requirement |
|----|-------------|
| FR-6.1 | All chains are **traced** in LangSmith (query rewriting, retrieval, reranking, generation). |
| FR-6.2 | A **test dataset of 30–50 questions**, each with reference answer, company, fiscal year and reference page(s), is stored in the repo and uploaded to LangSmith. |
| FR-6.3 | Evaluation runs report **RAGAS** metrics: faithfulness, answer relevancy, context precision and context recall, logged as LangSmith experiment feedback. |
| FR-6.4 | Results are also written to `docs/results/` and shown in the web UI. |

### FR-7 API and UI
| ID | Requirement |
|----|-------------|
| FR-7.1 | FastAPI backend: `POST /api/ask`, `POST /api/ask/vectorless`, `GET /api/reports`, `GET /api/eval`, `GET /api/health`. |
| FR-7.2 | Web UI: question box, company and year filters, mode switch (Hybrid RAG / Vectorless), pipeline toggles, example questions. |
| FR-7.3 | The answer view shows clickable citation chips. Clicking one shows the snippet and opens the PDF page. |
| FR-7.4 | A "How it works" panel shows the generated sub-queries and the retrieved chunks with their scores. |
| FR-7.5 | An "Evaluation" page shows RAGAS scores, the chunking comparison and the vector vs vectorless comparison. |
| FR-7.6 | The UI works on mobile and desktop. |

---

## 2. Technical requirements

### TR-1 Stack
| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11, managed with `uv` | Reproducible installs |
| Orchestration | LangChain (`langchain-core`, `langchain-community`, `langchain-groq`) | Required by the brief |
| LLM | Groq: `openai/gpt-oss-120b` for answers, `openai/gpt-oss-20b` for rewriting, navigation and judging (configurable) | Free tier, fast |
| PDF parsing | PyMuPDF (`pymupdf`) — text blocks with font sizes, `find_tables()` for tables, `get_toc()` for outlines | Fast, handles tables, pure wheel |
| Embeddings | `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX, CPU) | Free, runs locally and on a free CPU host |
| Vector store | Chroma (persistent, metadata `where` filters) | Native filtering |
| Keyword search | `rank_bm25` (BM25Okapi) with filter masking | Classic lexical baseline |
| Reranker | Cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` via FastEmbed | Free, CPU-friendly |
| Evaluation | LangSmith SDK (datasets, experiments) + RAGAS | Required by the brief |
| API | FastAPI + Uvicorn, Pydantic v2 | Required by the brief |
| Frontend | React + TypeScript + Vite + Tailwind CSS | Static build → GitHub Pages |

### TR-2 Architecture

```
PDFs ─► parser (text + tables, page no.) ─► chunker (fixed | structure)
                                             │
                     ┌───────────────────────┴──────────────────┐
                     ▼                                          ▼
              Chroma (bge-small)                          BM25 index
                     │                                          │
question ─► query analyser (LLM: sub-queries + filters)         │
                     │                                          │
                     └──► per sub-query: vector ∪ BM25 ─► RRF ─► cross-encoder rerank
                                                                │
                                         top-k chunks ─► LLM answer with [n] citations

Vectorless: PDF outline/headings ─► section tree + LLM summaries (JSON)
            question ─► LLM picks nodes ─► read pages ─► LLM answer with page citations
```

### TR-3 Repository layout
```
src/annual_report_rag/
  config.py            settings from environment / .env
  registry.py          companies & reports
  ingest/              download, PDF parsing, chunking, index build
  retrieval/           bm25, vector, hybrid (RRF), reranker, query analysis
  rag/                 answer chain + citations
  vectorless/          tree builder + navigator
  eval/                dataset, LangSmith upload, RAGAS runs, comparisons
  api/                 FastAPI app
frontend/              React app (GitHub Pages)
data/                  pdfs/ (git-ignored), processed/, index/ (git-ignored), trees/
docs/results/          evaluation outputs (committed)
```

### TR-4 Configuration
Read from environment / `.env` (never committed):
`GROQ_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT`,
`ANSWER_MODEL`, `FAST_MODEL`, `JUDGE_MODEL`, `CHUNK_STRATEGY`, `ALLOWED_ORIGINS`.

### TR-5 Non-functional
| ID | Requirement |
|----|-------------|
| NFR-1 | Total hosting and model cost: ₹0 (free tiers only). |
| NFR-2 | p50 answer latency under ~8 s on the free host, excluding cold start. |
| NFR-3 | Embedding and reranking run on CPU in under 2 GB RAM. |
| NFR-4 | API keys stay server-side. The public API has a simple per-IP rate limit to protect the free LLM quota. |
| NFR-5 | Ingestion is idempotent and can be re-run. Index artifacts are rebuilt from PDFs with one command. |
| NFR-6 | Groq free-tier rate limits are handled with retries/backoff. Evaluation can run in batches (`--limit`, resume). |

### TR-6 Deployment (free)
| Part | Host | Notes |
|------|------|-------|
| Backend | Hugging Face Spaces (Docker SDK, free CPU: 2 vCPU / 16 GB) | Index files shipped with the Space. Secrets set in Space settings. Sleeps after 48 h idle and wakes on the next request. |
| Frontend | GitHub Pages via GitHub Actions | `VITE_API_URL` points to the Space. |
| CI | GitHub Actions | Lint + unit tests + frontend build on push. |

### TR-7 Testing
- Unit tests (pytest) for chunking, RRF fusion, filter masking, citation parsing and tree navigation parsing. These need no network or API keys.
- End-to-end evaluation through LangSmith + RAGAS (needs keys).
