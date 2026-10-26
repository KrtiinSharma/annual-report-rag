# Backend image for a Hugging Face Docker Space (also runs anywhere: docker run -p 7860:7860).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/hf

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first so code changes don't reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the local embedding and reranker models into the image
RUN uv run --no-sync python -c "from annual_report_rag.retrieval.models import get_embeddings, get_cross_encoder; get_embeddings().embed_query('warm'); get_cross_encoder()"

# Prebuilt indexes, parsed pages, contents trees, PDFs and evaluation results
COPY data ./data
COPY docs/results ./docs/results

# Spaces run as uid 1000; Chroma needs write access to its sqlite file
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["uv", "run", "--no-sync", "uvicorn", "annual_report_rag.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
