---
title: Annual Report Analyst API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: RAG over Indian annual reports with page-level citations
---

FastAPI backend for **Annual Report Analyst**: hybrid (BM25 + vector + reranker) and vectorless
RAG over the annual reports of Indian listed companies.

- API docs: `/docs`
- Health: `/api/health`

Source code and the web UI are on GitHub. This Space is deployed with `scripts/deploy_space.py`.
