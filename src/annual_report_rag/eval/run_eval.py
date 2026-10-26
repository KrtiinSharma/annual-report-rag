"""Evaluation CLI.

    uv run arr-eval upload                                  # push questions to LangSmith
    uv run arr-eval retrieval --strategy fixed              # hit@k / MRR, no LLM needed
    uv run arr-eval run --name hybrid-structure             # full RAG + RAGAS, logged to LangSmith
    uv run arr-eval run --name hybrid-fixed --strategy fixed
    uv run arr-eval run --name hybrid-no-rerank --no-rerank
    uv run arr-eval run --name vectorless --mode vectorless --subset vectorless
    uv run arr-eval run --name vector-on-indigo --subset vectorless --companies indigo --years FY24
    uv run arr-eval report                                  # docs/results/summary.json + RESULTS.md

Every run writes docs/results/<name>.json. Re-running with the same name resumes: questions that
already have scores are skipped, which helps with Groq's daily token limits.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from annual_report_rag.config import get_settings
from annual_report_rag.eval import ragas_metrics
from annual_report_rag.eval.dataset import DATASET_NAME, Question, load_questions, upload
from annual_report_rag.retrieval.filters import Filters
from annual_report_rag.retrieval.hybrid import RetrievalOptions, retrieve

K_VALUES = (1, 3, 5, 10)


def results_path(name: str) -> Path:
    return get_settings().results_dir / f"{name}.json"


def page_metrics(q: Question, ranked_pages: list[tuple[str, int]]) -> dict:
    gold = q.pages
    out = {f"hit@{k}": float(any(p in gold for p in ranked_pages[:k])) for k in K_VALUES}
    rank = next((i for i, p in enumerate(ranked_pages, 1) if p in gold), None)
    out["mrr"] = 1.0 / rank if rank else 0.0
    return out


def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


# --------------------------------------------------------------------------- retrieval only


def eval_retrieval(questions: list[Question], strategy: str, multi_query: bool) -> dict:
    opts = RetrievalOptions(strategy=strategy, use_multi_query=multi_query, top_n=10)
    rows = []
    for q in questions:
        r = retrieve(q.question, Filters(), opts)
        ranked = [(h.doc.metadata["report_id"], h.doc.metadata["page"]) for h in r.hits]
        rows.append({"id": q.id, **page_metrics(q, ranked)})
    keys = [f"hit@{k}" for k in K_VALUES] + ["mrr"]
    return {
        "name": f"retrieval-{strategy}{'-mq' if multi_query else ''}",
        "kind": "retrieval",
        "config": {"strategy": strategy, "multi_query": multi_query},
        "aggregate": {k: mean(r[k] for r in rows) for k in keys},
        "rows": rows,
    }


# --------------------------------------------------------------------------- full RAG + RAGAS


def make_target(args):
    if args.mode == "vectorless":
        from annual_report_rag.vectorless.navigator import answer_vectorless

        return lambda q: answer_vectorless(q, args.report)
    from annual_report_rag.rag.chain import answer_question

    opts = RetrievalOptions(
        strategy=args.strategy,
        use_bm25=not args.no_bm25,
        use_vector=not args.no_vector,
        use_rerank=not args.no_rerank,
        use_multi_query=not args.no_multi_query,
    )
    filters = Filters.of(args.companies, args.years)
    return lambda q: answer_question(q, filters, opts)


def evaluate_one(target, q: Question) -> dict:
    t = time.perf_counter()
    result = target(q.question)
    latency = time.perf_counter() - t
    ranked = [(c.report_id, c.page) for c in result.citations]
    cited = [(c.report_id, c.page) for c in result.citations if c.cited]
    scores = ragas_metrics.score(q.question, result.answer, result.contexts, q.reference)
    return {
        "id": q.id,
        "question": q.question,
        "answer": result.answer,
        "reference": q.reference,
        "pages": ranked,
        "cited_pages": cited,
        "latency_s": round(latency, 2),
        "cited_page_hit": float(any(p in q.pages for p in cited)),
        **page_metrics(q, ranked),
        **scores,
    }


def run_with_langsmith(target, questions, args, done: dict) -> None:
    """Run through langsmith.evaluate so each question is a traced run with RAGAS feedback."""
    from langsmith import Client, evaluate

    by_id = {q.id: q for q in questions}

    def ls_target(inputs: dict) -> dict:
        q = by_id[inputs["id"]]
        row = evaluate_one(target, q)
        done[q.id] = row
        save(args, done)
        return {"answer": row["answer"], "pages": row["pages"]}

    def ragas_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        row = done[inputs["id"]]
        keys = [*ragas_metrics.METRICS, "hit@5", "mrr", "cited_page_hit"]
        return {"results": [{"key": k, "score": row[k]} for k in keys if row.get(k) is not None]}

    todo = {q.id for q in questions if q.id not in done}
    data = [
        e for e in Client().list_examples(dataset_name=DATASET_NAME) if e.inputs.get("id") in todo
    ]
    evaluate(
        ls_target,
        data=data,
        evaluators=[ragas_evaluator],
        experiment_prefix=args.name,
        metadata={k: v for k, v in vars(args).items() if k != "func"},
        max_concurrency=1,
    )


def save(args, rows: dict) -> None:
    path = results_path(args.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows.values())
    keys = [*ragas_metrics.METRICS, "hit@1", "hit@5", "mrr", "cited_page_hit", "latency_s"]
    payload = {
        "name": args.name,
        "kind": "rag",
        "config": {k: v for k, v in vars(args).items() if k not in ("func", "limit")},
        "n": len(rows_list),
        "aggregate": {k: mean(r.get(k) for r in rows_list) for k in keys},
        "rows": rows_list,
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_run(args) -> None:
    questions = load_questions()
    if args.subset == "vectorless":
        questions = [q for q in questions if q.vectorless]
    if args.limit:
        questions = questions[: args.limit]
    path = results_path(args.name)
    done = {}
    if path.exists():
        done = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))["rows"]}
        done = {k: v for k, v in done.items() if v.get("faithfulness") is not None}
    target = make_target(args)
    print(f"{args.name}: {len(questions)} questions, {len(done)} already scored")

    if os.environ.get("LANGSMITH_API_KEY") and not args.local:
        run_with_langsmith(target, questions, args, done)
    else:
        for q in questions:
            if q.id in done:
                continue
            done[q.id] = evaluate_one(target, q)
            save(args, done)
            row = done[q.id]
            print(f"  {q.id}: faith={row['faithfulness']} recall={row['context_recall']}")
    save(args, done)
    agg = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
    print(json.dumps(agg, indent=1))


def cmd_retrieval(args) -> None:
    questions = load_questions()
    out = eval_retrieval(questions, args.strategy, args.multi_query)
    path = results_path(out["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out["name"], json.dumps(out["aggregate"]))


def cmd_upload(args) -> None:
    get_settings()
    print("dataset id:", upload(load_questions()))


def cmd_report(args) -> None:
    from annual_report_rag.eval.report import write_report

    print("wrote", write_report())


def main() -> None:
    get_settings()  # exports LangSmith env vars
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("upload")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("retrieval")
    p.add_argument("--strategy", default="structure", choices=["fixed", "structure"])
    p.add_argument("--multi-query", action="store_true")
    p.set_defaults(func=cmd_retrieval)

    p = sub.add_parser("run")
    p.add_argument("--name", required=True)
    p.add_argument("--mode", default="hybrid", choices=["hybrid", "vectorless"])
    p.add_argument("--report", default="indigo_FY24", help="report used by vectorless mode")
    p.add_argument("--strategy", default="structure", choices=["fixed", "structure"])
    p.add_argument("--subset", choices=["all", "vectorless"], default="all")
    p.add_argument("--companies", nargs="*")
    p.add_argument("--years", nargs="*")
    p.add_argument("--no-bm25", action="store_true")
    p.add_argument("--no-vector", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--no-multi-query", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--local", action="store_true", help="do not log to LangSmith")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
