"""Parse every downloaded report, chunk it with each strategy and build the search indexes.

uv run arr-ingest                      # all strategies, all reports present in data/pdfs
uv run arr-ingest --strategy structure
uv run arr-ingest --reparse            # ignore cached parses in data/processed
"""

import argparse
import time

from annual_report_rag.config import get_settings
from annual_report_rag.ingest.chunking import STRATEGIES, chunk
from annual_report_rag.ingest.pdf_parser import load_pages, parse_pdf, save_pages
from annual_report_rag.registry import REPORTS
from annual_report_rag.retrieval import store


def parsed_pages(report, reparse: bool = False):
    s = get_settings()
    cache = s.processed_dir / f"{report.id}.jsonl"
    if cache.exists() and not reparse:
        return load_pages(cache)
    t = time.perf_counter()
    pages = parse_pdf(s.pdf_dir / report.file)
    save_pages(pages, cache)
    tables = sum(1 for p in pages for e in p.elements if e.kind == "table")
    print(
        f"  parsed {report.id}: {len(pages)} pages, {tables} tables "
        f"in {time.perf_counter() - t:.0f}s"
    )
    return pages


def build(strategies, reparse: bool = False) -> None:
    s = get_settings()
    reports = [r for r in REPORTS if (s.pdf_dir / r.file).exists()]
    missing = [r.file for r in REPORTS if r not in reports]
    if missing:
        print("skipping (PDF not found, run arr-download):", ", ".join(missing))
    if not reports:
        raise SystemExit("no PDFs in " + str(s.pdf_dir))

    pages = {r.id: parsed_pages(r, reparse) for r in reports}

    for strategy in strategies:
        docs = []
        for r in reports:
            report_docs = chunk(r, pages[r.id], strategy)
            for i, d in enumerate(report_docs):
                d.metadata["chunk_index"] = i
            docs.extend(report_docs)
        store.save_chunks(docs, strategy)

        vs = store.get_vectorstore(strategy)
        vs.reset_collection()
        t = time.perf_counter()
        batch = 256
        for i in range(0, len(docs), batch):
            part = docs[i : i + batch]
            vs.add_documents(part, ids=[store.chunk_id(d) for d in part])
        avg = sum(len(d.page_content) for d in docs) / len(docs)
        print(
            f"[{strategy}] {len(docs)} chunks (avg {avg:.0f} chars), "
            f"embedded in {time.perf_counter() - t:.0f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, action="append")
    parser.add_argument("--reparse", action="store_true")
    args = parser.parse_args()
    build(args.strategy or STRATEGIES, args.reparse)


if __name__ == "__main__":
    main()
