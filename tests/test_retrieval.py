from langchain_core.documents import Document

from annual_report_rag.retrieval.bm25 import BM25Index, tokenize
from annual_report_rag.retrieval.filters import Filters
from annual_report_rag.retrieval.hybrid import rrf
from annual_report_rag.retrieval.query_analysis import (
    normalise_year,
    plan_queries,
    rule_based_filters,
)


def doc(text, company="indigo", year="FY24", page=1, idx=0):
    return Document(
        page_content=text,
        metadata={
            "report_id": f"{company}_{year}",
            "company": company,
            "fiscal_year": year,
            "page": page,
            "chunk_index": idx,
            "strategy": "structure",
        },
    )


CORPUS = [
    doc("Revenue from operations grew to 68,904 crore", "indigo", "FY24", 10, 0),
    doc("Revenue from operations was 1,53,670 crore", "infosys", "FY24", 20, 1),
    doc("Revenue from operations was 1,62,990 crore", "infosys", "FY25", 30, 2),
    doc("Fleet size and aircraft deliveries", "indigo", "FY24", 11, 3),
]


def test_tokenize_merges_indian_number_grouping():
    assert "162990" in tokenize("Revenue of ₹1,62,990 crore")
    assert "the" not in tokenize("the revenue")


def test_bm25_filters_mask_other_companies_and_years():
    index = BM25Index(CORPUS)
    hits = index.search("revenue from operations", 5, Filters.of(["infosys"], ["FY25"]))
    assert [d.metadata["page"] for d, _ in hits] == [30]
    unfiltered = index.search("revenue from operations", 5)
    assert len(unfiltered) == 3


def test_chroma_filter_shapes():
    assert Filters().to_chroma() is None
    assert Filters.of(["tcs"]).to_chroma() == {"company": {"$in": ["tcs"]}}
    both = Filters.of(["tcs"], ["FY24"]).to_chroma()
    assert both == {"$and": [{"company": {"$in": ["tcs"]}}, {"fiscal_year": {"$in": ["FY24"]}}]}


def test_narrow_prefers_user_filters():
    inferred = Filters.of(["indigo", "tcs"], ["FY24"])
    assert inferred.narrow(Filters.of(["tcs"])).companies == ("tcs",)
    assert inferred.narrow(Filters.of(["infosys"])).companies == ("infosys",)
    assert inferred.narrow(Filters()).fiscal_years == ("FY24",)


def test_rrf_rewards_documents_found_by_both_retrievers():
    a, b, c = CORPUS[:3]
    fused = rrf({"bm25": [a, b], "vector": [c, a]}, k=60)
    assert fused[0].doc is a
    assert sorted(fused[0].sources) == ["bm25#1", "vector#2"]


def test_year_and_company_detection():
    assert normalise_year("FY24 vs FY 2025") == ["FY24", "FY25"]
    assert normalise_year("in 2023-24") == ["FY24"]
    assert normalise_year("fiscal 2025 results") == ["FY25"]
    f = rule_based_filters("Compare IndiGo and Infosys revenue growth in FY24")
    assert set(f.companies) == {"indigo", "infosys"} and f.fiscal_years == ("FY24",)


def test_rule_based_plan_splits_comparisons_per_company():
    plan = plan_queries("Compare IndiGo and TCS revenue in FY24", Filters(), use_llm=False)
    assert {(t.company, t.fiscal_year) for t in plan.targets} == {
        ("indigo", "FY24"),
        ("tcs", "FY24"),
    }


def test_citation_parsing_handles_lists_and_ranges():
    from annual_report_rag.rag.citations import cited_ids

    answer = "Revenue rose [1], margins fell [2, 4] and costs grew [5-6]. See [99]."
    assert cited_ids(answer, 6) == {1, 2, 4, 5, 6}


def test_planner_targets_cover_every_named_company():
    from annual_report_rag.retrieval.filters import Filters
    from annual_report_rag.retrieval.query_analysis import (
        QueryPlan,
        SearchTarget,
        _cover_named_companies,
    )

    plan = QueryPlan(targets=[SearchTarget(queries=["q"], company="infosys", fiscal_year="FY24")])
    covered = _cover_named_companies(plan, Filters.of(["indigo", "infosys"], ["FY24"]))
    assert [(t.company, t.fiscal_year) for t in covered.targets] == [
        ("infosys", "FY24"),
        ("indigo", "FY24"),
    ]


def test_gpt_oss_citation_brackets_are_normalised():
    from annual_report_rag.rag.citations import cited_ids, normalise_citations

    answer = normalise_citations("Revenue grew 26.6% 【3】; see also 【2†L4-L9】 and 【1, 4】.")
    assert answer == "Revenue grew 26.6% [3]; see also [2] and [1, 4]."
    assert cited_ids(answer, 5) == {1, 2, 3, 4}


def test_split_number_cells_are_cleaned():
    from annual_report_rag.rag.citations import clean_table_text

    row = "| Revenues | 18,562 | 2 | 18,212 | 2 | 1.9 | 9 % | (0.3 | 3 ) |"
    assert clean_table_text(row) == "| Revenues | 18,562 | 18,212 | 1.9% | (0.3) |"


def test_reranker_sees_only_the_top_candidates(monkeypatch):
    from annual_report_rag.retrieval import hybrid
    from annual_report_rag.retrieval.query_analysis import SearchTarget

    docs = [doc(f"passage {i}", page=i, idx=i) for i in range(30)]
    monkeypatch.setattr(hybrid, "bm25_search", lambda *a: docs)
    monkeypatch.setattr(hybrid, "vector_search", lambda *a: docs[::-1])
    seen = []

    def fake_rerank(query, candidates, top_n):
        seen.extend(candidates)
        return [(d, 1.0) for d in candidates[:top_n]]

    monkeypatch.setattr(hybrid, "rerank", fake_rerank)
    hybrid.retrieve_target(SearchTarget(queries=["q"]), Filters(), hybrid.RetrievalOptions())
    assert len(seen) == hybrid.get_settings().rerank_candidates
