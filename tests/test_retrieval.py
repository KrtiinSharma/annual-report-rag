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
