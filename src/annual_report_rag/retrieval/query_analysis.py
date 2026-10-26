"""Turn a user question into targeted searches.

Multi-query rewriting: the LLM splits a question into one *target* per company / fiscal year it
needs (so "compare A and B" retrieves evidence for both) and writes several phrasings of the
search for each target. Rule-based detection of company names and years is the fallback when
rewriting is switched off or the LLM call fails.
"""

import re

from langsmith import traceable
from pydantic import BaseModel, Field

from annual_report_rag.registry import COMPANIES, REPORTS
from annual_report_rag.retrieval.filters import Filters

ALIASES = {
    "indigo": ["indigo", "interglobe", "inter globe"],
    "infosys": ["infosys", "infy"],
    "tcs": ["tcs", "tata consultancy"],
}


class SearchTarget(BaseModel):
    queries: list[str] = Field(
        description="2-3 differently worded search queries for this target, using annual-report "
        "vocabulary (e.g. 'revenue from operations', 'consolidated statement of profit and loss')"
    )
    company: str | None = Field(None, description="company slug, or null if not company-specific")
    fiscal_year: str | None = Field(None, description="fiscal year like FY24, or null")


class QueryPlan(BaseModel):
    targets: list[SearchTarget] = Field(
        description="one target per company/fiscal-year combination the question needs (max 6)"
    )


def normalise_year(text: str) -> list[str]:
    years = set()
    for m in re.finditer(r"\bfy\s?'?(\d{2}|\d{4})\b", text, re.I):
        years.add(f"FY{m.group(1)[-2:]}")
    for m in re.finditer(r"\b20(\d{2})\s?[-–/]\s?(?:20)?(\d{2})\b", text):
        years.add(f"FY{m.group(2)}")
    for m in re.finditer(r"\bfiscal(?: year)?\s+20(\d{2})\b", text, re.I):
        years.add(f"FY{m.group(1)}")
    return sorted(years)


def rule_based_filters(question: str) -> Filters:
    q = question.lower()
    companies = [slug for slug, names in ALIASES.items() if any(n in q for n in names)]
    return Filters.of(companies, normalise_year(question))


def _catalogue() -> str:
    lines = []
    for slug, c in COMPANIES.items():
        years = sorted(r.fiscal_year for r in REPORTS if r.company == slug)
        lines.append(f"- {slug}: {c.name} (reports: {', '.join(years)})")
    return "\n".join(lines)


PROMPT = """You plan searches over Indian companies' annual reports.

Available reports:
{catalogue}

Fiscal year FY24 means the year ended 31 March 2024 and is covered by the FY24 report.
Question: {question}

Return one target per company and fiscal year the question needs. If the question compares
companies, make a separate target for each. If no year is given, use the latest report year.
If the question is not about a specific company, set company to null.
Write 2-3 queries per target in the wording an annual report would use, and make one of them the
exact financial line item (e.g. "Revenue from operations", "Profit after tax", "Total income").
Infosys reports are Form 20-F filings: they say "fiscal 2024", "Revenues" and "Net profit"."""


@traceable(name="plan_queries")
def plan_queries(question: str, user_filters: Filters, use_llm: bool = True) -> QueryPlan:
    fallback = rule_based_filters(question).narrow(user_filters)
    if use_llm:
        try:
            from annual_report_rag.retrieval.models import get_structured_llm

            planner = get_structured_llm("fast", QueryPlan)
            plan = planner.invoke(PROMPT.format(catalogue=_catalogue(), question=question))
            return _cover_named_companies(_sanitise(plan, question, user_filters), fallback)
        except Exception as exc:  # rate limits, malformed tool calls
            print(f"query planning failed, using rules: {exc}")
    return _single_target(question, fallback)


def _single_target(question: str, filters: Filters) -> QueryPlan:
    if len(filters.companies) <= 1 and len(filters.fiscal_years) <= 1:
        return QueryPlan(
            targets=[
                SearchTarget(
                    queries=[question],
                    company=filters.companies[0] if filters.companies else None,
                    fiscal_year=filters.fiscal_years[0] if filters.fiscal_years else None,
                )
            ]
        )
    years = filters.fiscal_years or (None,)
    companies = filters.companies or (None,)
    return QueryPlan(
        targets=[
            SearchTarget(queries=[question], company=c, fiscal_year=y)
            for c in companies
            for y in years
        ]
    )


def _sanitise(plan: QueryPlan, question: str, user_filters: Filters) -> QueryPlan:
    valid_years = {r.fiscal_year for r in REPORTS}
    targets = []
    for t in plan.targets[:6]:
        company = t.company if t.company in COMPANIES else None
        year = t.fiscal_year.upper().replace(" ", "") if t.fiscal_year else None
        year = year if year in valid_years else None
        if user_filters.companies and company not in user_filters.companies:
            continue
        if user_filters.fiscal_years and year and year not in user_filters.fiscal_years:
            year = None
        queries = [q for q in t.queries if q.strip()][:3] or [question]
        targets.append(
            SearchTarget(queries=[question, *queries], company=company, fiscal_year=year)
        )
    return QueryPlan(targets=targets) if targets else _single_target(question, user_filters)


def _cover_named_companies(plan: QueryPlan, named: Filters) -> QueryPlan:
    """Add a target for any company the question names that the planner left out.

    LLM planners occasionally drop one side of a comparison; the alias rules never do.
    """
    planned = {t.company for t in plan.targets}
    missing = [c for c in named.companies if c not in planned]
    if not missing:
        return plan
    years = named.fiscal_years or sorted({t.fiscal_year for t in plan.targets if t.fiscal_year})
    template = plan.targets[0].queries if plan.targets else []
    extra = [
        SearchTarget(queries=template, company=c, fiscal_year=y)
        for c in missing
        for y in (years or [None])
    ]
    return QueryPlan(targets=[*plan.targets, *extra][:6])


def target_filters(target: SearchTarget, user_filters: Filters) -> Filters:
    f = Filters.of(
        [target.company] if target.company else None,
        [target.fiscal_year] if target.fiscal_year else None,
    )
    return f.narrow(user_filters)
