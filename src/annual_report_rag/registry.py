"""The annual reports in the corpus.

To add a report: append a `Report`, put the PDF at `data/pdfs/<report.file>` (or let
`arr-download` fetch it), then re-run `arr-ingest`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Company:
    slug: str
    name: str
    nse: str
    bse: str
    sector: str
    ir_page: str


@dataclass(frozen=True)
class Report:
    company: str  # Company.slug
    fiscal_year: str  # "FY24" = year ended 31 March 2024
    url: str  # direct PDF link when one is known, else the investor-relations page

    @property
    def id(self) -> str:
        return f"{self.company}_{self.fiscal_year}"

    @property
    def file(self) -> str:
        return f"{self.id}.pdf"


COMPANIES: dict[str, Company] = {
    c.slug: c
    for c in [
        Company(
            "indigo",
            "InterGlobe Aviation (IndiGo)",
            "INDIGO",
            "539448",
            "Aviation",
            "https://www.goindigo.in/information/investor-relations.html",
        ),
        Company(
            "infosys",
            "Infosys",
            "INFY",
            "500209",
            "IT Services",
            "https://www.infosys.com/investors/reports-filings/annual-report.html",
        ),
        Company(
            "tcs",
            "Tata Consultancy Services",
            "TCS",
            "532540",
            "IT Services",
            "https://www.tcs.com/investor-relations",
        ),
    ]
}

REPORTS: list[Report] = [
    Report(
        "indigo",
        "FY24",
        "https://nsearchives.nseindia.com/annual_reports/AR_24753_INDIGO_2023_2024_31072024222919.pdf",
    ),
    Report(
        "indigo",
        "FY25",
        "https://www.goindigo.in/content/dam/goindigo/investor-relations/policies/2025/Annual_Report_FY25.pdf",
    ),
    Report(
        "infosys",
        "FY24",
        "https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-24.pdf",
    ),
    Report(
        "infosys",
        "FY25",
        "https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-25.pdf",
    ),
    Report(
        "tcs",
        "FY24",
        "https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2023-24/ar/annual-report-2023-2024.pdf",
    ),
]

REPORTS_BY_ID: dict[str, Report] = {r.id: r for r in REPORTS}


def company_name(slug: str) -> str:
    return COMPANIES[slug].name if slug in COMPANIES else slug
