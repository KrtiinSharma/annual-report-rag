"""Company / fiscal-year filters shared by BM25 and the vector store."""

from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass(frozen=True)
class Filters:
    companies: tuple[str, ...] = field(default_factory=tuple)
    fiscal_years: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def of(cls, companies=None, fiscal_years=None) -> "Filters":
        return cls(tuple(companies or ()), tuple(fiscal_years or ()))

    def matches(self, doc: Document) -> bool:
        m = doc.metadata
        return (not self.companies or m.get("company") in self.companies) and (
            not self.fiscal_years or m.get("fiscal_year") in self.fiscal_years
        )

    def to_chroma(self) -> dict | None:
        clauses = []
        if self.companies:
            clauses.append({"company": {"$in": list(self.companies)}})
        if self.fiscal_years:
            clauses.append({"fiscal_year": {"$in": list(self.fiscal_years)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def narrow(self, other: "Filters") -> "Filters":
        """Intersect with filters the user set explicitly (theirs win when they conflict)."""

        def pick(mine, theirs):
            if not theirs:
                return mine
            both = tuple(x for x in mine if x in theirs)
            return both or theirs

        return Filters(
            pick(self.companies, other.companies), pick(self.fiscal_years, other.fiscal_years)
        )
