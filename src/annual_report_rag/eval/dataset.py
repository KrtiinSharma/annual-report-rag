"""The test questions (data/eval/questions.jsonl) and their LangSmith dataset."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from annual_report_rag.config import get_settings

DATASET_NAME = "annual-report-qa"


@dataclass
class Question:
    id: str
    question: str
    reference: str  # ground-truth answer written from the report
    reference_pages: list[dict]  # [{"report_id": "indigo_FY24", "page": 123}]
    companies: list[str] = field(default_factory=list)
    fiscal_years: list[str] = field(default_factory=list)
    kind: str = "fact"  # fact | numeric | comparison | table | qualitative
    vectorless: bool = False  # answerable from the single report used for vectorless RAG

    @property
    def pages(self) -> set[tuple[str, int]]:
        return {(p["report_id"], p["page"]) for p in self.reference_pages}


def questions_path() -> Path:
    return get_settings().data_dir / "eval" / "questions.jsonl"


def load_questions(path: Path | None = None) -> list[Question]:
    with open(path or questions_path(), encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def upload(questions: list[Question], name: str = DATASET_NAME) -> str:
    """Create (or replace the examples of) the LangSmith dataset. Returns the dataset id."""
    from langsmith import Client

    client = Client()
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        old = [e.id for e in client.list_examples(dataset_id=dataset.id)]
        if old:
            client.delete_examples(example_ids=old)
    else:
        dataset = client.create_dataset(
            name, description="Questions over Indian annual reports with reference answers/pages"
        )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": q.question, "id": q.id},
                "outputs": {"reference": q.reference, "reference_pages": q.reference_pages},
                "metadata": {
                    "kind": q.kind,
                    "companies": q.companies,
                    "fiscal_years": q.fiscal_years,
                    "vectorless": q.vectorless,
                },
            }
            for q in questions
        ],
    )
    return str(dataset.id)
