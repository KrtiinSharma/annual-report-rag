"""Deploy the backend to a free Hugging Face Docker Space.

    uv run python scripts/deploy_space.py --space <hf-user>/annual-report-analyst \
        --origins https://<github-user>.github.io

Uploads the code, the prebuilt indexes (data/index, data/processed, data/trees), the PDFs and the
evaluation results in one commit, and sets GROQ_API_KEY / LANGSMITH_API_KEY from .env as Space
secrets. Needs `huggingface-cli login` (or HF_TOKEN) with write access.
"""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from annual_report_rag.config import ROOT, get_settings

INCLUDE = [
    "Dockerfile",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "src/**/*.py",
    "data/index/**/*",
    "data/processed/*.jsonl",
    "data/trees/*.json",
    "data/pdfs/*.pdf",
    "docs/results/*",
]


def files() -> list[Path]:
    out = []
    for pattern in INCLUDE:
        out += [p for p in ROOT.glob(pattern) if p.is_file() and "__pycache__" not in p.parts]
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True, help="e.g. myname/annual-report-analyst")
    parser.add_argument("--origins", default="", help="comma-separated allowed CORS origins")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    s = get_settings()
    api = HfApi()
    api.create_repo(
        args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
    )

    ops = [
        CommitOperationAdd(path_in_repo=p.relative_to(ROOT).as_posix(), path_or_fileobj=str(p))
        for p in files()
        if p.name != "README.md"
    ]
    # the Space card doubles as the README.md the package metadata needs
    ops.append(CommitOperationAdd("README.md", str(ROOT / "deploy" / "space_README.md")))
    size = sum(Path(op.path_or_fileobj).stat().st_size for op in ops) / 1e6
    print(f"uploading {len(ops)} files ({size:.0f} MB) to spaces/{args.space}")
    api.create_commit(
        args.space, operations=ops, repo_type="space", commit_message="Deploy backend"
    )

    if s.groq_api_key:
        api.add_space_secret(args.space, "GROQ_API_KEY", s.groq_api_key)
    if s.langsmith_api_key:
        api.add_space_secret(args.space, "LANGSMITH_API_KEY", s.langsmith_api_key)
        api.add_space_variable(args.space, "LANGSMITH_TRACING", "true")
        api.add_space_variable(args.space, "LANGSMITH_PROJECT", s.langsmith_project)
    origins = ",".join(o for o in ["http://localhost:5173", args.origins] if o)
    api.add_space_variable(args.space, "ALLOWED_ORIGINS", origins)

    owner, name = args.space.split("/")
    print(f"Space: https://huggingface.co/spaces/{args.space}")
    print(f"API:   https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
