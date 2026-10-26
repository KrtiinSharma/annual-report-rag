"""Deploy the backend to Google Cloud Run (free tier: scales to zero, 2M requests/month).

    uv run python scripts/deploy_cloudrun.py --project <gcp-project-id> \
        --origins https://<github-user>.github.io

Needs the gcloud CLI, logged in (`gcloud auth login`) with billing enabled on the project.
Cloud Build builds the Dockerfile from the files .gcloudignore lets through (code + prebuilt
indexes + PDFs + evaluation results). GROQ_API_KEY / LANGSMITH_API_KEY are read from .env and set
as environment variables on the service; they are never part of the uploaded source or the image.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from annual_report_rag.config import ROOT, get_settings

SERVICE = "annual-report-api"
APIS = ["run.googleapis.com", "cloudbuild.googleapis.com", "artifactregistry.googleapis.com"]


def gcloud(*args: str, capture: bool = False) -> str:
    exe = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not exe:
        raise SystemExit("gcloud CLI not found: https://cloud.google.com/sdk/docs/install")
    out = subprocess.run([exe, *args], check=True, text=True, capture_output=capture)
    return out.stdout.strip() if capture else ""


def env_vars(origins: str) -> dict[str, str]:
    s = get_settings()
    if not s.groq_api_key:
        raise SystemExit("GROQ_API_KEY missing from .env")
    allowed = ",".join(o for o in ["http://localhost:5173", origins] if o)
    out = {
        "GROQ_API_KEY": s.groq_api_key,
        "ANSWER_MODEL": s.answer_model,
        "FAST_MODEL": s.fast_model,
        "ALLOWED_ORIGINS": allowed,
    }
    if s.langsmith_api_key:
        out |= {
            "LANGSMITH_API_KEY": s.langsmith_api_key,
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": s.langsmith_project,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--origins", default="", help="allowed CORS origin(s), comma-separated")
    args = parser.parse_args()

    gcloud("services", "enable", *APIS, "--project", args.project)

    # Keys go through a temporary YAML file so they never appear on a command line or in logs.
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "env.yaml"
        env_file.write_text(
            "".join(f'{k}: "{v}"\n' for k, v in env_vars(args.origins).items()), encoding="utf-8"
        )
        gcloud(
            "run", "deploy", SERVICE,
            "--source", str(ROOT),
            "--project", args.project,
            "--region", args.region,
            "--port", "7860",
            "--memory", "2Gi",
            "--cpu", "1",
            "--cpu-boost",
            "--min-instances", "0",
            "--max-instances", "2",
            "--concurrency", "4",
            "--timeout", "300",
            "--allow-unauthenticated",
            "--env-vars-file", str(env_file),
            "--quiet",
        )  # fmt: skip

    url = gcloud(
        "run", "services", "describe", SERVICE,
        "--project", args.project, "--region", args.region,
        "--format", "value(status.url)",
        capture=True,
    )  # fmt: skip
    print(f"API: {url}")
    print(f"Set the GitHub repository variable API_URL to {url}")


if __name__ == "__main__":
    main()
