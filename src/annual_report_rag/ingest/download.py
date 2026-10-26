"""Fetch the report PDFs listed in the registry into `data/pdfs/`.

Many company and exchange sites block scripted downloads. For those, this prints the page to open
in a browser and the file name to save the PDF as.
"""

import argparse

import httpx

from annual_report_rag.config import get_settings
from annual_report_rag.registry import REPORTS, company_name


def fetch(url: str, dest) -> bool:
    try:
        with httpx.stream("GET", url, timeout=60, follow_redirects=True) as r:
            if r.status_code != 200:
                return False
            chunks = r.iter_bytes()
            first = next(chunks, b"")
            if not first.startswith(b"%PDF"):
                return False
            with open(dest, "wb") as f:
                f.write(first)
                for chunk in chunks:
                    f.write(chunk)
        return True
    except httpx.HTTPError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    pdf_dir = get_settings().pdf_dir
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manual = []
    for report in REPORTS:
        dest = pdf_dir / report.file
        if dest.exists() and not args.force:
            print(f"ok       {report.file}")
            continue
        if report.url.lower().endswith(".pdf") and fetch(report.url, dest):
            print(f"fetched  {report.file}")
        else:
            dest.unlink(missing_ok=True)
            manual.append(report)
            print(f"MISSING  {report.file}")

    if manual:
        print("\nDownload these in a browser and save them into", pdf_dir)
        for r in manual:
            print(f"  {r.file:<22} {company_name(r.company)} {r.fiscal_year}: {r.url}")


if __name__ == "__main__":
    main()
