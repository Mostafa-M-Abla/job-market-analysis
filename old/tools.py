import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from crewai.tools import BaseTool
from PyPDF2 import PdfReader

try:
    import requests
except ImportError:
    requests = None



class ResumePDFTool(BaseTool):
    """
    Reads a resume PDF from disk and returns extracted text.
    """
    name: str = "resume_pdf_reader"
    description: str = "Reads Resume.pdf from disk and returns extracted text."

    def __init__(self, resume_path: Path):
        super().__init__()
        self.resume_path = resume_path

    def _run(self) -> str:
        if not self.resume_path.exists():
            raise FileNotFoundError(f"Resume file not found: {self.resume_path.resolve()}")
        reader = PdfReader(str(self.resume_path))
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)


class SerpApiGoogleJobsTool(BaseTool):
    """
    Collects job postings from SerpAPI Google Jobs.

    Notes:
    - Requires `SERPAPI_API_KEY` env var.
    - Requires `requests` package.
    """
    name: str = "serpapi_google_jobs"
    description: str = (
        "Fetches job postings using SerpAPI Google Jobs.\n"
        "Inputs: query (string), limit (int). Output: JSON list of postings."
    )

    def __init__(self, serp_api_key: str):
        super().__init__()
        self.serp_api_key = serp_api_key

    def _run(self, query: str, limit: int) -> str:
        if not self.serp_api_key:
            raise RuntimeError("SERPAPI_API_KEY is not set.")
        if requests is None:
            raise RuntimeError("requests is not installed. Run: pip install requests")

        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_jobs",
            "q": query,
            "api_key": self.serp_api_key,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        jobs = data.get("jobs_results", []) or []
        results: List[Dict[str, Any]] = []
        for j in jobs[:limit]:
            results.append(
                {
                    "title": j.get("title"),
                    "company": (j.get("company_name") or j.get("company")),
                    "location": j.get("location"),
                    "description": (j.get("description") or ""),
                    "source": "serpapi_google_jobs",
                    "url": (
                        j.get("related_links", [{}])[0].get("link")
                        if j.get("related_links") else j.get("job_id")
                    ),
                }
            )

        return json.dumps(results, ensure_ascii=False)
