import os
import time
import re
from typing import Any, Dict, List, Optional, Tuple
import requests

from crewai.tools import BaseTool


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _title_is_similar(found_title: str, target_titles: List[str]) -> bool:
    """
    Very lightweight similarity check: token overlap.
    You can improve later with embeddings / fuzzy matching.
    """
    ft = set(re.findall(r"[a-zA-Z]+", _normalize(found_title)))
    if not ft:
        return False

    for tt in target_titles:
        tt_tokens = set(re.findall(r"[a-zA-Z]+", _normalize(tt)))
        if not tt_tokens:
            continue
        overlap = len(ft.intersection(tt_tokens)) / max(1, len(tt_tokens))
        if overlap >= 0.6:  # tweak threshold if needed
            return True
    return False


class GoogleJobsCollectorTool(BaseTool):
    name: str = "Google Jobs Collector Tool"
    description: str = (
        "Uses SerpAPI Google Jobs API to find job postings by title and country, "
        "then uses Google Jobs Listing API to fetch full job content (description). "
        "Returns structured job postings."
    )
    api_key: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.getenv("SERPAPI_API_KEY", "")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY env var is required to use GoogleJobsCollectorTool.")

    def _run(
        self,
        job_titles: List[str],
        country: str,
        limit: int = 20,
        fetch_full_listing: bool = True,
        google_domain: str = "google.com",
        gl: Optional[str] = None,  # e.g. "eg" for Egypt (optional)
        hl: str = "en",
        sleep_s: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Returns list of:
        {title, company, location, via, job_id, apply_links, description, source, serpapi_url}
        """
        collected: List[Dict[str, Any]] = []
        seen_keys = set()

        for title in job_titles:
            # Query format: keep it simple and location-focused
            q = f"{title} jobs in {country}"

            jobs = self._serpapi_google_jobs_search(
                q=q, google_domain=google_domain, gl=gl, hl=hl
            )

            for j in jobs:
                job_id = j.get("job_id") or j.get("jobid") or j.get("id")
                found_title = j.get("title") or ""
                company = j.get("company_name") or j.get("company") or ""
                location = j.get("location") or ""

                # Filter: location contains country (best-effort)
                if country and _normalize(country) not in _normalize(location):
                    # Some listings omit country; you can relax if needed.
                    pass

                # Filter: title similar to one of your titles
                if not _title_is_similar(found_title, job_titles):
                    continue

                key = (_normalize(found_title), _normalize(company), _normalize(location))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                details = {}
                if fetch_full_listing and job_id:
                    details = self._serpapi_google_jobs_listing(job_id=job_id, hl=hl)
                    time.sleep(sleep_s)

                collected.append({
                    "title": found_title,
                    "company": company,
                    "location": location,
                    "via": j.get("via"),
                    "job_id": job_id,
                    "apply_links": j.get("apply_options") or j.get("related_links") or [],
                    # Prefer detailed description from listing API; fallback to snippet/description
                    "description": (
                        (details.get("job_description") or details.get("description") or "").strip()
                        or (j.get("description") or "").strip()
                    ),
                    "source": "serpapi_google_jobs",
                    "serpapi_job_result": j,
                    "serpapi_listing_result": details,
                })

                if len(collected) >= limit:
                    return collected

            time.sleep(sleep_s)

        return collected

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")

    def _serpapi_google_jobs_search(
        self,
        q: str,
        google_domain: str = "google.com",
        gl: Optional[str] = None,
        hl: str = "en",
    ) -> List[Dict[str, Any]]:
        """
        engine=google_jobs
        """
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_jobs",
            "q": q,
            "google_domain": google_domain,
            "hl": hl,
            "api_key": self.api_key,
        }
        if gl:
            params["gl"] = gl

        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("jobs_results", []) or []

    def _serpapi_google_jobs_listing(self, job_id: str, hl: str = "en") -> Dict[str, Any]:
        """
        engine=google_jobs_listing with q=job_id
        """
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_jobs_listing",
            "q": job_id,
            "hl": hl,
            "api_key": self.api_key,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json() or {}
