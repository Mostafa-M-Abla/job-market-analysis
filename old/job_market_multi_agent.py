import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import pandas as pd
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process

from tools import ResumePDFTool, SerpApiGoogleJobsTool


# ----------------------------
# Environment / Config
# ----------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # CrewAI/OpenAI provider will use this env var
serp_api_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env or environment variables.")
if not serp_api_key:
    raise RuntimeError("SERPAPI_API_KEY is not set. Put it in your .env or environment variables.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ARTIFACTS_DIR = Path("./artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

RESUME_PATH = Path("./Resume.pdf")
JOBS_CACHE_PATH = ARTIFACTS_DIR / "job_postings_cache.json"

# ----------------------------
# Helpers (minimal hardcoding)
# ----------------------------
def safe_lower(s: str) -> str:
    return (s or "").strip().lower()


def normalize_token(t: str) -> str:
    """
    Minimal normalization only:
    - lowercase
    - collapse whitespace
    - trim punctuation
    """
    t = (t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" ,.;:-_/\\()[]{}\"'")
    return t


def compute_document_frequency(items_per_doc: List[List[str]]) -> Tuple[Counter, int]:
    """
    Document frequency (DF): an item counts at most once per job posting.
    """
    df = Counter()
    for items in items_per_doc:
        unique = set(normalize_token(x) for x in (items or []) if x and x.strip())
        df.update(unique)
    return df, len(items_per_doc)


def to_table(counter: Counter, total_docs: int, top_n: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for k, v in counter.most_common(top_n):
        pct = (v / total_docs * 100.0) if total_docs else 0.0
        rows.append({"item": k, "count": v, "percent": round(pct, 1)})
    return pd.DataFrame(rows)


def print_df(title: str, df: pd.DataFrame):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if df.empty:
        print("(No results)")
        return
    print(df.to_string(index=False))


def save_json(path: Path, data: Any):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------
# LLM Prompts
# ----------------------------
EXTRACT_JOB_INFO_INSTRUCTIONS = """
You are extracting structured requirements from job postings.

Return ONLY valid JSON with this schema:
{
  "skills": [string, ...],            // technical skills (e.g., "RAG", "MLOps", "Python", "NLP")
  "tools": [string, ...],             // tools/frameworks (e.g., "LangGraph", "Docker", "Kubernetes", "Airflow")
  "cloud_platforms": [string, ...],   // from: AWS, Azure, GCP (only these)
  "certifications": [string, ...],    // certifications (e.g., "AWS Certified Solutions Architect", "Azure AI Engineer Associate")
  "other_keywords": [string, ...]     // optional: domain keywords (e.g., "Computer Vision")
}

Rules:
- Keep entries short (1-6 words).
- No duplicates.
- If something is not mentioned, keep the list empty.
- cloud_platforms must only contain: ["AWS","Azure","GCP"] if mentioned explicitly.
"""

EXTRACT_RESUME_INFO_INSTRUCTIONS = """
Extract structured skills/tools/certifications from a resume text.

Return ONLY valid JSON with schema:
{
  "skills": [string, ...],
  "tools": [string, ...],
  "cloud_platforms": [string, ...],   // from: AWS, Azure, GCP (only these)
  "certifications": [string, ...]
}

Rules:
- Keep entries short, no duplicates.
- cloud_platforms only from AWS/Azure/GCP if clearly present.
"""

CANONICALIZE_ITEMS_INSTRUCTIONS = """
You will canonicalize a list of technical items (skills/tools/certs) by merging synonyms and near-duplicates.

Return ONLY valid JSON with schema:
{
  "canonical_map": {
     "<original_item>": "<canonical_item>",
     ...
  }
}

Rules:
- Canonical items should be short, lowercase, and consistent.
- Merge obvious synonyms/variants (e.g., "amazon web services" -> "aws", "ci/cd" -> "cicd", "lang chain" -> "langchain").
- Do NOT invent items not in the list.
- If an item is already canonical, map it to itself.
"""


# ----------------------------
# Crew Agents
# ----------------------------
def build_agents() -> Dict[str, Agent]:
    collector = Agent(
        role="Job Postings Collector",
        goal="Collect relevant job postings for given job titles and country and provide clean structured postings.",
        backstory="You are careful about data quality and deduplication.",
        verbose=False,
        allow_delegation=False,
        llm=f"openai/{OPENAI_MODEL}",
        tools=[SerpApiGoogleJobsTool(serp_api_key)],
    )

    job_extractor = Agent(
        role="Job Requirements Extractor",
        goal="Extract skills, tools, cloud platforms, and certifications from job descriptions into strict JSON schema.",
        backstory="You are precise and avoid hallucinating items not present in the text.",
        verbose=False,
        allow_delegation=False,
        llm=f"openai/{OPENAI_MODEL}",
    )

    resume_extractor = Agent(
        role="Resume Extractor",
        goal="Extract the candidate's skills/tools/certifications from Resume.pdf text into strict JSON schema.",
        backstory="You are accurate and conservative when extracting.",
        verbose=False,
        allow_delegation=False,
        llm=f"openai/{OPENAI_MODEL}",
        tools=[ResumePDFTool(RESUME_PATH)],
    )

    normalizer = Agent(
        role="Skill Normalizer",
        goal="Create a canonical mapping to merge synonyms and near-duplicates for accurate counting.",
        backstory="You normalize skill names conservatively, without inventing new skills.",
        verbose=False,
        allow_delegation=False,
        llm=f"openai/{OPENAI_MODEL}",
    )

    return {
        "collector": collector,
        "job_extractor": job_extractor,
        "resume_extractor": resume_extractor,
        "normalizer": normalizer,
    }


# ----------------------------
# LLM call helpers (via tiny crews for reliability)
# ----------------------------
def _run_single_task(agent: Agent, description: str) -> str:
    t = Task(description=description, expected_output="Valid JSON only.", agent=agent)
    c = Crew(agents=[agent], tasks=[t], process=Process.sequential, verbose=False)
    out = c.kickoff()
    return str(out).strip()


def _parse_json_strict(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"Could not parse JSON from model output:\n{text}")
        return json.loads(m.group(0))


def extract_job_requirements(job_extractor: Agent, job: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
{EXTRACT_JOB_INFO_INSTRUCTIONS}

JOB POSTING:
Title: {job.get("title","")}
Company: {job.get("company","")}
Location: {job.get("location","")}
Description:
{(job.get("description","") or "")[:9000]}
"""
    raw = _run_single_task(job_extractor, prompt)
    data = _parse_json_strict(raw)

    for k in ["skills", "tools", "cloud_platforms", "certifications", "other_keywords"]:
        data.setdefault(k, [])
    return data


def extract_resume(resume_extractor: Agent) -> Dict[str, Any]:
    resume_text = ResumePDFTool(RESUME_PATH)._run()
    prompt = f"""
{EXTRACT_RESUME_INFO_INSTRUCTIONS}

RESUME TEXT:
{resume_text[:12000]}
"""
    raw = _run_single_task(resume_extractor, prompt)
    data = _parse_json_strict(raw)

    for k in ["skills", "tools", "cloud_platforms", "certifications"]:
        data.setdefault(k, [])
    return data


def canonicalize_items(normalizer: Agent, items: List[str]) -> Dict[str, str]:
    """
    Uses the LLM to map each item -> canonical item.
    To control cost, we canonicalize a unique list once, then apply mapping.
    """
    unique_items = sorted(set(normalize_token(x) for x in items if x and x.strip()))
    # Keep prompt size reasonable
    if not unique_items:
        return {}

    prompt = f"""
{CANONICALIZE_ITEMS_INSTRUCTIONS}

ITEMS (one per line):
{chr(10).join(unique_items)}
"""
    raw = _run_single_task(normalizer, prompt)
    data = _parse_json_strict(raw)
    cmap = data.get("canonical_map", {}) or {}

    # Ensure every item is mapped (fallback to itself)
    final = {}
    for it in unique_items:
        mapped = normalize_token(cmap.get(it, it))
        final[it] = mapped if mapped else it
    return final


# ----------------------------
# Data collection (SerpAPI only)
# ----------------------------
def collect_job_postings(collector: Agent, titles: List[str], country: str, num_posts: int, resume_path: str, refresh: bool) -> List[Dict[str, Any]]:
    if not refresh and JOBS_CACHE_PATH.exists():
        cached = load_json(JOBS_CACHE_PATH)
        return cached[:num_posts]

    tool = SerpApiGoogleJobsTool(serp_api_key)
    postings: List[Dict[str, Any]] = []
    for title in titles:
        query = f"{title} jobs in {country}"
        batch = json.loads(tool._run(query=query, limit=num_posts))
        postings.extend(batch)
        if len(postings) >= num_posts:
            break

    # Dedup by (title, company)
    seen = set()
    deduped = []
    for p in postings:
        key = (safe_lower(p.get("title", "")), safe_lower(p.get("company", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
        if len(deduped) >= num_posts:
            break

    save_json(JOBS_CACHE_PATH, deduped)
    return deduped


# ----------------------------
# Analysis
# ----------------------------
def analyze(job_extractions: List[Dict[str, Any]], resume_extraction: Dict[str, Any], canonical_map: Dict[str, str]) -> Dict[str, Any]:
    n = len(job_extractions)

    def apply_map(lst: List[str]) -> List[str]:
        out = []
        for x in (lst or []):
            nx = normalize_token(x)
            out.append(canonical_map.get(nx, nx))
        # remove empties
        return [o for o in out if o]

    skills_per_job = [apply_map(x.get("skills", [])) for x in job_extractions]
    tools_per_job = [apply_map(x.get("tools", [])) for x in job_extractions]
    certs_per_job = [apply_map(x.get("certifications", [])) for x in job_extractions]
    clouds_per_job = [apply_map(x.get("cloud_platforms", [])) for x in job_extractions]

    combined_per_job = [list(set((s or []) + (t or []))) for s, t in zip(skills_per_job, tools_per_job)]

    skill_df, total = compute_document_frequency(combined_per_job)
    cloud_df, _ = compute_document_frequency(clouds_per_job)
    cert_df, _ = compute_document_frequency(certs_per_job)

    # Resume set
    resume_items = set()
    for k in ["skills", "tools", "certifications", "cloud_platforms"]:
        for item in resume_extraction.get(k, []):
            resume_items.add(canonical_map.get(normalize_token(item), normalize_token(item)))

    # Missing ranked by market demand
    market = Counter()
    market.update(skill_df)
    market.update(cert_df)
    market.update(cloud_df)

    missing = [item for item, _ in market.most_common() if item and item not in resume_items]

    return {
        "n": n,
        "skills_table": to_table(skill_df, total_docs=total).rename(columns={"item": "skill_or_tool"}),
        "clouds_table": to_table(cloud_df, total_docs=total).rename(columns={"item": "cloud_platform"}),
        "certs_table": to_table(cert_df, total_docs=total, top_n=30).rename(columns={"item": "certification"}),
        "top10_missing": missing[:10],
    }


# ----------------------------
# Main entrypoint
# ----------------------------
def run(job_titles: List[str], country: str, num_posts: int, refresh_postings: bool = False):
    agents = build_agents()

    postings = collect_job_postings(agents["collector"], job_titles, country, num_posts=num_posts, refresh=refresh_postings)
    if len(postings) < num_posts:
        print(f"Warning: only found {len(postings)} postings (requested {num_posts}).")

    save_json(ARTIFACTS_DIR / "job_postings_used.json", postings)

    # Extract job requirements
    job_extractions = []
    for i, job in enumerate(postings, start=1):
        try:
            extracted = extract_job_requirements(agents["job_extractor"], job)
            extracted["_meta"] = {k: job.get(k) for k in ["title", "company", "location", "url"]}
            job_extractions.append(extracted)
            time.sleep(0.2)
        except Exception as e:
            print(f"[Extractor error] posting #{i}: {e}")

    save_json(ARTIFACTS_DIR / "job_requirements_extracted.json", job_extractions)

    # Extract resume
    resume_extraction = extract_resume(agents["resume_extractor"])
    save_json(ARTIFACTS_DIR / "resume_extracted.json", resume_extraction)

    # Build a canonical map using ALL items we will compare/count
    all_items = []
    for x in job_extractions:
        all_items.extend(x.get("skills", []))
        all_items.extend(x.get("tools", []))
        all_items.extend(x.get("certifications", []))
        all_items.extend(x.get("cloud_platforms", []))
    for k in ["skills", "tools", "certifications", "cloud_platforms"]:
        all_items.extend(resume_extraction.get(k, []))

    canonical_map = canonicalize_items(agents["normalizer"], all_items)
    save_json(ARTIFACTS_DIR / "canonical_map.json", canonical_map)

    results = analyze(job_extractions, resume_extraction, canonical_map)

    print_df(f"(a) Top needed skills/tools for {job_titles} in {country} (N={results['n']})", results["skills_table"])
    print_df(f"(b) Cloud platforms mentioned (N={results['n']})", results["clouds_table"])
    print_df(f"(c) Certifications mentioned (N={results['n']})", results["certs_table"])

    print("\n" + "=" * 90)
    print("(d) Top 10 missing skills/tools/certs not found in Resume.pdf (ranked by demand)")
    print("=" * 90)
    for idx, item in enumerate(results["top10_missing"], start=1):
        print(f"{idx}. {item}")

    save_json(ARTIFACTS_DIR / "final_results_summary.json", {
        "job_titles": job_titles,
        "country": country,
        "num_job_postings": results["n"],
        "top10_missing": results["top10_missing"],
    })


if __name__ == "__main__":
    target_job_titles = ["AI Engineer", "GenAI Engineer", "Generative AI Engineer"]

    # TODO later change num_posts to 20
    run(job_titles=target_job_titles, country="Egypt", num_posts=2, refresh_postings=False)
