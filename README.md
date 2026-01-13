# Job Market Multi-Agent Analysis System

A **CrewAI multi-agent workflow** that:
1. Collects real job postings for one or more target titles in a given country (via **SerpAPI Google Jobs**).
2. Extracts and aggregates **in-demand requirements** (skills/tools, cloud platforms, certifications).
3. Extracts your **resume skills** from a local PDF.
4. Compares resume vs. market needs and generates **top skill/tool recommendations**.
5. Produces a **polished HTML report** saved to `outputs/`.

---

## Project Structure

```text
job-market-analysis/
├─ job_market_multi_agent.py          # Main entry point (kickoff + saves outputs)
├─ build_crew.py                      # Builds the CrewAI agents & tasks pipeline
├─ tools/
│  ├─ google_jobs_tool.py             # SerpAPI Google Jobs collector tool
│  ├─ resume_pdf_tool.py              # PDF -> text extractor tool
│  └─ save_html_tool.py               # Saves final HTML report to outputs/
├─ evaluation/
│  └─ evaluate_html_report.py         # Optional: LLM-based HTML report scorer (LangSmith)
├─ outputs/                           # Generated artifacts (HTML/JSON/MD)
├─ Resume.pdf                         # Your resume (expected file name by default)
├─ .env                               # Environment variables (you create this)
└─ .gitignore
```

---

## How It Works (Agent Pipeline)

The system runs **sequentially**:

1. **Job Postings Finder**
   - Uses `GoogleJobsCollectorTool` to fetch job posts for your target titles/country.

2. **Job Requirements Extractor**
   - Parses each job description and extracts:
     - `technical_skills_and_tools`
     - `cloud_platforms` (limited to AWS / Azure / GCP)
     - `certifications`

3. **Job Requirements Analyzer**
   - Aggregates requirements across postings, deduplicates synonyms, and produces a **markdown summary**.

4. **Resume Content Extractor**
   - Uses `ResumePDFTextTool` to extract text from `Resume.pdf`
   - Extracts skills/tools, cloud platforms, certifications.

5. **Resume Booster Specialist**
   - Compares market demand vs. resume strengths and recommends **top 5 skills/tools to learn next** (with frequencies).

6. **HTML Reports Writer Specialist**
   - Combines the market analysis + recommendations into a **light-themed HTML report**
   - Saves the report via `SaveHTMLTool` into `outputs/`.

---

## Requirements

- Python 3.10+ recommended
- API keys:
  - `SERPAPI_API_KEY` (required for job collection)
  - `OPENAI_API_KEY` (required for LLM agents)
  - `LANGSMITH_API_KEY` (only if you run the evaluation script)

---

## Installation

```bash
# 1) Create & activate a virtual environment (example)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -U pip
pip install crewai langchain-openai langchain-core python-dotenv requests PyPDF2 langsmith
```

> Tip: if you already have a `requirements.txt`, prefer `pip install -r requirements.txt`.

---

## Configuration (.env)

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
SERPAPI_API_KEY=your_serpapi_key

# Optional (only needed for evaluation/evaluate_html_report.py)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=job-market-multi-agent
```

---

## Usage

### 1) Put your resume PDF in the project root

The resume tool defaults to `Resume.pdf`.  
Make sure the file name matches exactly:

```text
Resume.pdf
```

### 2) Run the multi-agent workflow

```bash
python job_market_multi_agent.py
```

By default, the script:
- Uses a list of AI/ML-oriented job titles
- Uses `country="Egypt"`
- Collects `total_num_posts=30`

You can edit the inputs in `job_market_multi_agent.py` to match your target roles and location.

---

## Outputs

All generated artifacts are written into the `outputs/` directory:

- `job_market_report_<timestamp>.html`  
  Final HTML report (written by `SaveHTMLTool`)

- `crew_output_<timestamp>.json`  
  Full CrewAI execution trace: all tasks, agent outputs, metadata

- `final_result_<timestamp>.json` and `final_result_<timestamp>.md`  
  Convenience files containing the final task output

---

## Optional: Evaluate the HTML report (LangSmith)

There is an evaluation script that loads a previously generated HTML report and scores it on:

- relevance, accuracy, completeness, clarity, visual appeal, insights, final_score

Run it from the `evaluation/` folder (or adjust paths as needed):

```bash
python evaluation/evaluate_html_report.py
```

> Notes:
> - You must set `OPENAI_API_KEY` and `LANGSMITH_API_KEY`.
> - The script currently points to a specific HTML file path; update `OUTPUT_HTML_PATH` to your latest report.

---

## Customization Ideas

- Replace the simple title similarity filter with embeddings / fuzzy matching.
- Add more job sources (LinkedIn, Indeed, etc.) behind additional tools.
- Improve the HTML report with charts (e.g., skill frequency bar chart).
- Add OCR support for scanned resumes (if `Resume.pdf` is image-only).

---

## Troubleshooting

**SerpAPI key error**  
If you see: `SERPAPI_API_KEY env var is required`  
→ add `SERPAPI_API_KEY` to your `.env`.

**Resume text extraction fails**  
If you see: `No text could be extracted from the PDF... needs OCR`  
→ your PDF is likely scanned images; convert to text-based PDF or add OCR.

**No job results / irrelevant titles**  
- Broaden the job titles list
- Adjust the similarity threshold logic in `google_jobs_tool.py`
- Increase the `limit` in the collector tool call

---

## License

Add your preferred license (MIT, Apache-2.0, etc.) or remove this section.
