import warnings
warnings.filterwarnings("ignore")
import time
from datetime import datetime
from langchain_openai import ChatOpenAI

import json
from pathlib import Path

import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from google_jobs_tool import GoogleJobsCollectorTool


# from crewai_tools import (
#   FileReadTool,
#   ScrapeWebsiteTool,
#   MDXSearchTool,
#   SerperDevTool
# )
#
# from download_page_tool import DownloadPageTool

#search_tool = SerperDevTool()
#scrape_tool = ScrapeWebsiteTool()
#read_resume = FileReadTool(file_path='./Resume.pdf')
#semantic_search_resume = MDXSearchTool(mdx='./Resume.pdf')

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


load_dotenv()

# Agent 1: Job Postings Finder
job_agent = Agent(
    role="Job Postings Finder",
    goal="Find relevant job postings for the given job titles and country and return structured results.",
    backstory="You are an expert job market data collector using Google Jobs API.",
    # llm=ChatOpenAI(model="gpt-4.5-turbo", temperature=0),
    tools=[GoogleJobsCollectorTool()],
    allow_delegation=False,
    verbose=True,
)

# Task 1: Find Job Postings
job_search_task = Task(
    description=(
        "Use the Google Jobs Collector Tool to retrieve {total_num_posts} job postings for these titles: {job_titles} "
        "in {country}, you can use one or more of the given job titles for your search. Ensure each posting has title/company/location and "
        "include full job description when available. Make sure that the all retrieved job postings are very close to the given job titles."
    ),
    expected_output=(
        "A JSON-like list of job postings (length = {total_num_posts}). Each item must include: "
        "title, company, location, description, job_id."
    ),
    agent=job_agent,
)

# Agent 2: Job Requirements Extractor
requirements_extractor_agent = Agent(
    role="Job Requirements Extractor",
    goal="Extract technical skills, tools, cloud platforms, and required certifications from job descriptions into strict JSON schema.",
    backstory="You are an expert Job Position analyzer, you can understand the needed requirements in a Job post. "
              "You are precise and avoid hallucinating items not present in the text. And you avoid duplications",
    verbose=True,
    allow_delegation=False,
    # llm=f"openai/{OPENAI_MODEL}",
)

# Task 2: Extract Job Requirements
job_requirements_task = Task(
    description=(
        "Analyze the retrieved job postings and extract the required technical skills, tools, cloud platforms (only 3 "
        "cloud platforms are allowed in that category: AWS 'Amazon Webs services' or GCP 'Google Cloud Platform' or Azure), "
        "and certifications into a JSON array. Each entry should correspond to a job posting and include: "
        "'job_id', 'technical_skills' (list), 'tools' (list), 'cloud_platforms' (list), 'certifications' (list). "
        "Avoid duplications and only include items explicitly mentioned in the descriptions."
    ),
    expected_output=(
        "A JSON-like list where each item has: job_id, technical_skills (list), tools (list), "
        "cloud_platforms'AWS, Azure or GCP' (list), certifications (list)."
    ),
    agent=requirements_extractor_agent,
)

# Agent 3: Job Requirements Analyzer
requirements_analyzer_agent = Agent(
    role="Job Requirements Analyzer",
    goal="Create an overview of most in demand skills and tools",
    backstory="You are an expert Job requirements analyzer, you can understand and group job requirements, "
              "you understand synonyms. e.g. ML and Machine Learning both point teh same skill"
              "You are precise and avoid hallucinating items not present in the text. And you avoid duplications",
    verbose=True,
    allow_delegation=False,
    # llm=f"openai/{OPENAI_MODEL}",
)

#Task 3: Analyze Job Requirements and create markdown Report
job_requirements_analysis_task = Task(
    description=(
        "Given the extracted job requirements from multiple job postings, Analyze the extracted job requirements and "
        "create a markdown report, providing teh following info:"
        "1- Top 15 technical skills mentioned across all job postings."
        "2- Top 15 tools mentioned across all job postings."
        "3- Top 3 cloud platforms mentioned across all job postings (only 3 platforms are allowed in that category: AWS, Azure, GCP)."
        "4- Top 5 certifications mentioned across all job postings."
        "Note: If a tool or skill appears multiple times in a single job posting, count it only once for that posting."
        "The output for each category (technical skills, tools, cloud platforms and certifications) should show a "
        "a column mentioning number of positions where the item appeared and a column mentioning % of positions where item appeared."
        "Avoid duplication of skills and avoid hallucinating items not present in the text."
    ),
    expected_output=(
        "A markdown report containing the requested info and the report should be well formatted with proper headings and tables/Charts."
    ),
    agent=requirements_analyzer_agent,
)


#TODO tell tasks/agemts



crew = Crew(
    agents=[job_agent, requirements_extractor_agent, requirements_analyzer_agent],
    tasks=[job_search_task, job_requirements_task, job_requirements_analysis_task],
    process=Process.sequential,
)


if __name__ == "__main__":
    start_time = time.time()

    result = crew.kickoff(inputs={
        "job_titles": ["AI Engineer", "Generative AI Engineer", "GenAI Engineer", "Agentic AI Engineer", "Machine Learning Engineer", "ML Engineer"],
        "country": "Egypt",
        "total_num_posts": 2,
    })
    print("\nFINAL OUTPUT:\n", result)

    crew_dict = result.model_dump(mode="json")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Save full crew output (debugging / traces)
    full_path = OUTPUT_DIR / f"crew_output_{ts}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(crew_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved full crew output: {full_path.resolve()}")

    # 2) Save each task output separately
    tasks_output = crew_dict.get("tasks_output", [])
    for i, task_out in enumerate(tasks_output, start=1):
        task_file = OUTPUT_DIR / f"task_{i}_output_{ts}.json"
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_out, f, indent=2, ensure_ascii=False)
        print(f"Saved task {i} output: {task_file.resolve()}")

    # 3) Save final output (last task) as JSON + Markdown
    final_text = None
    if tasks_output:
        last = tasks_output[-1]
        final_text = last.get("output") or last.get("raw") or last.get("result")

    final_json_path = OUTPUT_DIR / f"final_result_{ts}.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump({"final_result": final_text}, f, indent=2, ensure_ascii=False)
    print(f"Saved final JSON: {final_json_path.resolve()}")

    final_md_path = OUTPUT_DIR / f"final_result_{ts}.md"
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write(final_text or "")
    print(f"Saved final Markdown: {final_md_path.resolve()}")

    print("Execution time: ",  ((time.time() - start_time)/60), " minutes")