import warnings
warnings.filterwarnings("ignore")

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


load_dotenv()


print("SERPAPI_API_KEY present?", bool(os.getenv("SERPAPI_API_KEY")))
print("SERPAPI_API_KEY starts with:", (os.getenv("SERPAPI_API_KEY") or "")[:6])


# Agent
job_agent = Agent(
    role="Job Postings Finder",
    goal="Find relevant job postings for the given job titles and country and return structured results.",
    backstory="You are an expert job market data collector using Google Jobs API.",
    tools=[GoogleJobsCollectorTool()],  # IMPORTANT: instance
    verbose=True,
)

# Task
job_search_task = Task(
    description=(
        "Use the Google Jobs Collector Tool to retrieve {total_num_posts} job postings for these titles: {job_titles} "
        "in {country}. Ensure each posting has title/company/location and include full job description when available."
    ),
    expected_output=(
        "A JSON-like list of job postings (length = {total_num_posts}). Each item must include: "
        "title, company, location, description, job_id."
    ),
    agent=job_agent,
)

crew = Crew(
    agents=[job_agent],
    tasks=[job_search_task],
    process=Process.sequential,
)

# "job_titles": "AI Engineer, Generative AI Engineer, GenAI Engineer, Agentic AI Engineer",


if __name__ == "__main__":
    # Use list for job_titles (not a single string)
    result = crew.kickoff(inputs={
        "job_titles": ["AI Engineer"],
        "country": "Egypt",
        "total_num_posts": 2,
    })
    print("\nFINAL OUTPUT:\n", result)














#
# # Agent 1:
# job_agent = Agent(
#     role="Job Postings Finder and Verifier",
#     goal="Do amazing Job in searching the internet and find Job listings relevant to the search.",
#     backstory=(
#         "You are an expert in online job hunting. You know how to find job listings, "
#         "extract and validate them, and ensure they match the specified job role."
#     ),
#     tools=[SerperDevTool(),DownloadPageTool()],
#     verbose=True
# )
#
# # Task
# job_search_task = Task(
#     description=(
#         "Search Google for 20 job listings matching the title: {job_title}. "
#         "For each URL, download the HTML using your tool. "
#         "Analyze the content and confirm if it's a real job posting related to the title. "
#         "Only keep listings that are valid and relevant. "
#         "Save the valid HTML pages locally and return a summary list with:\n"
#         "- URL\n"
#         "- Page title (cleaned)\n"
#         "- Local HTML file path"
#     ),
#     expected_output=(
#         "A list of 20 validated job listings. Each entry must include:\n"
#         "- The original job listing URL\n"
#         "- The page title or identified job title\n"
#         "- Path to the saved HTML file"
#     ),
#     agent=job_agent
# )
#
# # Crew setup
# crew = Crew(
#     agents=[job_agent],
#     tasks=[job_search_task],
#     process=Process.sequential
# )
#
# # "job_titles": "AI Engineer, Generative AI Engineer, GenAI Engineer, Agentic AI Engineer",
#
# # Run the crew
# if __name__ == "__main__":
#     result = crew.kickoff(inputs={
#         "job_title": "AI Engineer",
#         "country": "Egypt",
#         "total_num_posts": 2,
#     })
#     print("\nFINAL OUTPUT:\n", result)
#




#
# # Agent 1: Researcher
# researcher = Agent(
#     role="Job Postings Collector",
#     goal="Make sure to do amazing analysis on "
#          "job posting to help job applicants",
#     tools = [scrape_tool, search_tool],
#     verbose=True,
#     backstory=(
#         "As a Job Researcher, your prowess in "
#         "navigating and extracting critical "
#         "information from job postings is unmatched."
#         "Your skills help pinpoint the necessary "
#         "qualifications and skills sought "
#         "by employers, forming the foundation for "
#         "effective application tailoring."
#     )
# )
#
#
# # Agent 2: Profiler
# profiler = Agent(
#     role="Personal Profiler for Engineers",
#     goal="Do increditble research on job applicants "
#          "to help them stand out in the job market",
#     tools = [scrape_tool, search_tool,
#              read_resume, semantic_search_resume],
#     verbose=True,
#     backstory=(
#         "Equipped with analytical prowess, you dissect "
#         "and synthesize information "
#         "from diverse sources to craft comprehensive "
#         "personal and professional profiles, laying the "
#         "groundwork for personalized resume enhancements."
#     )
# )
#
#
# # Task 1: Task for Researcher Agent: Extract Job Requirements
# research_task = Task(
#     description=(
#         "Analyze the job posting URL provided ({job_posting_url}) "
#         "to extract key skills, experiences, and qualifications "
#         "required. Use the tools to gather content and identify "
#         "and categorize the requirements."
#     ),
#     expected_output=(
#         "A structured list of job requirements, including necessary "
#         "skills, qualifications, and experiences."
#     ),
#     agent=researcher,
#     async_execution=True
# )
#
#
#
# # Task2: Task for Profiler Agent: Compile Comprehensive Profile
# profile_task = Task(
#     description=(
#         "Compile a detailed personal and professional profile "
#         "using the GitHub ({github_url}) URLs, and personal write-up "
#         "({personal_writeup}). Utilize tools to extract and "
#         "synthesize information from these sources."
#     ),
#     expected_output=(
#         "A comprehensive profile document that includes skills, "
#         "project experiences, contributions, interests, and "
#         "communication style."
#     ),
#     agent=profiler,
#     async_execution=True
# )
#
