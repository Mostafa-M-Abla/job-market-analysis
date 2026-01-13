from crewai import Agent, Task, Crew, Process
from tools.google_jobs_tool import GoogleJobsCollectorTool
from tools.resume_pdf_tool import ResumePDFTextTool
from tools.save_html_tool import SaveHTMLTool


def build_muti_agent_crew():
    """
    Function that builds and returns the full CrewAI multi-agent system for job market analysis.

    This crew includes agents for:
    - Finding job listings
    - Extracting job requirements
    - Analyzing required skills
    - Extracting resume skills
    - Suggesting improvements
    - Writing final HTML reports

    Each agent performs one task and passes structured data to the next step.
    The process is run sequentially.
    """

    # Agent 1: Job Postings Finder
    job_postings_finder_agent = Agent(
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
            "include full job description and requirements when available. Make sure that the all retrieved job postings are very close to the given job titles."
        ),
        expected_output=(
            "A JSON-like list of job postings (length = {total_num_posts}). Each item must include: "
            "title, company, location, description, job_id."
        ),
        agent=job_postings_finder_agent,
    )

    # Agent 2: Job Requirements Extractor
    requirements_extractor_agent = Agent(
        role="Job Requirements Extractor",
        goal="Extract technical skills, tools, cloud platforms, and required certifications from job descriptions into strict JSON schema.",
        backstory="You are an expert Job Position analyzer, you can understand the needed requirements in a Job post. "
                  "You are precise and avoid hallucinating items not present in the text. And you avoid duplications",
        verbose=True,
        allow_delegation=False,
    )

    # Task 2: Extract Job Requirements
    job_requirements_task = Task(
        description=(
            "Analyze the retrieved job postings and extract the required technical skills, tools, cloud platforms (only 3 "
            "cloud platforms are allowed in that category: AWS 'Amazon Webs services' or GCP 'Google Cloud Platform' or Azure), "
            "and certifications into a JSON array. Each entry should correspond to a job posting and include: "
            "'job_id', 'technical_skills and tools' (list), 'cloud_platforms' (list), 'certifications' (list). "
            "Avoid duplications and only include items explicitly mentioned in the descriptions."
        ),
        expected_output=(
            "A JSON-like list where each item has: job_id, technical_skills and tools (list), "
            "cloud_platforms'AWS, Azure or GCP' (list), certifications (list)."
        ),
        agent=requirements_extractor_agent,
    )

    # Agent 3: Job Requirements Analyzer
    requirements_analyzer_agent = Agent(
        role="Job Requirements Analyzer",
        goal="Create an overview of most in demand requirements",
        backstory="You are an expert Job requirements analyzer, you can understand and group job requirements, "
                  "you understand synonyms. e.g. ML and Machine Learning both point the same skill"
                  "You are precise and avoid hallucinating items not present in the text. And you avoid duplications",
        verbose=True,
        allow_delegation=False,
        llm=f"openai/gpt-5.2",
    )

    # Task 3: Analyze Job Requirements and create markdown Report
    job_requirements_analysis_task = Task(
        description=(
            "Given the extracted job requirements from multiple job postings, Analyze the extracted job requirements and "
            "create a markdown report, providing the following info:"
            "1- All technical skills and tools mentioned across all job postings."
            "2- Top 3 cloud platforms mentioned across all job postings (only 3 platforms are allowed in that category: AWS, Azure, GCP)."
            "3- Top 5 certifications mentioned across all job postings."
            "Note: If a tool or skill appears multiple times in a single job posting, count it only once for that posting."
            "The output for each category (technical skills and tools, cloud platforms and certifications) should show a "
            "a column mentioning number of positions where the item appeared and a column mentioning % of positions where item appeared."
            "Avoid duplication of skills and avoid hallucinating items not present in the text. "
            "In the final answer group skills and tools that has the same meaning together to avoid duplication e.g. CI/CD and CI /CD are teh same and should be listed only once."
        ),
        expected_output=(
            "A markdown report containing the requested info and the report should be well formatted with proper headings and tables/Charts."
        ),
        agent=requirements_analyzer_agent,
    )

    # Agent 4: Resume Extractor
    resume_skills_extractor_agent = Agent(
        role="Resume Content Extractor",
        goal="Extract Technical skills and tools, certifications and other points of strength of a candidate from a given Resume",
        backstory="You are an expert Skills extractor you extract all info from a Resume that can be useful for the recruiters decisoion to hire the candiaate or no",
        verbose=True,
        allow_delegation=False,
        tools=[ResumePDFTextTool()],
        # llm=f"openai/{OPENAI_MODEL}",
    )

    # Task 4: Resume Skills Extraction
    resume_skills_extraction_task = Task(
        description=(
            "Use the ResumePDFTextTool tool to read the given Resume pdf file."
            "Extract all technical skills and tools, certifications, cloud platforms (AWS, Azure, GCP) and other points of "
            "strength of the candidate from the Resume pdf file."
        ),
        expected_output=(
            "A JSON-like output of extracted items from the resume document: technical_skills and tools (list), cloud_platforms'AWS, Azure or GCP' (list), certifications (list)."
        ),
        agent=resume_skills_extractor_agent,
        context=[],
    )

    # Agent 5: Resume booster agent
    resume_booster_agent = Agent(
        role="Resume Booster Specialist",
        goal="Suggest Technical skills and tools, certifications and cloud platforms for the user that will boost "
             "his Resume for a given job title",
        backstory="You are an expert Career coach, you have knowledge of most needed Job requirements for certain job "
                  "title/s and also you have knowledge of the candidate skills and tool set. Based on this data suggest "
                  "the skill and tools that the user should attain to better suit the job market."
                  "You don't hallucinate or make up data, you use the data attained from previous tasks to suggest the needed skills and tools "
                  "You avoid duplication and you understand synonyms of skills and tools e.g. ML and Machine learning are the same.",
        verbose=True,
        allow_delegation=False,
        llm=f"openai/gpt-5.2",
    )

    # Task 5: Resume boost task
    resume_boost_task = Task(
        description=(
            "Using the info about most needed technical skills and tools, cloud platforms, certifications about the target job "
            "title from the job_requirements_analysis_task and using the user's technical skills and tools, "
            "cloud platforms, certifications and other points given by the previous resume_skills_extraction_task, suggest "
            "the top 5 technical skills and tools that the user should learn next to further boost his/her Resume and have a better chance"
            "in the job market (i.e. check what the user already has and suggest the items he/she needs to acquire that are most needed). "
            "Also list in what percentage of the job postings scanned was each of those suggested 5 skills/tools mentioned."
            "The wording should be positive, you are providing next step improvements for the user!"
        ),
        expected_output=(
            "A markdown report containing the requested info and the report should be well formatted with proper headings and tables/Charts."
        ),
        agent=resume_booster_agent,
        context=[job_requirements_analysis_task, resume_skills_extraction_task],
    )

    # Agent 6: html repot Creator agent
    report_writer_agent = Agent(
        role="HTML Reports Writer Specialist",
        goal="Create an informative visually appealing HTML Report",
        backstory="You are an expert Reports writer, You use available data to create information rich, "
                  "good looking and visually appealing HTML reports"
                  "You don't hallucinate or make up data, you use the data attained from previous tasks to write the reports "
                  "You avoid duplication and you understand synonyms of skills and tools e.g. ML and Machine learning are the same.",
        verbose=True,
        allow_delegation=False,
        tools=[SaveHTMLTool()],
        llm=f"openai/gpt-5.2",
    )

    # Task 6: Report writing task
    report_writing_task = Task(
        description=(
            "Combine the outputs from the job market analysis and resume boost tasks into a single, informative, and visually appealing HTML report. "
            "The report should have the title 'Job Market Analysis and Resume Boost Report'. Directly underneath, list: "
            "- Target Job Titles: {job_titles}  \n"
            "- Country: {country}  \n"
            "- Number of job listings analyzed: {total_num_posts}  \n\n"
        
            "Then include the following sections:\n"
            "1. **Job Market Analysis** — summarize the top 20 most in-demand technical skills, tools, cloud platforms, and certifications using charts or tables.\n"
            "2. **Resume Boosting Suggestions** — briefly explain that this section compares the user’s resume with the market needs and presents top 5 suggested skills/tools to learn next. Include frequency stats.\n\n"
        
            "Style Requirements:\n"
            "- Use polite, concise, and encouraging tone.\n"
            "- Use colorful, light-themed HTML with professional formatting.\n"
            "- Include visually attractive tables and charts where appropriate.\n"
            "- Avoid hallucinations and duplications.\n"
            "- Avoid repeating the same data across tables and charts.\n"
            "- Do not suggest how to phrase resume content; only present insights.\n\n"
            
            "Use SaveHTMLTool to save the final report to disk."
        ),
        expected_output=(
            "A complete, professionally formatted HTML string containing the combined report. "
            "It must include the report title, meta info (job titles, country, number of listings), "
            "two structured sections (Job Market Analysis and Resume Boosting Suggestions), and use light-themed design with clean tables and charts. "
            "The HTML should be ready to be saved to disk using SaveHTMLTool and rendered well in modern browsers."
        ),
        agent=report_writer_agent,
        context=[job_requirements_analysis_task, resume_boost_task],
    )

    crew = Crew(
        agents=[job_postings_finder_agent, requirements_extractor_agent, requirements_analyzer_agent,
                resume_skills_extractor_agent, resume_booster_agent, report_writer_agent],
        tasks=[job_search_task, job_requirements_task, job_requirements_analysis_task, resume_skills_extraction_task,
               resume_boost_task, report_writing_task],
        process=Process.sequential,
    )

    return crew