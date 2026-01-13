import os
import dotenv
from langsmith import traceable, Client
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

"""
evaluate_html_report.py

This script evaluates the quality of a generated HTML report produced by the multi-agent job market analysis system.
It uses an LLM (e.g., GPT-4) to score the report on key criteria such as relevance, accuracy, completeness, clarity,
visual appeal, and insightfulness. Each criterion is rated from 1 to 5, with a final composite score out of 10.

The evaluation is performed using a LangChain Runnable chain and is fully traceable via LangSmith.
LangSmith provides full visibility into the prompt, LLM reasoning, and evaluation outputs.

Example output:
{
    "relevance": 5,
    "accuracy": 4,
    "completeness": 5,
    "clarity": 4,
    "visual_appeal": 5,
    "insights": 5,
    "final_score": 9,
    "comments": "Great structure and helpful suggestions. Could add more visual distinction between sections."
}
"""

# Load .env vars
dotenv.load_dotenv()

# Check required keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY in .env")
if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("Missing LANGSMITH_API_KEY in .env")

# Report path
OUTPUT_HTML_PATH = "../outputs/job_market_report_20260113_133015.html"
with open(OUTPUT_HTML_PATH, "r", encoding="utf-8") as f:
    html_content = f.read()

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert evaluator of job market analysis reports."),
    ("user", """
Evaluate the following HTML report on these criteria, and return JSON with keys:
relevance, accuracy, completeness, clarity, visual_appeal, insights, and final_score (out of 10).
Each is scored out of 5. Add comments if needed.

Criteria:
1. Relevance: Does it match job title + country?
2. Accuracy: Are the mentioned skills actually from job posts?
3. Completeness: All required sections present?
4. Clarity: Is the writing clear?
5. Visual Appeal: Is it styled well?
6. Insights: Are resume suggestions meaningful?

Format:
{{
  "relevance": <int>,
  "accuracy": <int>,
  "completeness": <int>,
  "clarity": <int>,
  "visual_appeal": <int>,
  "insights": <int>,
  "final_score": <int>,
  "comments": "..."
}}

Report HTML:
---------------------
{html}
""")
]).partial(html=html_content)

# LLM
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

# Parser
parser = JsonOutputParser()

# Chain with LangSmith-traceable wrapping
@traceable(name="Evaluate Job Market HTML Report")
def evaluate_html_report():
    chain: Runnable = prompt | llm | parser
    result = chain.invoke({})
    return result

# Entry point
if __name__ == "__main__":
    result = evaluate_html_report()
    print("\n✅ LangSmith Evaluation Summary:")
    for k, v in result.items():
        print(f"{k}: {v}")
