import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime

import json
from pathlib import Path
from dotenv import load_dotenv
from build_crew import build_muti_agent_crew

load_dotenv()


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)



def format_output(crew_result):
    """
    Formats and saves the output of a CrewAI execution run.

    - Serializes the full result object (crew_result) as a JSON trace for debugging.
    - Extracts the final task's output (last agent's result) and saves it separately:
        - As a JSON file (for programmatic consumption)
        - As a Markdown file (for readable reports)
    - All output files are timestamped and saved in the `outputs/` directory.

    Args:
        crew_result: The result object returned by `crew.kickoff()`, containing task outputs.

    Returns:
        None. Side effect: Writes JSON and Markdown files to disk.
        None. Side effect: Writes JSON and Markdown files to disk.
    """

    crew_dict = crew_result.model_dump(mode="json")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full crew output (debugging / traces)
    full_path = OUTPUT_DIR / f"crew_output_{ts}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(crew_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved full crew output: {full_path.resolve()}")

    # Save each task output separately
    tasks_output = crew_dict.get("tasks_output", [])

    # Save final output (last task) as JSON + Markdown
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


if __name__ == "__main__":
    """
    Main entry point for the CrewAI multi-agent job market analysis system.

    Loads inputs (job titles, country), builds the agent crew, and kicks off
    the full workflow to produce a final HTML report comparing job requirements
    with resume strengths and improvement suggestions.
    """

    start_time = time.time()

    crew = build_muti_agent_crew()

    result = crew.kickoff(inputs={
        "job_titles": ["AI Engineer", "Generative AI Engineer", "GenAI Engineer", "Agentic AI Engineer", "AI Developer", "AI Team Leader", "Machine Learning Engineer", "ML Engineer"],
        "country": "Egypt",
        "total_num_posts": 30,
    })
    print("\nFINAL OUTPUT:\n", result)

    format_output(result)

    print("Execution time: ",  ((time.time() - start_time)/60), " minutes")