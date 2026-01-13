from crewai.tools import BaseTool
import os
from datetime import datetime

class SaveHTMLTool(BaseTool):
    """
    Tool for saving generated HTML reports to the local filesystem.

    Takes an HTML string and writes it to a timestamped file under the `outputs/` directory.
    Returns the full path to the saved file.
    """

    name: str = "SaveHTMLTool"
    description: str = "Saves a given HTML string to a local file and returns the path."

    def _run(self, html_text: str) -> str:
        try:
            os.makedirs("../outputs", exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"job_market_report_{timestamp}.html"
            path = os.path.join("../outputs", safe_filename)

            with open(path, "w", encoding="utf-8") as f:
                f.write(html_text)

            return f"HTML saved at {path}"

        except Exception as e:
            return f"Failed to save HTML: {e}"

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
