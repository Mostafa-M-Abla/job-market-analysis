# download_page_tool.py

from crewai.tools import BaseTool
import requests
import os
from urllib.parse import urlparse
from bs4 import BeautifulSoup

class DownloadPageTool(BaseTool):
    name: str= "DownloadPageTool"
    description: str = "Downloads HTML content from a job listing URL and saves it locally."

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            parsed_url = urlparse(url)
            filename = parsed_url.netloc.replace('.', '_') + "_" + os.path.basename(parsed_url.path).replace('/', '_') + ".html"
            save_path = os.path.join("downloads", filename)

            os.makedirs("downloads", exist_ok=True)

            soup = BeautifulSoup(response.text, "html.parser")
            pretty_html = soup.prettify()

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(pretty_html)

            return f"Page saved at {save_path}"
        except Exception as e:
            return f"Failed to download {url}: {str(e)}"

