from crewai.tools import BaseTool
from pathlib import Path
from PyPDF2 import PdfReader

class ResumePDFTextTool(BaseTool):
    name: str = "Resume PDF Text Extractor"
    description: str = "Extracts readable text from Resume.pdf and returns it as a single string."

    def _run(self, file_path: str = "Resume.pdf") -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume file not found: {path.resolve()}")

        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")

        text = "\n".join(pages).strip()
        if not text:
            raise ValueError(
                "No text could be extracted from the PDF. "
                "The resume might be scanned images (needs OCR)."
            )
        return text

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
