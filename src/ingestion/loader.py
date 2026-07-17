import os
import re
import tempfile
from pathlib import Path

from src.config import DEFAULT_RESUME_PATH


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(file_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def extract_text_from_docx(file_path: str) -> str:
    from docx import Document

    doc = Document(file_path)
    text_parts = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text.strip())

    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = clean_text("\n".join(p.text for p in cell.paragraphs))
                if cell_text:
                    row_text.append(cell_text)
            if row_text:
                text_parts.append(" | ".join(row_text))

    for section in doc.sections:
        for paragraph in section.header.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())
        for paragraph in section.footer.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())

    return clean_text("\n".join(text_parts))


def extract_text_from_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        if suffix == ".pdf":
            return extract_text_from_pdf(temp_path)
        if suffix == ".docx":
            return extract_text_from_docx(temp_path)
        if suffix == ".txt":
            return clean_text(Path(temp_path).read_text(encoding="utf-8", errors="ignore"))
        raise ValueError("Please upload PDF, DOCX, or TXT.")
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def load_default_resume_text() -> str:
    if not DEFAULT_RESUME_PATH.exists():
        return ""
    return extract_text_from_pdf(str(DEFAULT_RESUME_PATH))
