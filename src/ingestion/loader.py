import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

from src.config import DEFAULT_RESUME_PATH, PORTFOLIO_URL


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


def load_portfolio_text(url: str = PORTFOLIO_URL) -> str:
    os.environ.setdefault(
        "USER_AGENT",
        "AgenticRAGCareerAssistant/1.0 portfolio loader by Vatsal Dhuvad",
    )

    try:
        from langchain_community.document_loaders import WebBaseLoader

        loader = WebBaseLoader(
            web_paths=[url],
            header_template={
                "User-Agent": (
                    "Mozilla/5.0 AgenticRAGCareerAssistant/1.0 "
                    "(portfolio loader for career assistant project)"
                )
            },
        )
        docs = loader.load()
    except Exception:
        docs = []

    page_text = "\n\n".join(doc.page_content for doc in docs)
    page_text = clean_text(page_text)
    if len(page_text) > 100:
        return page_text

    return load_portfolio_text_from_static_site(url)


def load_portfolio_text_from_static_site(url: str) -> str:
    try:
        import requests

        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": os.environ["USER_AGENT"]},
        )
        response.raise_for_status()
    except Exception:
        return ""

    html_text = response.text
    text_parts = []

    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.I | re.S)
    if title_match:
        text_parts.append(clean_html_text(title_match.group(1)))

    for content in re.findall(r'<meta[^>]+content=["\']([^"\']+)["\']', html_text, re.I):
        if content:
            text_parts.append(clean_html_text(content))

    script_sources = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html_text, re.I)
    for script_src in script_sources:
        script_text = load_script_text(urljoin(url, script_src))
        if script_text:
            text_parts.append(script_text)

    return clean_text("\n".join(dict.fromkeys(text_parts)))


def clean_html_text(text: str) -> str:
    import html

    text = re.sub(r"<[^>]+>", " ", text or "")
    return clean_text(html.unescape(text))


def load_script_text(script_url: str, visited: set[str] | None = None) -> str:
    visited = visited or set()
    if script_url in visited or len(visited) > 8:
        return ""
    visited.add(script_url)

    try:
        import html
        import requests

        response = requests.get(
            script_url,
            timeout=20,
            headers={"User-Agent": os.environ["USER_AGENT"]},
        )
        response.raise_for_status()
    except Exception:
        return ""

    script_text = response.text
    nested_text_parts = []
    nested_assets = re.findall(r'["\']([^"\']*assets/[^"\']+\.js)["\']', script_text)
    parsed_url = urlparse(script_url)
    site_root = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    for asset_path in nested_assets:
        nested_text = load_script_text(urljoin(site_root, asset_path), visited)
        if nested_text:
            nested_text_parts.append(nested_text)

    useful_lines = []
    field_values = re.findall(
        r"(?:name|title|description|bio|heroBio|email|phone|location|category|github|linkedin|demo):\"([^\"]{3,700})\"",
        script_text,
    )
    array_values = re.findall(r'"([^"]{3,120})"', script_text)
    strings = [(value, "") for value in field_values + array_values]
    important_words = (
        "vatsal",
        "portfolio",
        "project",
        "skill",
        "python",
        "machine",
        "learning",
        "data",
        "science",
        "engineer",
        "github",
        "linkedin",
        "resume",
        "certification",
        "education",
        "house",
        "price",
        "predictor",
        "cnn",
        "classification",
        "forecasting",
        "dashboard",
        "recommendation",
        "tensorflow",
        "xgboost",
        "power bi",
    )

    for double_quoted, single_quoted in strings:
        value = html.unescape(double_quoted or single_quoted)
        value = value.replace("\\n", " ").replace("\\t", " ")
        value = re.sub(r"\\u[0-9a-fA-F]{4}", " ", value)
        value = clean_text(value)

        if not value or len(value) < 5:
            continue
        if not any(word in value.lower() for word in important_words):
            continue
        if re.search(r"[{}<>=]{2,}|function|=>|className|children|props", value):
            continue

        useful_lines.append(value)

    return "\n".join(dict.fromkeys(useful_lines + nested_text_parts))
