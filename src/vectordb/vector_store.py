from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.chunking.chunker import split_documents
from src.embeddings.embedder import get_embedding_model
from src.ingestion.loader import (
    build_portfolio_first_impression,
    build_portfolio_skills_summary,
    build_portfolio_summary,
)
from src.utils.skills import IMPORTANT_SKILLS


def build_vector_store(resume_text: str, job_description: str, portfolio_text: str = ""):
    documents = [
        Document(page_content=resume_text, metadata={"source": "resume"}),
        Document(page_content=job_description, metadata={"source": "job_description"}),
        Document(
            page_content="Important target skills: " + ", ".join(IMPORTANT_SKILLS),
            metadata={"source": "target_skills"},
        ),
    ]

    if portfolio_text.strip():
        first_impression = build_portfolio_first_impression(portfolio_text)
        if first_impression:
            documents.append(
                Document(
                    page_content=first_impression,
                    metadata={"source": "portfolio_first_impression"},
                )
            )
        portfolio_summary = build_portfolio_summary(portfolio_text)
        if portfolio_summary:
            documents.append(
                Document(
                    page_content=portfolio_summary,
                    metadata={"source": "portfolio_projects"},
                )
            )
        portfolio_skills = build_portfolio_skills_summary(portfolio_text)
        if portfolio_skills:
            documents.append(
                Document(
                    page_content=portfolio_skills,
                    metadata={"source": "portfolio_skills"},
                )
            )
        documents.append(
            Document(page_content=portfolio_text, metadata={"source": "portfolio"})
        )

    chunks = split_documents(documents)
    return FAISS.from_documents(chunks, get_embedding_model())
