IMPORTANT_SKILLS = [
    "Python",
    "SQL",
    "Data Science",
    "Machine Learning",
    "AI/ML",
    "Generative AI",
    "LLM",
    "LangChain",
    "LangGraph",
    "RAG",
    "Agentic AI",
    "Prompt Engineering",
    "Vector Database",
    "FAISS",
    "Hugging Face Embeddings",
    "NLP",
    "Text Preprocessing",
    "Resume Screening",
    "Job Matching",
    "ATS Optimization",
    "Skill Gap Analysis",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "OpenCV",
    "Streamlit",
    "LCEL",
    "Prompt Templates",
    "Output Parsers",
    "Document Loaders",
    "Text Splitters",
    "Retrievers",
    "Multi-Agent Workflow",
    "GitHub",
]


def extract_known_skills(text: str) -> dict:
    text_lower = text.lower()
    matched = []
    missing = []

    for skill in IMPORTANT_SKILLS:
        if skill.lower() in text_lower:
            matched.append(skill)
        else:
            missing.append(skill)

    return {
        "matched_important_skills": matched,
        "missing_important_skills": missing,
        "matched_count": len(matched),
        "missing_count": len(missing),
    }

