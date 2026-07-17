import hashlib
import re

import streamlit as st

from src.config import (
    APP_IDENTITY,
    DEFAULT_GROQ_MODEL,
    DEFAULT_RESUME_PATH,
    EMBEDDING_MODEL,
)
from src.ingestion.loader import (
    clean_text,
    extract_text_from_upload,
    load_default_resume_text,
)
from src.llm.llm_client import get_llm
from src.prompts.prompt_templates import run_text_prompt
from src.retrieval.retriever import ask_rag_question
from src.utils.skills import IMPORTANT_SKILLS, extract_known_skills
from src.vectordb.vector_store import build_vector_store


DEFAULT_RESUME_OVERVIEW = """
Vatsal Dhuvad is a Computer Engineering student focused on Data Science, Machine Learning, AI/ML, Generative AI, Agentic AI, LLM applications, LangChain, LangGraph, RAG, and Computer Vision. His profile is well suited for AI/ML internships, Data Science internships, Python Developer internships, Generative AI internships, and entry-level roles involving intelligent automation or AI-powered tools.

Strong technical skills include Python, SQL, MongoDB, Firebase, Supabase, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, OpenCV, LangChain, LangGraph, RAG, LLMs, Prompt Engineering, FAISS vector database, local Hugging Face embeddings, and Streamlit. These skills show a strong foundation in both classical machine learning and modern Generative AI application development.

Strong project highlights include Student Placement Prediction using Machine Learning, House Price Prediction, Customer Segmentation using K-Means Clustering, Real-Time Face, Eye, and Smile Detection using OpenCV, and this Agentic RAG Career Assistant project. These projects show practical experience in classification, regression, clustering, data preprocessing, feature engineering, model evaluation, computer vision, vector search, retrieval-augmented generation, and multi-agent AI workflows.

This resume is a strong fit for internship opportunities where Python, machine learning, data analysis, Generative AI, LangChain, LangGraph, RAG, LLMs, and AI agents are useful. Vatsal can contribute to projects involving resume screening systems, job matching tools, AI assistants, data-driven dashboards, computer vision tools, and intelligent automation workflows.

Recruiter-friendly pitch: Vatsal Dhuvad is a motivated Computer Engineering student with hands-on project experience in Machine Learning, Data Science, Computer Vision, and Agentic RAG applications. He is ready to contribute to AI/ML and Data Science internship projects while continuing to grow in real-world Generative AI development.
"""

DEFAULT_JOB_MATCH_DETAILS = """
### Candidate Strengths
Vatsal Dhuvad is strongest for technology-sector roles in Data Science, Machine Learning, AI/ML, Generative AI, Agentic AI, LangChain, LangGraph, RAG, Python development, and Computer Vision.

### Matching Technical Skills
Python, SQL, MongoDB, Firebase, Supabase, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, OpenCV, LangChain, LangGraph, RAG, LLMs, Prompt Engineering, FAISS vector database, local Hugging Face embeddings, and Streamlit.

### Strong Project Matches
- Student Placement Prediction using Machine Learning
- House Price Prediction using Machine Learning
- Customer Segmentation using K-Means Clustering
- Real-Time Face, Eye, and Smile Detection using OpenCV
- Agentic RAG Career Assistant using LangChain, LangGraph, FAISS, Groq, and local Hugging Face embeddings

### Best-Fit Roles
AI/ML Intern, Data Science Intern, Python Developer Intern, Generative AI Intern, Agentic AI Intern, RAG Application Intern, Computer Vision Intern, and entry-level AI application development roles.
"""

TECH_JOB_KEYWORDS = [
    "ai",
    "ml",
    "machine learning",
    "data science",
    "data scientist",
    "data analyst",
    "python",
    "sql",
    "analytics",
    "deep learning",
    "nlp",
    "llm",
    "generative ai",
    "rag",
    "langchain",
    "langgraph",
    "agentic ai",
    "computer vision",
    "opencv",
    "scikit-learn",
    "pandas",
    "numpy",
    "streamlit",
]

NON_TECH_JOB_KEYWORDS = [
    "sales",
    "marketing",
    "accounting",
    "finance",
    "hr",
    "human resource",
    "business development",
    "telecaller",
    "customer support",
    "graphic design",
]


def make_text_key(*values: str) -> str:
    joined_text = "\n---\n".join(values)
    return hashlib.sha256(joined_text.encode("utf-8", errors="ignore")).hexdigest()


def show_resume_in_sidebar(title: str, pdf_bytes: bytes | None = None, text_preview: str = "") -> None:
    st.sidebar.header("Active Resume")
    st.sidebar.caption(title)

    if pdf_bytes is None and not text_preview:
        st.sidebar.warning("No resume is loaded.")
        return

    if pdf_bytes:
        resume_button_col, skills_button_col = st.sidebar.columns(2)
        with resume_button_col:
            st.download_button(
                "Download Resume",
                data=pdf_bytes,
                file_name=title,
                mime="application/pdf",
            )
        with skills_button_col:
            st.download_button(
                "Download AI Skills",
                data="\n".join(IMPORTANT_SKILLS),
                file_name="important_ai_ml_skills.txt",
            )
        st.sidebar.success("Your resume loaded successfully.")
        st.sidebar.caption("Use Download Resume to view the PDF.")
    else:
        st.sidebar.download_button(
            "Download AI Skills",
            data="\n".join(IMPORTANT_SKILLS),
            file_name="important_ai_ml_skills.txt",
        )
        st.sidebar.text_area("Resume Preview", text_preview, height=420)


def is_identity_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(who are you|what are you|your name|introduce yourself)\b",
            question,
            re.I,
        )
    )


def is_tech_job_description(job_text: str) -> bool:
    text = job_text.lower()
    tech_hits = sum(1 for keyword in TECH_JOB_KEYWORDS if keyword in text)
    non_tech_hits = sum(1 for keyword in NON_TECH_JOB_KEYWORDS if keyword in text)
    return tech_hits > 0 and tech_hits >= non_tech_hits


def build_default_job_match_text(job_text: str) -> str:
    if is_tech_job_description(job_text):
        return """
### Job Match
Yes, this candidate matches the requirements for this AI/ML, Data Science, Generative AI, or technology-related role.

The resume shows strong alignment through Python, SQL, Machine Learning, Data Science, OpenCV, LangChain, LangGraph, RAG, LLMs, Agentic AI, Generative AI, FAISS, Streamlit, and project-based technical experience.
"""

    return """
### Job Match
Sorry, this role does not match the candidate's main field.

This candidate is strongest for technology-sector opportunities, especially Data Science, Machine Learning, AI/ML, Generative AI, Agentic AI, LangChain, LangGraph, RAG, Python development, and Computer Vision roles.
"""


def build_uploaded_resume_match_text(resume_text: str, job_text: str) -> str:
    skill_result = extract_known_skills(resume_text)
    matched_skills = skill_result.get("matched_important_skills", [])
    skill_line = ", ".join(matched_skills[:18]) if matched_skills else "technical and project-based skills"

    if is_tech_job_description(job_text):
        return f"""
### Job Match
Yes, this resume is suitable for this technology-related job description.

### Matching Skills Found
{skill_line}

### Resume Direction
This profile can be presented positively for AI/ML, Data Science, Python, Analytics, Generative AI, or technology internship opportunities.
"""

    return f"""
### Job Match
Sorry, this role does not match the candidate's main technical field.

### Best-Fit Direction
This resume is better aligned with technology-sector opportunities such as AI/ML, Data Science, Python, Analytics, Generative AI, RAG, Computer Vision, or software-related internships.

### Skills Found
{skill_line}
"""


def get_default_resume_match_line(llm, job_text: str) -> str:
    return run_text_prompt(
        llm,
        (
            "You are a strict but positive job matching assistant. "
            "Return only one short sentence. No explanation."
        ),
        f"""
Candidate profile:
Computer Engineering student with Data Science, Machine Learning, AI/ML, Generative AI, Agentic AI, LangChain, LangGraph, RAG, LLM, Python, SQL, OpenCV, FAISS, and Streamlit skills.

Job description:
{job_text}

Task:
If the job is related to technology, AI, ML, Data Science, Python, software, analytics, RAG, LLM, Generative AI, Computer Vision, or similar technical work, say:
Yes, this candidate matches your job description.

If the job is outside this technical field, say:
Sorry, this candidate does not match this job description because his skills are focused on Data Science, Machine Learning, AI, and technology roles.
""",
    )


def show_project_stack() -> None:
    st.sidebar.header("Project Stack")
    st.sidebar.write("LangChain")
    st.sidebar.write("LangGraph")
    st.sidebar.write("RAG")
    st.sidebar.write("LCEL")
    st.sidebar.write("Prompt Templates")
    st.sidebar.write("Output Parsers")
    st.sidebar.write("FAISS Vector DB")
    st.sidebar.write("Local Hugging Face Embeddings")
    st.sidebar.write("Groq LLM")
    st.sidebar.write("Streamlit")


def show_friendly_llm_error(error: Exception) -> None:
    error_text = str(error).lower()
    if "rate" in error_text or "limit" in error_text or "quota" in error_text or "429" in error_text:
        st.warning(
            "The current AI model limit was reached. If you add MISTRAL_API_KEY in .env, "
            "the app can automatically try Mistral as a fallback."
        )
    else:
        st.warning("The AI response could not be generated right now. Please try again after a moment.")


def build_positive_resume_overview(llm, resume_text: str) -> str:
    return run_text_prompt(
        llm,
        (
            "You are a positive career assistant. Highlight only strengths and suitable opportunities. "
            "Do not show scores, weak points, missing skills, or negative comments."
        ),
        f"""
Resume:
{resume_text}

Create a positive resume overview with:
1. Best professional positioning
2. Strong technical skills
3. Strong project highlights
4. Best-fit internship or job roles
5. Short recruiter-friendly pitch

Rules:
- Do not show any score.
- Do not mention weak points.
- Do not use words like weak, missing, gap, lack, poor, or not suitable.
- Keep it useful for finding better internships or jobs.
""",
    )


def main() -> None:
    st.set_page_config(
        page_title="Agentic RAG Career Assistant",
        page_icon="AI",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        [data-testid="stSidebarUserContent"] {
            padding-top: 1rem;
        }
        [data-testid="stSidebar"] iframe {
            border-radius: 8px;
        }
        .block-container {
            padding-top: 2.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Agentic RAG Career Assistant")
    st.caption(APP_IDENTITY)

    st.markdown(
        """
        This project highlights matching skills, strong project matches, best-fit opportunities,
        cover letter support, and interview preparation for the selected resume.
        """
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        resume_source = st.radio(
            "Resume source",
            ["Use default Vatsal_Dhuvad_Resume.pdf", "Upload another resume"],
        )

        uploaded_resume = None
        if resume_source == "Upload another resume":
            uploaded_resume = st.file_uploader(
                "Upload resume",
                type=["pdf", "docx", "txt"],
            )

        job_description = st.text_area(
            "Job description",
            height=230,
            placeholder="Paste internship/job description here...",
        )
        show_job_match = st.button("Show", type="primary")

    with right_col:
        st.subheader("Ask AI")
        st.caption("Ask anything about the active resume, job matching, project strengths, or interview preparation.")
        question = st.text_area(
            "Your question",
            height=150,
            placeholder="Example: What skills should I improve for this role?",
        )
        ask_clicked = st.button("Ask AI", type="primary")
        ask_answer_box = st.empty()

    resume_text = ""
    active_resume_name = "Vatsal_Dhuvad_Resume.pdf"
    active_resume_pdf_bytes = None

    if resume_source == "Use default Vatsal_Dhuvad_Resume.pdf":
        resume_text = load_default_resume_text()
        if DEFAULT_RESUME_PATH.exists():
            active_resume_pdf_bytes = DEFAULT_RESUME_PATH.read_bytes()
    elif uploaded_resume:
        active_resume_name = uploaded_resume.name
        resume_text = extract_text_from_upload(uploaded_resume)
        if uploaded_resume.name.lower().endswith(".pdf"):
            active_resume_pdf_bytes = uploaded_resume.getvalue()

    show_resume_in_sidebar(
        active_resume_name,
        pdf_bytes=active_resume_pdf_bytes,
        text_preview=resume_text,
    )

    with st.sidebar:
        st.divider()
        st.header("Model Settings")
        model_name = st.selectbox(
            "Groq model",
            [
                DEFAULT_GROQ_MODEL,
                "meta-llama/llama-4-scout-17b-16e-instruct",
            ],
            index=0,
            key="sidebar_model_after_resume",
        )
        st.write("Embedding model:", EMBEDDING_MODEL)
        st.write("Vector database: FAISS")
        st.divider()
        show_project_stack()

    job_text = clean_text(job_description)

    if not resume_text:
        st.stop()

    should_build_vector_store = resume_source != "Use default Vatsal_Dhuvad_Resume.pdf" or ask_clicked
    vector_key = make_text_key(resume_text, job_text or "general-career-assistant")
    if should_build_vector_store and st.session_state.get("vector_key") != vector_key:
        vector_store = build_vector_store(resume_text, job_text or "General AI/ML internship preparation")
        st.session_state["resume_text"] = resume_text
        st.session_state["job_description"] = job_text
        st.session_state["vector_store"] = vector_store
        st.session_state["vector_key"] = vector_key
        if not job_text:
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("final_report", None)
            st.session_state.pop("analysis_key", None)
        if resume_source == "Use default Vatsal_Dhuvad_Resume.pdf":
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("final_report", None)
            st.session_state.pop("analysis_key", None)

    if ask_clicked:
        if not question.strip():
            with ask_answer_box.container():
                st.warning("Please enter a question.")
        elif is_identity_question(question):
            with ask_answer_box.container():
                st.write(APP_IDENTITY)
        else:
            with ask_answer_box.container():
                with st.spinner("Thinking..."):
                    llm = get_llm(model_name)
                    try:
                        if "vector_store" not in st.session_state:
                            st.session_state["vector_store"] = build_vector_store(
                                resume_text,
                                job_text or "General AI/ML internship preparation",
                            )
                        answer = ask_rag_question(llm, st.session_state["vector_store"], question)
                        st.write(answer)
                    except Exception as error:
                        show_friendly_llm_error(error)

    if resume_source == "Use default Vatsal_Dhuvad_Resume.pdf":
        st.session_state["positive_overview"] = DEFAULT_RESUME_OVERVIEW
        st.session_state["overview_key"] = "default-saved-overview"
    else:
        overview_key = make_text_key(resume_text, model_name, "positive-overview")
        if st.session_state.get("overview_key") != overview_key:
            try:
                llm = get_llm(model_name)
                st.session_state["positive_overview"] = build_positive_resume_overview(llm, resume_text)
                st.session_state["overview_key"] = overview_key
            except Exception as error:
                show_friendly_llm_error(error)

    analysis_key = make_text_key(resume_text, job_text, model_name)
    if show_job_match:
        if not job_text:
            st.warning("Please enter a job description first.")
        elif st.session_state.get("analysis_key") != analysis_key:
            if resume_source == "Use default Vatsal_Dhuvad_Resume.pdf":
                try:
                    llm = get_llm(model_name)
                    match_line = get_default_resume_match_line(llm, job_text)
                except Exception as error:
                    show_friendly_llm_error(error)
                    match_line = ""

                if match_line:
                    st.session_state["default_job_result"] = (
                        f"### Job Match\n{match_line}\n\n{DEFAULT_JOB_MATCH_DETAILS}"
                    )
                    st.session_state["analysis_key"] = analysis_key
                    st.session_state.pop("analysis_result", None)
                    st.session_state.pop("final_report", None)
            else:
                st.session_state["default_job_result"] = build_uploaded_resume_match_text(resume_text, job_text)
                st.session_state["analysis_key"] = analysis_key
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("final_report", None)

    if not job_text and "positive_overview" in st.session_state:
        st.subheader("Resume Overview")
        st.markdown(st.session_state["positive_overview"])

    if (
        "default_job_result" in st.session_state
        and job_text
        and st.session_state.get("analysis_key") == analysis_key
    ):
        st.subheader("Job Match")
        st.markdown(st.session_state["default_job_result"])

    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]

        tab1, tab2, tab3 = st.tabs(
            [
                "Job Match",
                "Interview Prep",
                "Cover Letter",
            ]
        )

        with tab1:
            st.subheader("Positive Job Match")
            st.markdown(result["match_report"])

        with tab2:
            st.subheader("Interview Preparation Questions")
            st.markdown(result["interview_questions"])

        with tab3:
            st.subheader("Generated Cover Letter")
            st.markdown(result["cover_letter"])

        st.download_button(
            "Download Full Career Report",
            data=st.session_state["final_report"],
            file_name="agentic_rag_career_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
