from typing import TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from src.prompts.prompt_templates import run_json_prompt, run_text_prompt
from src.retrieval.retriever import retrieve_context
from src.utils.skills import IMPORTANT_SKILLS, extract_known_skills
from src.vectordb.vector_store import build_vector_store


class CareerState(TypedDict):
    resume_text: str
    job_description: str
    retrieved_context: str
    skill_keywords_json: str
    match_report: str
    skill_gap_report: str
    ats_report: str
    cover_letter: str
    interview_questions: str
    final_report: str


skill_extractor_runnable = RunnableLambda(extract_known_skills)


def create_career_workflow(llm):
    def retrieval_agent(state: CareerState) -> CareerState:
        vector_store = build_vector_store(state["resume_text"], state["job_description"])
        state["retrieved_context"] = retrieve_context(
            vector_store,
            "resume strengths matching skills project matches job requirements interview preparation",
        )
        return state

    def skill_keyword_agent(state: CareerState) -> CareerState:
        local_result = skill_extractor_runnable.invoke(
            state["resume_text"] + " " + state["job_description"]
        )
        matched_local_result = {
            "matched_important_skills": local_result.get("matched_important_skills", []),
            "matched_count": local_result.get("matched_count", 0),
        }

        llm_result = run_json_prompt(
            llm,
            "You are an AI Skill Extraction Agent. Only identify skills and strengths that are already present or clearly supported by the resume. Do not mention the response style.",
            f"""
Resume:
{state["resume_text"]}

Job Description:
{state["job_description"]}

Extract JSON with these keys:
- core_ai_skills
- programming_skills
- data_science_skills
- rag_agentic_ai_skills
- strongest_resume_keywords
- role_keywords
""",
        )

        state["skill_keywords_json"] = (
            "Matched skill extractor:\n"
            f"{matched_local_result}\n\n"
            "Structured skill extractor:\n"
            f"{llm_result}"
        )
        return state

    def job_match_agent(state: CareerState) -> CareerState:
        state["match_report"] = run_text_prompt(
            llm,
            (
                "You are a Job Match Agent. Show only helpful, confident, and opportunity-focused points. "
                "Do not provide an overall score. Do not mention weak areas, gaps, missing skills, negatives, or limitations. "
                "Do not mention that the answer is positive, supportive, or strength-focused."
            ),
            f"""
Resume:
{state["resume_text"]}

Job Description:
{state["job_description"]}

Retrieved Context:
{state["retrieved_context"]}

Structured Skill Keywords:
{state["skill_keywords_json"]}

Give:
1. Fit summary
2. Matching skills
3. Strong project matches
4. Best-fit internship or job roles
5. Short recruiter-friendly application pitch

Rules:
- Do not show any score.
- Do not show weak points.
- Do not show negative comments.
- Do not use words like weak, missing, gap, lack, poor, low, or not suitable.
- Keep the response useful for finding better internships or jobs.
""",
        )
        return state

    def skill_gap_agent(state: CareerState) -> CareerState:
        state["skill_gap_report"] = run_text_prompt(
            llm,
            (
                "You are a Career Growth Agent. Recommend only helpful next-step skills in a motivating way. "
                "Do not describe the resume negatively. Do not mention the response style."
            ),
            f"""
Resume:
{state["resume_text"]}

Job Description:
{state["job_description"]}

Important Skills:
{", ".join(IMPORTANT_SKILLS)}

Structured Skill Keywords:
{state["skill_keywords_json"]}

Give:
1. Current skill strengths
2. High-value skills to learn next
3. Learning roadmap
4. Best 5 keywords to add only after the user builds or learns them

Rules:
- Do not use negative wording.
- Do not say missing, weak, lack, or poor.
- Make it encouraging and internship-focused.
""",
        )
        return state

    def ats_optimizer_agent(state: CareerState) -> CareerState:
        state["ats_report"] = run_text_prompt(
            llm,
            (
                "You are a Resume Strengthening Agent. Suggest only constructive improvements that make the resume stronger. "
                "Do not provide scores or negative comments. Do not mention the response style."
            ),
            f"""
Resume:
{state["resume_text"]}

Job Description:
{state["job_description"]}

Reports:
{state["match_report"]}
{state["skill_gap_report"]}

Structured Skill Keywords:
{state["skill_keywords_json"]}

Give:
1. Strong resume summary suggestion
2. Stronger skills section suggestion
3. Stronger project bullet suggestions
4. Keywords to include naturally

Rules:
- Do not show ATS score.
- Do not mention problems or weaknesses.
- Keep all suggestions constructive and honest.
""",
        )
        return state

    def cover_letter_agent(state: CareerState) -> CareerState:
        state["cover_letter"] = run_text_prompt(
            llm,
            "You are a Cover Letter Agent. Write a short internship cover letter based only on the provided resume and job description.",
            f"""
Resume:
{state["resume_text"]}

Job Description:
{state["job_description"]}

Write a concise cover letter for an internship application.
Avoid overclaiming.
Use simple professional English.
""",
        )
        return state

    def interview_agent(state: CareerState) -> CareerState:
        state["interview_questions"] = run_text_prompt(
            llm,
            "You are an Interview Preparation Agent. Generate practical interview questions and model answer hints.",
            f"""
Resume:
{state["resume_text"]}

Job Description:
{state["job_description"]}

Generate:
1. 8 technical interview questions
2. 5 project-based questions
3. 5 HR questions
4. Short answer hints for each section
""",
        )
        return state

    def final_report_agent(state: CareerState) -> CareerState:
        state["final_report"] = f"""
## Job Match
{state["match_report"]}

## Career Growth Suggestions
{state["skill_gap_report"]}

## Resume Strengthening Suggestions
{state["ats_report"]}

## Internship Cover Letter
{state["cover_letter"]}

## Interview Preparation
{state["interview_questions"]}
"""
        return state

    graph = StateGraph(CareerState)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("skill_keyword_agent", skill_keyword_agent)
    graph.add_node("job_match_agent", job_match_agent)
    graph.add_node("skill_gap_agent", skill_gap_agent)
    graph.add_node("ats_optimizer_agent", ats_optimizer_agent)
    graph.add_node("cover_letter_agent", cover_letter_agent)
    graph.add_node("interview_agent", interview_agent)
    graph.add_node("final_report_agent", final_report_agent)

    graph.set_entry_point("retrieval_agent")
    graph.add_edge("retrieval_agent", "skill_keyword_agent")
    graph.add_edge("skill_keyword_agent", "job_match_agent")
    graph.add_edge("job_match_agent", "skill_gap_agent")
    graph.add_edge("skill_gap_agent", "ats_optimizer_agent")
    graph.add_edge("ats_optimizer_agent", "cover_letter_agent")
    graph.add_edge("cover_letter_agent", "interview_agent")
    graph.add_edge("interview_agent", "final_report_agent")
    graph.add_edge("final_report_agent", END)

    return graph.compile()
