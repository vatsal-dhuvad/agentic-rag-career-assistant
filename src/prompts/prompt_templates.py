from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from src.config import APP_IDENTITY


def escape_prompt_text(text: str) -> str:
    """Prevent normal text/dicts from being treated as prompt variables."""
    return text.replace("{", "{{").replace("}", "}}")


def run_text_prompt(llm, system_prompt: str, user_prompt: str) -> str:
    safe_system_prompt = escape_prompt_text(system_prompt)
    safe_user_prompt = escape_prompt_text(user_prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                safe_system_prompt
                + "\n\nIf anyone asks who you are, answer exactly: "
                + APP_IDENTITY,
            ),
            ("human", safe_user_prompt),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({}).strip()


def run_json_prompt(llm, system_prompt: str, user_prompt: str) -> dict:
    parser = JsonOutputParser()
    safe_system_prompt = escape_prompt_text(system_prompt)
    safe_user_prompt = escape_prompt_text(user_prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                safe_system_prompt
                + "\nReturn only valid JSON. Do not wrap it in markdown."
                + "\nIf anyone asks who you are, answer exactly: "
                + APP_IDENTITY,
            ),
            ("human", safe_user_prompt + "\n\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    return chain.invoke({})
