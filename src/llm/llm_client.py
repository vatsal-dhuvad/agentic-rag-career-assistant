import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None


def get_secret_value(name: str) -> str:
    load_dotenv()
    env_key = os.getenv(name, "").strip()
    if env_key:
        return env_key
    return str(st.secrets.get(name, "")).strip()


def get_gemini_api_key() -> str:
    return get_secret_value("GEMINI_API_KEY") or get_secret_value("GOOGLE_API_KEY")


def get_groq_api_key() -> str:
    return get_secret_value("GROQ_API_KEY")


def get_mistral_api_key() -> str:
    return get_secret_value("MISTRAL_API_KEY")


def get_llm(
    model_name: str,
    temperature: float = 0.2,
    gemini_model: str = "gemini-2.5-flash-lite",
    mistral_model: str = "mistral-small-latest",
):
    llms = []

    gemini_api_key = get_gemini_api_key()
    if gemini_api_key and ChatGoogleGenerativeAI is not None:
        llms.append(
            ChatGoogleGenerativeAI(
                google_api_key=gemini_api_key,
                model=gemini_model,
                temperature=temperature,
            )
        )

    groq_api_key = get_groq_api_key()
    if groq_api_key:
        llms.append(
            ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=temperature,
            )
        )

    mistral_api_key = get_mistral_api_key()
    if mistral_api_key and ChatMistralAI is not None:
        llms.append(
            ChatMistralAI(
                api_key=mistral_api_key,
                model=mistral_model,
                temperature=temperature,
            )
        )

    if not llms:
        st.error(
            "No AI API key found. Add GEMINI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, "
            "or MISTRAL_API_KEY in .env or Streamlit secrets."
        )
        st.stop()

    primary_llm = llms[0]
    fallback_llms = llms[1:]

    if fallback_llms:
        return primary_llm.with_fallbacks(fallback_llms)

    return primary_llm
