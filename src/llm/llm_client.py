import os

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None


def get_groq_api_key() -> str:
    load_dotenv()
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    return str(st.secrets.get("GROQ_API_KEY", "")).strip()


def get_mistral_api_key() -> str:
    load_dotenv()
    env_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if env_key:
        return env_key
    return str(st.secrets.get("MISTRAL_API_KEY", "")).strip()


def get_llm(model_name: str, temperature: float = 0.2, fallback_model: str = "mistral-small-latest"):
    api_key = get_groq_api_key()

    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it in your .env file and restart Streamlit.")
        st.stop()

    groq_llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )

    mistral_api_key = get_mistral_api_key()
    if not mistral_api_key:
        return groq_llm

    if ChatMistralAI is None:
        st.info("Mistral fallback is configured, but langchain-mistralai is not installed.")
        return groq_llm

    mistral_llm = ChatMistralAI(
        api_key=mistral_api_key,
        model=fallback_model,
        temperature=temperature,
    )

    return groq_llm.with_fallbacks([mistral_llm])
