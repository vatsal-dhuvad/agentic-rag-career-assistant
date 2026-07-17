# Agentic RAG Career Assistant

An AI-powered career assistant built by **Vatsal Dhuvad**, a Computer Engineering student. The app helps match a resume with AI/ML, Data Science, Generative AI, and technology job descriptions using Streamlit, LangChain, LangGraph, FAISS, local Hugging Face embeddings, Groq, and optional Mistral fallback.

## Features

- Shows a default resume: `Vatsal_Dhuvad_Resume.pdf`
- Supports uploading another resume in PDF, DOCX, or TXT format
- Shows a saved default resume overview without calling any API
- Uses API only for lightweight job match decision for the default resume
- Highlights positive job match, matching skills, strong project matches, and best-fit roles
- Supports RAG-based Ask AI from the active resume
- Uses local embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Stores vectors locally with FAISS
- Uses Groq as the main LLM provider
- Supports Mistral fallback if `MISTRAL_API_KEY` is added

## Project Structure

```text
agentic_rag_career_assistant/
  app.py
  README.md
  requirements.txt
  .env.example
  config.yaml
  Vatsal_Dhuvad_Resume.pdf
  src/
    config.py
    ingestion/
      loader.py
    chunking/
      chunker.py
    embeddings/
      embedder.py
    vectordb/
      vector_store.py
    retrieval/
      retriever.py
    prompts/
      prompt_templates.py
    llm/
      llm_client.py
    graph/
      workflow.py
    utils/
      skills.py
  tests/
```

## Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- FAISS
- Hugging Face sentence-transformers
- Groq API
- Mistral API fallback
- pypdf
- python-docx

## Local Setup

Create and activate a virtual environment:

```powershell
uv venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
uv pip install -r requirements.txt
```

Create a `.env` file:

```text
GROQ_API_KEY=your_groq_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

`MISTRAL_API_KEY` is optional, but useful when the Groq limit is reached.

Run the app:

```powershell
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Add secrets in Streamlit Cloud:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
MISTRAL_API_KEY = "your_mistral_api_key_here"
```

6. Deploy the app.

## Security

Do not push `.env` to GitHub. Use `.env.example` for public reference and Streamlit secrets for deployment.

## Identity Prompt

If someone asks "who are you?", the assistant responds:

```text
I am an Agentic RAG-based Career Assistant built by Vatsal Dhuvad, a Computer Engineering student.
```
