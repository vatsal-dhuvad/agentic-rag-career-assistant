from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESUME_PATH = BASE_DIR / "Vatsal_Dhuvad_Resume.pdf"
PORTFOLIO_URL = "https://vatsal-dhuvad.vercel.app/"
FIRESTORE_PORTFOLIO_URL = (
    "https://firestore.googleapis.com/v1/projects/vatsal-portfolio-5699f/"
    "databases/(default)/documents/portfolio/data"
)

APP_IDENTITY = (
    "I am an Agentic RAG-based Career Assistant built by Vatsal Dhuvad, "
    "a Computer Engineering student."
)

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
RETRIEVAL_K = 6
