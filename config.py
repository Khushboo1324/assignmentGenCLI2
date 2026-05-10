from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Groq
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Chroma
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "documents")

    # Retrieval
    top_k_default: int = int(os.getenv("TOP_K_DEFAULT", "5"))

    # Chunking
    chunk_strategy_default: str = os.getenv("CHUNK_STRATEGY_DEFAULT", "fixed")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))


settings = Settings()