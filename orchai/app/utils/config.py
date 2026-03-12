import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MongoDB
    MONGO_URI: str
    MONGO_DB_NAME: str = "orchai_db"

    # OpenAI - küçük/hızlı model
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Claude - büyük/uzun context model
    CLAUDE_API_KEY: str
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"

    # Routing eşikleri
    RAG_SIMILARITY_THRESHOLD: float = 0.75
    MAX_SHORT_TERM_MESSAGES: int = 10
    LONG_TERM_SUMMARY_THRESHOLD: int = 20

    # Token limitleri
    MAX_TOKENS_DIRECT: int = 1000
    MAX_TOKENS_RAG: int = 2000
    MAX_TOKENS_WEB: int = 2000

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    model_config = {
        "env_file": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "protected_namespaces": ()
    }


settings = Settings()
