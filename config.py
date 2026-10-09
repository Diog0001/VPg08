"""Чтение .env и настройки логирования."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def _first_env(*names: str, default: str | None = None, required: bool = False) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    if required:
        joined = " / ".join(names)
        raise RuntimeError(f"Не задана переменная окружения: {joined}")
    return default


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telebot").setLevel(logging.INFO)


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_model: str
    embedding_dimensions: int
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str
    retrieval_top_k: int
    chat_session_max_messages: int

    @classmethod
    def load(cls) -> "Settings":
        base_url = _first_env("OPENAI_BASE_URL", required=True)
        assert base_url is not None
        return cls(
            telegram_bot_token=_first_env("TELEGRAM_BOT_TOKEN", required=True) or "",
            openai_api_key=_first_env("OPENAI_API_KEY", required=True) or "",
            openai_base_url=base_url.rstrip("/"),
            openai_model=_first_env("OPENAI_MODEL", "OPENAI_CHAT_MODEL", required=True) or "",
            embedding_model=_first_env("EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL", required=True)
            or "",
            embedding_dimensions=int(_first_env("OPENAI_EMBEDDING_DIMENSIONS", default="512") or "512"),
            pinecone_api_key=_first_env("PINECONE_API_KEY", required=True) or "",
            pinecone_index_name=_first_env("PINECONE_INDEX_NAME", required=True) or "",
            pinecone_cloud=_first_env("PINECONE_CLOUD", default="aws") or "aws",
            pinecone_region=_first_env("PINECONE_REGION", default="us-east-1") or "us-east-1",
            pinecone_namespace=_first_env("PINECONE_MEMORY_NAMESPACE", default="telegram-users")
            or "telegram-users",
            retrieval_top_k=int(_first_env("MEMORY_RETRIEVAL_TOP_K", default="6") or "6"),
            chat_session_max_messages=int(
                _first_env("CHAT_SESSION_MAX_MESSAGES", default="20") or "20"
            ),
        )
