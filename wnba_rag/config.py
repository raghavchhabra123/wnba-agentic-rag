import os
from functools import lru_cache

from dotenv import load_dotenv

from wnba_rag.paths import ROOT

load_dotenv(ROOT / ".env")

CHAT_MODEL = os.getenv("WNBA_RAG_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("WNBA_RAG_EMBED_MODEL", "text-embedding-3-small")


@lru_cache(maxsize=1)
def get_openai_client():
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment or in .env (see .env.example).")
    return OpenAI()
