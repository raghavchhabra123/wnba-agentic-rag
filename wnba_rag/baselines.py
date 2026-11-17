"""Comparison systems for the evaluation.

  llm_only     - no retrieval (what the course "Base LLM" did)
  naive_rag    - top-k similarity search over ALL games, stuffed into the prompt
                 (the course "Basic RAG" design, with the improved chunks)
  scoped_rag   - the agent with game resolution + game-scoped search but NO
                 stats tools (ablation: is retrieval alone enough once the game is known?)
  agentic      - full agent: game resolution + stats tools + scoped search
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from wnba_rag.agent import ANSWER_FORMAT, _add_usage, build_agent
from wnba_rag.config import CHAT_MODEL


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def llm_only(question: str) -> dict:
    prompt = f"Answer this question about the 2025 WNBA season.\n{question}\n\n{ANSWER_FORMAT}"
    reply = _llm().invoke(prompt)
    return {"answer": reply.content, "trace": [], "usage": _add_usage(None, reply)}


def naive_rag(question: str, k: int = 6) -> dict:
    from wnba_rag.retrieval import get_index

    hits = get_index().search(question, k=k)
    context = "\n\n".join(h["text"] for h in hits)
    prompt = (f"Use the WNBA play-by-play excerpts below to answer the question.\n\n"
              f"{context}\n\nQuestion: {question}\n\n{ANSWER_FORMAT}")
    reply = _llm().invoke(prompt)
    return {"answer": reply.content, "usage": _add_usage(None, reply),
            "trace": [{"step": "retrieve", "game_ids": sorted({h['game_id'] for h in hits})}]}


_GRAPHS: dict[bool, object] = {}


def _graph(use_stats: bool):
    if use_stats not in _GRAPHS:
        _GRAPHS[use_stats] = build_agent(use_stats)
    return _GRAPHS[use_stats]


def scoped_rag(question: str) -> dict:
    r = _graph(False).invoke({"question": question}, {"recursion_limit": 30})
    return {"answer": r.get("answer", ""), "game_id": r.get("game_id"), "mode": r.get("mode"),
            "trace": r.get("trace", []), "usage": r.get("usage", {})}


def agentic(question: str) -> dict:
    r = _graph(True).invoke({"question": question}, {"recursion_limit": 30})
    return {"answer": r.get("answer", ""), "game_id": r.get("game_id"), "mode": r.get("mode"),
            "trace": r.get("trace", []), "usage": r.get("usage", {})}


SYSTEMS = {"llm_only": llm_only, "naive_rag": naive_rag, "scoped_rag": scoped_rag, "agentic": agentic}
