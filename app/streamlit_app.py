"""Streamlit demo: ask a question, see the answer and every tool call behind it.

    streamlit run app/streamlit_app.py
"""

import datetime
import os
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from wnba_rag.agent import ask, build_agent
from wnba_rag.stats import load_tables

st.set_page_config(page_title="WNBA Agentic RAG", page_icon="🏀", layout="wide")

# Spend guards for the public demo: every question costs ~$0.0007 of the owner's
# OpenAI credit, so cap questions per visitor and per day across all visitors.
MAX_PER_SESSION = int(os.getenv("WNBA_RAG_MAX_PER_SESSION", "20"))
MAX_PER_DAY = int(os.getenv("WNBA_RAG_MAX_PER_DAY", "1000"))
MAX_CHARS = 300


@st.cache_resource
def daily_counter() -> dict:
    return {"date": None, "count": 0, "lock": threading.Lock()}


def take_quota() -> str | None:
    """Return an error message if a limit is hit, else count the question."""
    if st.session_state.get("asked", 0) >= MAX_PER_SESSION:
        return f"This demo allows {MAX_PER_SESSION} questions per visit. Refresh later or run it locally."
    c = daily_counter()
    with c["lock"]:
        today = datetime.date.today().isoformat()
        if c["date"] != today:
            c["date"], c["count"] = today, 0
        if c["count"] >= MAX_PER_DAY:
            return "The demo's daily question limit is reached. Please try again tomorrow."
        c["count"] += 1
    st.session_state["asked"] = st.session_state.get("asked", 0) + 1
    return None


@st.cache_resource
def graph():
    return build_agent()


@st.cache_data
def games() -> pd.DataFrame:
    g = load_tables().games
    g = g.assign(label=g["date"] + "  " + g["away"] + " at " + g["home"]
                 + "  (" + g["away_score"].astype(str) + "-" + g["home_score"].astype(str) + ")")
    return g.sort_values("date", ascending=False)


st.title("WNBA Agentic RAG")
st.markdown("[Source code, method and evaluation on GitHub](https://github.com/raghavchhabra123/wnba-agentic-rag)")
st.caption("Ask about any 2025 WNBA game or the season as a whole. Counting questions use exact stat "
           "tools validated against official box scores; descriptive questions use retrieval scoped "
           "to one game. Every answer shows the tool calls behind it.")

with st.sidebar:
    st.subheader("Pick a game")
    choice = st.selectbox("Game", games()["label"].tolist(), index=None,
                          placeholder="optional: fills in the game for you")
    st.markdown("**Try asking** (click to fill in)")
    examples = [
        "Who scored the most points in Game 3 of the Finals?",
        "How did Caitlin Clark do against the Sky this year?",
        "In A'ja Wilson's best scoring game, who led the other team in points?",
        "Who led the league in assists per game?",
        "What was the largest scoring run in Game 4 of the Finals, and how did it happen?",
    ]
    for i, ex in enumerate(examples):
        st.button(ex, key=f"ex{i}", use_container_width=True,
                  on_click=lambda ex=ex: st.session_state.update(question=ex))

question = st.text_input("Question", max_chars=MAX_CHARS, key="question",
                         placeholder="who went off for the lynx in the 4th vs golden state on sept 11?")
if choice and question:
    row = games().set_index("label").loc[choice]
    question = f"{question} (game: {row['away']} at {row['home']} on {row['date']})"

if st.button("Ask", type="primary", disabled=not question):
    blocked = take_quota()
    if blocked:
        st.warning(blocked)
        st.stop()
    with st.spinner("Thinking..."):
        result = ask(question, graph=graph())
    # The agent ends with a "FINAL ANSWER:" line for automatic grading; show it as a highlight.
    m = re.search(r"FINAL ANSWER:\s*(.+)", result["answer"])
    body = result["answer"][:m.start()].strip() if m else result["answer"]
    if m:
        st.success(m.group(1).strip())
    if body:
        st.markdown(body)
    usage = result.get("usage", {})
    mode = {"game": "single game", "season": "season", "done": "needs more detail"}.get(
        result.get("mode"), result.get("mode"))
    st.caption(f"Mode: {mode} · {usage.get('llm_calls', 0)} LLM calls · "
               f"{usage.get('input_tokens', 0) + usage.get('output_tokens', 0):,} tokens · "
               f"${usage.get('cost_usd', 0):.4f}")
    with st.expander("How the agent got there", expanded=True):
        for n, step in enumerate(result["trace"], 1):
            if step["step"] == "resolve_scope":
                found = {k: v for k, v in (step.get("extracted") or {}).items()
                         if v not in (None, False, "")}
                extra = {k: v for k, v in step.items()
                         if k not in ("step", "extracted") and v not in (None, "", [], {})}
                st.markdown(f"{n}. **resolve_scope** → `{found or 'season-wide question'}`"
                            + (f" `{extra}`" if extra else ""))
            else:
                args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
                st.markdown(f"{n}. **{step['name']}**(`{args}`)")
