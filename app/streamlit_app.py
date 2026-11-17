"""Streamlit demo: ask a question, see the answer and every tool call behind it.

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from wnba_rag.agent import ask, build_agent
from wnba_rag.stats import load_tables

st.set_page_config(page_title="WNBA Agentic RAG", page_icon="🏀", layout="wide")


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
st.caption("Ask about any 2025 WNBA game or the season as a whole. Counting questions use exact stat "
           "tools validated against official box scores; descriptive questions use retrieval scoped "
           "to one game. Every answer shows the tool calls behind it.")

with st.sidebar:
    st.subheader("Pick a game")
    choice = st.selectbox("Game", games()["label"].tolist(), index=None,
                          placeholder="optional: fills in the game for you")
    st.markdown("**Try asking**")
    examples = [
        "Who scored the most points in Game 3 of the Finals?",
        "How did Caitlin Clark do against the Sky this year?",
        "In A'ja Wilson's best scoring game, who led the other team in points?",
        "Who led the league in assists per game?",
        "With a game picked: what was the largest scoring run, and how did it happen?",
    ]
    for ex in examples:
        st.markdown(f"- {ex}")

question = st.text_input("Question", placeholder="who went off for the lynx in the 4th vs golden state on sept 11?")
if choice and question:
    row = games().set_index("label").loc[choice]
    question = f"{question} (game: {row['away']} at {row['home']} on {row['date']})"

if st.button("Ask", type="primary", disabled=not question):
    with st.spinner("Thinking..."):
        result = ask(question, graph=graph())
    st.markdown(result["answer"])
    usage = result.get("usage", {})
    st.caption(f"mode: {result.get('mode')} · {usage.get('llm_calls', 0)} LLM calls · "
               f"{usage.get('input_tokens', 0) + usage.get('output_tokens', 0):,} tokens · "
               f"${usage.get('cost_usd', 0):.4f}")
    with st.expander("How the agent got there", expanded=True):
        for step in result["trace"]:
            if step["step"] == "resolve_scope":
                extra = {k: v for k, v in step.items() if k not in ("step", "extracted")}
                st.markdown(f"**resolve_scope** → extracted `{step.get('extracted')}` {extra or ''}")
            else:
                st.markdown(f"**{step['name']}** `{step['args']}`")
