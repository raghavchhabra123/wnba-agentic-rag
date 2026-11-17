"""LangGraph agent: decide the scope, then answer with tools.

    question
       |
    [resolve_scope]  LLM extracts a structured reference (teams, date, playoff round /
       |             game number, is it about ONE game?) -> deterministic lookup
       |-- one game  -> GAME mode: game tools locked to that game_id (+ season tools)
       |-- 0 / 2+ games for a single-game question -> ask which game, stop
       |-- not about one game -> SEASON mode: season tools + game tools by game_id
    [agent] <-----> [tools]   tool-calling loop (max MAX_TOOL_ROUNDS rounds)
       |
    answer + trace (every tool call, token usage, cost)
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from wnba_rag import season, stats
from wnba_rag.config import CHAT_MODEL
from wnba_rag.games import describe_game, find_games, resolve_team

MAX_TOOL_ROUNDS = 6

# USD per 1M tokens (input, output). gpt-4o-mini list price.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}

ANSWER_FORMAT = (
    "End your reply with one line in the form 'FINAL ANSWER: <short answer>'. "
    "Put the player or team name, and the number when the question asks for one, in that line. "
    "If several players are tied, list all of them. If the data cannot answer the question, "
    "say so in the FINAL ANSWER line instead of guessing."
)


class ScopeRef(BaseModel):
    single_game: bool = Field(
        description="True if the question is about ONE specific game (a matchup on a date, "
                    "'Game 3 of the Finals', 'the last game of the season vs X'). False for "
                    "season-level questions (records, averages, leaders, head-to-head, a player's "
                    "best game, 'how did X do against Y this season').")
    team_a: str | None = Field(None, description="First TEAM named in the question, copied as written "
                               "(city, nickname or abbreviation). Never a player's name. Null if no team is named.")
    team_b: str | None = Field(None, description="Second team named, if any")
    date: str | None = Field(None, description="Game date as YYYY-MM-DD ONLY if the question states a "
                             "month and day. Assume 2025 if no year. Never invent a date.")
    playoff_round: str | None = Field(None, description="'First Round', 'Semifinals' or 'Finals' if mentioned")
    series_game: int | None = Field(None, description="Game number within a playoff series, e.g. 3 for 'Game 3'")


class AgentState(TypedDict, total=False):
    question: str
    mode: str                 # game | season | done
    game_id: int | None
    game_desc: str
    messages: Annotated[list[BaseMessage], add_messages]
    rounds: int
    answer: str
    trace: list[dict]
    usage: dict


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def _add_usage(usage: dict | None, msg: Any) -> dict:
    usage = dict(usage or {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0})
    meta = getattr(msg, "usage_metadata", None) or {}
    usage["input_tokens"] += meta.get("input_tokens", 0)
    usage["output_tokens"] += meta.get("output_tokens", 0)
    usage["llm_calls"] += 1
    pin, pout = PRICES.get(CHAT_MODEL, (0, 0))
    usage["cost_usd"] = round((usage["input_tokens"] * pin + usage["output_tokens"] * pout) / 1e6, 6)
    return usage


# ---------------------------------------------------------------- tools

def _dump(x: Any) -> str:
    return json.dumps(x, default=str)


def _check_game(game_id: int) -> None:
    stats.game_row(int(game_id))  # raises a clear error for an unknown id


def game_tool_functions():
    """Single-game tools. Each takes game_id; in GAME mode it is overwritten with the
    resolved game, so the model cannot query the wrong game."""

    def player_leaders(game_id: int, stat: str, periods: list[int] | None = None,
                       team: str | None = None) -> str:
        """Rank players in one game by a stat counted from every play.
        stat: points, rebounds, assists, steals, blocks, turnovers, fouls, threes.
        periods: quarters to include, e.g. [4] for the 4th quarter, [3, 4] for the second half,
        [5] for overtime. Omit periods for the whole game. team: full team name, only when
        the question asks about one team's players (e.g. 'who led the Lynx in assists')."""
        _check_game(game_id)
        return _dump(stats.player_leaders(int(game_id), stat, periods or None, team))

    def team_totals(game_id: int, stat: str, periods: list[int] | None = None) -> str:
        """Compare the two teams' totals for a stat (points, rebounds, assists, steals,
        blocks, turnovers, fouls, threes) in one game, optionally for specific quarters."""
        _check_game(game_id)
        return _dump(stats.team_totals(int(game_id), stat, periods or None))

    def game_summary(game_id: int) -> str:
        """Final score, winner, date, and points scored by each team in each quarter."""
        _check_game(game_id)
        return _dump(stats.game_summary(int(game_id)))

    def first_or_last_event(game_id: int, event: str, which: str = "first",
                            periods: list[int] | None = None, team: str | None = None) -> str:
        """Find the first or last event of a kind in one game. event: basket (any made shot
        incl. free throws, i.e. 'points'), field_goal (made shot, no free throws), three,
        turnover, foul. which: first or last. Leave periods and team empty unless the
        question explicitly asks about a specific quarter or a specific team."""
        _check_game(game_id)
        return _dump(stats.first_or_last_event(int(game_id), event, which, periods or None, team))

    def scoring_runs(game_id: int) -> str:
        """Largest unanswered scoring runs (points in a row by one team) in one game."""
        _check_game(game_id)
        return _dump(stats.scoring_runs(int(game_id)))

    def largest_lead(game_id: int) -> str:
        """Each team's largest lead in one game and when it happened, plus lead changes."""
        _check_game(game_id)
        return _dump(stats.largest_lead(int(game_id)))

    def player_game_line(game_id: int, player: str) -> str:
        """Official box-score line (points, rebounds, assists, etc.) for one player in one game."""
        _check_game(game_id)
        return _dump(stats.player_game_line(int(game_id), player))

    def search_plays(game_id: int, query: str, periods: list[int] | None = None) -> str:
        """Semantic search over one game's play-by-play text. Use for descriptive questions
        (how a run happened, what a player did late, key moments), not for counting."""
        from wnba_rag.retrieval import get_index

        _check_game(game_id)
        hits = get_index().search(query, k=5, game_id=int(game_id), periods=periods or None)
        return "\n\n".join(f"[chunk {h['chunk_id']}]\n{h['text']}" for h in hits) or "No plays found."

    return {f.__name__: f for f in [player_leaders, team_totals, game_summary, first_or_last_event,
                                    scoring_runs, largest_lead, player_game_line, search_plays]}


def season_tool_functions():
    def team_record(team: str, season_type: str = "regular") -> str:
        """A team's record (overall, home, away, last 10), points for/against per game.
        season_type: regular (default), playoffs, or all."""
        return _dump(season.team_record(team, season_type))

    def standings() -> str:
        """2025 regular-season standings for all 13 teams with games back."""
        return _dump(season.standings())

    def player_season(player: str, season_type: str = "regular", opponent: str | None = None) -> str:
        """A player's games, per-game averages, totals and best scoring game. Pass opponent
        for 'how did X do against Y'. season_type: regular (default), playoffs, or all."""
        return _dump(season.player_season(player, season_type, opponent))

    def player_game_log(player: str, season_type: str = "regular", opponent: str | None = None,
                        sort_by: str = "date", limit: int = 10) -> str:
        """A player's individual game lines. sort_by='date' for chronological, or a stat
        (points, rebounds, assists, ...) to list her best games by that stat."""
        return _dump(season.player_game_log(player, season_type, opponent, sort_by, limit))

    def season_leaders(stat: str, per_game: bool = True, season_type: str = "regular",
                       team: str | None = None, min_games: int = 10) -> str:
        """League or team leaders. stat: points, rebounds, assists, steals, blocks, turnovers,
        threes, minutes. per_game=True ranks averages (min_games applies), False ranks totals."""
        return _dump(season.season_leaders(stat, per_game, season_type, team, min_games))

    def head_to_head(team_a: str, team_b: str, season_type: str = "all") -> str:
        """Every 2025 meeting between two teams with scores and the series record."""
        return _dump(season.head_to_head(team_a, team_b, season_type))

    def find_games(team: str, opponent: str | None = None, playoff_round: str | None = None,
                   series_game: int | None = None, month: int | None = None) -> str:
        """Find game_ids for a team, optionally vs an opponent, in a playoff round
        ('First Round', 'Semifinals', 'Finals'), a series game number, or a month (1-12)."""
        return _dump(season.find_games_by(team, opponent, playoff_round, series_game, month))

    return {f.__name__: f for f in [team_record, standings, player_season, player_game_log,
                                    season_leaders, head_to_head, find_games]}


def make_tools(mode: str, use_stats: bool = True) -> list[StructuredTool]:
    game_fns = game_tool_functions()
    if not use_stats:  # ablation: retrieval only
        return [StructuredTool.from_function(game_fns["search_plays"])]
    fns = list(game_fns.values()) + list(season_tool_functions().values())
    return [StructuredTool.from_function(f) for f in fns]


# ---------------------------------------------------------------- prompts

COMMON_RULES = """Use tools for every fact; never rely on memory. The data covers the 2025 WNBA
season (regular season and playoffs); questions without a year refer to 2025. Only if a
question is clearly about another season or league, say the data does not cover it.
- Counting or ranking questions -> use the exact stats tools; trust their numbers over
  anything inferred from text.
- Descriptive questions (how a lead was built, key moments) -> first locate WHEN it happened
  with game_summary / scoring_runs / largest_lead, then search_plays for those quarters.
  Support each claim with specific plays (quarter, clock, player, score) and cite chunk ids
  like [chunk 123]. Do not make claims you did not see in a tool result.
- Quarters are periods 1-4; "first half" = [1, 2], "second half" = [3, 4]; overtime is 5+.
- Only pass optional filters (team, periods, opponent) when the question itself restricts to
  them. Naming the matchup is NOT a team filter, and "the first three-pointer of the game" is
  NOT a first-quarter filter. If a filtered call returns nothing, retry without the filter.
- "This season" / no qualifier means the regular season; include playoffs only if asked.
- Never map a team name to a team yourself: pass the user's wording to the tools, which
  resolve WNBA team names (e.g. "suns" means the Connecticut Sun). This is the WNBA, not the NBA."""

GAME_PROMPT = """You are a WNBA analyst. The question is about ONE game: {game} (game_id {game_id}).
This game is in the data. Game tools are locked to it; season tools are available for context.
{rules}
{answer_format}"""

SEASON_PROMPT = """You are a WNBA analyst answering a question about the 2025 season.
Use season tools for records, averages, leaders and matchups. For questions that need a
specific game's details, find its game_id with find_games or player_game_log, then use the
game tools with that game_id. If the question is about "the game" but nothing identifies
which game, ask the user which game they mean instead of guessing.
{rules}
{answer_format}"""

SEARCH_ONLY_PROMPT = """You are a WNBA analyst answering questions about ONE game: {game}.
You can only search the game's play-by-play text with search_plays. Base every fact on
retrieved plays and cite chunk ids. Quarters are periods 1-4.
{answer_format}"""


# ---------------------------------------------------------------- graph

def build_agent(use_stats: bool = True):
    llm = _llm()

    def resolve_scope(state: AgentState) -> dict:
        raw = llm.with_structured_output(ScopeRef, include_raw=True).invoke(
            "Extract the game reference from this WNBA question. Use null for anything not "
            f"stated.\nQuestion: {state['question']}")
        usage = _add_usage(None, raw["raw"])
        ref: ScopeRef = raw["parsed"]
        trace = [{"step": "resolve_scope", "extracted": ref.model_dump()}]

        # Guard against extraction mistakes: a date whose day is not in the question was
        # invented, and a "team" that is not a team (often a player's name) is dropped.
        if ref.date and not re.search(rf"\b0?{int(ref.date[-2:])}(st|nd|rd|th)?\b", state["question"]):
            trace[0]["dropped_date"] = ref.date
            ref.date = None
        for attr in ("team_a", "team_b"):
            if getattr(ref, attr) and not resolve_team(getattr(ref, attr)):
                trace[0][f"dropped_{attr}"] = getattr(ref, attr)
                setattr(ref, attr, None)
        if ref.team_b and not ref.team_a:
            ref.team_a, ref.team_b = ref.team_b, None

        if not use_stats and not ref.single_game:
            ref.single_game = True  # the search-only ablation can only handle single games

        # A single game we cannot pin down from the text alone (e.g. "A'ja Wilson's best
        # scoring game") goes to season mode, where tools can find it.
        if ref.single_game and not ref.team_a and not ref.playoff_round:
            ref.single_game = False if use_stats else ref.single_game

        if not ref.single_game:
            prompt = SEASON_PROMPT.format(rules=COMMON_RULES, answer_format=ANSWER_FORMAT)
            return {"mode": "season", "game_id": None, "rounds": 0, "trace": trace, "usage": usage,
                    "messages": [SystemMessage(prompt), HumanMessage(state["question"])]}

        if not ref.team_a and not ref.playoff_round:
            return {"mode": "done", "game_id": None, "trace": trace, "usage": usage,
                    "answer": "Which game do you mean? Please name the teams and, ideally, the date."}
        if ref.playoff_round or ref.series_game:
            matches = season.find_games_by(ref.team_a, ref.team_b, ref.playoff_round,
                                           ref.series_game)["matches"]
            if ref.date:
                matches = [m for m in matches if m["date"] == ref.date] or matches
            found: dict = {"matches": matches}
        else:
            found = find_games(ref.team_a, ref.team_b, ref.date)
        matches = found["matches"]
        trace[0]["matches"] = len(matches)
        if len(matches) > 1 and not (ref.date or ref.playoff_round or ref.series_game) and use_stats:
            # Several games match and the text gives no date/round: it may still be
            # identifiable with tools ("A'ja Wilson's best scoring game"). Let season mode
            # find it, or ask.
            trace[0]["fallback"] = "season"
            prompt = SEASON_PROMPT.format(rules=COMMON_RULES, answer_format=ANSWER_FORMAT)
            note = (f"Note: {len(matches)} games match the teams named. Unless the question itself "
                    "pins down one game through other details (for example 'her highest-scoring "
                    "game'), do not pick one: ask the user which game they mean.")
            return {"mode": "season", "game_id": None, "rounds": 0, "trace": trace, "usage": usage,
                    "messages": [SystemMessage(prompt + "\n" + note), HumanMessage(state["question"])]}
        if len(matches) != 1:
            if not matches:
                msg = found.get("error") or "I couldn't find that game in the 2025 data."
            else:
                options = "; ".join(f"{m['date']}: {m['matchup']}" for m in matches[:8])
                more = f" (and {len(matches) - 8} more)" if len(matches) > 8 else ""
                msg = f"That matches {len(matches)} games. Which one? {options}{more}"
            return {"mode": "done", "game_id": None, "trace": trace, "usage": usage,
                    "answer": f"{msg}\nFINAL ANSWER: need clarification"}

        game_id = matches[0]["game_id"]
        desc = describe_game(game_id)
        template = GAME_PROMPT if use_stats else SEARCH_ONLY_PROMPT
        prompt = template.format(game=desc, game_id=game_id, rules=COMMON_RULES,
                                 answer_format=ANSWER_FORMAT)
        return {"mode": "game", "game_id": game_id, "game_desc": desc, "rounds": 0,
                "trace": trace, "usage": usage,
                "messages": [SystemMessage(prompt), HumanMessage(state["question"])]}

    def agent(state: AgentState) -> dict:
        tools = make_tools(state["mode"], use_stats)
        model = llm.bind_tools(tools) if state["rounds"] < MAX_TOOL_ROUNDS else llm
        reply = model.invoke(state["messages"])
        out: dict[str, Any] = {"messages": [reply], "rounds": state["rounds"] + 1,
                               "usage": _add_usage(state.get("usage"), reply)}
        if not reply.tool_calls:
            out["answer"] = reply.content
        return out

    def run_tools(state: AgentState) -> dict:
        tools = {t.name: t for t in make_tools(state["mode"], use_stats)}
        last: AIMessage = state["messages"][-1]
        msgs, trace = [], list(state.get("trace", []))
        for call in last.tool_calls:
            args = dict(call["args"])
            if state["mode"] == "game" and "game_id" in tools[call["name"]].args:
                args["game_id"] = state["game_id"]  # lock game tools to the resolved game
            try:
                result = tools[call["name"]].invoke(args)
            except Exception as exc:  # surface tool errors so the model can recover
                result = f"Tool error: {exc}"
            trace.append({"step": "tool", "name": call["name"], "args": args})
            msgs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": msgs, "trace": trace}

    def after_resolve(state: AgentState) -> str:
        return END if state["mode"] == "done" else "agent"

    def after_agent(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("resolve_scope", resolve_scope)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.set_entry_point("resolve_scope")
    graph.add_conditional_edges("resolve_scope", after_resolve, {"agent": "agent", END: END})
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def ask(question: str, use_stats: bool = True, graph=None) -> dict:
    graph = graph or build_agent(use_stats)
    r = graph.invoke({"question": question}, {"recursion_limit": 30})
    return {"answer": r.get("answer", ""), "game_id": r.get("game_id"), "mode": r.get("mode"),
            "trace": r.get("trace", []), "usage": r.get("usage", {})}
