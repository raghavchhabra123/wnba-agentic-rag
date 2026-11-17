"""Deterministic stat calculations over the play-by-play table.

Counting questions ("who had the most assists in the 4th quarter?") need every
play in a game, not the ten most similar ones, so the agent calls these
functions as tools instead of asking the LLM to count retrieved text.

Every function here is checked against ESPN's official box scores in
tests/test_stats.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from wnba_rag.paths import PROCESSED_DIR

STATS = ("points", "rebounds", "assists", "steals", "blocks", "turnovers", "fouls", "threes")

# Fouls that count toward a player's personal-foul total. "Offensive Foul Turnover"
# is the turnover half of an offensive foul (ESPN logs both), and technicals are
# not personal fouls.
PERSONAL_FOUL_TYPES = {
    "Shooting Foul", "Personal Foul", "Offensive Foul", "Loose Ball Foul",
    "Personal Take Foul", "Transition Take Foul", "Flagrant Foul Type 1",
    "Flagrant Foul Type 2", "Away from Play Foul", "Clear Path Foul",
    "Offensive Charge", "Personal Block", "Inbound Foul",
}


@dataclass(frozen=True)
class Tables:
    games: pd.DataFrame
    plays: pd.DataFrame
    box: pd.DataFrame


@lru_cache(maxsize=1)
def load_tables() -> Tables:
    games = pd.read_csv(PROCESSED_DIR / "games.csv")
    plays = pd.read_csv(PROCESSED_DIR / "plays.csv")
    for col in ["team_id", "p1_id", "p2_id"]:
        plays[col] = plays[col].astype("Int64")
    box = pd.read_csv(PROCESSED_DIR / "box.csv")
    return Tables(games, plays, box)


def game_plays(game_id: int, periods: list[int] | None = None) -> pd.DataFrame:
    plays = load_tables().plays
    out = plays[plays["game_id"] == game_id]
    if periods:
        out = out[out["period"].isin(periods)]
    return out


def game_row(game_id: int) -> pd.Series:
    games = load_tables().games
    rows = games[games["game_id"] == game_id]
    if rows.empty:
        raise ValueError(f"Unknown game_id {game_id}")
    return rows.iloc[0]


# --------------------------------------------------------------------------
# Event extraction: one row per (player, team, stat credit)
# --------------------------------------------------------------------------

def _credits(plays: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Rows of (player_id, player, team_id, team, value) credited for `stat`."""
    text = plays["text"].fillna("")
    ptype = plays["type"].fillna("")

    if stat == "points":
        rows = plays[plays["scoring"]]
        return _frame(rows, "p1", rows["score_value"])
    if stat == "threes":
        rows = plays[is_made_three(plays)]
        return _frame(rows, "p1", 1)
    if stat == "assists":
        rows = plays[plays["scoring"] & text.str.contains(r"assists\)", regex=True)]
        return _frame(rows, "p2", 1, same_team=True)
    if stat == "rebounds":
        rows = plays[ptype.isin(["Offensive Rebound", "Defensive Rebound"]) & plays["p1_id"].notna()]
        return _frame(rows, "p1", 1)
    if stat == "steals":
        rows = plays[text.str.contains(r"steals\)", regex=True)]
        return _frame(rows, "p2", 1, same_team=False)
    if stat == "blocks":
        rows = plays[text.str.contains(" blocks ", regex=False)]
        return _frame(rows, "p2", 1, same_team=False)
    if stat == "turnovers":
        rows = plays[is_turnover(plays)]
        return _frame(rows, "p1", 1, team_fallback=True)
    if stat == "fouls":
        rows = plays[ptype.isin(PERSONAL_FOUL_TYPES)]
        return _frame(rows, "p1", 1)
    raise ValueError(f"Unknown stat '{stat}'. Choose from {STATS}.")


def is_made_three(plays: pd.DataFrame) -> pd.Series:
    """Made threes, identified by points scored. The text is not reliable: some threes
    are written as e.g. '25-foot running pullup jump shot' with no 'three point'."""
    text = plays["text"].fillna("")
    return (plays["scoring"] & (plays["score_value"] == 3)
            & ~text.str.contains("free throw", case=False))


def is_turnover(plays: pd.DataFrame) -> pd.Series:
    """Turnover plays. ESPN logs traveling as its own type without 'Turnover' in it."""
    ptype = plays["type"].fillna("")
    return (ptype.str.contains("Turnover") & (ptype != "No Turnover")) | (ptype == "Traveling")


def _frame(rows: pd.DataFrame, who: str, value, same_team: bool = True,
           team_fallback: bool = False) -> pd.DataFrame:
    """Build credit rows. Players on the other team (stealer, blocker) get the
    opposing team id, looked up from the box score."""
    out = pd.DataFrame({
        "game_id": rows["game_id"],
        "player_id": rows[f"{who}_id"],
        "player": rows[who],
        "value": value,
        "play_team_id": rows["team_id"],
    })
    if not team_fallback:
        out = out[out["player_id"].notna()]
    box = load_tables().box[["game_id", "player_id", "team_id", "team"]].drop_duplicates()
    out = out.merge(box, on=["game_id", "player_id"], how="left")
    # Team-only events (e.g. shot-clock turnovers) have no player: use the play's team.
    missing = out["team_id"].isna()
    if missing.any():
        games = load_tables().games.set_index("game_id")
        out.loc[missing, "team_id"] = out.loc[missing, "play_team_id"]
        names = {}
        for gid, row in games.iterrows():
            names[(gid, row["home_id"])] = row["home"]
            names[(gid, row["away_id"])] = row["away"]
        out.loc[missing, "team"] = [
            names.get((g, t)) for g, t in zip(out.loc[missing, "game_id"], out.loc[missing, "team_id"])
        ]
    return out.drop(columns=["play_team_id"])


# --------------------------------------------------------------------------
# Tool-facing functions (return small, JSON-friendly dicts)
# --------------------------------------------------------------------------

def player_totals(game_id: int, stat: str, periods: list[int] | None = None,
                  team: str | None = None) -> pd.DataFrame:
    credits = _credits(game_plays(game_id, periods), stat)
    credits = credits[credits["player_id"].notna()]
    if team:
        credits = credits[credits["team"] == team]
    totals = (credits.groupby(["player", "team"], as_index=False)["value"].sum()
              .sort_values("value", ascending=False))
    return totals.rename(columns={"value": stat}).reset_index(drop=True)


def player_leaders(game_id: int, stat: str, periods: list[int] | None = None,
                   team: str | None = None, top_n: int = 5) -> dict:
    """Top players for a stat, with ties at the top listed explicitly."""
    totals = player_totals(game_id, stat, periods, team)
    if totals.empty:
        return {"stat": stat, "periods": periods or "all", "leaders": [], "top_value": 0}
    top_value = int(totals[stat].iloc[0])
    tied = totals[totals[stat] == top_value]["player"].tolist()
    return {
        "stat": stat,
        "periods": periods or "all",
        "team_filter": team,
        "top_value": top_value,
        "tied_for_lead": tied,
        "leaders": totals.head(max(top_n, len(tied))).to_dict(orient="records"),
    }


def team_totals(game_id: int, stat: str, periods: list[int] | None = None) -> dict:
    g = game_row(game_id)
    credits = _credits(game_plays(game_id, periods), stat)
    sums = credits.groupby("team")["value"].sum()
    result = {g["home"]: int(sums.get(g["home"], 0)), g["away"]: int(sums.get(g["away"], 0))}
    return {"stat": stat, "periods": periods or "all", "totals": result}


def game_summary(game_id: int) -> dict:
    g = game_row(game_id)
    import json
    hq, aq = json.loads(g["home_quarters"]), json.loads(g["away_quarters"])
    winner = g["home"] if g["home_score"] > g["away_score"] else g["away"]
    return {
        "game_id": int(game_id),
        "date": g["date"],
        "season_type": g["season_type"],
        "home": g["home"], "away": g["away"],
        "final_score": {g["home"]: int(g["home_score"]), g["away"]: int(g["away_score"])},
        "winner": winner,
        "points_by_period": {g["home"]: hq, g["away"]: aq},
    }


def first_or_last_event(game_id: int, event: str, which: str = "first",
                        periods: list[int] | None = None, team: str | None = None) -> dict:
    """event: 'basket' (any made shot incl. free throws), 'field_goal', 'three', 'turnover', 'foul'."""
    plays = game_plays(game_id, periods)
    text = plays["text"].fillna("")
    if event == "basket":
        rows = plays[plays["scoring"]]
    elif event == "field_goal":
        rows = plays[plays["scoring"] & ~text.str.contains("free throw", case=False)]
    elif event == "three":
        rows = plays[is_made_three(plays)]
    elif event == "turnover":
        rows = plays[is_turnover(plays)]
    elif event == "foul":
        rows = plays[plays["type"].isin(PERSONAL_FOUL_TYPES)]
    else:
        raise ValueError("event must be basket, field_goal, three, turnover or foul")
    def describe(r: pd.Series) -> dict:
        return {"player": r["p1"], "team": r["team"], "period": int(r["period"]),
                "clock": r["clock"], "description": r["text"],
                "score_after": f"{int(r['home_score'])}-{int(r['away_score'])} (home-away)"}

    def pick(df: pd.DataFrame) -> pd.Series:
        return df.iloc[0] if which == "first" else df.iloc[-1]

    result: dict = {"event": event, "which": which, "periods": periods or "all"}
    if rows.empty:
        return {**result, "found": False}
    out = {**result, "found": True, **describe(pick(rows))}  # whole game is the main answer
    if team:
        # Models often add a team filter just because the question names a team, so the
        # whole-game answer stays primary and the team-only one is clearly secondary.
        filtered = rows[rows["team"] == team]
        out["only_if_question_asks_for_this_team"] = {
            "team": team,
            **(describe(pick(filtered)) if not filtered.empty else {"found": False}),
            "use_when": f"the question asks for the {team}' own {which} {event} "
                        f"(e.g. 'the {team}' {which} {event}'), not the {which} {event} of the game",
        }
    return out


def scoring_runs(game_id: int, top_n: int = 3) -> dict:
    """Largest unanswered scoring runs: consecutive points by one team before the
    other team scores. Free throws count."""
    g = game_row(game_id)
    plays = game_plays(game_id)
    scoring = plays[plays["scoring"] & plays["team"].notna()]
    runs, current = [], None
    for _, row in scoring.iterrows():
        if current and current["team"] == row["team"]:
            current["points"] += int(row["score_value"])
            current["end"] = (int(row["period"]), row["clock"])
            current["end_score"] = f"{int(row['home_score'])}-{int(row['away_score'])}"
        else:
            if current:
                runs.append(current)
            current = {"team": row["team"], "points": int(row["score_value"]),
                       "start": (int(row["period"]), row["clock"]),
                       "end": (int(row["period"]), row["clock"]),
                       "end_score": f"{int(row['home_score'])}-{int(row['away_score'])}"}
    if current:
        runs.append(current)
    runs.sort(key=lambda r: r["points"], reverse=True)
    fmt = [{"team": r["team"], "points": r["points"],
            "from": f"Q{r['start'][0]} {r['start'][1]}", "to": f"Q{r['end'][0]} {r['end'][1]}",
            "score_after_run": f"{r['end_score']} ({g['home']}-{g['away']})"} for r in runs[:top_n]]
    return {"largest_runs": fmt}


def largest_lead(game_id: int) -> dict:
    g = game_row(game_id)
    plays = game_plays(game_id)
    margin = plays["home_score"] - plays["away_score"]
    out = {}
    for team, series in ((g["home"], margin), (g["away"], -margin)):
        idx = series.idxmax()
        lead = int(series.max())
        out[team] = {"largest_lead": max(lead, 0),
                     "when": f"Q{int(plays.loc[idx, 'period'])} {plays.loc[idx, 'clock']}" if lead > 0 else None}
    lead_changes = int(((margin.shift() * margin) < 0).sum())
    return {"largest_lead": out, "lead_changes": lead_changes}


def player_game_line(game_id: int, player: str) -> dict:
    """Official box-score line for one player (name matched case-insensitively,
    last name accepted if unique)."""
    box = load_tables().box
    rows = box[box["game_id"] == game_id]
    match = rows[rows["player"].str.lower() == player.lower()]
    if match.empty:
        last = player.split()[-1].lower()
        match = rows[rows["player"].str.lower().str.split().str[-1] == last]
    if match.empty:
        return {"found": False, "players_in_game": sorted(rows["player"].tolist())}
    if len(match) > 1:
        return {"found": False, "ambiguous": match["player"].tolist()}
    r = match.iloc[0]
    cols = ["player", "team", "minutes", "points", "rebounds", "assists", "steals",
            "blocks", "turnovers", "fouls"]
    return {"found": True, **{c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in cols}}
