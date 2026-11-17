"""Season-level tools: records, standings, player season lines, leaders, matchups.

All numbers come from official box scores and final scores. Season stats follow
WNBA convention: the All-Star Game and the Commissioner's Cup final do not count,
and regular season and playoffs are reported separately.
"""

from __future__ import annotations

import difflib
import unicodedata
from functools import lru_cache

import pandas as pd

from wnba_rag.games import resolve_team
from wnba_rag.stats import load_tables

PER_GAME_STATS = ["points", "rebounds", "assists", "steals", "blocks", "turnovers", "threes", "minutes"]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("'", "").replace(".", "").split())


@lru_cache(maxsize=1)
def _player_box() -> pd.DataFrame:
    """Box rows joined with game info: opponent, result, category, date."""
    t = load_tables()
    g = t.games
    b = t.box.merge(g[["game_id", "date", "category", "playoff_round", "series_game", "home", "away",
                       "home_score", "away_score"]], on="game_id")
    b["is_home"] = b["team"] == b["home"]
    b["opponent"] = b["away"].where(b["is_home"], b["home"])
    team_pts = b["home_score"].where(b["is_home"], b["away_score"])
    opp_pts = b["away_score"].where(b["is_home"], b["home_score"])
    b["result"] = (team_pts > opp_pts).map({True: "W", False: "L"})
    b["score"] = team_pts.astype(str) + "-" + opp_pts.astype(str)
    return b


def _filter_season(df: pd.DataFrame, season_type: str) -> pd.DataFrame:
    if season_type == "regular":
        return df[df["category"] == "regular"]
    if season_type == "playoffs":
        return df[df["category"] == "playoffs"]
    if season_type == "all":
        return df[df["category"].isin(["regular", "playoffs"])]
    raise ValueError("season_type must be regular, playoffs or all")


def resolve_player(name: str) -> dict:
    """Match a player name: exact, accent/case-insensitive, unique last name, or close spelling."""
    players = sorted(load_tables().box["player"].unique())
    by_norm = {_norm(p): p for p in players}
    key = _norm(name)
    if key in by_norm:
        return {"player": by_norm[key]}
    last = [p for p in players if _norm(p).split()[-1] == key.split()[-1]]
    if len(last) == 1 and (len(key.split()) == 1 or _norm(last[0]).split()[0][0] == key[0]):
        return {"player": last[0]}
    close = difflib.get_close_matches(key, list(by_norm), n=3, cutoff=0.8)
    if len(close) == 1:
        return {"player": by_norm[close[0]]}
    options = [by_norm[c] for c in close] or last
    return {"player": None, "did_you_mean": options[:5]}


def _team_games(team: str, season_type: str) -> pd.DataFrame:
    g = _filter_season(load_tables().games, season_type)
    g = g[(g["home"] == team) | (g["away"] == team)].copy()
    g["is_home"] = g["home"] == team
    g["opponent"] = g["away"].where(g["is_home"], g["home"])
    g["pts"] = g["home_score"].where(g["is_home"], g["away_score"])
    g["opp_pts"] = g["away_score"].where(g["is_home"], g["home_score"])
    g["result"] = (g["pts"] > g["opp_pts"]).map({True: "W", False: "L"})
    return g.sort_values("date")


# --------------------------------------------------------------------- tools

def team_record(team: str, season_type: str = "regular") -> dict:
    full = resolve_team(team)
    if not full:
        return {"found": False, "error": f"Unknown team '{team}'"}
    g = _team_games(full, season_type)
    if g.empty:
        return {"team": full, "season_type": season_type, "games": 0}
    w = int((g["result"] == "W").sum())
    home, away = g[g["is_home"]], g[~g["is_home"]]
    streak_res = g["result"].iloc[-1]
    streak = int((g["result"][::-1] != streak_res).cumsum().eq(0).sum())
    return {
        "team": full, "season_type": season_type, "wins": w, "losses": len(g) - w,
        "home_record": f"{(home['result'] == 'W').sum()}-{(home['result'] == 'L').sum()}",
        "away_record": f"{(away['result'] == 'W').sum()}-{(away['result'] == 'L').sum()}",
        "points_per_game": round(g["pts"].mean(), 1),
        "opp_points_per_game": round(g["opp_pts"].mean(), 1),
        "avg_point_diff": round((g["pts"] - g["opp_pts"]).mean(), 1),
        "last_10": f"{(g['result'].tail(10) == 'W').sum()}-{(g['result'].tail(10) == 'L').sum()}",
        "final_streak": f"{streak_res}{streak}",
    }


def standings() -> dict:
    """Regular-season standings, ranked by win percentage (ties by point differential)."""
    teams = sorted(set(_filter_season(load_tables().games, "regular")["home"]))
    rows = [team_record(t) for t in teams]
    rows.sort(key=lambda r: (r["wins"] / (r["wins"] + r["losses"]), r["avg_point_diff"]), reverse=True)
    top = rows[0]
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["games_back"] = ((top["wins"] - r["wins"]) + (r["losses"] - top["losses"])) / 2
    return {"standings": [{k: r[k] for k in ("rank", "team", "wins", "losses", "games_back",
                                            "avg_point_diff")} for r in rows]}


def player_season(player: str, season_type: str = "regular", opponent: str | None = None) -> dict:
    """Totals and per-game averages for a player, optionally only against one opponent."""
    res = resolve_player(player)
    if not res["player"]:
        return {"found": False, **res}
    b = _filter_season(_player_box(), season_type)
    b = b[b["player"] == res["player"]]
    if opponent:
        opp = resolve_team(opponent)
        if not opp:
            return {"found": False, "error": f"Unknown team '{opponent}'"}
        b = b[b["opponent"] == opp]
    if b.empty:
        return {"found": True, "player": res["player"], "games": 0, "season_type": season_type,
                "opponent": opponent}
    totals = {s: int(b[s].sum()) for s in PER_GAME_STATS}
    per_game = {s: round(b[s].mean(), 1) for s in PER_GAME_STATS}
    best = b.sort_values(["points", "date"], ascending=[False, True]).iloc[0]
    return {
        "found": True, "player": res["player"], "team(s)": sorted(b["team"].unique()),
        "season_type": season_type, "opponent": resolve_team(opponent) if opponent else None,
        "games": len(b), "record_in_those_games": f"{(b['result'] == 'W').sum()}-{(b['result'] == 'L').sum()}",
        "per_game": per_game, "totals": totals,
        "best_scoring_game": {"points": int(best["points"]), "date": best["date"], "opponent": best["opponent"],
                              "result": f"{best['result']} {best['score']}", "game_id": int(best["game_id"])},
    }


def player_game_log(player: str, season_type: str = "regular", opponent: str | None = None,
                    sort_by: str = "date", limit: int = 10) -> dict:
    """A player's individual games. sort_by: date, or a stat (points, rebounds, ...)
    to get her best games by that stat."""
    res = resolve_player(player)
    if not res["player"]:
        return {"found": False, **res}
    b = _filter_season(_player_box(), season_type)
    b = b[b["player"] == res["player"]]
    if opponent:
        b = b[b["opponent"] == resolve_team(opponent)]
    if sort_by == "date":
        b = b.sort_values("date")
    else:
        b = b.sort_values([sort_by, "date"], ascending=[False, True])
    cols = ["date", "game_id", "opponent", "result", "score", "minutes", "points", "rebounds",
            "assists", "steals", "blocks", "turnovers", "threes"]
    return {"found": True, "player": res["player"], "games_total": len(b),
            "games": b[cols].head(limit).to_dict(orient="records")}


def season_leaders(stat: str, per_game: bool = True, season_type: str = "regular",
                   team: str | None = None, min_games: int = 10, top_n: int = 10) -> dict:
    """League (or team) leaders for a stat. per_game=False ranks season totals.
    min_games is applied to per-game rankings (default 10; lowered automatically for playoffs)."""
    if stat not in PER_GAME_STATS:
        return {"error": f"stat must be one of {PER_GAME_STATS}"}
    b = _filter_season(_player_box(), season_type)
    if team:
        full = resolve_team(team)
        if not full:
            return {"error": f"Unknown team '{team}'"}
        b = b[b["team"] == full]
    if season_type == "playoffs":
        min_games = min(min_games, 2)
    agg = b.groupby("player").agg(games=("game_id", "nunique"), total=(stat, "sum"),
                                  avg=(stat, "mean"), team=("team", "last"))
    if per_game:
        agg = agg[agg["games"] >= min_games]
    key = "avg" if per_game else "total"
    agg = agg.sort_values(key, ascending=False).head(top_n).reset_index()
    agg["avg"] = agg["avg"].round(1)
    return {"stat": stat, "ranking": "per game" if per_game else "season total",
            "season_type": season_type, "team_filter": team,
            "min_games": min_games if per_game else None,
            "leaders": agg[["player", "team", "games", "total", "avg"]].to_dict(orient="records")}


def head_to_head(team_a: str, team_b: str, season_type: str = "all") -> dict:
    a, b = resolve_team(team_a), resolve_team(team_b)
    if not a or not b:
        return {"found": False, "error": "Unknown team"}
    g = _team_games(a, season_type)
    g = g[g["opponent"] == b]
    games = [{"date": r.date, "game_id": int(r.game_id), "category": r.category,
              "playoff_round": r.playoff_round if isinstance(r.playoff_round, str) else None,
              "home": r.home, "score": f"{a} {r.pts}, {b} {r.opp_pts}", "winner": a if r.result == "W" else b}
             for r in g.itertuples()]
    wins = sum(1 for x in games if x["winner"] == a)
    return {"teams": [a, b], "season_type": season_type,
            "series": f"{a} {wins}-{len(games) - wins}", "games": games}


def find_games_by(team: str | None = None, opponent: str | None = None,
                  playoff_round: str | None = None, series_game: int | None = None,
                  month: int | None = None) -> dict:
    """Look up games for 'Game 3 of the Finals', 'the Fever's games in June', etc.
    team may be omitted when the round identifies the series (e.g. the Finals)."""
    g = load_tables().games
    if team:
        full = resolve_team(team)
        if not full:
            return {"matches": [], "error": f"Unknown team '{team}'"}
        g = g[(g["home"] == full) | (g["away"] == full)]
    elif not playoff_round:
        return {"matches": [], "error": "Give a team or a playoff round"}
    if opponent:
        o = resolve_team(opponent)
        g = g[(g["home"] == o) | (g["away"] == o)]
    if playoff_round:
        g = g[g["playoff_round"].fillna("").str.lower() == playoff_round.lower()]
    if series_game:
        g = g[g["series_game"] == series_game]
    if month:
        g = g[pd.to_datetime(g["date"]).dt.month == month]
    return {"matches": [
        {"game_id": int(r.game_id), "date": r.date, "category": r.category,
         "playoff_round": r.playoff_round if isinstance(r.playoff_round, str) else None,
         "series_game": int(r.series_game) if pd.notna(r.series_game) else None,
         "matchup": f"{r.away} at {r.home}", "final": f"{r.away} {r.away_score}, {r.home} {r.home_score}"}
        for r in g.sort_values("date").itertuples()][:50]}
