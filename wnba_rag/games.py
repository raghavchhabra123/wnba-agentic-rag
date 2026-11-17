"""Resolve a natural-language game reference to an ESPN game_id.

The original project asked about "the game" while indexing 312 games, so
retrieval mixed plays from different games. Every question now has to be pinned
to one game before any stats or retrieval happen.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd

from wnba_rag.stats import load_tables


def _team_aliases(games: pd.DataFrame) -> dict[str, str]:
    """Map lower-case aliases (full name, nickname, city, abbreviation) to full name.
    All-Star rosters ('TEAM CLARK') are matched only by their exact full name."""
    aliases: dict[str, str] = {}
    for side in ("home", "away"):
        cols = [side, f"{side}_nickname", f"{side}_abbr"]
        for full, nick, abbr in games[cols].drop_duplicates().itertuples(index=False):
            if str(full).startswith("TEAM "):
                aliases[full.lower()] = full
                continue
            nick = "" if pd.isna(nick) else str(nick)
            city = full[: -len(nick)].strip() if nick and full.endswith(nick) else ""
            for alias in (full, nick, abbr, city):
                alias = str(alias).strip().lower()
                if len(alias) >= 2:  # never index an empty alias: "" is inside every string
                    aliases[alias] = full
    return aliases


def resolve_team(name: str) -> str | None:
    """Resolve 'Aces', 'las vegas', 'LV', 'the suns' (plural of Sun) to a full team name.
    Matching is on whole words, never raw substrings."""
    if not name:
        return None
    games = load_tables().games
    aliases = _team_aliases(games)
    if name.strip().lower() in aliases:
        return aliases[name.strip().lower()]
    key = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    key = [w for w in key if w not in {"the", "team", "s"}]
    if not key:
        return None
    text = " ".join(key)
    candidates = {text, text.rstrip("s"), " ".join(w.rstrip("s") for w in key)}
    for c in candidates:
        if c in aliases:
            return aliases[c]
    hits = set()
    padded = f" {text} "
    for alias, full in aliases.items():
        if len(alias) < 3 or full.startswith("TEAM "):
            continue
        if f" {alias} " in padded or f" {alias}s " in padded:
            hits.add(full)
    if len(hits) == 1:
        return hits.pop()
    if hits:
        return None
    # Fan shorthand: a unique nickname prefix of 3+ letters ("lib", "valks").
    if len(key) == 1 and len(key[0].rstrip("s")) >= 3:
        stem = key[0].rstrip("s")
        nicknames = {str(n).lower(): f for f, n in zip(games["home"], games["home_nickname"])
                     if not str(f).startswith("TEAM ")}
        pref = {f for n, f in nicknames.items() if n.startswith(stem)}
        if len(pref) == 1:
            return pref.pop()
    return None


def find_games(team_a: str, team_b: str | None = None, game_date: str | None = None) -> dict:
    """Return matching games. game_date is YYYY-MM-DD (local US date); +/- 1 day is
    tolerated because people often use the UTC date shown on some sites."""
    games = load_tables().games
    full_a = resolve_team(team_a)
    if full_a is None:
        return {"matches": [], "error": f"Unknown team '{team_a}'", "teams": sorted(set(games["home"]))}
    mask = (games["home"] == full_a) | (games["away"] == full_a)
    if team_b:
        full_b = resolve_team(team_b)
        if full_b is None:
            return {"matches": [], "error": f"Unknown team '{team_b}'"}
        mask &= (games["home"] == full_b) | (games["away"] == full_b)
    candidates = games[mask]
    if game_date:
        d = date.fromisoformat(game_date)
        exact = candidates[candidates["date"] == d.isoformat()]
        if not exact.empty:
            candidates = exact
        else:
            near = {(d + timedelta(days=k)).isoformat() for k in (-1, 1)}
            candidates = candidates[candidates["date"].isin(near)]
    matches = [
        {"game_id": int(r.game_id), "date": r.date, "matchup": f"{r.away} at {r.home}",
         "final": f"{r.away} {r.away_score}, {r.home} {r.home_score}", "season_type": r.season_type}
        for r in candidates.sort_values("date").itertuples()
    ]
    return {"matches": matches}


def describe_game(game_id: int) -> str:
    g = load_tables().games.set_index("game_id").loc[game_id]
    return f"{g['away']} at {g['home']} on {g['date']}"
