"""Turn cached ESPN game summaries (data/raw/*.json) into three clean tables.

    data/processed/games.csv  one row per game: date, teams, final score, quarter scores
    data/processed/plays.csv  one row per play: clock, team, score, players involved
    data/processed/box.csv    official per-player box score (used as ground truth)

Usage:
    python -m wnba_rag.build_tables
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wnba_rag.paths import PROCESSED_DIR, RAW_DIR

SEASON_TYPES = {1: "preseason", 2: "regular", 3: "playoffs"}
BOX_COLUMNS = {
    "MIN": "minutes", "PTS": "points", "REB": "rebounds", "AST": "assists",
    "TO": "turnovers", "STL": "steals", "BLK": "blocks", "OREB": "off_rebounds",
    "DREB": "def_rebounds", "PF": "fouls",
}


def clock_to_seconds(clock: str) -> float:
    """'9:43' -> 583.0 ; '24.5' -> 24.5 (ESPN drops the minutes under one minute)."""
    clock = str(clock)
    if ":" in clock:
        minutes, seconds = clock.split(":")
        return int(minutes) * 60 + float(seconds)
    return float(clock)


def parse_game(path: Path) -> tuple[dict, list[dict], list[dict]]:
    data = json.loads(path.read_text())
    comp = data["header"]["competitions"][0]
    game_id = int(data["header"]["id"])

    teams = {}
    for c in comp["competitors"]:
        side = c["homeAway"]
        teams[side] = {
            "id": int(c["team"]["id"]),
            "name": c["team"]["displayName"],          # "Las Vegas Aces"
            "nickname": c["team"].get("name", ""),     # "Aces"
            "abbr": c["team"]["abbreviation"],         # "LV"
            "score": int(c["score"]),
            "quarters": [int(float(q.get("displayValue", 0))) for q in c.get("linescores", [])],
        }
    team_by_id = {t["id"]: t for t in teams.values()}

    season = data["header"].get("season", {})
    game = {
        "game_id": game_id,
        "date_utc": comp["date"],
        "season_type": SEASON_TYPES.get(season.get("type"), str(season.get("type"))),
        "home_id": teams["home"]["id"], "home": teams["home"]["name"],
        "home_nickname": teams["home"]["nickname"], "home_abbr": teams["home"]["abbr"],
        "away_id": teams["away"]["id"], "away": teams["away"]["name"],
        "away_nickname": teams["away"]["nickname"], "away_abbr": teams["away"]["abbr"],
        "home_score": teams["home"]["score"], "away_score": teams["away"]["score"],
        "home_quarters": json.dumps(teams["home"]["quarters"]),
        "away_quarters": json.dumps(teams["away"]["quarters"]),
    }

    # Athlete id -> (name, team) from the box score, used to label players in plays.
    athletes: dict[int, tuple[str, int]] = {}
    box_rows = []
    for team_block in data.get("boxscore", {}).get("players", []):
        team_id = int(team_block["team"]["id"])
        stats = team_block["statistics"][0]
        labels = stats["labels"]
        for a in stats["athletes"]:
            aid = int(a["athlete"]["id"])
            name = a["athlete"]["displayName"]
            athletes[aid] = (name, team_id)
            if a.get("didNotPlay") or not a.get("stats"):
                continue
            row = {"game_id": game_id, "player_id": aid, "player": name,
                   "team_id": team_id, "team": team_by_id[team_id]["name"]}
            for label, value in zip(labels, a["stats"]):
                if label == "3PT":  # "made-attempted", e.g. "3-7"
                    row["threes"] = int(value.split("-")[0]) if "-" in value else 0
                if label in BOX_COLUMNS:
                    try:
                        row[BOX_COLUMNS[label]] = int(value)
                    except ValueError:
                        row[BOX_COLUMNS[label]] = 0
            box_rows.append(row)

    play_rows = []
    for p in data.get("plays", []):
        clock = p.get("clock", {}).get("displayValue", "0:00")
        period = int(p["period"]["number"])
        remaining = clock_to_seconds(clock)
        period_len = 600 if period <= 4 else 300  # WNBA: 10-min quarters, 5-min OT
        # Seconds of game time before this period started.
        elapsed = (period - 1) * 600 if period <= 4 else 2400 + (period - 5) * 300
        team_id = int(p["team"]["id"]) if "team" in p else None
        participants = [int(x["athlete"]["id"]) for x in p.get("participants", [])]
        names = [athletes.get(pid, (None, None))[0] for pid in participants]
        play_rows.append({
            "game_id": game_id,
            "seq": int(p["sequenceNumber"]),
            "period": period,
            "clock": clock,
            "elapsed_sec": elapsed + (period_len - remaining),
            "team_id": team_id,
            "team": team_by_id[team_id]["name"] if team_id in team_by_id else None,
            "type": p["type"]["text"].replace("\n", " "),
            "text": p.get("text", ""),
            "scoring": bool(p.get("scoringPlay")),
            "score_value": int(p.get("scoreValue") or 0),
            "shooting": bool(p.get("shootingPlay")),
            "home_score": int(p.get("homeScore", 0)),
            "away_score": int(p.get("awayScore", 0)),
            "p1_id": participants[0] if len(participants) > 0 else None,
            "p1": names[0] if len(names) > 0 else None,
            "p2_id": participants[1] if len(participants) > 1 else None,
            "p2": names[1] if len(names) > 1 else None,
        })
    return game, play_rows, box_rows


ROUND_NAMES = ["First Round", "Semifinals", "Finals"]


def add_game_categories(games: pd.DataFrame) -> pd.DataFrame:
    """ESPN's summaries do not label special games, so derive:
    category: regular | all_star | cup_final | playoffs
      - All-Star: rosters named 'TEAM <captain>'
      - Commissioner's Cup final: the one regular-season game that is the 45th game
        for both teams (it does not count toward standings or season stats)
    playoff_round / series_game for playoff games: series are grouped by team pair
    and ordered by start date (4 first-round, 2 semifinal, 1 final series)."""
    g = games.copy()
    g["category"] = g["season_type"]
    all_star = g["home"].str.startswith("TEAM ") | g["away"].str.startswith("TEAM ")
    g.loc[all_star, "category"] = "all_star"

    reg = g[g["category"] == "regular"].sort_values("date")
    counts = pd.concat([reg["home"], reg["away"]]).value_counts()
    extra_teams = set(counts[counts > counts.mode().iloc[0]].index)
    if len(extra_teams) == 2:
        cand = reg[reg["home"].isin(extra_teams) & reg["away"].isin(extra_teams)]
        # The Cup final is played mid-season, at a neutral-style one-off date: take the
        # meeting whose date is closest to July 1 (the Cup final window).
        cup = cand.iloc[(pd.to_datetime(cand["date"]) - pd.Timestamp("2025-07-01")).abs().argsort()[:1]]
        g.loc[cup.index, "category"] = "cup_final"

    g["playoff_round"] = None
    g["series_game"] = pd.NA
    po = g[g["category"] == "playoffs"].copy()
    if not po.empty:
        po["pair"] = [" vs ".join(sorted([h, a])) for h, a in zip(po["home"], po["away"])]
        starts = po.groupby("pair")["date"].min().sort_values()
        start_dates = sorted(starts.unique())
        for pair, start in starts.items():
            rnd = ROUND_NAMES[min(start_dates.index(start), len(ROUND_NAMES) - 1)]
            idx = po[po["pair"] == pair].sort_values("date").index
            g.loc[idx, "playoff_round"] = rnd
            g.loc[idx, "series_game"] = range(1, len(idx) + 1)
    return g


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    games, plays, box = [], [], []
    for path in sorted(RAW_DIR.glob("*.json")):
        g, p, b = parse_game(path)
        games.append(g)
        plays.extend(p)
        box.extend(b)

    games_df = pd.DataFrame(games)
    games_df["date"] = (
        pd.to_datetime(games_df["date_utc"], utc=True)
        .dt.tz_convert("America/New_York").dt.date.astype(str)
    )  # ESPN dates are UTC; evening games would otherwise land on the next day
    games_df = add_game_categories(games_df)
    plays_df = pd.DataFrame(plays).sort_values(["game_id", "seq"])
    for col in ["team_id", "p1_id", "p2_id"]:
        plays_df[col] = plays_df[col].astype("Int64")
    box_df = pd.DataFrame(box)

    games_df.to_csv(PROCESSED_DIR / "games.csv", index=False)
    plays_df.to_csv(PROCESSED_DIR / "plays.csv", index=False)
    box_df.to_csv(PROCESSED_DIR / "box.csv", index=False)
    print(f"games={len(games_df)} plays={len(plays_df)} box_rows={len(box_df)}")
    print(games_df["season_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
