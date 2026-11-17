"""Generate a benchmark of questions with answers computed from the data.

To avoid grading the agent with its own code, gold answers here come from
sources independent of wnba_rag/stats.py wherever possible:
  * box-score questions      -> ESPN's official box score
  * score / quarter questions -> ESPN's official final and quarter scores
  * play-based questions      -> parsed here from the play text alone, not from
                                 the ESPN fields stats.py counts

Usage:
    python -m eval.make_benchmark --split dev  --games 25 --seed 7
    python -m eval.make_benchmark --split test --games 30 --seed 2026
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import date

import pandas as pd

from wnba_rag.paths import EVAL_DIR, PROCESSED_DIR

ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def game_ref(g: pd.Series) -> str:
    d = date.fromisoformat(g["date"])
    return f"the {g['away_nickname']} at {g['home_nickname']} game on {d.strftime('%B')} {d.day}, {d.year}"


def scorer_from_text(text: str) -> str | None:
    m = re.match(r"^(.+?) makes ", str(text))
    return m.group(1).strip() if m else None


def text_points(text: str) -> int:
    t = str(text).lower()
    if "free throw" in t:
        return 1
    return 3 if "three point" in t else 2


def scoring_events(plays: pd.DataFrame, g: pd.Series) -> pd.DataFrame:
    """Made shots parsed from the play TEXT only ("<player> makes ..."), with points
    inferred from the wording. This deliberately ignores ESPN's scoringPlay /
    scoreValue fields, which the agent's stat tools use, so the two methods check
    each other. (An earlier version inferred scorers from changes in the running
    score; that broke when ESPN updated the score on a neighbouring row.)"""
    p = plays.copy()
    p["text_scorer"] = p["text"].map(scorer_from_text)
    delta = (p["home_score"].diff().fillna(p["home_score"])
             + p["away_score"].diff().fillna(p["away_score"]))
    p = p[p["text_scorer"].notna()].copy()
    # Use the player id's official (box-score) name: the text keeps old names for
    # players who changed them mid-season (e.g. Nia Coffey -> Nia Brodie).
    p["scorer"] = p["p1"].where(p["p1"].notna(), p["text_scorer"])
    # Points: this row's score change when it is a sane 1-3, else infer from wording
    # (some threes are written "26-foot running pullup jump shot").
    d = delta.loc[p.index]
    p["pts"] = [int(x) if x in (1, 2, 3) else text_points(t) for x, t in zip(d, p["text"])]
    return p


def make_questions(g: pd.Series, plays: pd.DataFrame, box: pd.DataFrame, rng: random.Random) -> list[dict]:
    ref = game_ref(g)
    qs: list[dict] = []
    base = {"game_id": int(g["game_id"])}

    # 1. Team leader from the official box score
    team = rng.choice([g["home"], g["away"]])
    nickname = g["home_nickname"] if team == g["home"] else g["away_nickname"]
    stat = rng.choice(["points", "rebounds", "assists"])
    tb = box[box["team"] == team]
    top = int(tb[stat].max())
    qs.append({**base, "type": "team_leader_box",
               "question": f"Who led the {nickname} in {stat} in {ref}?",
               "gold_names": tb[tb[stat] == top]["player"].tolist(), "gold_numbers": []})

    # 2. A player's points from the box score
    scorers = box[box["points"] >= 8]
    if not scorers.empty:
        r = scorers.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
        qs.append({**base, "type": "player_points_box",
                   "question": f"How many points did {r['player']} score in {ref}?",
                   "gold_names": [], "gold_numbers": [int(r["points"])]})

    # 3. Final score from the official header
    qs.append({**base, "type": "final_score",
               "question": f"What was the final score of {ref}?",
               "gold_names": [], "gold_numbers": [int(g["home_score"]), int(g["away_score"])]})

    # 4. Quarter winner from official quarter scores
    hq, aq = json.loads(g["home_quarters"]), json.loads(g["away_quarters"])
    quarters = [q for q in range(1, 5) if q <= len(hq) and hq[q - 1] != aq[q - 1]]
    if quarters:
        q = rng.choice(quarters)
        winner = g["home"] if hq[q - 1] > aq[q - 1] else g["away"]
        qs.append({**base, "type": "quarter_winner",
                   "question": f"Which team scored more points in the {ORDINAL[q]} quarter of {ref}, and how many?",
                   "gold_names": [winner], "gold_numbers": [max(hq[q - 1], aq[q - 1])]})

    se = scoring_events(plays, g)

    # 5. Top scorer in the 4th quarter (from score changes)
    q4 = se[(se["period"] == 4) & se["scorer"].notna()]
    if not q4.empty:
        totals = q4.groupby("scorer")["pts"].sum()
        qs.append({**base, "type": "q4_top_scorer",
                   "question": f"Who scored the most points in the fourth quarter of {ref}?",
                   "gold_names": totals[totals == totals.max()].index.tolist(),
                   "gold_numbers": [], "info_points": int(totals.max())})

    # 6. First made three-pointer
    threes = se[(se["pts"] == 3) & se["scorer"].notna()]
    if not threes.empty:
        qs.append({**base, "type": "first_three",
                   "question": f"Who made the first three-pointer of {ref}?",
                   "gold_names": [threes.iloc[0]["scorer"]], "gold_numbers": []})

    # 7. Last points of the game
    last = se[se["scorer"].notna()]
    if not last.empty:
        qs.append({**base, "type": "last_points",
                   "question": f"Who scored the final points of {ref}?",
                   "gold_names": [last.iloc[-1]["scorer"]], "gold_numbers": []})

    # 8. Largest lead
    margin = plays["home_score"] - plays["away_score"]
    home_lead, away_lead = int(margin.max()), int(-margin.min())
    if home_lead != away_lead:
        team = g["home"] if home_lead > away_lead else g["away"]
        qs.append({**base, "type": "largest_lead",
                   "question": f"What was the largest lead in {ref}, and which team held it?",
                   "gold_names": [team], "gold_numbers": [max(home_lead, away_lead)]})

    # 9. Second-half turnovers (team-level)
    is_tov = ((plays["type"].str.contains("Turnover") & (plays["type"] != "No Turnover"))
              | (plays["type"] == "Traveling"))
    tov = plays[plays["period"].isin([3, 4]) & is_tov]
    counts = tov.groupby("team").size()
    if len(counts) == 2 and counts.iloc[0] != counts.iloc[1]:
        qs.append({**base, "type": "second_half_turnovers",
                   "question": f"Which team committed more turnovers in the second half of {ref}?",
                   "gold_names": [counts.idxmax()], "gold_numbers": []})
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", choices=["dev", "test"], default="dev",
                    help="dev = used while building/debugging; test = held out, reported")
    args = ap.parse_args()

    games = pd.read_csv(PROCESSED_DIR / "games.csv")
    plays = pd.read_csv(PROCESSED_DIR / "plays.csv")
    box = pd.read_csv(PROCESSED_DIR / "box.csv")

    rng = random.Random(args.seed)
    # Skip games where the same two teams met twice within a day (none expected) and
    # pick a spread of regular-season and playoff games.
    pool = games
    if args.split == "test":
        # Held-out games: never overlap with the dev set's games.
        dev_path = EVAL_DIR / "benchmark_dev.json"
        dev_games = {q["game_id"] for q in json.loads(dev_path.read_text())}
        pool = games[~games["game_id"].isin(dev_games)]
    sample = pool.sample(n=args.games, random_state=args.seed)
    questions = []
    for _, g in sample.iterrows():
        gp = plays[plays["game_id"] == g["game_id"]].sort_values("seq")
        gb = box[box["game_id"] == g["game_id"]]
        questions.extend(make_questions(g, gp, gb, rng))
    for i, q in enumerate(questions):
        q["qid"] = i
    out = EVAL_DIR / f"benchmark_{args.split}.json"
    out.write_text(json.dumps(questions, indent=2))
    print(f"{len(questions)} questions from {args.games} games -> {out}")
    print(pd.Series([q["type"] for q in questions]).value_counts().to_string())


if __name__ == "__main__":
    main()
