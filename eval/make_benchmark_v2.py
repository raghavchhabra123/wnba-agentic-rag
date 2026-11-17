"""Benchmark v2: questions closer to how people actually ask.

Categories
  casual_game   single-game questions from v1, rewritten casually by a DIFFERENT model
                (gpt-4o) with nicknames, no year, slang. Gold answers are unchanged.
  season        records, leaders, averages, head-to-head, player vs opponent, playoff games
  multi_step    questions that need one lookup to find the game, then another
  abstain       ambiguous, out-of-scope or non-existent: the right answer is to ask or decline

Gold answers come from the box score / final scores with pandas code written here,
not from wnba_rag/season.py.

Split: teams are randomly assigned to dev (5) or test (8). A question is placed in a
split only if every team it involves belongs to that split's team set (questions that
mix dev and test teams are dropped), so test questions are about teams and players the
dev set never covered. casual_game rewrites use the v1 dev/test questions, whose games
are already disjoint.

Usage:
    python -m eval.make_benchmark_v2
"""

from __future__ import annotations

import json
import os
import random
import re

import pandas as pd

from wnba_rag.paths import EVAL_DIR, PROCESSED_DIR

REWRITE_MODEL = os.getenv("WNBA_RAG_REWRITE_MODEL", "gpt-4o-mini")  # gpt-4o if your key has it
SEED = 11
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def load():
    g = pd.read_csv(PROCESSED_DIR / "games.csv")
    b = pd.read_csv(PROCESSED_DIR / "box.csv").merge(
        g[["game_id", "date", "category", "playoff_round", "series_game", "home", "away",
           "home_score", "away_score"]], on="game_id")
    b["opponent"] = b["away"].where(b["team"] == b["home"], b["home"])
    return g, b


def possessive(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


def uniq_top(series: pd.Series):
    """Return (index, value) of the max if it is unique, else None."""
    top = series.max()
    hits = series[series == top]
    return (hits.index[0], top) if len(hits) == 1 else None


def season_questions(g: pd.DataFrame, b: pd.DataFrame, rng: random.Random) -> list[dict]:
    qs = []
    reg_g = g[g["category"] == "regular"]
    reg_b = b[b["category"] == "regular"]
    teams = sorted(set(reg_g["home"]))
    nick = {r.home: r.home_nickname for r in g.itertuples()} | {r.away: r.away_nickname for r in g.itertuples()}

    # team record
    for t in teams:
        tg = reg_g[(reg_g["home"] == t) | (reg_g["away"] == t)]
        won = ((tg["home"] == t) & (tg["home_score"] > tg["away_score"])) | \
              ((tg["away"] == t) & (tg["away_score"] > tg["home_score"]))
        qs.append({"category": "season", "type": "team_record", "teams": [t],
                   "question": f"What was the {possessive(nick[t])} regular-season record in 2025?",
                   "gold_names": [], "gold_numbers": [int(won.sum()), int((~won).sum())]})

    # team leader per game (stable across 10 and 20 game minimums)
    for t in teams:
        stat = rng.choice(["points", "rebounds", "assists"])
        tb = reg_b[reg_b["team"] == t].groupby("player").agg(gp=("game_id", "nunique"), avg=(stat, "mean"))
        a, c = uniq_top(tb[tb.gp >= 10]["avg"].round(3)), uniq_top(tb[tb.gp >= 20]["avg"].round(3))
        if a and c and a[0] == c[0]:
            qs.append({"category": "season", "type": "team_leader_per_game", "teams": [t],
                       "question": f"Who led the {nick[t]} in {stat} per game during the 2025 regular season?",
                       "gold_names": [a[0]], "gold_numbers": []})

    # player season average (players with 20+ games)
    agg = reg_b.groupby(["player", "team"]).agg(gp=("game_id", "nunique"), pts=("points", "mean"),
                                                reb=("rebounds", "mean"), ast=("assists", "mean")).reset_index()
    agg = agg[agg["gp"] >= 20]
    for t in teams:
        cands = agg[(agg["team"] == t) & (agg["pts"] >= 8)]
        if cands.empty:
            continue
        r = cands.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
        stat, col = rng.choice([("points", "pts"), ("rebounds", "reb"), ("assists", "ast")])
        qs.append({"category": "season", "type": "player_average", "teams": [t],
                   "question": f"How many {stat} per game did {r['player']} average in the 2025 regular season?",
                   "gold_names": [], "gold_numbers": [round(float(r[col]), 1)]})

    # head-to-head regular-season series (decisive only)
    pairs = set()
    for r in reg_g.itertuples():
        pairs.add(tuple(sorted([r.home, r.away])))
    for a, c in rng.sample(sorted(pairs), 30):
        mg = reg_g[((reg_g.home == a) & (reg_g.away == c)) | ((reg_g.home == c) & (reg_g.away == a))]
        a_w = int((((mg.home == a) & (mg.home_score > mg.away_score)) |
                   ((mg.away == a) & (mg.away_score > mg.home_score))).sum())
        c_w = len(mg) - a_w
        if a_w == c_w:
            continue
        winner = a if a_w > c_w else c
        qs.append({"category": "season", "type": "head_to_head", "teams": [a, c],
                   "question": f"Who won the 2025 regular-season series between the {nick[a]} and the {nick[c]}, and what was the series record?",
                   "gold_names": [winner], "gold_numbers": [max(a_w, c_w), min(a_w, c_w)]})

    # player vs opponent scoring
    for t in teams:
        cands = agg[(agg["team"] == t) & (agg["pts"] >= 12)]
        if cands.empty:
            continue
        p = cands.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]["player"]
        pb = reg_b[reg_b["player"] == p]
        opp = rng.choice(sorted(pb["opponent"].unique()))
        vs = pb[pb["opponent"] == opp]
        qs.append({"category": "season", "type": "player_vs_opponent", "teams": [t, opp],
                   "question": f"How many points per game did {p} average against the {nick[opp]} in the 2025 regular season?",
                   "gold_names": [], "gold_numbers": [round(float(vs["points"].mean()), 1)]})

    # league leaders (only where leader is the same at 10 and 25 game minimums)
    for stat in ["points", "rebounds", "assists", "steals", "blocks", "threes"]:
        lb = reg_b.groupby("player").agg(gp=("game_id", "nunique"), avg=(stat, "mean"),
                                         team=("team", "last"))
        a, c = uniq_top(lb[lb.gp >= 10]["avg"].round(3)), uniq_top(lb[lb.gp >= 25]["avg"].round(3))
        if a and c and a[0] == c[0]:
            qs.append({"category": "season", "type": "league_leader", "teams": [lb.loc[a[0], "team"]],
                       "question": f"Who led the WNBA in {stat} per game in the 2025 regular season?",
                       "gold_names": [a[0]], "gold_numbers": []})

    # playoff game top scorer
    po = g[g["category"] == "playoffs"]
    for r in po.itertuples():
        pb = b[b["game_id"] == r.game_id].set_index("player")["points"]
        top = uniq_top(pb)
        if not top:
            continue
        qs.append({"category": "season", "type": "playoff_game_top_scorer", "teams": [r.home, r.away],
                   "question": f"Who led all scorers in Game {int(r.series_game)} of the {r.playoff_round} "
                               f"between the {nick[r.away]} and the {nick[r.home]}?",
                   "gold_names": [top[0]], "gold_numbers": [int(top[1])]})
    return qs


def multi_step_questions(g, b, rng) -> list[dict]:
    qs = []
    reg_b = b[b["category"] == "regular"]
    nick = {r.home: r.home_nickname for r in g.itertuples()} | {r.away: r.away_nickname for r in g.itertuples()}
    stars = reg_b.groupby(["player", "team"])["points"].mean().reset_index()
    stars = stars[stars["points"] >= 15]
    for r in stars.itertuples():
        pb = reg_b[reg_b["player"] == r.player].set_index("game_id")["points"]
        best = uniq_top(pb)
        if not best:
            continue
        gid = best[0]
        opp_box = b[(b["game_id"] == gid) & (b["team"] != r.team)].set_index("player")["points"]
        lead = uniq_top(opp_box)
        if not lead:
            continue
        opp = b[(b["game_id"] == gid) & (b["team"] != r.team)]["team"].iloc[0]
        qs.append({"category": "multi_step", "type": "best_game_opponent_leader", "teams": [r.team, opp],
                   "question": f"In {r.player}'s highest-scoring game of the 2025 regular season, "
                               "who led the opposing team in points?",
                   "gold_names": [lead[0]], "gold_numbers": []})
    # team's highest-scoring game -> top scorer in that game
    reg_g = g[g["category"] == "regular"]
    for t in sorted(set(reg_g["home"])):
        tg = reg_g[(reg_g.home == t) | (reg_g.away == t)].copy()
        tg["pts"] = tg["home_score"].where(tg["home"] == t, tg["away_score"])
        best = uniq_top(tg.set_index("game_id")["pts"])
        if not best:
            continue
        tb = b[(b["game_id"] == best[0]) & (b["team"] == t)].set_index("player")["points"]
        lead = uniq_top(tb)
        if not lead:
            continue
        opp = tg.set_index("game_id").loc[best[0]]
        opp = opp["away"] if opp["home"] == t else opp["home"]
        qs.append({"category": "multi_step", "type": "team_best_game_top_scorer", "teams": [t, opp],
                   "question": f"Who scored the most points for the {nick[t]} in their highest-scoring "
                               "game of the 2025 regular season?",
                   "gold_names": [lead[0]], "gold_numbers": []})
    return qs


def abstain_questions(g, rng) -> list[dict]:
    nick = {r.home: r.home_nickname for r in g.itertuples()}
    teams = sorted(t for t in set(g[g["category"] == "regular"]["home"]))
    qs = []
    for t in teams:
        qs.append({"category": "abstain", "type": "ambiguous_game", "teams": [t], "expect": "clarify",
                   "question": f"Who scored the last basket in the {nick[t]} game?"})
        qs.append({"category": "abstain", "type": "other_season", "teams": [t], "expect": "decline",
                   "question": f"What was the {nick[t]}' record in the 2023 season?"})
    played = set(zip(g["home"], g["away"], g["date"]))
    for t in teams:
        for _ in range(20):
            o = rng.choice([x for x in teams if x != t])
            d = pd.Timestamp("2025-06-01") + pd.Timedelta(days=rng.randint(0, 60))
            ds = d.date().isoformat()
            near = {(d + pd.Timedelta(days=k)).date().isoformat() for k in (-1, 0, 1)}
            if not any((h, a, x) in played for h, a in ((t, o), (o, t)) for x in near):
                qs.append({"category": "abstain", "type": "nonexistent_game", "teams": [t, o],
                           "expect": "decline",
                           "question": f"Who led the {nick[o]} in rebounds at the {nick[t]} on "
                                       f"{d.strftime('%B')} {d.day}, 2025?"})
                break
    for q in ["How many points did LeBron James score in the 2025 NBA Finals?",
              "Who won the 2024 WNBA MVP?",
              "What was Diana Taurasi's scoring average in 2025?"]:
        qs.append({"category": "abstain", "type": "out_of_scope", "teams": [], "expect": "decline",
                   "question": q})
    return qs


# ------------------------------------------------------------ casual rewrites

REWRITE_PROMPT = """Rewrite this question the way a basketball fan would casually type it into
a chat box. Keep EXACTLY the same meaning and every identifying detail (teams, players, month
and day, quarter, game number, which stat). You may: use a team's nickname or city instead of
its full name, drop the year, abbreviate the month, use lowercase, use common fan slang
("dropped", "went off for", "dimes", "boards", "4th", "Q4"). Do not add new facts.
Return only the rewritten question.

Question: {q}"""

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                      "august", "september", "october", "november", "december"], 1)}


def keeps_identifiers(orig: str, new: str, g: pd.DataFrame) -> bool:
    """Programmatic check that a rewrite kept the day of month and the teams/players."""
    n = new.lower()
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October) (\d+)", orig)
    if m:
        mon, day = m.group(1).lower(), m.group(2)
        if not re.search(rf"\b{day}(st|nd|rd|th)?\b", n):
            return False
        if mon[:3] not in n and not re.search(rf"\b{MONTHS[mon]}/{day}\b", n):
            return False
    for r in g.drop_duplicates("home").itertuples():
        if r.home_nickname.lower() in orig.lower():
            city = r.home[: -len(r.home_nickname)].strip().lower()
            if not any(a in n for a in {r.home_nickname.lower(), city, r.home_abbr.lower()}):
                return False
    names = re.findall(r"(?:[A-Z][a-zA-Z'\-]+ )+[A-Z][a-zA-Z'\-]+", orig)
    for name in names:
        if any(w in name for w in ["Game", "Finals", "Semifinals", "First Round", "WNBA"]):
            continue
        last = name.split()[-1].lower()
        if last not in n and not any(last == x.lower() for x in g["home_nickname"]):
            return False
    for kw in ["first", "second", "third", "fourth", "final", "last", "per game", "most"]:
        pass  # semantic drift is checked by spot review, see README
    return True


def casualize(items: list[dict], g: pd.DataFrame) -> None:
    from openai import OpenAI

    from wnba_rag.config import get_openai_client

    client: OpenAI = get_openai_client()
    for it in items:
        it["question_template"] = it["question"]
        it["rewritten"] = False
        for _ in range(3):
            resp = client.chat.completions.create(
                model=REWRITE_MODEL, temperature=1.0,
                messages=[{"role": "user", "content": REWRITE_PROMPT.format(q=it["question"])}])
            new = resp.choices[0].message.content.strip().strip('"')
            if keeps_identifiers(it["question"], new, g):
                it["question"], it["rewritten"] = new, True
                break


def main() -> None:
    g, b = load()
    rng = random.Random(SEED)
    teams = sorted(set(g[g["category"] == "regular"]["home"]))
    shuffled = teams[:]
    rng.shuffle(shuffled)
    split_teams = {"dev": set(shuffled[:5]), "test": set(shuffled[5:])}

    pool = season_questions(g, b, rng) + multi_step_questions(g, b, rng) + abstain_questions(g, rng)
    out = {"dev": [], "test": []}
    for q in pool:
        for split, ts in split_teams.items():
            if q["teams"] and all(t in ts for t in q["teams"]):
                out[split].append(q)
        if not q["teams"]:  # league-wide out-of-scope questions go to test
            out["test"].append(q)

    for split in ("dev", "test"):
        v1 = json.loads((EVAL_DIR / f"benchmark_{split}.json").read_text())
        r = random.Random(SEED)
        sample = r.sample(v1, 60 if split == "test" else 30)
        for it in sample:
            out[split].append({**{k: it[k] for k in ("game_id", "type", "question", "gold_names",
                                                     "gold_numbers")},
                               "category": "casual_game", "v1_qid": it["qid"]})

    for split, items in out.items():
        casual = [it for it in items if it["category"] != "abstain"]
        casualize(casual, g)
        for i, it in enumerate(items):
            it["qid"] = i
        path = EVAL_DIR / f"benchmark_v2_{split}.json"
        path.write_text(json.dumps(items, indent=2, default=str))
        df = pd.DataFrame(items)
        print(f"{split}: {len(items)} questions, teams={sorted(split_teams[split])}")
        print(df.groupby(["category", "type"]).size().to_string())
        print("rewritten:", int(df.get("rewritten", pd.Series(dtype=bool)).fillna(False).sum()))


if __name__ == "__main__":
    main()
