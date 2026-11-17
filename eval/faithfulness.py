"""Hallucination check for descriptive answers ("how did they win?").

These answers have no single gold string, so they are checked claim by claim:
  1. an LLM splits the answer into atomic factual claims,
  2. a judge checks every claim against the game's FULL play-by-play and box score
     (not just what the system retrieved), labelling it supported / contradicted /
     not_verifiable.

The judge is itself validated first (`--validate-judge`): it is given planted claims
built from the data, half true and half made false by changing one detail (a
number, a player, a quarter), and its accuracy on those is reported.

Usage:
    python -m eval.faithfulness --validate-judge
    python -m eval.faithfulness --systems naive_rag scoped_rag agentic --games 15
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pydantic import BaseModel

from wnba_rag.config import CHAT_MODEL, get_openai_client
from wnba_rag.paths import EVAL_DIR
from wnba_rag.stats import load_tables

JUDGE_MODEL = CHAT_MODEL  # gpt-4o-mini: the only chat model this API key can use
OUT_DIR = EVAL_DIR / "results_faithfulness"


class Claims(BaseModel):
    claims: list[str]


class Verdict(BaseModel):
    # Field order matters: the model writes the evidence BEFORE committing to a label.
    claim_id: int
    evidence: str
    label: str        # supported | contradicted | not_verifiable


class Verdicts(BaseModel):
    verdicts: list[Verdict]


def game_context(game_id: int) -> str:
    t = load_tables()
    g = t.games.set_index("game_id").loc[game_id]
    plays = t.plays[t.plays["game_id"] == game_id]
    box = t.box[t.box["game_id"] == game_id]
    hq, aq = json.loads(g["home_quarters"]), json.loads(g["away_quarters"])
    lines = [f"GAME: {g['away']} at {g['home']}, {g['date']}. Final: {g['away']} {g['away_score']}, "
             f"{g['home']} {g['home_score']}.",
             f"Points by quarter: {g['away']} {aq}; {g['home']} {hq}.", "", "BOX SCORE:"]
    for r in box.itertuples():
        lines.append(f"{r.player} ({r.team}): {r.minutes} min, {r.points} pts, {r.rebounds} reb, "
                     f"{r.assists} ast, {r.steals} stl, {r.blocks} blk, {r.turnovers} to, {r.fouls} pf, "
                     f"{r.threes} 3pm")
    lines += ["", f"PLAY-BY-PLAY (score shown as {g['away_abbr']}-{g['home_abbr']}):"]
    for p in plays.itertuples():
        if p.type == "Substitution":
            continue
        lines.append(f"Q{p.period} {p.clock} {p.text} ({p.away_score}-{p.home_score})")
    return "\n".join(lines)


def _parse(model_cls, messages, temperature: float = 0):
    client = get_openai_client()
    resp = client.beta.chat.completions.parse(model=JUDGE_MODEL, temperature=temperature,
                                              messages=messages, response_format=model_cls)
    return resp.choices[0].message.parsed


def extract_claims(answer: str) -> list[str]:
    body = answer.split("FINAL ANSWER")[0]
    return _parse(Claims, [{"role": "user", "content":
        "List every checkable factual claim in this basketball answer as short standalone "
        "sentences (who did what, when, scores, numbers, runs, leads). Skip opinions and vague "
        "statements ('played well', 'strong defense'). Drop chunk citations.\n\n" + body}]).claims


JUDGE_PROMPT = """You are fact-checking claims about one WNBA game against the official data below.
For each claim, answer:
  supported      - the data clearly shows it (small rounding is fine)
  contradicted   - the data shows something different (wrong number, player, quarter, team, order)
  not_verifiable - the data cannot confirm or refute it
For each claim id, first quote the deciding line(s) as evidence, then give the label that
the evidence supports. Return one verdict per claim id.

{context}

CLAIMS:
{claims}"""


def judge(game_id: int, claims: list[str]) -> list[Verdict]:
    if not claims:
        return []
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    out = _parse(Verdicts, [{"role": "user", "content":
                             JUDGE_PROMPT.format(context=game_context(game_id), claims=numbered)}]).verdicts
    # Align by claim id, never by position; a claim the judge skipped is not_verifiable.
    by_id = {v.claim_id: v for v in out}
    return [by_id.get(i + 1, Verdict(claim_id=i + 1, evidence="(judge returned no verdict)",
                                     label="not_verifiable")) for i in range(len(claims))]


# ------------------------------------------------------------------ judge validation

def planted_claims(n_games: int = 20, seed: int = 3) -> list[dict]:
    """True claims from the data, and false twins with one detail changed."""
    t = load_tables()
    rng = random.Random(seed)
    games = t.games[t.games["category"] == "regular"].sample(n_games, random_state=seed)
    out = []
    for g in games.itertuples():
        box = t.box[(t.box["game_id"] == g.game_id) & (t.box["points"] >= 6)]
        r = box.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
        out.append({"game_id": g.game_id, "claim": f"{r['player']} scored {r['points']} points.", "truth": True})
        out.append({"game_id": g.game_id, "claim": f"{r['player']} scored {r['points'] + rng.choice([-4, -3, 3, 5])} points.",
                    "truth": False})
        made = t.plays[(t.plays["game_id"] == g.game_id) & t.plays["scoring"] & (t.plays["score_value"] >= 2)]
        p = made.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
        out.append({"game_id": g.game_id, "claim": f"{p['p1']} made a basket with {p['clock']} left in quarter {p['period']}.",
                    "truth": True})
        wrong_q = rng.choice([q for q in (1, 2, 3, 4) if q != p["period"]])
        hit = made[(made["p1"] == p["p1"]) & (made["period"] == wrong_q) & (made["clock"] == p["clock"])]
        if hit.empty:
            out.append({"game_id": g.game_id, "claim": f"{p['p1']} made a basket with {p['clock']} left in quarter {wrong_q}.",
                        "truth": False})
        winner, loser = (g.home, g.away) if g.home_score > g.away_score else (g.away, g.home)
        out.append({"game_id": g.game_id, "claim": f"The {winner} beat the {loser}.", "truth": True})
        out.append({"game_id": g.game_id, "claim": f"The {loser} beat the {winner}.", "truth": False})
    return out


def validate_judge() -> dict:
    items = planted_claims()
    by_game: dict[int, list[dict]] = {}
    for it in items:
        by_game.setdefault(it["game_id"], []).append(it)

    def run(gid_items):
        gid, its = gid_items
        verdicts = judge(gid, [i["claim"] for i in its])
        return [(i, v.label) for i, v in zip(its, verdicts)]

    with ThreadPoolExecutor(8) as pool:
        results = [x for chunk in pool.map(run, by_game.items()) for x in chunk]
    rows = pd.DataFrame([{"truth": i["truth"], "label": lab} for i, lab in results])
    tp = ((rows.truth) & (rows.label == "supported")).sum()
    tn = ((~rows.truth) & (rows.label == "contradicted")).sum()
    summary = {
        "planted_claims": len(rows),
        "true_claims_marked_supported": f"{tp}/{int(rows.truth.sum())}",
        "false_claims_marked_contradicted": f"{tn}/{int((~rows.truth).sum())}",
        "false_claims_marked_supported": int(((~rows.truth) & (rows.label == "supported")).sum()),
        "accuracy": round((tp + tn) / len(rows), 3),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "judge_validation.json").write_text(json.dumps(summary, indent=2, default=int))
    return summary


# ------------------------------------------------------------------ main eval

def descriptive_questions(n_games: int, seed: int = 21) -> list[dict]:
    """Held-out style: games not in the v1 dev split."""
    t = load_tables()
    dev_games = {q["game_id"] for q in json.loads((EVAL_DIR / "benchmark_dev.json").read_text())}
    pool = t.games[(t.games["category"].isin(["regular", "playoffs"])) & ~t.games["game_id"].isin(dev_games)]
    games = pool.sample(n_games, random_state=seed)
    rng = random.Random(seed)
    qs = []
    for g in games.itertuples():
        d = pd.Timestamp(g.date)
        ref = f"the {g.away_nickname} at {g.home_nickname} game on {d.strftime('%B')} {d.day}, {d.year}"
        winner = g.home_nickname if g.home_score > g.away_score else g.away_nickname
        box = t.box[t.box["game_id"] == g.game_id].sort_values("points", ascending=False)
        star = box.iloc[rng.randint(0, 2)]["player"]
        qs += [
            {"game_id": int(g.game_id), "question": f"How did the {winner} win {ref}?"},
            {"game_id": int(g.game_id), "question": f"What happened in the fourth quarter of {ref}?"},
            {"game_id": int(g.game_id), "question": f"Describe {star}'s performance in {ref}."},
        ]
    return qs


def evaluate(systems: list[str], n_games: int) -> str:
    from wnba_rag.baselines import SYSTEMS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qs = descriptive_questions(n_games)
    rows = []
    for name in systems:
        path = OUT_DIR / f"{name}.jsonl"
        done = {json.loads(l)["question"] for l in path.read_text().splitlines()} if path.exists() else set()

        def one(q):
            ans = SYSTEMS[name](q["question"])["answer"]
            claims = extract_claims(ans)
            verdicts = judge(q["game_id"], claims)
            return {**q, "system": name, "answer": ans,
                    "verdicts": [v.model_dump() for v in verdicts]}

        todo = [q for q in qs if q["question"] not in done]
        print(f"{name}: {len(todo)} to run", flush=True)
        with path.open("a") as f, ThreadPoolExecutor(6) as pool:
            for rec in pool.map(one, todo):
                f.write(json.dumps(rec) + "\n")
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            for v in rec["verdicts"]:
                rows.append({"system": name, "question": rec["question"], "label": v["label"]})

    df = pd.DataFrame(rows)
    tab = df.groupby("system")["label"].value_counts(normalize=False).unstack(fill_value=0)
    for col in ("supported", "contradicted", "not_verifiable"):
        if col not in tab:
            tab[col] = 0
    tab["claims"] = tab[["supported", "contradicted", "not_verifiable"]].sum(axis=1)
    per_q = df.groupby(["system", "question"])["label"].apply(lambda s: (s == "contradicted").any())
    lines = ["# Descriptive-answer faithfulness", "",
             f"{len(qs)} descriptive questions over {n_games} games. Claims checked against the full "
             "play-by-play and box score by an LLM judge (validated, see judge_validation.json).", "",
             "| System | Claims | Supported | Contradicted | Not verifiable | Answers with ≥1 false claim |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in [x for x in systems if x in tab.index]:
        r = tab.loc[s]
        lines.append(f"| {s} | {r['claims']} | {r['supported'] / r['claims']:.1%} | "
                     f"{r['contradicted'] / r['claims']:.1%} | {r['not_verifiable'] / r['claims']:.1%} | "
                     f"{per_q[s].mean():.0%} |")
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-judge", action="store_true")
    ap.add_argument("--systems", nargs="+", default=["naive_rag", "scoped_rag", "agentic"])
    ap.add_argument("--games", type=int, default=15)
    args = ap.parse_args()
    if args.validate_judge:
        print(json.dumps(validate_judge(), indent=2, default=int))
    else:
        print(evaluate(args.systems, args.games))


if __name__ == "__main__":
    main()
