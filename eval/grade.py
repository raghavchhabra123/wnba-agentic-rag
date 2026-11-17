"""Deterministic grading (no LLM judge).

Answer questions: only the 'FINAL ANSWER:' line is graded. It passes when
  * at least one gold player/team name appears (ties: any tied player counts),
  * every gold number appears (decimals within 0.05, e.g. 23.4 per game), and
  * for player questions, no other player is named (listing candidates to hedge fails).
Player names are checked against the game's roster for single-game questions and
against the whole league for season questions.

Abstain questions ('expect' = clarify | decline): the whole reply is checked.
  clarify -> must ask which game, and must not name a player as the answer
  decline -> must say the data does not cover it / the game does not exist, and the
             FINAL ANSWER line must not contain a stat (years are allowed)
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

from wnba_rag.paths import PROCESSED_DIR

CLARIFY = ["which game", "which one", "which specific", "specify", "clarif", "more details",
           "more information", "which of", "need clarification", "provide the date", "which aces",
           "please name", "can you tell me which"]
DECLINE = ["not cover", "doesn't cover", "does not cover", "don't have", "do not have", "no data",
           "not available", "couldn't find", "could not find", "no game", "not in the data",
           "only covers", "unable", "no record", "not include", "doesn't include", "no such",
           "did not play", "didn't play", "there was no", "not find", "no information",
           "isn't in", "is not in", "not part of", "cannot", "can't", "only includes",
           "only contains", "outside", "not available"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9.' ]+", " ", s)


def final_line(answer: str) -> str:
    m = re.findall(r"final answer\s*[:\-]\s*(.+)", answer or "", flags=re.I)
    if m:
        return m[-1]
    lines = [ln for ln in (answer or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


@lru_cache(maxsize=1)
def _tables():
    return (pd.read_csv(PROCESSED_DIR / "games.csv").set_index("game_id"),
            pd.read_csv(PROCESSED_DIR / "box.csv"))


@lru_cache(maxsize=1)
def _team_aliases() -> dict[str, set[str]]:
    games, _ = _tables()
    out: dict[str, set[str]] = {}
    for side in ("home", "away"):
        for full, nick, abbr in games[[side, f"{side}_nickname", f"{side}_abbr"]].drop_duplicates().itertuples(index=False):
            city = full[: -len(nick)].strip() if full.endswith(nick) else full
            out.setdefault(full, set()).update({full, nick, abbr, city})
    return out


def _mentioned(phrase: str, text: str) -> bool:
    return f" {norm(phrase).strip()} " in f" {norm(text)} "


def _name_mentioned(name: str, text: str, surname_unique: bool) -> bool:
    if _mentioned(name, text):
        return True
    return surname_unique and _mentioned(norm(name).split()[-1], text)


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\d.])\d+(?:\.\d+)?", text)]


def _number_present(gold: float, found: list[float]) -> bool:
    if float(gold).is_integer():
        return any(abs(f - gold) < 1e-9 for f in found)
    return any(abs(f - gold) <= 0.051 for f in found)


def grade(item: dict, answer: str) -> dict:
    if item.get("expect"):
        return _grade_abstain(item, answer)

    games, box = _tables()
    line = final_line(answer)
    gid = item.get("game_id")
    if gid is not None and not pd.isna(gid):
        players = box[box["game_id"] == int(gid)]["player"].unique().tolist()
    else:
        players = box["player"].unique().tolist()
    surnames = pd.Series([norm(p).split()[-1] for p in players]).value_counts()

    def uniq(p: str) -> bool:
        return surnames.get(norm(p).split()[-1], 0) == 1

    aliases = _team_aliases()
    names_ok, extra = True, []
    if item["gold_names"]:
        if all(n in aliases for n in item["gold_names"]):
            names_ok = any(any(_mentioned(a, line) for a in aliases[n]) for n in item["gold_names"])
        else:
            names_ok = any(_name_mentioned(n, line, uniq(n)) for n in item["gold_names"])
            extra = [p for p in players if p not in item["gold_names"] and _name_mentioned(p, line, uniq(p))]
    found = _numbers(line)
    numbers_ok = all(_number_present(float(n), found) for n in item["gold_numbers"])
    passed = names_ok and numbers_ok and not extra
    return {"pass": passed, "names_ok": names_ok, "numbers_ok": numbers_ok,
            "extra_players": extra, "graded_line": line}


def _grade_abstain(item: dict, answer: str) -> dict:
    text = norm(answer)
    line = final_line(answer)
    if item["expect"] == "clarify":
        asked = any(p in text for p in CLARIFY)
        _, box = _tables()
        named = [p for p in box["player"].unique() if _mentioned(p, line)]
        return {"pass": asked and not named, "asked": asked, "named_players": named[:3],
                "graded_line": line}
    declined = any(p in text for p in DECLINE)
    stats_in_line = [n for n in _numbers(line) if not (1990 <= n <= 2030)]
    return {"pass": declined and not stats_in_line, "declined": declined,
            "stats_in_line": stats_in_line, "graded_line": line}
