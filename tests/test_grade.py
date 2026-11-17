import json

import pytest

from eval.grade import final_line, grade
from wnba_rag.paths import EVAL_DIR


@pytest.fixture(scope="module")
def items():
    return {q["type"]: q for q in json.loads((EVAL_DIR / "benchmark_dev.json").read_text())}


def test_final_line_extraction():
    assert final_line("blah\nFINAL ANSWER: Jane Doe, 21") == "Jane Doe, 21"
    assert final_line("no marker\nlast line") == "last line"


def test_player_answer_passes_and_hedging_fails(items):
    q = items["first_three"]
    gold = q["gold_names"][0]
    assert grade(q, f"Reasoning...\nFINAL ANSWER: {gold}")["pass"]
    # Naming a second player from the same game is treated as hedging.
    from eval.grade import _tables
    _, box = _tables()
    other = box[(box.game_id == q["game_id"]) & (box.player != gold)].iloc[0]["player"]
    assert not grade(q, f"FINAL ANSWER: {gold} or {other}")["pass"]


def test_numbers_required(items):
    q = items["final_score"]
    a, b = q["gold_numbers"]
    assert grade(q, f"FINAL ANSWER: {a}-{b}")["pass"]
    assert not grade(q, f"FINAL ANSWER: {a}-{b + 1}")["pass"]


def test_team_alias_accepted(items):
    q = items["second_half_turnovers"]
    nickname = q["gold_names"][0].split()[-1]
    assert grade(q, f"FINAL ANSWER: the {nickname}")["pass"]


def test_refusal_fails(items):
    q = items["last_points"]
    assert not grade(q, "I don't have data on that game.")["pass"]
