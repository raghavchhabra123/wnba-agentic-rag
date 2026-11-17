"""Season tools: internal consistency and known 2025 facts."""

from wnba_rag import season
from wnba_rag.stats import load_tables


def test_game_categories():
    cats = load_tables().games["category"].value_counts().to_dict()
    assert cats == {"regular": 286, "playoffs": 24, "cup_final": 1, "all_star": 1}


def test_standings_are_consistent():
    rows = season.standings()["standings"]
    assert len(rows) == 13
    assert all(r["wins"] + r["losses"] == 44 for r in rows)
    assert sum(r["wins"] for r in rows) == sum(r["losses"] for r in rows)
    assert rows[0]["team"] == "Minnesota Lynx" and rows[0]["wins"] == 34


def test_finals_sweep():
    assert season.head_to_head("Aces", "Mercury", "playoffs")["series"] == "Las Vegas Aces 4-0"


def test_scoring_title():
    top = season.season_leaders("points")["leaders"][0]
    assert top["player"] == "A'ja Wilson" and top["avg"] == 23.4


def test_player_season_excludes_special_games():
    # The All-Star Game and the Commissioner's Cup final do not count toward season stats.
    log = season.player_game_log("Caitlin Clark", limit=100)["games"]
    special = set(load_tables().games.query("category in ['all_star', 'cup_final']")["game_id"])
    assert not special & {g["game_id"] for g in log}


def test_player_resolution():
    assert season.resolve_player("aja wilson")["player"] == "A'ja Wilson"
    assert season.resolve_player("Brittney Grinner")["player"] == "Brittney Griner"
    assert season.resolve_player("Smith")["player"] is None  # ambiguous surname


def test_find_finals_game_without_team():
    m = season.find_games_by(playoff_round="Finals", series_game=3)["matches"]
    assert len(m) == 1 and m[0]["game_id"] == 401820326
