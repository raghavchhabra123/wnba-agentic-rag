"""Check the play-by-play stat engine against ESPN's official box scores.

The agent trusts these tools for every counting question, so they are tested
on all 312 games (~5,900 player-game rows), not a handful of examples.
Run: pytest -q
"""

import pandas as pd
import pytest

from wnba_rag import stats
from wnba_rag.games import find_games, resolve_team

# Minimum share of player-game rows where the play-by-play count equals the
# official box score. Residual misses are ESPN logging gaps (e.g. an assist
# credited in the box score but not written into the play text).
THRESHOLDS = {"points": 1.0, "rebounds": 0.999, "assists": 0.998, "steals": 0.998,
              "blocks": 0.999, "turnovers": 0.99, "fouls": 0.998, "threes": 0.999}


def agreement(stat: str) -> float:
    t = stats.load_tables()
    credits = stats._credits(t.plays, stat)
    credits = credits[credits["player_id"].notna()]
    pbp = credits.groupby(["game_id", "player_id"])["value"].sum().rename("pbp")
    merged = t.box.set_index(["game_id", "player_id"])[[stat]].join(pbp).fillna(0)
    return float((merged[stat] == merged["pbp"]).mean())


@pytest.mark.parametrize("stat", THRESHOLDS)
def test_player_stats_match_box_score(stat):
    assert agreement(stat) >= THRESHOLDS[stat]


def test_team_points_match_final_score():
    games = stats.load_tables().games
    for g in games.itertuples():
        totals = stats.team_totals(g.game_id, "points")["totals"]
        assert totals[g.home] == g.home_score
        assert totals[g.away] == g.away_score


def test_quarter_points_match_official_linescore():
    games = stats.load_tables().games.sample(40, random_state=0)
    for g in games.itertuples():
        summary = stats.game_summary(g.game_id)["points_by_period"]
        for q in range(1, 5):
            totals = stats.team_totals(g.game_id, "points", [q])["totals"]
            assert totals[g.home] == summary[g.home][q - 1]
            assert totals[g.away] == summary[g.away][q - 1]


def test_team_resolution():
    assert resolve_team("Aces") == "Las Vegas Aces"
    assert resolve_team("las vegas") == "Las Vegas Aces"
    assert resolve_team("LV") == "Las Vegas Aces"
    assert resolve_team("the Fever") == "Indiana Fever"
    assert resolve_team("Atlantis") is None
    # Regressions found in the v2 dev set:
    assert resolve_team("suns") == "Connecticut Sun"          # plural of a nickname
    assert resolve_team("Kelsey Mitchell") is None            # a player is not a team
    assert resolve_team("Caitlin Clark's team") is None       # (empty alias bug matched this)
    assert resolve_team("TEAM CLARK") == "TEAM CLARK"         # All-Star roster, exact only
    assert resolve_team("lib") == "New York Liberty"          # unique nickname prefix
    assert resolve_team("valks") == "Golden State Valkyries"


def test_find_game_by_teams_and_date():
    games = stats.load_tables().games
    g = games.iloc[0]
    hits = find_games(g["home_nickname"], g["away_nickname"], g["date"])["matches"]
    assert [h["game_id"] for h in hits] == [g["game_id"]]


def test_scoring_runs_sum_is_consistent():
    gid = int(stats.load_tables().games.iloc[0]["game_id"])
    runs = stats.scoring_runs(gid, top_n=1000)["largest_runs"]
    total = sum(r["points"] for r in runs)
    g = stats.game_row(gid)
    assert total == g["home_score"] + g["away_score"]
