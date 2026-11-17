# Evaluation results: v2 test split

140 questions; deterministic grading (see eval/grade.py). Model: gpt-4o-mini for every system.

| System | Accuracy | Correct | Median latency (s) | Avg cost / question |
|---|---:|---:|---:|---:|
| agentic | 95.7% | 134/140 | 3.5 | $0.00076 |

## Accuracy by category

| category | n | agentic |
|---|---:|---:|
| abstain | 22 | 95% |
| casual_game | 60 | 97% |
| multi_step | 12 | 75% |
| season | 46 | 100% |

## Accuracy by question type

| type | n | agentic |
|---|---:|---:|
| ambiguous_game | 8 | 100% |
| best_game_opponent_leader | 9 | 100% |
| final_score | 9 | 100% |
| first_three | 9 | 89% |
| head_to_head | 8 | 100% |
| largest_lead | 10 | 100% |
| last_points | 2 | 50% |
| league_leader | 3 | 100% |
| nonexistent_game | 3 | 100% |
| other_season | 8 | 100% |
| out_of_scope | 3 | 67% |
| player_average | 8 | 100% |
| player_points_box | 7 | 100% |
| player_vs_opponent | 6 | 100% |
| playoff_game_top_scorer | 5 | 100% |
| q4_top_scorer | 4 | 100% |
| quarter_winner | 9 | 100% |
| second_half_turnovers | 5 | 100% |
| team_best_game_top_scorer | 3 | 0% |
| team_leader_box | 5 | 100% |
| team_leader_per_game | 8 | 100% |
| team_record | 8 | 100% |

## Game resolution (single-game questions)

- agentic: resolved the correct game for 59/60 questions
