# Evaluation results: v2 test split

140 questions; deterministic grading (see eval/grade.py). Model: gpt-4o-mini for every system.

| System | Accuracy | Correct | Median latency (s) | Avg cost / question |
|---|---:|---:|---:|---:|
| llm_only | 5.7% | 8/140 | 1.0 | $0.00004 |
| naive_rag | 18.6% | 26/140 | 1.6 | $0.00038 |
| scoped_rag | 20.0% | 28/140 | 1.7 | $0.00064 |
| agentic | 96.4% | 135/140 | 3.4 | $0.00071 |

## Accuracy by category

| category | n | llm_only | naive_rag | scoped_rag | agentic |
|---|---:|---:|---:|---:|---:|
| abstain | 22 | 36% | 64% | 50% | 95% |
| casual_game | 60 | 0% | 17% | 28% | 97% |
| multi_step | 12 | 0% | 0% | 0% | 83% |
| season | 46 | 0% | 4% | 0% | 100% |

## Accuracy by question type

| type | n | llm_only | naive_rag | scoped_rag | agentic |
|---|---:|---:|---:|---:|---:|
| ambiguous_game | 8 | 0% | 0% | 100% | 100% |
| best_game_opponent_leader | 9 | 0% | 0% | 0% | 100% |
| final_score | 9 | 0% | 44% | 56% | 100% |
| first_three | 9 | 0% | 11% | 56% | 89% |
| head_to_head | 8 | 0% | 25% | 0% | 100% |
| largest_lead | 10 | 0% | 10% | 10% | 100% |
| last_points | 2 | 0% | 0% | 0% | 50% |
| league_leader | 3 | 0% | 0% | 0% | 100% |
| nonexistent_game | 3 | 100% | 100% | 100% | 100% |
| other_season | 8 | 25% | 100% | 0% | 100% |
| out_of_scope | 3 | 100% | 100% | 0% | 67% |
| player_average | 8 | 0% | 0% | 0% | 100% |
| player_points_box | 7 | 0% | 0% | 0% | 100% |
| player_vs_opponent | 6 | 0% | 0% | 0% | 100% |
| playoff_game_top_scorer | 5 | 0% | 0% | 0% | 100% |
| q4_top_scorer | 4 | 0% | 0% | 25% | 100% |
| quarter_winner | 9 | 0% | 33% | 33% | 100% |
| second_half_turnovers | 5 | 0% | 20% | 20% | 100% |
| team_best_game_top_scorer | 3 | 0% | 0% | 0% | 33% |
| team_leader_box | 5 | 0% | 0% | 20% | 100% |
| team_leader_per_game | 8 | 0% | 0% | 0% | 100% |
| team_record | 8 | 0% | 0% | 0% | 100% |

## Game resolution (single-game questions)

- agentic: resolved the correct game for 59/60 questions
- scoped_rag: resolved the correct game for 59/60 questions
