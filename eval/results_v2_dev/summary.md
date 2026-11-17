# Evaluation results: v2 dev split

75 questions; deterministic grading (see eval/grade.py). Model: gpt-4o-mini for every system.

| System | Accuracy | Correct | Median latency (s) | Avg cost / question |
|---|---:|---:|---:|---:|
| llm_only | 2.7% | 2/75 | 1.1 | $0.00004 |
| naive_rag | 13.3% | 10/75 | 1.8 | $0.00039 |
| scoped_rag | 22.7% | 17/75 | 2.6 | $0.00077 |
| agentic | 100.0% | 75/75 | 3.8 | $0.00064 |

## Accuracy by category

| category | n | llm_only | naive_rag | scoped_rag | agentic |
|---|---:|---:|---:|---:|---:|
| abstain | 12 | 17% | 50% | 83% | 100% |
| casual_game | 30 | 0% | 10% | 23% | 100% |
| multi_step | 3 | 0% | 0% | 0% | 100% |
| season | 30 | 0% | 3% | 0% | 100% |

## Accuracy by question type

| type | n | llm_only | naive_rag | scoped_rag | agentic |
|---|---:|---:|---:|---:|---:|
| ambiguous_game | 5 | 0% | 0% | 100% | 100% |
| best_game_opponent_leader | 3 | 0% | 0% | 0% | 100% |
| final_score | 2 | 0% | 0% | 0% | 100% |
| first_three | 3 | 0% | 0% | 0% | 100% |
| head_to_head | 4 | 0% | 25% | 0% | 100% |
| largest_lead | 5 | 0% | 0% | 0% | 100% |
| last_points | 4 | 0% | 25% | 25% | 100% |
| league_leader | 3 | 0% | 0% | 0% | 100% |
| nonexistent_game | 2 | 100% | 50% | 100% | 100% |
| other_season | 5 | 0% | 100% | 60% | 100% |
| player_average | 5 | 0% | 0% | 0% | 100% |
| player_points_box | 5 | 0% | 0% | 0% | 100% |
| player_vs_opponent | 1 | 0% | 0% | 0% | 100% |
| playoff_game_top_scorer | 8 | 0% | 0% | 0% | 100% |
| q4_top_scorer | 4 | 0% | 0% | 50% | 100% |
| quarter_winner | 1 | 0% | 100% | 0% | 100% |
| second_half_turnovers | 4 | 0% | 25% | 75% | 100% |
| team_leader_box | 2 | 0% | 0% | 50% | 100% |
| team_leader_per_game | 4 | 0% | 0% | 0% | 100% |
| team_record | 5 | 0% | 0% | 0% | 100% |

## Game resolution (single-game questions)

- agentic: resolved the correct game for 30/30 questions
- scoped_rag: resolved the correct game for 28/30 questions
