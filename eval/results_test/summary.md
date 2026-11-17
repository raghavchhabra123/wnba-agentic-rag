# Evaluation results: v1 test split

266 questions; deterministic grading (see eval/grade.py). Model: gpt-4o-mini for every system.

| System | Accuracy | Correct | Median latency (s) | Avg cost / question |
|---|---:|---:|---:|---:|
| llm_only | 0.0% | 0/266 | 1.4 | - |
| naive_rag | 23.7% | 63/266 | 1.7 | - |
| scoped_rag | 38.0% | 101/266 | 3.8 | - |
| agentic | 99.2% | 264/266 | 3.4 | $0.00066 |

## Accuracy by question type

| type | n | llm_only | naive_rag | scoped_rag | agentic |
|---|---:|---:|---:|---:|---:|
| final_score | 30 | 0% | 53% | 53% | 100% |
| first_three | 30 | 0% | 7% | 43% | 93% |
| largest_lead | 28 | 0% | 7% | 7% | 100% |
| last_points | 30 | 0% | 40% | 73% | 100% |
| player_points_box | 30 | 0% | 0% | 3% | 100% |
| q4_top_scorer | 30 | 0% | 30% | 33% | 100% |
| quarter_winner | 30 | 0% | 30% | 37% | 100% |
| second_half_turnovers | 28 | 0% | 32% | 39% | 100% |
| team_leader_box | 30 | 0% | 13% | 50% | 100% |

## Game resolution (single-game questions)

- agentic: resolved the correct game for 266/266 questions
- scoped_rag: resolved the correct game for 265/266 questions
