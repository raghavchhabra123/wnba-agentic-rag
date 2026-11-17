# Evaluation results (test split)

266 questions from 30 games; deterministic grading (see eval/grade.py). Model: gpt-4o-mini for every system.

| System | Accuracy | Correct | Median latency (s) |
|---|---:|---:|---:|
| llm_only | 0.0% | 0/266 | 1.4 |
| naive_rag | 23.3% | 62/266 | 1.7 |
| scoped_rag | 38.0% | 101/266 | 3.8 |
| agentic | 99.6% | 265/266 | 3.1 |

## Accuracy by question type

| Question type | llm_only | naive_rag | scoped_rag | agentic |
|---|---:|---:|---:|---:|
| final_score | 0% | 53% | 53% | 100% |
| first_three | 0% | 7% | 43% | 97% |
| largest_lead | 0% | 7% | 7% | 100% |
| last_points | 0% | 40% | 73% | 100% |
| player_points_box | 0% | 0% | 3% | 100% |
| q4_top_scorer | 0% | 30% | 33% | 100% |
| quarter_winner | 0% | 30% | 37% | 100% |
| second_half_turnovers | 0% | 29% | 39% | 100% |
| team_leader_box | 0% | 13% | 50% | 100% |

## Game resolution

- agentic: resolved the correct game for 266/266 questions
- scoped_rag: resolved the correct game for 265/266 questions
